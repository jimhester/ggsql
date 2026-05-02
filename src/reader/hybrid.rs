//! Hybrid reader: data source + in-process DuckDB staging.
//!
//! Wraps a primary [`Reader`] (e.g. a remote analytic backend) and a staging
//! [`DuckDBReader`]. [`Reader::register`] writes to staging; [`Reader::execute_sql`]
//! routes queries that reference registered names to staging, everything else to
//! the primary data source.
//!
//! Designed for backends where `register()` is unavailable (read-only Flight SQL
//! servers, anonymous Trino, etc.) or where round-tripping during visualization
//! iteration is wasteful. Staging in a local DuckDB sidesteps both: the primary
//! query runs against the remote source; subsequent `register()`-based machinery
//! (stat transforms, layer filters, temp-table DDL) runs against in-process DuckDB.
//!
//! # Known limitations
//!
//! A single SQL statement cannot reference both staged names and primary-data
//! tables. Queries are dispatched whole to either staging or the primary backend
//! based on whether they mention a staged name, so cross-backend joins are not
//! supported; materialize one side into staging first if you need to combine them.
//!
//! Staged data lives in the in-process DuckDB instance and is released when the
//! `HybridReader` is dropped. There is no spill-to-disk and no shared cache across
//! readers — plan staging volume against available RAM.
//!
//! All internally-generated SQL (stat transforms, layer filters, temp-table DDL)
//! is emitted in DuckDB dialect, which is why [`Reader::dialect`] on `HybridReader`
//! returns staging's dialect. That is the correct choice for queries over staged
//! data; when you need SQL targeted at the remote source (e.g. schema introspection
//! of the remote catalog), use [`HybridReader::data_dialect`] instead.
//!
//! # Query-Result Cache
//!
//! `HybridReader::execute_sql` memoizes remote query results in the staging
//! DuckDB under hashed table names (`__ggsql_cache_<hex>`). Repeat calls
//! with identical `(reader_uri, sql)` within the configured TTL return
//! cached data in <1 ms instead of a round-trip to the primary reader.
//!
//! - Cache is enabled by default; set `GGSQL_HYBRID_CACHE_DISABLED=1` to
//!   turn it off, or construct via `HybridReader::with_cache_config` to
//!   supply a custom `CacheConfig` (TTL, byte budget).
//! - Eviction is LRU by last-access once the cumulative `byte_estimate`
//!   of entries exceeds `CacheConfig::max_bytes` (default 512 MB).
//! - Manual invalidation: `HybridReader::clear_cache()` on the Rust side,
//!   `-- @uncache` meta-command in the Jupyter kernel.
//! - Scope: the cache lives in the staging DuckDB instance owned by a
//!   single `HybridReader`. No cross-session/disk persistence in v1.

use crate::reader::hybrid_cache::{self, CacheConfig};
use crate::reader::{DuckDBReader, Reader, Spec, SqlDialect};
use crate::{DataFrame, Result};
use std::cell::RefCell;
use std::collections::HashSet;

pub struct HybridReader {
    data: Box<dyn Reader + Send>,
    staging: DuckDBReader,
    staged_names: RefCell<HashSet<String>>,
    cache: CacheConfig,
}

impl HybridReader {
    /// Construct a `HybridReader` from a primary data reader and a staging
    /// DuckDB instance, with default cache config (enabled unless
    /// `GGSQL_HYBRID_CACHE_DISABLED=1` is set).
    pub fn new(data: Box<dyn Reader + Send>, staging: DuckDBReader) -> Self {
        Self::with_cache_config(data, staging, CacheConfig::default())
    }

    /// Construct with caller-supplied cache config. Ensures the meta
    /// table exists eagerly so the first remote query doesn't pay the
    /// schema-creation cost. Ignoring errors here is intentional: if
    /// meta-init fails, subsequent cache ops will fail too and bubble up.
    pub fn with_cache_config(
        data: Box<dyn Reader + Send>,
        staging: DuckDBReader,
        cache: CacheConfig,
    ) -> Self {
        if cache.enabled {
            let _ = hybrid_cache::ensure_meta(&staging);
        }
        Self {
            data,
            staging,
            staged_names: RefCell::new(HashSet::new()),
            cache,
        }
    }

    /// Read-only access to this reader's cache configuration.
    pub fn cache_config(&self) -> &CacheConfig {
        &self.cache
    }

    /// Dialect of the primary data backend. Useful for SQL targeted at the
    /// remote source rather than the staging DuckDB (e.g. schema introspection
    /// of the remote catalog).
    pub fn data_dialect(&self) -> &dyn SqlDialect {
        self.data.dialect()
    }
}

impl Reader for HybridReader {
    fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
        use crate::naming::quote_ident;

        let names = self.staged_names.borrow();
        if references_staged_name(sql, &names) {
            return self.staging.execute_sql(sql);
        }
        drop(names);

        // Cache disabled → direct passthrough.
        if !self.cache.enabled {
            return self.data.execute_sql(sql);
        }

        // We need a reader URI for the cache key. The Reader trait doesn't
        // expose one — we use a fixed placeholder per HybridReader instance
        // (each instance has its own staging DuckDB namespace, so keys don't
        // need to cross-collide between instances).
        let reader_uri = "hybrid-primary";
        let key = hybrid_cache::cache_key(reader_uri, sql);
        let table = hybrid_cache::cache_table_name(&key);

        // Hit: meta row exists AND within TTL AND table exists.
        if let Some(entry) = hybrid_cache::lookup(&self.staging, &key)? {
            let age_ms = hybrid_cache::now_ms() - entry.fetched_at_epoch_ms;
            let ttl_ms = (self.cache.ttl_secs as i64) * 1000;
            // Strict `<` so that ttl=0 always misses, even when racing within
            // the same millisecond (age_ms = 0, ttl_ms = 0).
            if age_ms < ttl_ms {
                let select = format!("SELECT * FROM {}", quote_ident(&table));
                match self.staging.execute_sql(&select) {
                    Ok(df) => {
                        let _ = hybrid_cache::touch(&self.staging, &key);
                        return Ok(df);
                    }
                    Err(_) => {
                        // Cache table vanished (manual drop, crash mid-insert).
                        // Fall through to miss path.
                        let _ = hybrid_cache::drop_entry(&self.staging, &key);
                    }
                }
            } else {
                // Stale — evict and refetch.
                let _ = hybrid_cache::drop_entry(&self.staging, &key);
            }
        }

        // Miss (or stale after eviction) — fetch from primary and stage.
        let df = self.data.execute_sql(sql)?;
        let row_count = df.height() as i64;
        let byte_estimate = estimate_bytes(&df);

        // DuckDB's `arrow(...)` table function (used by `register`) rejects
        // schemas with zero columns. The viz pipeline occasionally issues
        // metadata-only queries that produce empty results; cache those by
        // value instead of by table.
        if df.width() == 0 {
            return Ok(df);
        }

        self.staging.register(&table, df.clone(), true)?;
        if let Err(e) = hybrid_cache::insert_meta(
            &self.staging,
            &key,
            reader_uri,
            sql,
            row_count,
            byte_estimate,
        ) {
            // insert_meta failed AFTER register succeeded — clean up the
            // orphan cache table so it doesn't linger forever without a
            // tracking meta row (which would prevent it from ever being
            // evicted or garbage-collected).
            let _ = self
                .staging
                .execute_sql(&format!("DROP TABLE IF EXISTS {}", quote_ident(&table)));
            return Err(e);
        }
        // Eviction is bookkeeping — if it hiccups (transient DuckDB error,
        // internal SQL bug), the user's data is already cached and the
        // primary fetch succeeded. Don't surface the failure as a query
        // error; just log and continue.
        if let Err(e) = hybrid_cache::evict_over_budget(&self.staging, self.cache.max_bytes) {
            eprintln!("ggsql: cache eviction failed (non-fatal): {e}");
        }
        Ok(df)
    }

    fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
        self.staging.register(name, df, replace)?;
        self.staged_names.borrow_mut().insert(name.to_string());
        Ok(())
    }

    fn unregister(&self, name: &str) -> Result<()> {
        self.staging.unregister(name)?;
        self.staged_names.borrow_mut().remove(name);
        Ok(())
    }

    fn execute(&self, query: &str) -> Result<Spec> {
        crate::reader::execute_with_reader(self, query)
    }

    fn dialect(&self) -> &dyn SqlDialect {
        // All generated SQL (stats, layer filters, temp-table DDL) targets
        // the staging backend, so return the staging dialect. Callers that
        // need the primary-data dialect (e.g. schema introspection of the
        // remote catalog) can access it via `HybridReader::data_dialect()`.
        self.staging.dialect()
    }

    fn clear_cache(&self) -> Result<()> {
        if self.cache.enabled {
            hybrid_cache::clear_all(&self.staging)?;
        }
        Ok(())
    }
}

fn estimate_bytes(df: &DataFrame) -> i64 {
    df.get_columns()
        .iter()
        .map(|col| col.get_array_memory_size())
        .sum::<usize>() as i64
}

/// Check whether `sql` references any name in `staged_names` as a SQL
/// identifier (not as part of a longer identifier, and not inside a
/// single-quoted string literal).
///
/// Matches when the name appears bare (`SELECT * FROM orders`), as a
/// double-quoted identifier (`FROM "orders"`), or adjacent to a qualified
/// prefix (`FROM catalog.schema.orders`). Does **not** match substrings of
/// longer identifiers (`orders_detail`) or string-literal contents
/// (`'orders of magnitude'`).
///
/// This is deliberately a lightweight scanner — it doesn't fully parse SQL.
/// False-positive cases we accept:
/// - Backslash-escaped quotes inside string literals (SQL standard escapes
///   a single quote as `''`, which we do handle).
/// - Comments containing what looks like an identifier: a primary-data
///   query whose only mention of a staged name is inside a SQL comment
///   will be misrouted to staging and fail with a clear error rather than
///   succeeding against the primary backend.
fn references_staged_name(sql: &str, staged_names: &HashSet<String>) -> bool {
    staged_names
        .iter()
        .any(|name| sql_references_identifier(sql, name))
}

fn sql_references_identifier(sql: &str, name: &str) -> bool {
    let bytes = sql.as_bytes();
    let name_bytes = name.as_bytes();
    let n = name_bytes.len();
    if n == 0 {
        return false;
    }
    let mut i = 0;
    while i + n <= bytes.len() {
        if &bytes[i..i + n] == name_bytes {
            let before_ok = i == 0 || !is_identifier_byte(bytes[i - 1]);
            let after_ok = i + n == bytes.len() || !is_identifier_byte(bytes[i + n]);
            if before_ok && after_ok && !is_inside_string_literal(bytes, i) {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Walk from start to `pos` tracking whether we're inside a single-quoted
/// string literal. SQL-standard doubled-single-quote (`''`) is an escape
/// that keeps us inside the literal.
fn is_inside_string_literal(bytes: &[u8], pos: usize) -> bool {
    let mut inside = false;
    let mut i = 0;
    while i < pos && i < bytes.len() {
        if bytes[i] == b'\'' {
            if inside && i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                i += 2;
                continue;
            }
            inside = !inside;
        }
        i += 1;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_references_staged_name_empty_set() {
        let set = HashSet::new();
        assert!(!references_staged_name("SELECT * FROM foo", &set));
    }

    #[test]
    fn test_references_staged_name_no_match() {
        let mut set = HashSet::new();
        set.insert("__ggsql_global_abc123__".to_string());
        assert!(!references_staged_name(
            "SELECT * FROM iceberg.dse.foo",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_match() {
        let mut set = HashSet::new();
        set.insert("__ggsql_global_abc123__".to_string());
        assert!(references_staged_name(
            "SELECT * FROM __ggsql_global_abc123__ WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_rejects_longer_identifier() {
        // The query references `orders_detail`, NOT `orders`. Must not route.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_staged_name(
            "SELECT * FROM orders_detail WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_rejects_prefix_of_longer_identifier() {
        // The name is `col`; query uses `col_id`. Must not route.
        let mut set = HashSet::new();
        set.insert("col".to_string());
        assert!(!references_staged_name("SELECT col_id FROM users", &set));
    }

    #[test]
    fn test_references_staged_name_rejects_inside_string_literal() {
        // `orders` appears only inside a string literal. Must not route.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(!references_staged_name(
            "SELECT 'orders of magnitude' AS label",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_matches_quoted_identifier() {
        // Double-quoted identifier — our boundary check lets this through
        // because `"` is not an identifier char.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(r#"SELECT * FROM "orders""#, &set));
    }

    #[test]
    fn test_references_staged_name_matches_qualified_reference() {
        // `catalog.schema.orders` — the dot is a non-identifier byte, so
        // `orders` at the end still matches.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(
            "SELECT * FROM catalog.schema.orders WHERE x > 1",
            &set
        ));
    }

    #[test]
    fn test_references_staged_name_handles_escaped_quotes_in_literal() {
        // SQL-standard '' is an escaped quote inside a string literal, so
        // the staged name appearing after should still be detected as
        // outside any literal.
        let mut set = HashSet::new();
        set.insert("orders".to_string());
        assert!(references_staged_name(
            "SELECT 'it''s fine' FROM orders",
            &set
        ));
    }

    #[test]
    fn test_register_delegates_to_staging_and_tracks_name() {
        use crate::df;
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        let df = df! { "x" => vec![1_i64, 2, 3] }.unwrap();
        reader.register("my_table", df, true).unwrap();

        // The name is tracked so subsequent queries route correctly.
        assert!(reader.staged_names.borrow().contains("my_table"));
    }

    #[test]
    fn test_execute_sql_routes_staged_queries_to_staging() {
        use crate::array_util::as_i64;
        use crate::df;
        // Make the data reader a DuckDB that does NOT have the table.
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        let df = df! { "x" => vec![1_i64, 2, 3] }.unwrap();
        reader.register("my_table", df, true).unwrap();

        // Query referencing the registered name routes to staging (which has it)
        let result = reader
            .execute_sql("SELECT COUNT(*) AS n FROM my_table")
            .unwrap();
        let n = as_i64(result.column("n").unwrap()).unwrap().value(0);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_execute_sql_routes_unstaged_queries_to_data() {
        use crate::array_util::as_i64;
        use crate::df;
        // Data reader has a distinctive table; staging is empty.
        let data_reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_reader
            .register("data_table", df! { "y" => vec![42_i64] }.unwrap(), true)
            .unwrap();

        let data = Box::new(data_reader) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        // Nothing registered in staging; query for `data_table` must hit data reader
        let result = reader.execute_sql("SELECT y FROM data_table").unwrap();
        let y = as_i64(result.column("y").unwrap()).unwrap().value(0);
        assert_eq!(y, 42);
    }

    #[test]
    fn test_unregister_delegates_to_staging_and_untracks() {
        use crate::df;
        let data = Box::new(DuckDBReader::from_connection_string("duckdb://memory").unwrap())
            as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        reader
            .register("tmp", df! { "x" => vec![1_i64] }.unwrap(), true)
            .unwrap();
        assert!(reader.staged_names.borrow().contains("tmp"));

        reader.unregister("tmp").unwrap();
        assert!(!reader.staged_names.borrow().contains("tmp"));
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn test_dialect_returns_staging_not_data() {
        use crate::reader::SqliteReader;
        // Use a SqliteReader on the data side so the data dialect (SQLite,
        // CASE-WHEN fallback for sql_greatest) differs from the staging
        // dialect (DuckDB, native GREATEST). This way the test would fail if
        // the impl returned the data dialect by mistake.
        let data = Box::new(SqliteReader::new().unwrap()) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        // dialect() returns the staging (DuckDB) dialect.
        let greatest = reader.dialect().sql_greatest(&["a", "b"]);
        assert_eq!(greatest, "GREATEST(a, b)");

        // data_dialect() returns the data-side (SQLite) dialect, whose
        // sql_greatest falls back to a portable CASE form.
        let data_greatest = reader.data_dialect().sql_greatest(&["a", "b"]);
        assert_ne!(data_greatest, "GREATEST(a, b)");
        assert!(
            data_greatest.contains("CASE"),
            "expected SQLite's CASE fallback, got: {data_greatest}"
        );
    }

    #[test]
    fn test_query_referencing_both_staged_and_remote_routes_to_staging() {
        use crate::df;
        // Primary has `remote_only` and ALSO `staged_only` (with different
        // values from the staging copy). Staging only has `staged_only`. If
        // the router incorrectly sent the query to the data side, the join
        // would succeed against the primary's two tables. Since routing must
        // pick staging on `staged_only`, the query fails because staging
        // lacks `remote_only`.
        let data_reader = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_reader
            .register(
                "remote_only",
                df! { "y" => vec![10_i64, 20] }.unwrap(),
                true,
            )
            .unwrap();
        data_reader
            .register(
                "staged_only",
                df! { "x" => vec![999_i64, 999] }.unwrap(),
                true,
            )
            .unwrap();
        let data = Box::new(data_reader) as Box<dyn Reader + Send>;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let reader = HybridReader::new(data, staging);

        reader
            .register("staged_only", df! { "x" => vec![1_i64, 2] }.unwrap(), true)
            .unwrap();

        // Query references BOTH names. Routing matches on `staged_only`, so the
        // whole query goes to staging — which doesn't have `remote_only`. The
        // wrong-route case (data side) would silently succeed because the
        // primary has both tables. So `is_err()` plus a staging-side error
        // message mentioning `remote_only` confirms correct routing.
        let result = reader.execute_sql("SELECT s.x, r.y FROM staged_only s, remote_only r");
        assert!(
            result.is_err(),
            "cross-side query must error when staging lacks the remote table"
        );
        let err_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            err_msg.contains("remote_only"),
            "expected staging-side error mentioning the missing `remote_only` table; got: {err_msg}"
        );
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A reader that wraps an inner `DuckDBReader` and increments a shared
    /// counter on every `execute_sql` call. Cache-hit tests use the
    /// counter to assert "the second call did NOT reach the primary."
    /// The counter is `Arc<AtomicUsize>` so a clone can be retained by
    /// the test after the reader itself is moved into `Box<dyn Reader>`.
    struct CountingReader {
        inner: DuckDBReader,
        calls: Arc<AtomicUsize>,
    }

    impl CountingReader {
        fn new(inner: DuckDBReader, calls: Arc<AtomicUsize>) -> Self {
            Self { inner, calls }
        }
    }

    impl Reader for CountingReader {
        fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.execute_sql(sql)
        }
        fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
            self.inner.register(name, df, replace)
        }
        fn unregister(&self, name: &str) -> Result<()> {
            self.inner.unregister(name)
        }
        fn execute(&self, query: &str) -> Result<Spec> {
            crate::reader::execute_with_reader(self, query)
        }
        fn dialect(&self) -> &dyn SqlDialect {
            self.inner.dialect()
        }
    }

    #[test]
    fn new_has_cache_enabled_by_default() {
        let data = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(data), staging);
        assert!(r.cache_config().enabled);
        assert_eq!(r.cache_config().ttl_secs, 300);
    }

    #[test]
    fn with_cache_config_applies_custom_settings() {
        use crate::reader::hybrid_cache::CacheConfig;
        let data = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let cfg = CacheConfig {
            enabled: false,
            ttl_secs: 60,
            max_bytes: 1024,
        };
        let r = HybridReader::with_cache_config(Box::new(data), staging, cfg);
        assert!(!r.cache_config().enabled);
        assert_eq!(r.cache_config().ttl_secs, 60);
        assert_eq!(r.cache_config().max_bytes, 1024);
    }

    #[test]
    fn repeat_query_hits_cache_not_data() {
        // Two identical execute_sql calls must result in only ONE
        // execute_sql reaching the primary reader — the second call hits
        // the cache.
        let data_inner = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_inner
            .execute_sql("CREATE TABLE t AS SELECT 1 AS x UNION ALL SELECT 2")
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingReader::new(data_inner, calls.clone());
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(counting), staging);

        // The CREATE TABLE above ran on `data_inner` BEFORE wrapping in
        // CountingReader, so `calls` is still 0. Two identical SELECTs
        // through the HybridReader: first miss → counter to 1; second
        // hit → counter stays at 1.
        let sql = "SELECT x FROM t ORDER BY x";
        r.execute_sql(sql).unwrap();
        r.execute_sql(sql).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second execute_sql must be served by cache"
        );
    }

    #[test]
    fn ttl_zero_forces_miss_every_call() {
        use crate::reader::hybrid_cache::CacheConfig;
        let data_inner = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_inner
            .execute_sql("CREATE TABLE t AS SELECT 1 AS x")
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingReader::new(data_inner, calls.clone());
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let cfg = CacheConfig {
            enabled: true,
            ttl_secs: 0,
            max_bytes: 1 << 30,
        };
        let r = HybridReader::with_cache_config(Box::new(counting), staging, cfg);

        // No sleep between calls — the boundary check must be strict
        // (`age_ms < ttl_ms`) so ttl=0 always misses, even when racing
        // within the same millisecond.
        r.execute_sql("SELECT x FROM t").unwrap();
        r.execute_sql("SELECT x FROM t").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "ttl=0 must always miss");
    }

    #[test]
    fn lru_evicts_oldest_when_over_budget() {
        use crate::reader::hybrid_cache::CacheConfig;
        let data_inner = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_inner
            .execute_sql("CREATE TABLE t AS SELECT 1 AS x UNION ALL SELECT 2")
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingReader::new(data_inner, calls.clone());
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        // Tiny budget: 1 byte. Every cached entry exceeds it, so eviction
        // fires deterministically on every insert.
        let cfg = CacheConfig {
            enabled: true,
            ttl_secs: 300,
            max_bytes: 1,
        };
        let r = HybridReader::with_cache_config(Box::new(counting), staging, cfg);

        r.execute_sql("SELECT x FROM t WHERE x = 1").unwrap(); // miss → stage A
        std::thread::sleep(std::time::Duration::from_millis(5));
        r.execute_sql("SELECT x FROM t WHERE x = 2").unwrap(); // miss → stage B; A evicted
        let before = calls.load(Ordering::SeqCst);
        r.execute_sql("SELECT x FROM t WHERE x = 1").unwrap(); // re-query A — must MISS
        assert_eq!(
            calls.load(Ordering::SeqCst),
            before + 1,
            "A should have been evicted by LRU and re-fetched on re-query"
        );
    }

    #[test]
    fn clear_cache_drops_everything() {
        let data_inner = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_inner
            .execute_sql("CREATE TABLE t AS SELECT 1 AS x")
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingReader::new(data_inner, calls.clone());
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(counting), staging);

        r.execute_sql("SELECT x FROM t").unwrap(); // miss
        r.execute_sql("SELECT x FROM t").unwrap(); // hit
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        r.clear_cache().unwrap();

        r.execute_sql("SELECT x FROM t").unwrap(); // miss again — cache wiped
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn viz_execute_shares_cache_with_execute_sql() {
        // Cache is keyed on (reader_uri, sql). The viz pipeline wraps the
        // SQL portion in DDL (`CREATE OR REPLACE TEMP TABLE __ggsql_global__
        // AS <body>`) and issues additional schema/range/data sub-queries
        // against the staged temp table. The DDL itself is uncacheable
        // (DuckDB returns a 0-column result for DDL, which the cache skips
        // to avoid `arrow(...)` rejecting an empty schema), but every
        // result-bearing sub-query routes through the cache.
        //
        // Load-bearing claim verified here: caching is wired into the viz
        // path. After running a viz query, manually issuing one of the
        // sub-queries the viz pipeline emits (e.g. the data fetch over
        // the global temp table) hits the cache and does NOT advance the
        // data counter. This guarantees that any pipeline that re-emits
        // the same SQL string within the TTL is memoized.
        let data_inner = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        data_inner
            .execute_sql("CREATE TABLE t AS SELECT 1 AS x, 10 AS y UNION ALL SELECT 2, 20")
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingReader::new(data_inner, calls.clone());
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(counting), staging);

        // Run the viz pipeline. This emits the temp-table DDL plus
        // schema/range/data sub-queries, all of which (except the DDL)
        // populate the cache.
        let viz_query = "SELECT x, y FROM t ORDER BY x\nVISUALISE x AS x, y AS y\nDRAW point";
        let _spec = r.execute(viz_query).unwrap();
        let after_viz = calls.load(Ordering::SeqCst);
        assert!(
            after_viz >= 1,
            "viz call must hit data at least once on cache miss; got {after_viz}"
        );

        // Re-issue the schema-fetch sub-query that the viz pipeline
        // emitted internally. The exact SQL depends on the global-table
        // name (which embeds a process-stable session UUID), so we
        // reconstruct it the same way the pipeline does.
        let global = crate::naming::quote_ident(&crate::naming::global_table());
        let schema_sql = format!("SELECT * FROM (SELECT * FROM {global}) AS __schema__ LIMIT 1");
        let before_replay = calls.load(Ordering::SeqCst);
        let _ = r.execute_sql(&schema_sql).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            before_replay,
            "replaying a sub-query the viz pipeline already issued must hit \
             the cache — data counter must not advance"
        );
    }
}

#[cfg(all(test, feature = "sqlite", feature = "adbc"))]
mod cache_equivalence_tests {
    //! Cache equivalence tests against a real ADBC SQLite primary reader.
    //!
    //! Drives `HybridReader` with `AdbcReader<sqlite via ManagedDriver>` as
    //! the primary and a local in-memory DuckDB as staging. Verifies that
    //! the cache returns correct data on miss and on hit, that hits avoid
    //! round-tripping back to the primary, and that `clear_cache()` /
    //! `ttl=0` correctly force re-fetches. The mirror image of the
    //! per-staged-data assertions in `super::tests`, but with a real
    //! external-backend primary instead of an in-process `DuckDBReader`.
    //!
    //! Skipped by default (each test is `#[ignore]`). To run them:
    //!
    //! 1. Install dbc: `curl -LsSf https://dbc.columnar.tech/install.sh | sh`
    //! 2. Install the SQLite driver: `dbc install sqlite`
    //! 3. Run: `cargo test --features "adbc duckdb sqlite" -- --ignored cache_equivalence`
    //!
    //! `dbc install` writes the driver to a manifest location that
    //! `ManagedDriver::load_from_name("sqlite", ...)` discovers automatically
    //! (on macOS: `~/Library/Application Support/ADBC/Drivers/sqlite.toml`).

    use super::HybridReader;
    use crate::reader::hybrid_cache::CacheConfig;
    use crate::reader::sqlite::SqliteDialect;
    use crate::reader::{AdbcReader, DuckDBReader, Reader, Spec, SqlDialect};
    use crate::{DataFrame, Result};
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::LOAD_FLAG_DEFAULT;
    use adbc_driver_manager::ManagedDriver;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    /// Construct an `AdbcReader<ManagedDriver>` pointed at a SQLite file via
    /// the `adbc_driver_sqlite` shared library installed by `dbc`.
    fn make_adbc_reader(db_path: &str) -> AdbcReader<ManagedDriver> {
        let driver = ManagedDriver::load_from_name(
            "sqlite",
            None,
            AdbcVersion::V110,
            LOAD_FLAG_DEFAULT,
            None,
        )
        .expect("`dbc install sqlite` first; see module docs");
        let dialect: Box<dyn SqlDialect + Send> = Box::new(SqliteDialect);
        AdbcReader::new_with_database_opts(
            driver,
            dialect,
            std::iter::once((
                OptionDatabase::Uri,
                OptionValue::String(format!("file:{}", db_path)),
            )),
        )
        .expect("construct AdbcReader<sqlite>")
    }

    /// A wrapper around `AdbcReader<ManagedDriver>` that counts every
    /// `execute_sql` call. Used to assert that cache hits do NOT round-trip
    /// back to the ADBC primary.
    struct CountingAdbcReader {
        inner: AdbcReader<ManagedDriver>,
        calls: Arc<AtomicUsize>,
    }

    impl Reader for CountingAdbcReader {
        fn execute_sql(&self, sql: &str) -> Result<DataFrame> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.execute_sql(sql)
        }
        fn register(&self, name: &str, df: DataFrame, replace: bool) -> Result<()> {
            self.inner.register(name, df, replace)
        }
        fn unregister(&self, name: &str) -> Result<()> {
            self.inner.unregister(name)
        }
        fn execute(&self, query: &str) -> Result<Spec> {
            crate::reader::execute_with_reader(self, query)
        }
        fn dialect(&self) -> &dyn SqlDialect {
            self.inner.dialect()
        }
    }

    /// Seed the SQLite file with a small test table via the ADBC reader's
    /// `register()` (the bulk-ingest path). Done with a bare AdbcReader so
    /// the seed doesn't show up in any later call counters.
    fn seed(path: &str) {
        let bare = make_adbc_reader(path);
        let df = crate::df! {
            "x" => vec![1i64, 2, 3, 4, 5],
            "y" => vec![10i64, 20, 30, 40, 50],
        }
        .unwrap();
        bare.register("t", df, false).unwrap();
    }

    /// Compare two DataFrames by per-column Arrow array contents. Mirrors
    /// the helper in `adbc::equivalence_tests`, scoped down to the
    /// dimensions PR3 cares about.
    fn assert_dataframes_equal(a: &DataFrame, b: &DataFrame, ctx: &str) {
        assert_eq!(a.height(), b.height(), "{ctx}: row count");
        assert_eq!(a.width(), b.width(), "{ctx}: col count");
        for f in a.schema().fields() {
            assert_eq!(
                a.column(f.name()).unwrap().as_ref(),
                b.column(f.name()).unwrap().as_ref(),
                "{ctx}: column '{}' mismatch",
                f.name()
            );
        }
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_cache_returns_same_data_as_bare_adbc() {
        // First call (miss) AND second call (hit) must both return data
        // identical to a bare AdbcReader. Validates that the
        // miss → register-in-staging → return-fresh path doesn't corrupt
        // data, and the hit → SELECT-from-staging path returns a faithful
        // copy.
        let db = NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap();
        seed(path);

        let bare = make_adbc_reader(path);
        let sql = "SELECT x, y, x*y AS xy FROM t WHERE y > 15 ORDER BY x";
        let golden = bare.execute_sql(sql).unwrap();

        let primary = make_adbc_reader(path);
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(primary), staging);

        let miss = r.execute_sql(sql).unwrap();
        assert_dataframes_equal(&golden, &miss, "first call (cache miss)");

        let hit = r.execute_sql(sql).unwrap();
        assert_dataframes_equal(&golden, &hit, "second call (cache hit)");
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_cache_hit_avoids_adbc_call() {
        // Counter-based assertion: identical query twice → ADBC reached
        // exactly once. The cache-hit path must serve from staging without
        // touching the primary.
        let db = NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap();
        seed(path);

        let primary = make_adbc_reader(path);
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingAdbcReader {
            inner: primary,
            calls: calls.clone(),
        };
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(counting), staging);

        let sql = "SELECT x FROM t ORDER BY x";
        r.execute_sql(sql).unwrap();
        r.execute_sql(sql).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second call must hit cache, not ADBC"
        );
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_clear_cache_forces_adbc_refetch() {
        // After clear_cache(), the next query must round-trip back to ADBC
        // — proving clear_cache actually drops cache state, not just meta
        // rows.
        let db = NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap();
        seed(path);

        let primary = make_adbc_reader(path);
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingAdbcReader {
            inner: primary,
            calls: calls.clone(),
        };
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let r = HybridReader::new(Box::new(counting), staging);

        let sql = "SELECT x FROM t";
        r.execute_sql(sql).unwrap(); // miss → 1
        r.execute_sql(sql).unwrap(); // hit  → still 1
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        r.clear_cache().unwrap();
        r.execute_sql(sql).unwrap(); // miss again → 2
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "clear_cache must force re-fetch from ADBC"
        );
    }

    #[test]
    #[ignore = "requires `dbc install sqlite`; see module docs"]
    fn equiv_ttl_zero_always_hits_adbc() {
        // ttl=0 must always evict + refetch, even within the same
        // millisecond. Each call must reach ADBC.
        let db = NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap();
        seed(path);

        let primary = make_adbc_reader(path);
        let calls = Arc::new(AtomicUsize::new(0));
        let counting = CountingAdbcReader {
            inner: primary,
            calls: calls.clone(),
        };
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        let cfg = CacheConfig {
            enabled: true,
            ttl_secs: 0,
            max_bytes: 1 << 30,
        };
        let r = HybridReader::with_cache_config(Box::new(counting), staging, cfg);

        let sql = "SELECT x FROM t";
        r.execute_sql(sql).unwrap();
        r.execute_sql(sql).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "ttl=0 must always hit ADBC"
        );
    }
}
