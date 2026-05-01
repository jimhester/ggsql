//! Query-result caching for `HybridReader`.
//!
//! Hashes `(reader_uri, sql)` → stable per-query table name in the staging
//! DuckDB. See `hybrid.rs` docstring for the wider design.

use arrow::array::Array;
use sha2::{Digest, Sha256};

use crate::array_util::{as_i64, as_str};
use crate::reader::{DuckDBReader, Reader};
use crate::Result;

pub const META_TABLE: &str = "__ggsql_cache_meta__";

/// Runtime configuration for the query-result cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Disable entirely if false — execute_sql always routes straight to
    /// the primary reader, no meta-table touched.
    pub enabled: bool,
    /// Time-to-live in seconds. Entries older than this get evicted on
    /// lookup and re-fetched. Default 300s.
    pub ttl_secs: u64,
    /// Cumulative byte budget across all cache entries. When exceeded
    /// after an insert, LRU eviction fires. Default 512 MB.
    pub max_bytes: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        let enabled = std::env::var("GGSQL_HYBRID_CACHE_DISABLED")
            .ok()
            .filter(|v| !v.is_empty() && v != "0")
            .is_none();
        Self {
            enabled,
            ttl_secs: 300,
            max_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Create the meta table if it does not already exist. DuckDB-specific
/// DDL; safe to call on every `HybridReader::new`.
pub fn ensure_meta(staging: &DuckDBReader) -> Result<()> {
    let ddl = format!(
        "CREATE TABLE IF NOT EXISTS {META_TABLE} (
           cache_key             VARCHAR PRIMARY KEY,
           reader_uri            VARCHAR NOT NULL,
           sql                   VARCHAR NOT NULL,
           fetched_at_epoch_ms   BIGINT  NOT NULL,
           last_accessed_epoch_ms BIGINT NOT NULL,
           row_count             BIGINT  NOT NULL,
           byte_estimate         BIGINT  NOT NULL
         )"
    );
    staging.execute_sql(&ddl).map(|_| ())
}

/// Stable cache-key derived from the remote reader's URI and the SQL text.
/// Uses SHA-256 hex truncated to 16 chars — 64 bits of key space, collision
/// odds negligible at any realistic cache size.
///
/// Inputs are joined with a newline separator so a `\n`-containing URI can't
/// be confused with the SQL half of a different pair.
pub fn cache_key(reader_uri: &str, sql: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(reader_uri.as_bytes());
    hasher.update(b"\n");
    hasher.update(sql.as_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..8]) // 16 hex chars = 64 bits
}

/// Table name for a cached result. Kept out-of-band from user-visible
/// names by the `__ggsql_cache_` prefix, matching the convention used for
/// other framework tables (`__ggsql_aes_*`, `__ggsql_layer_*`).
pub fn cache_table_name(key: &str) -> String {
    format!("__ggsql_cache_{}", key)
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub cache_key: String,
    pub reader_uri: String,
    pub sql: String,
    pub fetched_at_epoch_ms: i64,
    pub last_accessed_epoch_ms: i64,
    pub row_count: i64,
    pub byte_estimate: i64,
}

pub fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// SQL-escape a string for embedding in a DuckDB string literal.
fn esc(s: &str) -> String {
    s.replace('\'', "''")
}

pub fn lookup(staging: &DuckDBReader, key: &str) -> Result<Option<CacheEntry>> {
    let sql = format!(
        "SELECT cache_key, reader_uri, sql, fetched_at_epoch_ms,
                last_accessed_epoch_ms, row_count, byte_estimate
         FROM {META_TABLE} WHERE cache_key = '{}'",
        esc(key)
    );
    let df = staging.execute_sql(&sql)?;
    if df.height() == 0 {
        return Ok(None);
    }
    let get_str = |col: &str| -> Result<String> {
        let s = df.column(col).map_err(|e| {
            crate::GgsqlError::ReaderError(format!("cache lookup col {col}: {e}"))
        })?;
        let arr = as_str(s).map_err(|e| {
            crate::GgsqlError::ReaderError(format!("cache lookup str {col}: {e}"))
        })?;
        if arr.is_empty() || arr.is_null(0) {
            Ok(String::new())
        } else {
            Ok(arr.value(0).to_string())
        }
    };
    let get_i64 = |col: &str| -> Result<i64> {
        let s = df.column(col).map_err(|e| {
            crate::GgsqlError::ReaderError(format!("cache lookup col {col}: {e}"))
        })?;
        let arr = as_i64(s).map_err(|e| {
            crate::GgsqlError::ReaderError(format!("cache lookup i64 {col}: {e}"))
        })?;
        if arr.is_empty() || arr.is_null(0) {
            Ok(0)
        } else {
            Ok(arr.value(0))
        }
    };
    Ok(Some(CacheEntry {
        cache_key: get_str("cache_key")?,
        reader_uri: get_str("reader_uri")?,
        sql: get_str("sql")?,
        fetched_at_epoch_ms: get_i64("fetched_at_epoch_ms")?,
        last_accessed_epoch_ms: get_i64("last_accessed_epoch_ms")?,
        row_count: get_i64("row_count")?,
        byte_estimate: get_i64("byte_estimate")?,
    }))
}

pub fn insert_meta(
    staging: &DuckDBReader,
    key: &str,
    reader_uri: &str,
    sql: &str,
    row_count: i64,
    byte_estimate: i64,
) -> Result<()> {
    // INSERT OR REPLACE so a stale row from a previous attempt (e.g. a
    // partial failure where the cache table was registered but the meta
    // row insert errored) gets cleanly overwritten on retry instead of
    // raising a PK conflict on `cache_key`.
    let now = now_ms();
    let dml = format!(
        "INSERT OR REPLACE INTO {META_TABLE}
         (cache_key, reader_uri, sql, fetched_at_epoch_ms,
          last_accessed_epoch_ms, row_count, byte_estimate)
         VALUES ('{}', '{}', '{}', {}, {}, {}, {})",
        esc(key),
        esc(reader_uri),
        esc(sql),
        now,
        now,
        row_count,
        byte_estimate,
    );
    staging.execute_sql(&dml).map(|_| ())
}

pub fn touch(staging: &DuckDBReader, key: &str) -> Result<()> {
    let dml = format!(
        "UPDATE {META_TABLE} SET last_accessed_epoch_ms = {}
         WHERE cache_key = '{}'",
        now_ms(),
        esc(key)
    );
    staging.execute_sql(&dml).map(|_| ())
}

pub fn drop_entry(staging: &DuckDBReader, key: &str) -> Result<()> {
    let del = format!(
        "DELETE FROM {META_TABLE} WHERE cache_key = '{}'",
        esc(key)
    );
    staging.execute_sql(&del)?;
    let drop = format!("DROP TABLE IF EXISTS {}", cache_table_name(key));
    staging.execute_sql(&drop).map(|_| ())
}

pub fn clear_all(staging: &DuckDBReader) -> Result<()> {
    // Find all cache_keys; drop each entry.
    let df = staging.execute_sql(&format!(
        "SELECT cache_key FROM {META_TABLE}"
    ))?;
    if df.height() > 0 {
        let col = df.column("cache_key").map_err(|e| {
            crate::GgsqlError::ReaderError(format!("clear_all col: {e}"))
        })?;
        let s = as_str(col).map_err(|e| {
            crate::GgsqlError::ReaderError(format!("clear_all str: {e}"))
        })?;
        // Collect keys first to avoid holding borrow during drop_entry mutations.
        let keys: Vec<String> = (0..s.len())
            .filter(|&i| !s.is_null(i))
            .map(|i| s.value(i).to_string())
            .collect();
        for k in keys {
            let _ = drop_entry(staging, &k);
        }
    }
    // Defensive final wipe in case stale rows linger.
    staging.execute_sql(&format!("DELETE FROM {META_TABLE}"))?;
    Ok(())
}

/// Drop the oldest-accessed entries until total bytes <= max_bytes.
/// Run after every insert.
pub fn evict_over_budget(staging: &DuckDBReader, max_bytes: u64) -> Result<()> {
    // Cast SUM result to BIGINT — DuckDB's SUM over BIGINT promotes to HUGEINT,
    // which the arrow adapter materializes as Float64 (or unsupported type)
    // and would silently break the i64 extractor below.
    let sum_sql = format!(
        "SELECT CAST(COALESCE(SUM(byte_estimate), 0) AS BIGINT) AS n FROM {META_TABLE}"
    );
    loop {
        let df = staging.execute_sql(&sum_sql)?;
        let total = df
            .column("n")
            .ok()
            .and_then(|c| as_i64(c).ok())
            .and_then(|arr| {
                if arr.is_empty() || arr.is_null(0) {
                    None
                } else {
                    Some(arr.value(0))
                }
            })
            .unwrap_or(0) as u64;
        if total <= max_bytes {
            return Ok(());
        }
        // Find the single oldest-accessed key.
        let pick = format!(
            "SELECT cache_key FROM {META_TABLE}
             ORDER BY last_accessed_epoch_ms ASC LIMIT 1"
        );
        let df = staging.execute_sql(&pick)?;
        if df.height() == 0 {
            return Ok(()); // empty but still over budget — impossible, safety
        }
        let key = df
            .column("cache_key")
            .ok()
            .and_then(|c| as_str(c).ok())
            .and_then(|arr| {
                if arr.is_empty() || arr.is_null(0) {
                    None
                } else {
                    Some(arr.value(0).to_string())
                }
            })
            .unwrap_or_default();
        if key.is_empty() {
            return Ok(());
        }
        drop_entry(staging, &key)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable() {
        let a = cache_key("backend://prod", "SELECT 1");
        let b = cache_key("backend://prod", "SELECT 1");
        assert_eq!(a, b);
    }

    #[test]
    fn uri_matters() {
        let a = cache_key("backend://prod", "SELECT 1");
        let b = cache_key("backend://test", "SELECT 1");
        assert_ne!(a, b);
    }

    #[test]
    fn sql_matters() {
        let a = cache_key("q+t://p", "SELECT 1");
        let b = cache_key("q+t://p", "SELECT 2");
        assert_ne!(a, b);
    }

    #[test]
    fn uri_sql_cannot_collide() {
        // If we didn't include the separator, ("ab", "c") and ("a", "bc")
        // would hash to the same concatenation.
        let a = cache_key("ab", "c");
        let b = cache_key("a", "bc");
        assert_ne!(a, b);
    }

    #[test]
    fn table_name_prefix() {
        assert!(cache_table_name("deadbeef").starts_with("__ggsql_cache_"));
    }

    #[test]
    fn ensure_meta_is_idempotent() {
        use crate::reader::DuckDBReader;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        ensure_meta(&staging).unwrap();
        // Running twice must not error
        ensure_meta(&staging).unwrap();
        // Meta table should be queryable
        let df = staging
            .execute_sql("SELECT COUNT(*) AS n FROM __ggsql_cache_meta__")
            .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn insert_meta_replaces_existing_row() {
        // insert_meta must be idempotent on the cache_key PK so a stale row
        // from a previous attempt doesn't prevent the miss path from
        // overwriting it. Without `INSERT OR REPLACE`, the second call here
        // raises a PK conflict.
        use crate::reader::DuckDBReader;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        ensure_meta(&staging).unwrap();
        let key = "samekey";
        insert_meta(&staging, key, "uri-a", "SELECT 1", 1, 100).unwrap();
        // Second insert with same key must NOT raise PK conflict.
        insert_meta(&staging, key, "uri-b", "SELECT 2", 2, 200).unwrap();
        let entry = lookup(&staging, key).unwrap().unwrap();
        assert_eq!(entry.reader_uri, "uri-b");
        assert_eq!(entry.sql, "SELECT 2");
        assert_eq!(entry.row_count, 2);
        assert_eq!(entry.byte_estimate, 200);
    }

    #[test]
    fn lookup_insert_touch_cycle() {
        use crate::reader::DuckDBReader;
        let staging = DuckDBReader::from_connection_string("duckdb://memory").unwrap();
        ensure_meta(&staging).unwrap();
        let key = "abc123";

        // empty
        assert!(lookup(&staging, key).unwrap().is_none());

        // insert
        insert_meta(&staging, key, "backend://prod", "SELECT 1", 1, 128).unwrap();
        let entry = lookup(&staging, key).unwrap().unwrap();
        assert_eq!(entry.row_count, 1);
        assert_eq!(entry.byte_estimate, 128);

        // touch advances last_accessed
        let before = entry.last_accessed_epoch_ms;
        std::thread::sleep(std::time::Duration::from_millis(10));
        touch(&staging, key).unwrap();
        let after = lookup(&staging, key).unwrap().unwrap();
        assert!(after.last_accessed_epoch_ms > before);

        // drop
        drop_entry(&staging, key).unwrap();
        assert!(lookup(&staging, key).unwrap().is_none());
    }
}
