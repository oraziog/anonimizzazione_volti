//! Test-only scratch directories.
//!
//! Two problems are solved here.
//!
//! **Flakiness.** The old per-module helpers named their directory from the pid
//! and a counter only, and never deleted it. When the OS recycled a pid
//! (Windows does this eagerly), a later run re-opened a directory left behind
//! by an earlier one and found stale state in it — an already-migrated SQLite
//! database, a `CAM_001` row, random images — so unrelated tests failed at
//! random. The name now embeds the pid, a nanosecond timestamp *and* a
//! counter, so no two runs can ever share a path.
//!
//! **Leftovers.** `TempDir` deletes its directory on drop. When the directory
//! still holds an open SQLite file (`sqlx` releases the handle only when the
//! process exits, so on Windows the delete is a sharing violation), the delete
//! is retried briefly and then given up on: the first `TempDir::new` of the
//! *next* process sweeps directories older than [`ORPHAN_AGE`], by which time
//! the owning process is gone and the file can be removed.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Leftovers younger than this may belong to a test run that is still alive.
const ORPHAN_AGE: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

static SEQ: AtomicU64 = AtomicU64::new(0);
static SWEPT: std::sync::Once = std::sync::Once::new();

fn unique_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "av_{tag}_{}_{}_{}",
        std::process::id(),
        nanos,
        SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Remove scratch directories abandoned by runs that are no longer alive.
///
/// Only paths carrying the `av_` prefix this module creates are considered,
/// and only if they have not been touched for [`ORPHAN_AGE`]; a concurrently
/// running suite always refreshes its own directories, so it is never touched.
fn sweep_orphans() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("av_") || !path.is_dir() {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .is_some_and(|age| age >= ORPHAN_AGE);
        if stale {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

/// A freshly created scratch directory, deleted on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    /// Create an empty, uniquely named scratch directory for `tag`.
    pub fn new(tag: &str) -> Self {
        SWEPT.call_once(sweep_orphans);
        let dir = unique_path(tag);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A closed `SqlitePool` releases the file from its own worker thread,
        // so on Windows the first `remove_dir_all` can be a sharing violation
        // for a few milliseconds. Retry for up to a second: tests that close
        // their pool normally succeed on the very first attempt, the waiting is
        // only paid by the rare straggler under a fully parallel run, and
        // anything still stuck is picked up by the next process' sweep.
        for _ in 0..50 {
            if std::fs::remove_dir_all(&self.0).is_ok() || !self.0.exists() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}
