//! SQLite persistence (spec §4, §8 `db.rs`): camera FSM state, detection
//! coordinates for ROI extraction, persisted ROI polygons.
//!
//! Uses runtime queries (no compile-time macros) so `cargo check` never needs
//! a live database.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::Row;

/// Lifecycle state of a camera (spec §4 FSM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraState {
    Initial,
    Learning,
    Active,
}

impl CameraState {
    pub fn as_str(&self) -> &'static str {
        match self {
            CameraState::Initial => "INITIAL",
            CameraState::Learning => "LEARNING",
            CameraState::Active => "ACTIVE",
        }
    }

    fn from_db(s: &str) -> Self {
        match s {
            "ACTIVE" => CameraState::Active,
            "LEARNING" => CameraState::Learning,
            _ => CameraState::Initial,
        }
    }
}

/// Persisted camera record.
#[derive(Debug, Clone)]
pub struct Camera {
    pub id: String,
    pub state: CameraState,
    pub learning_started_at: Option<DateTime<Utc>>,
    pub roi_json: Option<String>,
    /// Frame geometry (fixed per camera) — required to express ROI
    /// validations and margins in pixel space (spec §5).
    pub frame_width: Option<u32>,
    pub frame_height: Option<u32>,
}

/// One persisted async job row (S3/queue consumers, `/status/:job_id`). Plain
/// string/number projection so `db.rs` stays feature-independent (the typed
/// `S3JobStatus` mapping lives in `zip_worker_s3.rs`).
#[cfg_attr(not(feature = "s3"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3JobRow {
    pub job_id: String,
    pub input_key: String,
    pub output_key: String,
    /// "queued" | "running" | "done" | "failed" (matches `S3JobState`).
    pub state: String,
    pub processed_images: i64,
    pub error_count: i64,
    pub output_size_bytes: i64,
    pub etag: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub error: Option<String>,
}

/// A detection center point collected during LEARNING (spec §4).
#[derive(Debug, Clone)]
pub struct Detection {
    pub camera_id: String,
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub captured_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct Db {
    pool: sqlx::SqlitePool,
}

impl Db {
    /// Opens (creating if needed) the database file and applies migrations.
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create db dir {}", parent.display()))?;
            }
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal);
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(options)
            .await
            .with_context(|| format!("open sqlite at {}", path.display()))?;
        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS cameras (
                id                  TEXT PRIMARY KEY,
                state               TEXT NOT NULL DEFAULT 'INITIAL',
                learning_started_at TEXT,
                roi_json            TEXT,
                frame_width         INTEGER,
                frame_height        INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        // Idempotent upgrade for databases created before the frame-size
        // columns existed.
        for (col, sql) in [
            (
                "frame_width",
                "ALTER TABLE cameras ADD COLUMN frame_width INTEGER",
            ),
            (
                "frame_height",
                "ALTER TABLE cameras ADD COLUMN frame_height INTEGER",
            ),
        ] {
            match sqlx::query(sql).execute(&self.pool).await {
                Ok(_) => {}
                Err(e) if e.to_string().contains("duplicate column name") => {}
                Err(_) => {}
            }
            let _ = col;
        }
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT NOT NULL,
                x           REAL NOT NULL,
                y           REAL NOT NULL,
                confidence  REAL NOT NULL,
                captured_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id)")
            .execute(&self.pool)
            .await?;
        // Async S3/queue job registry (feature `s3` and the `queue`/`rabbitmq`
        // consumers). Persisted so `/status/:job_id` survives a restart — the
        // durable trace is this table + the audit-log object in the logs bucket.
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS s3_jobs (
                job_id            TEXT PRIMARY KEY,
                input_key         TEXT NOT NULL,
                output_key        TEXT NOT NULL,
                state             TEXT NOT NULL,
                processed_images  INTEGER NOT NULL DEFAULT 0,
                error_count       INTEGER NOT NULL DEFAULT 0,
                output_size_bytes INTEGER NOT NULL DEFAULT 0,
                etag              TEXT,
                started_at        TEXT,
                finished_at       TEXT,
                error             TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Inserts (or replaces) a persisted S3/queue job row. Used by the async
    /// job tracker so job state survives process restarts.
    #[cfg_attr(not(feature = "s3"), allow(dead_code))]
    pub async fn job_upsert(&self, job: &S3JobRow) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO s3_jobs (
                job_id, input_key, output_key, state, processed_images,
                error_count, output_size_bytes, etag, started_at, finished_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&job.job_id)
        .bind(&job.input_key)
        .bind(&job.output_key)
        .bind(&job.state)
        .bind(job.processed_images)
        .bind(job.error_count)
        .bind(job.output_size_bytes)
        .bind(&job.etag)
        .bind(&job.started_at)
        .bind(&job.finished_at)
        .bind(&job.error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Loads one persisted job row by id.
    #[cfg_attr(not(feature = "s3"), allow(dead_code))]
    pub async fn job_get(&self, job_id: &str) -> Result<Option<S3JobRow>> {
        let row = sqlx::query(
            r#"
            SELECT job_id, input_key, output_key, state, processed_images,
                   error_count, output_size_bytes, etag, started_at, finished_at, error
            FROM s3_jobs WHERE job_id = ?
            "#,
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| S3JobRow {
            job_id: r.get("job_id"),
            input_key: r.get("input_key"),
            output_key: r.get("output_key"),
            state: r.get("state"),
            processed_images: r.get("processed_images"),
            error_count: r.get("error_count"),
            output_size_bytes: r.get("output_size_bytes"),
            etag: r.get("etag"),
            started_at: r.get("started_at"),
            finished_at: r.get("finished_at"),
            error: r.get("error"),
        }))
    }

    /// Fetches the camera row, creating it in INITIAL on first sight
    /// (spec §4: unknown camera → INITIAL → immediate transition to LEARNING).
    pub async fn get_or_create_camera(&self, id: &str) -> Result<Camera> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT id, state, learning_started_at, roi_json, frame_width, frame_height FROM cameras WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&mut *tx)
        .await?;
        let camera = match row {
            Some(row) => row_to_camera(&row),
            None => {
                let now = Utc::now().to_rfc3339();
                sqlx::query(
                    "INSERT INTO cameras (id, state, learning_started_at) VALUES (?, 'LEARNING', ?)",
                )
                .bind(id)
                .bind(&now)
                .execute(&mut *tx)
                .await?;
                Camera {
                    id: id.to_string(),
                    state: CameraState::Learning,
                    learning_started_at: Some(Utc::now()),
                    roi_json: None,
                    frame_width: None,
                    frame_height: None,
                }
            }
        };
        tx.commit().await?;
        Ok(camera)
    }

    /// Sets the FSM state; `learning_started_at` is stamped on entry to
    /// LEARNING and cleared on exit.
    pub async fn set_state(&self, id: &str, state: CameraState) -> Result<()> {
        let ts = match state {
            CameraState::Learning => Some(Utc::now().to_rfc3339()),
            _ => None,
        };
        sqlx::query("UPDATE cameras SET state = ?, learning_started_at = ? WHERE id = ?")
            .bind(state.as_str())
            .bind(ts)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Records the fixed frame geometry of a camera (first image seen wins;
    /// later mismatches are logged by the caller but not stored).
    pub async fn ensure_camera_frame_size(&self, id: &str, width: u32, height: u32) -> Result<()> {
        sqlx::query(
            "UPDATE cameras SET frame_width = COALESCE(frame_width, ?), frame_height = COALESCE(frame_height, ?) WHERE id = ?",
        )
        .bind(width as i64)
        .bind(height as i64)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Operator reset to INITIAL: zeroes ROI and learning start.
    pub async fn reset_to_initial(&self, id: &str) -> Result<bool> {
        let res = sqlx::query(
            "UPDATE cameras SET state = 'INITIAL', learning_started_at = NULL, roi_json = NULL WHERE id = ?",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Operator reset to LEARNING: keeps the production ROI but reopens data
    /// collection (spec §4 "Intervento Operatore").
    pub async fn reset_to_learning(&self, id: &str) -> Result<bool> {
        let res = sqlx::query(
            "UPDATE cameras SET state = 'LEARNING', learning_started_at = ? WHERE id = ?",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Stores the computed ROI as JSON and transitions to ACTIVE.
    pub async fn set_roi_and_activate(&self, id: &str, roi_json: &str) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("UPDATE cameras SET roi_json = ?, state = 'ACTIVE' WHERE id = ?")
            .bind(roi_json)
            .bind(id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Appends detection center points collected during LEARNING. Inserted in
    /// multi-row VALUES chunks (500 rows per statement — one statement is
    /// atomic on SQLite), so a camera with thousands of detections writes a
    /// handful of short transactions instead of one row per roundtrip.
    pub async fn insert_detections(&self, detections: &[Detection]) -> Result<()> {
        if detections.is_empty() {
            return Ok(());
        }
        const CHUNK: usize = 500;
        for chunk in detections.chunks(CHUNK) {
            let values = vec!["(?, ?, ?, ?, ?)"; chunk.len()].join(",");
            let sql = format!(
                "INSERT INTO detections (camera_id, x, y, confidence, captured_at) VALUES {values}"
            );
            let mut q = sqlx::query(&sql);
            for d in chunk {
                q = q
                    .bind(&d.camera_id)
                    .bind(d.x)
                    .bind(d.y)
                    .bind(d.confidence)
                    .bind(d.captured_at.to_rfc3339());
            }
            q.execute(&self.pool).await?;
        }
        Ok(())
    }

    /// All detection coordinates for a camera (ROI extraction input).
    pub async fn detections_for_camera(&self, id: &str) -> Result<Vec<(f32, f32)>> {
        let rows = sqlx::query("SELECT x, y FROM detections WHERE camera_id = ?")
            .bind(id)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .into_iter()
            .map(|r| (r.get::<f64, _>("x") as f32, r.get::<f64, _>("y") as f32))
            .collect())
    }

    /// Detection coordinates recorded since `since` for a camera (dynamic-ROI
    /// re-extraction window — see `ROI_REEXTRACT_WINDOW_DAYS`).
    pub async fn detections_for_camera_since(
        &self,
        id: &str,
        since: &DateTime<Utc>,
    ) -> Result<Vec<(f32, f32)>> {
        let rows = sqlx::query(
            "SELECT x, y FROM detections WHERE camera_id = ? AND captured_at >= ?",
        )
        .bind(id)
        .bind(since.to_rfc3339())
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| (r.get::<f64, _>("x") as f32, r.get::<f64, _>("y") as f32))
            .collect())
    }

    /// Deletes detections older than `cutoff`; returns the number of rows
    /// removed. Bounds the table to the retention horizon of the ROI
    /// extraction (nightly task, see `background_loop`).
    pub async fn prune_detections_older_than(&self, cutoff: &DateTime<Utc>) -> Result<u64> {
        let res = sqlx::query("DELETE FROM detections WHERE captured_at < ?")
            .bind(cutoff.to_rfc3339())
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected())
    }

    /// Cameras whose LEARNING period has elapsed (for the nightly ROI task).
    pub async fn cameras_in_learning(&self) -> Result<Vec<Camera>> {
        let rows = sqlx::query(
            "SELECT id, state, learning_started_at, roi_json, frame_width, frame_height FROM cameras WHERE state = 'LEARNING'",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_camera).collect())
    }

    /// All cameras (operator listing endpoint).
    pub async fn all_cameras(&self) -> Result<Vec<Camera>> {
        let rows = sqlx::query(
            "SELECT id, state, learning_started_at, roi_json, frame_width, frame_height FROM cameras ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_camera).collect())
    }

    pub async fn camera_by_id(&self, id: &str) -> Result<Option<Camera>> {
        let row = sqlx::query(
            "SELECT id, state, learning_started_at, roi_json, frame_width, frame_height FROM cameras WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(row_to_camera))
    }
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

fn row_to_camera(row: &sqlx::sqlite::SqliteRow) -> Camera {
    Camera {
        id: row.get("id"),
        state: CameraState::from_db(&row.get::<String, _>("state")),
        learning_started_at: row
            .get::<Option<String>, _>("learning_started_at")
            .and_then(|s| parse_ts(&s)),
        roi_json: row.get("roi_json"),
        frame_width: row.get::<Option<i64>, _>("frame_width").map(|v| v as u32),
        frame_height: row.get::<Option<i64>, _>("frame_height").map(|v| v as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_db() -> Db {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "av_db_test_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Db::open(&dir.join("test.sqlite3")).await.unwrap()
    }

    #[tokio::test]
    async fn unknown_camera_gets_learning_state() {
        let db = temp_db().await;
        let cam = db.get_or_create_camera("CAM_001").await.unwrap();
        assert_eq!(cam.state, CameraState::Learning);
        assert!(cam.learning_started_at.is_some());
        // Second read must not re-create or change state.
        let cam2 = db.get_or_create_camera("CAM_001").await.unwrap();
        assert_eq!(cam2.state, CameraState::Learning);
    }

    #[tokio::test]
    async fn fsm_transitions_and_detections_roundtrip() {
        let db = temp_db().await;
        db.get_or_create_camera("CAM_002").await.unwrap();
        db.insert_detections(&[
            Detection {
                camera_id: "CAM_002".into(),
                x: 100.0,
                y: 200.0,
                confidence: 0.9,
                captured_at: Utc::now(),
            },
            Detection {
                camera_id: "CAM_002".into(),
                x: 110.0,
                y: 210.0,
                confidence: 0.8,
                captured_at: Utc::now(),
            },
        ])
        .await
        .unwrap();
        let pts = db.detections_for_camera("CAM_002").await.unwrap();
        assert_eq!(pts.len(), 2);

        db.set_roi_and_activate("CAM_002", r#"[{"x":1,"y":2}]"#)
            .await
            .unwrap();
        let cam = db.camera_by_id("CAM_002").await.unwrap().unwrap();
        assert_eq!(cam.state, CameraState::Active);
        assert!(cam.roi_json.is_some());

        db.reset_to_learning("CAM_002").await.unwrap();
        let cam = db.camera_by_id("CAM_002").await.unwrap().unwrap();
        assert_eq!(cam.state, CameraState::Learning);

        db.reset_to_initial("CAM_002").await.unwrap();
        let cam = db.camera_by_id("CAM_002").await.unwrap().unwrap();
        assert_eq!(cam.state, CameraState::Initial);
        assert!(cam.roi_json.is_none());
    }

    #[tokio::test]
    async fn s3_job_rows_roundtrip_and_upsert() {
        let db = temp_db().await;
        let row = S3JobRow {
            job_id: "job-1".into(),
            input_key: "cam.zip".into(),
            output_key: "elaborati/cam_elaborato.zip".into(),
            state: "queued".into(),
            processed_images: 0,
            error_count: 0,
            output_size_bytes: 0,
            etag: None,
            started_at: None,
            finished_at: None,
            error: None,
        };
        db.job_upsert(&row).await.unwrap();

        // Roundtrip.
        let got = db.job_get("job-1").await.unwrap().unwrap();
        assert_eq!(got, row);

        // Re-submit (queue redelivery) replaces the row, it does not duplicate.
        let updated = S3JobRow {
            state: "running".into(),
            processed_images: 7,
            error_count: 1,
            output_size_bytes: 12345,
            etag: Some("\"abc\"".into()),
            started_at: Some("2026-09-09T10:00:00Z".into()),
            finished_at: None,
            error: None,
            ..row.clone()
        };
        db.job_upsert(&updated).await.unwrap();
        let got = db.job_get("job-1").await.unwrap().unwrap();
        assert_eq!(got.state, "running");
        assert_eq!(got.processed_images, 7);
        assert_eq!(got.etag.as_deref(), Some("\"abc\""));

        // Unknown id → None.
        assert!(db.job_get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn detections_since_and_pruning() {
        let db = temp_db().await;
        db.get_or_create_camera("CAM_003").await.unwrap();
        let now = Utc::now();
        db.insert_detections(&[
            Detection {
                camera_id: "CAM_003".into(),
                x: 1.0,
                y: 2.0,
                confidence: 0.9,
                captured_at: now,
            },
            Detection {
                camera_id: "CAM_003".into(),
                x: 3.0,
                y: 4.0,
                confidence: 0.8,
                captured_at: now - chrono::Duration::days(10),
            },
        ])
        .await
        .unwrap();

        let since = now - chrono::Duration::days(3);
        let recent = db.detections_for_camera_since("CAM_003", &since).await.unwrap();
        assert_eq!(recent.len(), 1, "only the recent detection survives the window");

        // Pruning with a 5-day cutoff removes the 10-day-old row.
        let cutoff = now - chrono::Duration::days(5);
        let removed = db.prune_detections_older_than(&cutoff).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(db.detections_for_camera("CAM_003").await.unwrap().len(), 1);
    }
}
