//! Model management (spec §8 `model_loader.rs`): runtime download with
//! exponential backoff, SHA-256 verification, on-disk cache, fail-fast loading
//! into ONNX sessions.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

/// Result of resolving one model: where the ONNX lives on disk and whether it
/// had to be downloaded in this process lifetime.
#[derive(Debug)]
pub struct ResolvedModel {
    pub path: PathBuf,
    pub downloaded: bool,
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn normalize_sha(s: &str) -> String {
    s.trim().to_lowercase()
}

/// Downloads `url` to `dest_path` streaming to disk, retrying up to
/// `max_attempts` times with exponential backoff (spec §3: max 3 tentativi).
async fn download_with_retry(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    max_attempts: u32,
) -> Result<()> {
    let mut last_err = None;
    for attempt in 1..=max_attempts {
        match download_once(client, url, dest_path).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                tracing::warn!("download attempt {attempt}/{max_attempts} for {url} failed: {e}");
                last_err = Some(e);
                if attempt < max_attempts {
                    let backoff = Duration::from_secs(2u64.pow(attempt.saturating_sub(1)));
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("download failed")))
}

async fn download_once(client: &reqwest::Client, url: &str, dest_path: &Path) -> Result<()> {
    let tmp_path = dest_path.with_extension("part");
    let response = client
        .get(url)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .context("send request")?
        .error_for_status()
        .with_context(|| format!("HTTP error downloading {url}"))?;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .context("create temp file")?;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read response chunk")?;
        file.write_all(&chunk)
            .await
            .context("write chunk to disk")?;
    }
    file.flush().await?;
    drop(file);
    tokio::fs::rename(&tmp_path, dest_path)
        .await
        .context("finalize download (rename)")?;
    Ok(())
}

/// Ensures the model at `url` is available at `cache_dir/<filename>` with a
/// matching SHA-256 (when `expected_sha` is provided). Skips the download when
/// a valid cached copy exists (spec §3 logic).
pub async fn ensure_model(
    client: &reqwest::Client,
    url: &str,
    expected_sha: Option<&str>,
    cache_dir: &Path,
) -> Result<ResolvedModel> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("create cache dir {}", cache_dir.display()))?;
    let filename = url
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("model.onnx");
    let dest_path = cache_dir.join(filename);

    if dest_path.exists() {
        let cached = tokio::fs::read(&dest_path).await?;
        let cached_ok = match expected_sha {
            Some(expected) => sha256_hex(&cached) == normalize_sha(expected),
            None => looks_like_onnx(&cached),
        };
        if cached_ok {
            tracing::info!("model cache hit: {}", dest_path.display());
            return Ok(ResolvedModel {
                path: dest_path,
                downloaded: false,
            });
        }
        tracing::warn!(
            "cached model {} failed integrity check; re-downloading",
            dest_path.display()
        );
    }

    tracing::info!("downloading model from {url} …");
    download_with_retry(client, url, &dest_path, 3).await?;

    let bytes = tokio::fs::read(&dest_path).await?;
    if let Some(expected) = expected_sha {
        let actual = sha256_hex(&bytes);
        if actual != normalize_sha(expected) {
            tokio::fs::remove_file(&dest_path).await.ok();
            return Err(anyhow!(
                "SHA-256 mismatch for {url}: expected {expected}, got {actual}"
            ));
        }
    } else if !looks_like_onnx(&bytes) {
        tokio::fs::remove_file(&dest_path).await.ok();
        return Err(anyhow!(
            "downloaded file from {url} is not a valid ONNX model"
        ));
    }
    Ok(ResolvedModel {
        path: dest_path,
        downloaded: true,
    })
}

/// Minimal ONNX sanity check: real exports embed the "onnx" identifier in the
/// protobuf. This catches HTML error pages and truncated downloads.
fn looks_like_onnx(bytes: &[u8]) -> bool {
    if bytes.len() < 64 {
        return false;
    }
    bytes.windows(4).any(|w| w == b"onnx")
}

/// Loads an ONNX file into an `ort` session (CPU EP, single intra-op thread —
/// concurrency comes from the image-level semaphore, spec §9).
pub fn load_session(path: &Path) -> Result<ort::session::Session> {
    let session = ort::session::Session::builder()
        .map_err(|e| anyhow!("create ort SessionBuilder: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("set intra threads: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX model at {}", path.display()))?;
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha_and_onnx_heuristics() {
        assert_eq!(sha256_hex(b"abc"), sha256_hex(b"abc"));
        assert_ne!(sha256_hex(b"abc"), sha256_hex(b"abd"));
        let mut model_bytes = vec![0u8; 64];
        model_bytes[..4].copy_from_slice(b"onnx");
        assert!(looks_like_onnx(&model_bytes));
        assert!(!looks_like_onnx(b"<html>404 not found</html>"));
        assert!(!looks_like_onnx(b"short"));
    }
}
