//! Model management (spec §8 `model_loader.rs`): runtime download with
//! exponential backoff, SHA-256 verification, on-disk cache, fail-fast loading
//! into ONNX sessions with an execution-provider (CPU/CUDA/TensorRT/DirectML)
//! selected process-wide from the environment.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
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

/// ONNX Runtime execution provider (env `ORT_EXECUTION_PROVIDER`, default
/// `cpu`). `cuda`/`tensorrt` require the app to be **compiled** with the
/// matching `cargo` feature (`cuda`/`tensorrt`), plus a working NVIDIA stack at
/// runtime (CUDA 13 + cuDNN 9 for the prebuilt binaries ort's
/// `download-binaries` fetches); `directml` needs the `directml` feature and
/// is Windows-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    TensorRt,
    DirectML,
}

impl ExecutionProvider {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" => Ok(ExecutionProvider::Cpu),
            "cuda" => Ok(ExecutionProvider::Cuda),
            "tensorrt" | "trt" => Ok(ExecutionProvider::TensorRt),
            "directml" | "dml" => Ok(ExecutionProvider::DirectML),
            other => anyhow::bail!(
                "ORT_EXECUTION_PROVIDER '{other}' must be 'cpu', 'cuda', 'tensorrt' or 'directml'"
            ),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionProvider::Cpu => "cpu",
            ExecutionProvider::Cuda => "cuda",
            ExecutionProvider::TensorRt => "tensorrt",
            ExecutionProvider::DirectML => "directml",
        }
    }
}

/// Process-wide session-building settings, configured once at startup from
/// `Config` (see [`configure_execution`]). Defaults to plain CPU so unit tests
/// and the `eval-*` CLI subcommands need no environment at all.
#[derive(Debug, Clone)]
pub struct ExecutionSettings {
    pub provider: ExecutionProvider,
    pub device_id: i32,
/// Cap (bytes) of the single-GPU device-memory arena / TensorRT workspace;
/// `None` lets ONNX Runtime pick.
#[cfg_attr(not(any(feature = "cuda", feature = "tensorrt")), allow(dead_code))]
pub gpu_memory_limit_bytes: Option<u64>,
/// Let cuDNN use TensorFloat-32 on Ampere+ (CUDA EP, off by default).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub enable_tf32: bool,
/// Reduced-precision inference for TensorRT (fp16; no generic fp16 toggle
/// exists on the plain CUDA EP).
#[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
pub enable_fp16: bool,
}

impl Default for ExecutionSettings {
    fn default() -> Self {
        Self {
            provider: ExecutionProvider::Cpu,
            device_id: 0,
            gpu_memory_limit_bytes: None,
            enable_tf32: false,
            enable_fp16: false,
        }
    }
}

static EP_SETTINGS: OnceLock<ExecutionSettings> = OnceLock::new();

/// Sets the process-wide execution settings. Must be called once before the
/// first session is created (startup); a second call is an error.
pub fn configure_execution(settings: ExecutionSettings) -> Result<()> {
    EP_SETTINGS
        .set(settings)
        .map_err(|_| anyhow!("execution provider already configured"))
}

fn execution_settings() -> &'static ExecutionSettings {
    EP_SETTINGS.get_or_init(ExecutionSettings::default)
}

/// Loads an ONNX file into an `ort` session using the configured execution
/// provider (single intra-op thread — concurrency comes from the image-level
/// session pool, spec §9).
///
/// The selected provider is registered in **front** of the CPU EP so ops the
/// GPU cannot run fall back to CPU; if the provider itself cannot be used (no
/// CUDA/cuDNN, no driver, mismatched dylibs) session creation **fails** — the
/// app fails fast at startup rather than silently running slower-than-expected
/// on CPU.
pub fn load_session(path: &Path) -> Result<ort::session::Session> {
    let settings = execution_settings();
    let mut builder = ort::session::Session::builder()
        .map_err(|e| anyhow!("create ort SessionBuilder: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow!("set intra threads: {e}"))?;

    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    match settings.provider {
        ExecutionProvider::Cpu => {
            providers.push(ort::ep::CPU::default().build());
        }
        ExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let mut ep = ort::ep::CUDA::default().with_device_id(settings.device_id);
                if let Some(limit) = settings.gpu_memory_limit_bytes {
                    ep = ep.with_memory_limit(limit as usize);
                }
                if settings.enable_tf32 {
                    ep = ep.with_tf32(true);
                }
                providers.push(ep.build());
                providers.push(ort::ep::CPU::default().build());
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!(
                    "ORT_EXECUTION_PROVIDER=cuda requires compiling with `--features cuda`"
                ));
            }
        }
        ExecutionProvider::TensorRt => {
            #[cfg(feature = "tensorrt")]
            {
                let mut ep = ort::ep::TensorRT::default().with_device_id(settings.device_id);
                if let Some(limit) = settings.gpu_memory_limit_bytes {
                    ep = ep.with_max_workspace_size(limit as usize);
                }
                if settings.enable_fp16 {
                    ep = ep.with_fp16(true);
                }
                providers.push(ep.build());
                providers.push(ort::ep::CPU::default().build());
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                return Err(anyhow!(
                    "ORT_EXECUTION_PROVIDER=tensorrt requires compiling with `--features tensorrt`"
                ));
            }
        }
        ExecutionProvider::DirectML => {
            #[cfg(feature = "directml")]
            {
                providers.push(
                    ort::ep::DirectML::default()
                        .with_device_id(settings.device_id)
                        .build(),
                );
                providers.push(ort::ep::CPU::default().build());
            }
            #[cfg(not(feature = "directml"))]
            {
                return Err(anyhow!(
                    "ORT_EXECUTION_PROVIDER=directml requires compiling with `--features directml`"
                ));
            }
        }
    }
    builder = builder
        .with_execution_providers(providers)
        .map_err(|e| {
            anyhow!(
                "register {} execution provider(s): {e}",
                settings.provider.as_str()
            )
        })?;

    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("load ONNX model at {}", path.display()))?;
    tracing::info!(
        "ONNX session loaded with {} execution provider (device {})",
        settings.provider.as_str(),
        settings.device_id
    );
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

    #[test]
    fn execution_provider_parsing() {
        assert_eq!(ExecutionProvider::parse("").unwrap(), ExecutionProvider::Cpu);
        assert_eq!(ExecutionProvider::parse("cpu").unwrap(), ExecutionProvider::Cpu);
        assert_eq!(ExecutionProvider::parse("CPU").unwrap(), ExecutionProvider::Cpu);
        assert_eq!(
            ExecutionProvider::parse("cuda").unwrap(),
            ExecutionProvider::Cuda
        );
        assert_eq!(
            ExecutionProvider::parse("tensorrt").unwrap(),
            ExecutionProvider::TensorRt
        );
        assert_eq!(
            ExecutionProvider::parse("trt").unwrap(),
            ExecutionProvider::TensorRt
        );
        assert_eq!(
            ExecutionProvider::parse("directml").unwrap(),
            ExecutionProvider::DirectML
        );
        assert_eq!(
            ExecutionProvider::parse("dml").unwrap(),
            ExecutionProvider::DirectML
        );
        assert!(ExecutionProvider::parse("vgpu").is_err());
        assert_eq!(ExecutionProvider::Cuda.as_str(), "cuda");
        assert_eq!(ExecutionProvider::TensorRt.as_str(), "tensorrt");
        assert_eq!(ExecutionProvider::DirectML.as_str(), "directml");
        assert_eq!(ExecutionProvider::Cpu.as_str(), "cpu");
    }
}
