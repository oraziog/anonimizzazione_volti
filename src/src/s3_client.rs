//! S3 client for the S3 ingress/egress backend (feature `s3`, spec §8
//! "Scenario S3").
//!
//! Thin, task-focused wrapper over `aws-sdk-s3`: streaming upload/download of
//! on-disk archives (never buffering them in RAM), object listing, presigned
//! download URLs, delete/copy and bucket-existence checks. All calls are
//! AWS-SigV4-signed against the standard `AWS_` credential chain
//! (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) — with
//! `S3_ENDPOINT` set they talk to an S3-compatible store (MinIO) using
//! path-style addressing.

//! When `S3_ENDPOINT` is empty (real AWS) virtual-hosted style is kept.

use anyhow::{Context, Result};
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use std::path::Path;
use tokio::io::AsyncWriteExt;

use crate::config::S3Settings;

/// One object as returned by `list_objects` (a light projection of the SDK
/// `Object` with the fields this service actually reads).
#[derive(Debug, Clone)]
pub struct S3Object {
    pub key: String,
    pub size: u64,
    pub etag: String,
    pub last_modified: chrono::DateTime<chrono::Utc>,
    pub storage_class: String,
}

/// S3 client plus the three fixed logical buckets (input / output / logs).
#[derive(Clone)]
pub struct S3Client {
    client: Client,
    pub(crate) bucket_input: String,
    pub(crate) bucket_output: String,
    pub(crate) bucket_logs: String,
}

impl S3Client {
    /// Builds the client from `S3Settings` + the standard `AWS_*` env vars.
    /// `force_path_style` and `endpoint_url` are wired here (they belong to
    /// transport, not to the settings every caller passes around).
    pub async fn from_settings(settings: &S3Settings) -> Result<Self> {
        let sdk_config = aws_config::load_from_env().await;
        let mut builder = aws_sdk_s3::config::Builder::from(&sdk_config)
            .force_path_style(settings.force_path_style);
        if let Some(endpoint) = &settings.endpoint {
            builder = builder.endpoint_url(endpoint.clone());
        }
        let client = Client::from_conf(builder.build());

        Ok(Self {
            client,
            bucket_input: settings.bucket_input.clone(),
            bucket_output: settings.bucket_output.clone(),
            bucket_logs: settings.bucket_logs.clone(),
        })
    }

    /// Checks a bucket can actually be reached (fail-fast at startup). A
    /// `HeadBucket` is a cheap, credential-less-but-signed round-trip that
    /// validates both the endpoint and the credentials.
    pub async fn check_bucket(&self, bucket: &str) -> Result<()> {
        self.client
            .head_bucket()
            .bucket(bucket)
            .send()
            .await
            .with_context(|| format!("cannot reach S3 bucket '{bucket}' (endpoint/credentials?)"))?;
        tracing::info!("S3 bucket '{bucket}' reachable");
        Ok(())
    }

    /// Whether `key` exists in `bucket` (HEAD). A 404 is `Ok(false)`; every
    /// other transport/error result is a real failure.
    pub async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
        match self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if sdk_is_not_found(&e) => Ok(false),
            Err(e) => Err(e)
                .with_context(|| format!("head s3://{}/{key}", bucket)),
        }
    }

    /// Lists objects under `prefix` (one page, `max_keys` cap). `last_modified`
    /// defaults to `Utc::now()` if the object carries no timestamp (defensive;
    /// S3 always sets one).
    pub async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
        max_keys: Option<i32>,
    ) -> Result<Vec<S3Object>> {
        let mut request = self.client.list_objects_v2().bucket(bucket);
        if let Some(prefix) = prefix {
            request = request.prefix(prefix);
        }
        if let Some(max) = max_keys {
            request = request.max_keys(max);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("list objects under s3://{bucket}/{prefix:?}"))?;

        let mut objects = Vec::new();
        for obj in response.contents() {
            let last_modified =
                obj.last_modified()
                    .map(|ts| {
                        chrono::DateTime::from_timestamp(ts.secs(), ts.subsec_nanos())
                            .unwrap_or_else(chrono::Utc::now)
                    })
                    .unwrap_or_else(chrono::Utc::now);
            let mut entry = S3Object {
                key: obj.key().unwrap_or_default().to_string(),
                size: obj.size().unwrap_or(0) as u64,
                etag: obj.e_tag().unwrap_or_default().to_string(),
                last_modified,
                storage_class: obj
                    .storage_class()
                    .map(|s| s.as_str().to_string())
                    .unwrap_or_default(),
            };
            // Defensive: an empty key would break the downstream default-key
            // derivation and temp spooling; skip instead of failing the batch.
            if !entry.key.is_empty() {
                entry.etag = entry.etag.trim_matches('"').to_string();
                objects.push(entry);
            }
        }
        Ok(objects)
    }

    /// Streams an on-disk file into `s3://bucket/key` (64 KiB chunks via
    /// `ByteStream::from_path`, no full-file buffering). Returns the object
    /// ETag assigned by the store.
    pub async fn upload_streaming(&self, bucket: &str, key: &str, path: &Path) -> Result<String> {
        let body = ByteStream::from_path(path)
            .await
            .with_context(|| format!("open {} for S3 upload", path.display()))?;

        let result = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(body)
            .send()
            .await
            .with_context(|| format!("upload s3://{bucket}/{key}"))?;
        let etag = result.e_tag().unwrap_or_default().trim_matches('"').to_string();
        let size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or_default();
        tracing::info!("uploaded s3://{bucket}/{key} ({size} bytes, etag {etag})");
        Ok(etag)
    }

    /// Uploads a small in-memory payload (audit log JSON).
    pub async fn upload_bytes(&self, bucket: &str, key: &str, bytes: Vec<u8>) -> Result<()> {
        self.client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .with_context(|| format!("upload s3://{bucket}/{key}"))?;
        Ok(())
    }

    /// Streams `s3://bucket/key` to a local file, returning the byte count.
    pub async fn download_streaming(&self, bucket: &str, key: &str, dest: &Path) -> Result<u64> {
        let response = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("download s3://{bucket}/{key}"))?;

        let mut file = tokio::fs::File::create(dest)
            .await
            .with_context(|| format!("create local destination {}", dest.display()))?;
        let mut stream = response.body;
        let mut size = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read S3 download chunk")?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("write {}", dest.display()))?;
            size += chunk.len() as u64;
        }
        file.flush().await.ok();
        tracing::info!("downloaded s3://{bucket}/{key} ({size} bytes)");
        Ok(size)
    }

    /// Presigned GET URL valid for `expires_secs` (≤ 7 days, enforced by the
    /// SDK). Used for the completion webhook `download_url` payload.
    pub async fn generate_presigned_url(
        &self,
        bucket: &str,
        key: &str,
        expires_secs: u32,
    ) -> Result<String> {
        use aws_sdk_s3::presigning::PresigningConfig;
        use std::time::Duration;

        let presigning = PresigningConfig::expires_in(Duration::from_secs(expires_secs as u64))
            .context("invalid presigned-URL validity")?;
        let request = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .presigned(presigning)
            .await
            .with_context(|| format!("presign GET s3://{bucket}/{key}"))?;
        Ok(request.uri().to_string())
    }

    /// Deletes one object (idempotent on S3/MinIO).
    pub async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("delete s3://{bucket}/{key}"))?;
        tracing::info!("deleted s3://{bucket}/{key}");
        Ok(())
    }

    /// Server-side copy between buckets/keys (no download involved).
    pub async fn copy_object(
        &self,
        source_bucket: &str,
        source_key: &str,
        dest_bucket: &str,
        dest_key: &str,
    ) -> Result<()> {
        let source = format!("{}/{}", source_bucket, source_key);
        self.client
            .copy_object()
            .copy_source(&source)
            .bucket(dest_bucket)
            .key(dest_key)
            .send()
            .await
            .with_context(|| {
                format!("copy s3://{source} to s3://{dest_bucket}/{dest_key}")
            })?;
        tracing::info!("copied s3://{source} to s3://{dest_bucket}/{dest_key}");
        Ok(())
    }
}

/// S3 error-class detection used by `object_exists`: `true` when the SDK
/// response carries an HTTP 404 (the only "soft" failure we act on).
fn sdk_is_not_found<E>(e: &SdkError<E>) -> bool {
    e.raw_response()
        .map(|r| u16::from(r.status()) == 404)
        .unwrap_or(false)
}