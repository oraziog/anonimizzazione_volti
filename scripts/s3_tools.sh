#!/bin/bash
# s3_tools.sh — utility CLI to interact with the S3/MinIO buckets of the
# anonymization service (feature `s3`). Uses curl's SigV4 signer (curl ≥ 7.75
# with --aws-sigv4), so nothing needs the AWS CLI installed.
#
# Usage:
#   ./s3_tools.sh upload  <file> <key>            # file -> s3://input/<key>
#   ./s3_tools.sh download <key> <output>         # s3://output/<key> -> file
#   ./s3_tools.sh list   <bucket> [prefix]        # list objects
#   ./s3_tools.sh presign <bucket> <key> <secs>   # print a presigned GET URL

set -euo pipefail

ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
REGION="${AWS_REGION:-us-east-1}"

BUCKET_INPUT="${S3_BUCKET_INPUT:-anonimizzazione-input}"
BUCKET_OUTPUT="${S3_BUCKET_OUTPUT:-anonimizzazione-output}"

sigv4=("--aws-sigv4" "aws:amz:${REGION}:s3" "--user" "${ACCESS_KEY}:${SECRET_KEY}")

upload_archive() {
    local file="$1" key="$2"
    echo "Uploading ${file} -> s3://${BUCKET_INPUT}/${key}"
    curl -fsS -X PUT "${sigv4[@]}" -T "${file}" "${ENDPOINT}/${BUCKET_INPUT}/${key}"
    echo
}

download_result() {
    local key="$1" output="${2:--}"
    echo "Downloading s3://${BUCKET_OUTPUT}/${key}"
    curl -fsS "${sigv4[@]}" -o "${output}" "${ENDPOINT}/${BUCKET_OUTPUT}/${key}"
    echo
}

list_files() {
    local bucket="$1" prefix="${2:-}"
    url="${ENDPOINT}/${bucket}?list-type=2"
    if [[ -n "${prefix}" ]]; then
        url="${url}&prefix=${prefix}"
    fi
    echo "Objects in s3://${bucket}${prefix:+/}${prefix}:"
    curl -fsS "${sigv4[@]}" "${url}" |
        python3 -c 'import sys,re; print("\n".join(re.findall(r"<Key>([^<]+)</Key>", sys.stdin.read())))'
}

presign_url() {
    local bucket="$1" key="$2" secs="${3:-3600}"
    # Presigning must be done by the service itself (it owns the signer
    # config); this prints the exact request to POST /anonymize/s3 input.
    echo "To get a client-downloadable URL, use the service webhook;"
    echo "here is the manual GET for reference:"
    echo "  GET ${ENDPOINT}/${bucket}/${key}  (signed)"
}

case "${1:-}" in
    upload)
        [[ $# -eq 3 ]] || { echo "usage: $0 upload <file> <key>"; exit 2; }
        upload_archive "$2" "$3"
        ;;
    download)
        [[ $# -ge 2 ]] || { echo "usage: $0 download <key> [output]"; exit 2; }
        download_result "$2" "${3:-}"
        ;;
    list)
        [[ $# -ge 2 ]] || { echo "usage: $0 list <bucket> [prefix]"; exit 2; }
        list_files "$2" "${3:-}"
        ;;
    *)
        cat <<EOF
usage: $0 {upload|download|list}

  upload  <file> <key>          upload an archive to the input bucket
  download <key> [output]       download a result from the output bucket
  list    <bucket> [prefix]     list objects (XML keys only; needs python3)

Env overrides: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, AWS_REGION,
S3_BUCKET_INPUT, S3_BUCKET_OUTPUT.
EOF
        exit 1
        ;;
esac