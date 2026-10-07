#!/usr/bin/env bash
# Build fin-decider-colab.tar.gz for running GLiClass on a Google Colab GPU.
# Upload the tarball to Google Drive at MyDrive/fin-decider/ and open colab/run_gliclass.ipynb.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "commit your changes first: the Colab run must be tied to a clean commit" >&2
  exit 1
fi
branch=$(git rev-parse --abbrev-ref HEAD)
commit=$(git rev-parse HEAD)
out=dist/colab
rm -rf "$out" && mkdir -p "$out"

git bundle create "$out/repo.bundle" "$branch"
cp corpus/canonical.parquet corpus/manifest.json "$out/"
# Pinned dependencies from uv.lock; torch is left to Colab's CUDA build.
UV_PROJECT_ENVIRONMENT=.venv .venv/bin/uv export --frozen --no-hashes --extra gliclass --no-emit-project \
  | grep -v -E '^(torch|nvidia-|triton)==' > "$out/requirements-colab.txt"
printf '{"branch": "%s", "commit": "%s"}\n' "$branch" "$commit" > "$out/bundle.json"

tar -czf dist/fin-decider-colab.tar.gz -C "$out" .
ls -lh dist/fin-decider-colab.tar.gz
echo "commit $commit on $branch"
