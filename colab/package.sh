#!/usr/bin/env bash
# Build dist/fin-decider-colab.tar.gz for running the self-hosted models (Strands Decider, CLM) inside Google Colab.
# Upload it to Google Drive at MyDrive/fin-decider/, then open colab/run_strands.ipynb or colab/run_clm.ipynb.
#   colab/package.sh                 Reuters test corpus + Bloomberg corpus (~500 MB)
#   colab/package.sh --reuters-only  Reuters test corpus only (~2 MB)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "commit your changes first: the Colab run must be tied to a clean commit" >&2
  exit 1
fi
branch=$(git rev-parse --abbrev-ref HEAD)
commit=$(git rev-parse HEAD)
out=dist/colab
rm -rf "$out" && mkdir -p "$out/corpus_native/reuters_top10"

git bundle create "$out/repo.bundle" "$branch"
cp corpus_native/reuters_top10/canonical.parquet corpus_native/reuters_top10/manifest.json "$out/corpus_native/reuters_top10/"
if [[ "${1:-}" != "--reuters-only" ]]; then
  mkdir -p "$out/corpus"
  cp corpus/canonical.parquet corpus/manifest.json "$out/corpus/"
fi
printf '{"branch": "%s", "commit": "%s"}\n' "$branch" "$commit" > "$out/bundle.json"

tar -czf dist/fin-decider-colab.tar.gz -C "$out" .
ls -lh dist/fin-decider-colab.tar.gz
echo "commit $commit on $branch"
