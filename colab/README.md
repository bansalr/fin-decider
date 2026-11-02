# Running the self-hosted models in Google Colab

**CLM, every dataset (one click):** open
https://colab.research.google.com/github/bansalr/fin-decider/blob/isaac_cwb/colab/run_clm.ipynb
then choose *Runtime → Run all*. It runs Reuters (v1, v2.1, v3.1), every G-SIB benchmark, FinCausal and the full Bloomberg corpus, in that order. Progress is saved to `MyDrive/fin-decider-clm/`; after any stop, *Run all* again resumes. Each finished dataset downloads as `clm_<dataset>.zip`. Expect several days of L4 time across sessions (Colab Pro plus extra compute units).

The sections below describe the earlier package-based route, kept for Strands.

Strands Decider and CLM are open-weight models with no hosted API, so they run on a Colab GPU. The whole pipeline runs inside the notebook: Colab only lends the GPU. Results are written to your Google Drive and copied back into the repo afterwards.

**Strands also runs on the Mac (Apple silicon GPU), with no Colab needed.** It was verified there on 2026-11-01: about 0.5 stories a second, so roughly 1.3 hours for the Reuters test set. See "Strands on the Mac" below. CLM needs NVIDIA CUDA (vLLM), so it needs Colab or another NVIDIA GPU.

| Model | Notebook | Colab GPU | Approximate cost |
|---|---|---|---|
| Strands Decider 2B | `run_strands.ipynb` | **T4** (free tier) | free |
| CLM-8B (Qwen3-8B encoder in bf16, ~16.4 GB) | `run_clm.ipynb` | **L4** (24 GB). A T4 is too small | Colab Pro, or pay-as-you-go compute units |

## Steps

1. **On the Mac, build the package** from a clean commit:
   ```bash
   colab/package.sh                 # Reuters test set + Bloomberg corpus (~500 MB)
   colab/package.sh --reuters-only  # Reuters test set only (~2 MB): enough for the main comparison
   ```
2. **Upload** `dist/fin-decider-colab.tar.gz` to Google Drive, into a folder named `fin-decider` (so `MyDrive/fin-decider/fin-decider-colab.tar.gz`).
3. **Open the notebook in Colab:** colab.research.google.com → *File → Upload notebook* → choose `colab/run_strands.ipynb` (or `run_clm.ipynb`).
4. **Set the GPU:** *Runtime → Change runtime type* → T4 (Strands) or L4 (CLM) → *Save*.
5. **Run the cells top to bottom** (*Runtime → Run all* works). Approve the Google Drive permission prompt.
   - The server cell prints the model's reply to a test question and the model name it reports. A 2xx status and a probability between 0 and 1 mean it's working.
   - The Reuters cell classifies the 2,533 test stories. The Bloomberg cell is optional.
6. **If Colab disconnects**, reconnect and rerun the setup, install, server and run cells: `--resume` continues where it stopped.
7. **When finished,** download `MyDrive/fin-decider/results/` (or leave it on Drive) and tell Claude where it is. It is merged into `classification_results_native/` and `classification_results/` and scored there with the other models.

## What gets recorded
- The exact commit (`bundle.json`), the pinned model revisions, and the GPU and package versions (`gpu_<model>.txt`, `pip_freeze_<model>.txt`).
- The server's raw reply to the test question (`<model>_probe.json`) and server logs.
- Strands' base model (Qwen3.5-2B) and CLM's 75 MB head are fetched by their packages, which do not take a revision; the package versions are pinned instead.

## Expected time
- **Strands on a T4:** a few seconds per Reuters story (10 questions each), so roughly 1–3 hours for the test set. Keep the tab open; the free tier disconnects idle sessions.
- **CLM on an L4:** much faster per question (it embeds once and compares), roughly 15–45 minutes for the test set after a ~16 GB first-time download.

## Strands on the Mac
```bash
uv venv --python 3.12 ~/strands-venv && uv pip install --python ~/strands-venv/bin/python strands-decider==0.2.0
CK=$(~/strands-venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('StrandsAgents/strands-decider-2B-hobson-v19', revision='bb282d786bc251fd4e3068de3ada9ddbb38127cd'))")
~/strands-venv/bin/strands-decider serve "$CK" --port 8000 --device mps --model-name strands-decider-2B-hobson-v19 --strict-window &
.venv/bin/classify run -c configs/native_reuters_top10_local.yaml -m strands --no-dev-eval --resume
```

## Verified server behaviour (Strands, 2026-11-01)
- `POST /v1/systemone` with `noul` questions; questions may carry `criteria` (`true`/`false`).
- The response reports `model` (set with `--model-name`), `answers.<key>.noul`, and `usage.input_tokens`.
- `GET /health` exists.
- `--strict-window` makes over-long input an HTTP 422 error instead of a silent truncation; the adapter then shortens the text and records it.
