# Large files

Artifacts over GitHub's size limits are hosted on Hugging Face. All three repositories are public.

## Gemma-2-2B / TinySQL

**Dataset: [`primal-sage/tinysql-compression-results`](https://huggingface.co/datasets/primal-sage/tinysql-compression-results)**

| Path | Size | What |
|---|---|---|
| `ALL_SIGNALS_COMPLETE.npz` | 226 MB | six raw signals for all 239,616 MLP neurons and 208 heads (`eap_mlp`, `grad_mlp`, `mag_mlp`, `wd_mlp`, `act_delta_mlp`, `edge_mlp`, `*_attn`) |
| `tacq_vulnerability.npz` | 2 MB | per-neuron vulnerability scores (`vuln_3bit`, `vuln_2bit`) used for the TaCQ-style selection row |
| `TACQ_ANALYSIS.json` | 11 MB | vulnerability analysis output |
| `tier_arrays.npz` | 3 kB | canonical τ = 0.3 tier assignment (also in `results/gemma_original/`) |
| `BASELINE_CORRECT.json` | 1 kB | indices of the 105 retention samples (also in `results/gemma_original/`) |
| `reruns_sept2026/` | 43 kB | all Sept 2026 rerun JSONs and per-sample arrays (mirrored in `results/`) |
| `notebooks/camera_ready_reruns.ipynb` | | the rerun notebook (mirrored as `notebooks/gemma_04_camera_ready_reruns.ipynb`) |

**Model: [`primal-sage/interpretability-workspace-backup`](https://huggingface.co/primal-sage/interpretability-workspace-backup)** (10 GB)

| Path | What |
|---|---|
| `models/finetuned/final/` | Gemma-2-2B fine-tuned on TinySQL, checkpoint 5,500 (bf16, safetensors) |
| `models/base/` | Gemma-2-2B base checkpoint used for weight and activation deltas |
| `data/test_CS1..5.json` | evaluation split, 40 per level |
| `data/analysis_CS1..5.json` | analysis split, 100 per level |
| `data/train_CS1..5.json` | training split, 20,000 per level |

## Llama-3-8B / Text2Cypher

**Model: [`primal-sage/circuittier-llama8b-cypher`](https://huggingface.co/primal-sage/circuittier-llama8b-cypher)** (~17 GB)

| Path | What |
|---|---|
| `models/finetuned/final/` | Llama-3-8B fine-tuned on Text2Cypher, checkpoint 1,236 (bf16, safetensors) |
| `results/ALL_SIGNALS_NORMALIZED.npz` | six normalized signals for all 458,752 MLP neurons (`S1_eap_mlp` … `S6_edge_mlp`) |
| `results/ALL_SIGNALS.npz` | raw signals |
| `results/tier_maps.npz` | P = 95 tier assignment (skeleton, supporting, compressible masks and vote counts) |
| `results/compression/` | February/March sweep outputs (mirrored in `results/llama_original/`) |
| `data/test_processed.json` | 4,833 test prompts with reference Cypher and CY tags |
| `data/finetuned_accuracy.json` | baseline-correct indices |
| `data/train_processed.json` | 39,554 training examples |
| `notebooks/session1.ipynb` | fine-tuning notebook (mirrored as `notebooks/llama_01_finetune.ipynb`) |

The P = 99 skeleton (2,133 neurons) is not stored; it is rebuilt from `ALL_SIGNALS_NORMALIZED.npz` in `gemma_04_camera_ready_reruns.ipynb` (per-signal 99th percentile, at least 3 of 6 signals above threshold).

## Download

```python
from huggingface_hub import snapshot_download
snapshot_download("primal-sage/tinysql-compression-results", repo_type="dataset", local_dir="hf/gemma_results")
snapshot_download("primal-sage/interpretability-workspace-backup", local_dir="hf/gemma", allow_patterns=["models/**", "data/**"])
snapshot_download("primal-sage/circuittier-llama8b-cypher", local_dir="hf/llama", allow_patterns=["models/finetuned/**", "results/**", "data/**"])
```
