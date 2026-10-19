# CanonicalBridge endpoint-mixture × NAFBlock controlled study

## Design

The 2 × 4 experimental grid tests two optical backbones (CNN, NAF) against
rho = 0, 0.10, 0.25 and 0.50.

- CNN: exact original CanonicalBridge model, trainable parameters 7,653,037.
- NAF: replace all optical encoder (4), bottleneck (2) and decoder (3)
  residual blocks with existing NAFNet-style NAFBlocks. Not a reproduction
  of the entire NAFNet model. SAR CNN, gated fusion, FiLM, stems, skips,
  up/downsampling and head remain unchanged. No CAM, ASPP, PSP or SFBlocks.
- Training: t sampled per-example from (1-rho)*Uniform{0,...,T} + rho*delta_T.
  alpha = t/T; x_alpha = (1-alpha)*x0 + alpha*y. Single forward pass per
  example and mean L1 loss. Exact endpoint probability is
  rho+(1-rho)/(T+1), including the original uniform endpoint.
- Conditioning remains MR-inverse(r=3); geometry straight, beta_a=1.
- Random-t validation always uses unchanged uniform alpha; best validation
  endpoint L1 selects checkpoints. NFE=1 test unchanged and ROI-disjoint.
- Seed42, original widths 32/64/128/256, batch4, Adam LR=5e-5,
  1000 steps, original source outputs under ../mscloudr/outputs.
- Previous CNN rho=0 result is an existing reference, but an independent
  code-parity rerun may strengthen conclusions.

## On Ruby: new worktree and regression tests

~~~bash
cd "$HOME/Desktop/Source Code (Research)/mscloudr"
git fetch origin
git worktree add -b exp/canonical-endpoint-mixture-naf \
  "../mscloudr-canonical-endpoint-naf" \
  origin/exp/canonical-endpoint-mixture-naf
cd "../mscloudr-canonical-endpoint-naf"
conda activate pylrt
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_canonical_endpoint_naf.py \
  tests/test_canonical_bridge_capacity.py \
  tests/test_canonical_bridge_full_controls.py \
  tests/test_canonical_bridge_full_eval.py \
  tests/test_canonical_bridge_alpha_grid.py
~~~

## Stage 1: Smoke tests (completed on GPUs 6 and 7)

All eight smoke experiments completed on Ruby. No need to rerun.

## Stage 2: Four-GPU three-epoch pilots (full validation; no test)

Updated matrix launcher maps one endpoint probability to each GPU:

| GPU | rho | Job 1 | Job 2 |
|---|---|---|---|
| 4 | 0% | CNN-0 | NAF-0 |
| 5 | 10% | CNN-10 | NAF-10 |
| 6 | 25% | CNN-25 | NAF-25 |
| 7 | 50% | CNN-50 | NAF-50 |

Default: CNN and NAF **sequential per GPU** (four processes active
simultaneously across four GPUs, eight jobs total). This avoids GPU
OOM and contention.

~~~bash
nvidia-smi -i 4,5,6,7
bash scripts/launch_canonical_endpoint_matrix.sh pilot
~~~

Optional explicit two **concurrent** processes per GPU (eight active
training processes across four GPUs); only if there is sufficient free
VRAM and the cluster permits GPU sharing. This may be SLOWER if competing
jobs saturate each GPU:

~~~bash
bash scripts/launch_canonical_endpoint_matrix.sh pilot parallel
~~~

Do not launch both modes for the same pilot names: the runner refuses
to overwrite an existing output directory.

Each GPU writes a separate nohup log under the ORIGINAL mscloudr/outputs
directory. Monitor one last line per GPU (for sequential mode):

~~~bash
watch -n 5 'for gpu in 4 5 6 7; do case "$gpu" in 4) p=0;; 5) p=10;; 6) p=25;; 7) p=50;; esac; f=$(printf "../mscloudr/outputs/CanonicalBridge_Endpoint%s_pilot_GPU%s_sequential.log" "$p" "$gpu"); printf "GPU%s rho=%s%%: " "$gpu" "$p"; if [ -f "$f" ]; then tail -n 1 "$f"; else echo "not started"; fi; done'
~~~

Pilots run three epochs with full 7,176-sample validation every epoch.
They do NOT evaluate TEST. Early NAFBlock convergence can differ
because residual scales initialize at zero; pilot results are not
necessarily predictive of 50-epoch performance.

## Stage 3: Full 2x4 factorial, 50 epochs each, automatic test evaluation

All eight smoke tests have passed. Running the complete 2x4 factorial
instead of filtering using three-epoch pilots is reasonable (higher
compute cost, but avoids early-convergence screening bias). There are
eight runs on four GPUs.

First check that GPUs 4-7 are free and previous pilot workers have stopped:

~~~bash
nvidia-smi -i 4,5,6,7
~~~

Recommended: four detached GPU workers. Each GPU runs CNN for 50 epochs,
then NAF for 50 epochs, with the same rho on that GPU:

~~~bash
bash scripts/launch_canonical_endpoint_matrix.sh full
~~~

Optional if both jobs fit concurrently on each GPU: eight active
training jobs (two competing for GPU memory/compute on each GPU):

~~~bash
bash scripts/launch_canonical_endpoint_matrix.sh full parallel
~~~

Run only one of the options. Parallel may OOM or slow down; sequential is
safer. Confirm cluster policy permits long-running concurrent jobs.

Mappings:
- GPU 4 -> CNN/NAF rho=0%
- GPU 5 -> CNN/NAF rho=10%
- GPU 6 -> CNN/NAF rho=25%
- GPU 7 -> CNN/NAF rho=50%

Each worker is detached with nohup and disown; logs in the original
mscloudr/outputs. All full training runs use 50 epochs, 107,142 training
samples and 7,176 validation samples. The best validation endpoint
checkpoint is evaluated NFE=1 automatically on all 7,899 test samples
after each successful training job.

Monitor the final log line from each sequential worker:

~~~bash
watch -n 5 'for spec in 4:0 5:10 6:25 7:50; do gpu=${spec%%:*}; p=${spec#*:}; f=$(printf "../mscloudr/outputs/CanonicalBridge_Endpoint%s_full_GPU%s_sequential.log" "$p" "$gpu"); printf "GPU%s rho=%s%%: " "$gpu" "$p"; if [ -f "$f" ]; then tail -n 1 "$f"; else echo "not started"; fi; done'
~~~

For parallel mode, change the suffix "_sequential.log" to "_parallel.log".

Each run creates its own output directory:

~~~text
../mscloudr/outputs/CanonicalBridge_CNN_Endpoint0_full_seed42
../mscloudr/outputs/CanonicalBridge_CNN_Endpoint10_full_seed42
../mscloudr/outputs/CanonicalBridge_CNN_Endpoint25_full_seed42
../mscloudr/outputs/CanonicalBridge_CNN_Endpoint50_full_seed42
../mscloudr/outputs/CanonicalBridge_NAF_Endpoint0_full_seed42
../mscloudr/outputs/CanonicalBridge_NAF_Endpoint10_full_seed42
../mscloudr/outputs/CanonicalBridge_NAF_Endpoint25_full_seed42
../mscloudr/outputs/CanonicalBridge_NAF_Endpoint50_full_seed42
~~~

Evaluation writes eval_metrics.json in each full run directory.
The eight TEST evaluations are descriptive outcomes of a predeclared
factorial; do not repeatedly tune based on TEST winners. Replicate
selected outcomes with additional independent seeds for paper claims.

Checkpoint metadata records backbone, rho, sampling measure and endpoint
mass. The CNN rho=0 run is a code-parity control for historical CNN.
