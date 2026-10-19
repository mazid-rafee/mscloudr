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

## Stage 3: selected 50-epoch runs with automatic held-out test

Example CNN rho=.25 on GPU 6, and NAF rho=.25 on GPU 7:

~~~bash
nohup bash scripts/run_canonical_endpoint.sh cnn 0.25 6 full \
  > ../mscloudr/outputs/CanonicalBridge_CNN_Endpoint25_full_seed42.log \
  2>&1 < /dev/null &
disown

nohup bash scripts/run_canonical_endpoint.sh naf 0.25 7 full \
  > ../mscloudr/outputs/CanonicalBridge_NAF_Endpoint25_full_seed42.log \
  2>&1 < /dev/null &
disown
~~~

The full runner automatically invokes eval_canonical_bridge on
checkpoints/best_endpoint.pt after successful 50-epoch training, performing
NFE=1 full 7,899-sample TEST evaluation. Full results are saved in
eval_metrics.json. Each experiment uses its own log and output directory.

First full NAF rho=0 is important as a backbone-only control. Repeated
seeds are needed before claiming robust gains.

Checkpoint metadata records the backbone, rho, training measure, and
actual endpoint probability; resume rejects changes to backbone/rho.
