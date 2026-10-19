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

## Stage 1: detached smoke tests of all eight combinations

~~~bash
bash scripts/launch_canonical_endpoint_matrix.sh smoke
~~~

This launches separate nohup workers:
- GPU6 runs CNN rho=0,0.10,0.25,0.50 sequentially.
- GPU7 runs NAF rho=0,0.10,0.25,0.50 sequentially.
- Each smoke runs 2 training batches and 2 validation batches and is
  labeled paper_grade=false.

Watch the last line for both:

~~~bash
watch -n 5 'for m in cnn naf; do f=$(printf "../mscloudr/outputs/CanonicalBridge_%s_EndpointMatrix_smoke.log" "$m"); printf "%s: " "$m"; tail -n 1 "$f"; done'
~~~

## Stage 2: detached three-epoch pilots (full validation; no test)

~~~bash
bash scripts/launch_canonical_endpoint_matrix.sh pilot
~~~

~~~bash
watch -n 5 'for m in cnn naf; do f=$(printf "../mscloudr/outputs/CanonicalBridge_%s_EndpointMatrix_pilot.log" "$m"); printf "%s: " "$m"; tail -n 1 "$f"; done'
~~~

Three-epoch pilots are only for plumbing/early optimization. They are NOT
reliable final predictors (NAF residual scales initialize at zero).
Use the full val_endpoint_l1 and the historical full-trained CNN
baseline to choose full runs; do not select configs based on TEST.

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
