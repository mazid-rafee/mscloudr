# Training and validation batch protocol

This stage defines only the batch computations. It does not yet introduce an
optimizer, epoch loop, or checkpoint writer.

## Training

For each batch:

    t ~ Uniform{0, ..., T} using the explicit sampler generator
    alpha_t = schedule(t)
    x_t = (1 - alpha_t) x0 + alpha_t y
    x0_hat = legacy_dbcr(x_t, t, SAR)
    loss = mean absolute error(x0_hat, x0)

The loss remains attached to the graph. The future epoch loop will own
zero_grad, backward, and optimizer.step.

## Validation

Two validation quantities are intentionally distinct.

### val_random_t_l1

Uses the same random-t construction as training under torch.no_grad.

This is diagnostic only. Its sampling measure changes when alpha(t) changes, so
it is not a fair checkpoint-selection quantity across schedules.

### val_endpoint_l1

Uses exactly:

    t = T
    x_T = y

No schedule callable and no sampler generator are accepted by the endpoint
function. This makes endpoint validation schedule-independent.

For NFE=1 experiments, val_endpoint_l1 is the canonical checkpoint-selection
metric and will determine best_endpoint.pt in the future epoch loop.

## Aggregation

Batch L1 is a mean over all elements. Epoch aggregation must weight each batch
mean by its batch size so a smaller final batch does not receive equal weight to
a full batch. WeightedMean implements this rule.

The next stage may wrap these computations in the optimizer, epoch, and
checkpoint loop without changing their definitions.
