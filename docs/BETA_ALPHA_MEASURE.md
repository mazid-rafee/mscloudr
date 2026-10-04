# Smooth endpoint-biased physical-alpha measure

This pilot control changes only the training measure along the existing linear
DB-CR bridge while keeping physical-alpha conditioning `c=T*alpha`.

For `a>0`, sample a uniform base variable `u=t/T` on the historical discrete
grid and transform

`alpha = u^(1/a)`.

In the continuous limit this produces `alpha ~ Beta(a,1)` with density
`q(alpha)=a*alpha^(a-1)`. The first experiment uses `a=2`, so
`q(alpha)=2*alpha`: the whole bridge remains covered, but high-corruption states
receive smoothly increasing probability without a point mass at `alpha=1`.

The control is intentionally restricted to `--schedule canonical_alpha` and
`--conditioning physical_alpha`. It is mutually exclusive with the endpoint
mixture control.

Example:

```bash
python -m mscloudr.cli.pilot_train \
  --run-name DBCR_Beta_a2_b1_pilot10_seed42 \
  --schedule canonical_alpha \
  --conditioning physical_alpha \
  --beta-a 2.0
```
