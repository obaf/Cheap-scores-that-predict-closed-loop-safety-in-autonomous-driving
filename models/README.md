# Trained small baselines

Weights of the three small baselines of the study "Which cheap scores predict closed-loop safety?": an ego-status MLP,
nuPlan's simple vector model and nuPlan's raster model (ResNet-18), each trained on 50,000 samples of nuPlan's Pittsburgh training
logs. These are the 12 checkpoints the study evaluated: seed 0 early (after 15% of the optimizer steps) and final (end of training), and seeds 1 and 2 final for the seed-variance check. The training set-up is in
results/baselines/ (run_config.json, manifest.json) and code/train_baselines.sh.

| Planner (code/planners_baselines.yaml) | Checkpoint | Size |
|---|---|---|
| `mlp_early` | mlp/seed0/early.ckpt | 2.3 MB |
| `mlp_final` | mlp/seed0/final.ckpt | 2.3 MB |
| `vector_early` | vector/seed0/early.ckpt | 0.5 MB |
| `vector_final` | vector/seed0/final.ckpt | 0.5 MB |
| `raster_early` | raster/seed0/early.ckpt | 45.0 MB |
| `raster_final` | raster/seed0/final.ckpt | 45.0 MB |
| `mlp_final_s1` | mlp/seed1/final.ckpt | 2.3 MB |
| `mlp_final_s2` | mlp/seed2/final.ckpt | 2.3 MB |
| `vector_final_s1` | vector/seed1/final.ckpt | 0.5 MB |
| `vector_final_s2` | vector/seed2/final.ckpt | 0.5 MB |
| `raster_final_s1` | raster/seed1/final.ckpt | 45.0 MB |
| `raster_final_s2` | raster/seed2/final.ckpt | 45.0 MB |

How to use them: see "Using the trained models" in the repository's README.md.

Checksums (`sha256sum -c SHA256SUMS` in this folder):

```text
3a91f446248de429fe3c3b26b699443bb2ae55dbf5b86996266d218898efd03f  mlp/seed0/early.ckpt
979b939f82658fb2c6fadf0d9e8b69e61043d931b58dc527840a829237d36701  mlp/seed0/final.ckpt
c54e2cb225e1d869a0a4858ac70f74ba7eb4efc843f4a84921a410e2fdbc40e2  mlp/seed1/final.ckpt
c795dff40aaa21f1936ee47a91c8c183bfabd81204de7ba52d08eb4e6a9165bf  mlp/seed2/final.ckpt
a1ecb468555d270e68c93b99b87f63464a39e2cb35d0fabe15ecf8102be69d44  raster/seed0/early.ckpt
3b0da24a3652c109e98fda70ff65c5ca86e4e3378819759975b50efa9e38f43d  raster/seed0/final.ckpt
c1412a8e7668e5f4f60c9b82705a752501b7b6c7e1300631b5ed05d320e374b3  raster/seed1/final.ckpt
0aa52deab5fb601b6b16c8f2035d7bccae2e77a4154e9774d4c6e42b11b19af9  raster/seed2/final.ckpt
137bee84cca7ac0cbbcebf7f0969d774be6d3df1e9bcc0a3fc1c3f5d12c471b9  vector/seed0/early.ckpt
d97bc2b73a022d7ad249887f86903f54f005d2b3442fb4832591b8b86f54f2c9  vector/seed0/final.ckpt
8fe3d96806b6f96fdd202c17286828ef6cb90978dcc345bc9499a98791529733  vector/seed1/final.ckpt
6743c95c2c7b3e9cee7592acc1630ef3a904c721152b19c09d4f8d492f74bf31  vector/seed2/final.ckpt
```

The checkpoints are PyTorch Lightning files (Python pickles): load only copies whose checksums match. The weights were trained on nuPlan data and are shared for non-commercial research use under the nuPlan dataset terms (CC BY-NC-SA 4.0); the MIT licence covers the code only.
