# AdamK EF Non-IID variants

All configs use CIFAR-10, Dirichlet Non-IID `alpha=0.1`, 100 rounds, 300
steps per round, and the existing EF2 omniscient attack construction.

- `v1_ef_nowd`: current EF2 pipeline with optimizer weight decay disabled.
- `v2_ef_nowd_noclip`: V1 plus ARC norm clipping disabled. Top-K compression
  remains enabled.
- `v3_ef_clientmom_b10`: current EF2 pipeline with honest-client EMA momentum
  (`0.9`) before EF/Krum and server AdamK `beta1=0`. Current weight decay and
  ARC clipping remain enabled.
- `v4_ef_masked_active`: current EF2 pipeline with mask-aware AdamK. All
  moments decay normally every step, but only coordinates in the exact Top-K
  row mask update model parameters. Current weight decay and clipping remain.

Each variant has `withoutatt`, `omniscient_foe`, and
`omniscient_signflipping` configs. Results are written below
`results/results_adamk_ef_variants_noniid/`.

The three `new_adamk_ef21_*` configs implement `new_adamk.md` separately from
the preceding classical-EF ablations. They use full per-client EF21 trackers,
client momentum initialized with `u_0=g_0`, projected and full-space
Multi-Krum, a 2:1 Top-K/random-refresh split, pre-aggregation ARC, and an Adam
update whose cap affects only the second moment. No weight decay or learning
rate schedule is enabled.

Run all configs sequentially from `FL_compress`:

```bash
python run_adamk_ef_variants_noniid.py
```

For two persistent GPU queues, shard the sorted config list as follows (the
launcher maps each physical GPU to logical device 0):

```bash
python run_adamk_ef_variants_noniid.py --physical-gpu 2 --shard-index 0 --num-shards 2
python run_adamk_ef_variants_noniid.py --physical-gpu 3 --shard-index 1 --num-shards 2
```

Completed `state_round_99.pth` jobs are skipped and partial jobs resume from
their latest checkpoint. Per-job launcher output is stored in
`launch_logs_adamk_ef_variants_noniid/`.
