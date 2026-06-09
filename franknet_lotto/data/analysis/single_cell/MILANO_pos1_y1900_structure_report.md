# Structure report: `MILANO_pos1_y1900_series.csv`

## Sizes

- n: 9309, diffs: 9308, digital_root: 9309

## Markov (chronological train/test)

| stream | order | test acc | baseline (majority) |
|--------|-------|----------|---------------------|
| digital_root | 0 | 0.1155 | 0.1155 |
| digital_root | 1 | 0.1139 | 0.1155 |
| digital_root | 2 | 0.1230 | 0.1155 |
| digital_root | 3 | 0.1117 | 0.1155 |
| digit_sum | 1 | 0.1004 | 0.1015 |
| digit_sum | 2 | 0.0913 | 0.1015 |
| diff_signed_q12 | 1 | 0.1445 | 0.0849 |
| diff_signed_q12 | 2 | 0.1477 | 0.0849 |
| n_mod10 | 1 | 0.1042 | 0.0999 |
| n_mod10 | 2 | 0.1053 | 0.0999 |
| n_mod10 | 3 | 0.1074 | 0.0999 |

## AR(1) on raw n

```json
{
  "a": 0.008346,
  "b": 45.250927,
  "train_r2": 7e-05,
  "test_mae_ar1": 22.7857,
  "test_mae_predict_mean": 22.7871,
  "test_n": 1862
}
```

## Shuffle null — Markov order 1, digital_root

```json
{
  "real_gain_over_majority": -0.001611,
  "shuffle_gains_min_median_max": [
    -0.030075,
    -0.003759,
    0.017723
  ],
  "real_gain_exceeds_shuffle_fraction": 0.625,
  "n_shuffles": 40,
  "note": "If real_gain is not above most shuffles, Markov signal is weak."
}
```

## Shuffle null — Markov order 2, diff quantile-12 bins

```json
{
  "real_gain_over_majority": 0.062836,
  "shuffle_gains_min_median_max": [
    -0.017723,
    -0.003759,
    0.015575
  ],
  "real_gain_exceeds_shuffle_fraction": 1.0,
  "n_shuffles": 40,
  "note": "If real_gain is not above most shuffles, Markov signal is weak."
}
```

## Top repeated motifs (diff_signed, length 2–4)

- len 2 ×6: `[0, -2]`
- len 2 ×6: `[15, -2]`
- len 2 ×5: `[-22, 21]`
- len 2 ×5: `[-10, 1]`
- len 2 ×5: `[-7, -4]`
- len 2 ×5: `[7, 28]`
- len 2 ×5: `[17, 19]`
- len 2 ×5: `[23, 4]`
- len 2 ×5: `[27, -14]`
- len 2 ×4: `[-55, -2]`

## Top repeated motifs (digital_root, length 3–5)

- len 5 ×4: `[1, 8, 9, 7, 1]`
- len 5 ×4: `[2, 7, 3, 3, 8]`
- len 5 ×4: `[4, 5, 1, 8, 5]`
- len 5 ×4: `[6, 6, 6, 6, 6]`
- len 5 ×4: `[6, 8, 9, 7, 9]`
- len 5 ×3: `[1, 1, 6, 8, 1]`
- len 5 ×3: `[1, 8, 5, 4, 3]`
- len 5 ×3: `[2, 1, 4, 4, 5]`
- len 5 ×3: `[2, 8, 1, 5, 8]`
- len 5 ×3: `[3, 2, 7, 2, 5]`