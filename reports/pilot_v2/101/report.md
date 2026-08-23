# 5-arm pilot report

All conditions use the same initialization, batch schedule, tokenizer, and 500-million-token target.

Final NLL is recomputed from each final checkpoint on the held-out `final_evaluation` partition with padding targets masked.

| Condition | Tokens | Final NLL | Perplexity | Median tokens/s | Training hours |
|---|---:|---:|---:|---:|---:|
| flash_baseline | 500039680 | 1.293561 | 3.646 | 126661.8 | 1.096 |
| flex_noop | 500039680 | 1.292766 | 3.643 | 133857.4 | 1.039 |
| matched_program_prior | 500039680 | 1.284155 | 3.612 | 24415.2 | 5.694 |
| incorrect_program_prior | 500039680 | 1.294342 | 3.649 | 24371.4 | 5.702 |
| wide_window_control | 500039680 | 1.289365 | 3.630 | 28277.6 | 4.913 |

The matched arm uses programs fitted only on the reserved discovery partition. The incorrect arm preserves the exact number of preferred edges in every causal row.
