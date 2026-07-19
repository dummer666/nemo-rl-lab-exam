# qa_fill_sft_pilot_data_wanghaonan

Builds the immutable data pack for the user-authorized fill SFT pilot.

- Reads the 50 strictly accepted fill trajectories from `raysubmit_TKZScBBXRS7PpZgB`.
- Preserves the existing `38/5/7` train/validation/holdout split.
- Never restores the 19 question-answer leakage rows.
- Requires at least one true two-hop trajectory in the training split.
- Adds 15 balanced objective train rows and 6 objective validation rows.
- Rechecks official-validation and cross-split fingerprint isolation.
- Writes data only; it never loads policy weights or starts training.

```bash
uv run lab validate qa_fill_sft_pilot_data_wanghaonan
uv run lab submit qa_fill_sft_pilot_data_wanghaonan
```
