# qa_fill_sft_v3_data_wanghaonan

Builds the stronger two-hop fill SFT pack authorized by the user.

- Keeps 31 one-hop train fills at one exposure each.
- Repeats the 7 true two-hop train fills four times, producing 28 exposures.
- Selects 24 balanced objective replay rows, eight per objective type.
- Keeps validation and RL holdout unchanged.
- Never restores leakage rows or changes source splits.
- Writes data only; no policy model is loaded.
