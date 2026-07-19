# qa_grounded_cloze_data_wanghaonan

Builds a clean, document-grounded fill curriculum without teacher generation:

- exact answer spans are masked from trusted reference sentences;
- every answer must reappear in the exact visible Top-4 observation;
- one-hop rows supervise stopping immediately when evidence is complete;
- two-hop rows combine independent sources, require strict evidence increment,
  and exclude every first-hop source on the second search;
- source-level train/validation isolation and official-313 question isolation;
- balanced objective replay remains 25%–35% of training exposures.

This experiment builds and audits data only. It does not train a model.
