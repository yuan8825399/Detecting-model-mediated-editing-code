# Data licence and third-party content

This repository contains three kinds of material, with different terms.

**1. Code** (`provenance/`, `analysis/`, `scripts/`, `tests/`). MIT, see `LICENSE`.

**2. Generated data.** The edited variants and all derived tables and figures were
produced for this study and are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**3. Source abstracts.** The 493 human-written abstracts and their titles are third-party
published material, reproduced here to make the reported analyses reproducible. They are
**not** covered by the CC BY 4.0 dedication. Before redistributing
`data/benchmark_records.jsonl` outside the terms under which the original publications were
accessed, confirm that your use permits it.

If it does not, use `data/benchmark_records_metadata_only.jsonl`, which retains the record
structure, labels, hashes and all numeric fields but omits the four text fields
(`abstract_raw`, `abstract_clean`, `source_title_raw`, `source_title_clean`) and sets
`text_withheld = true`.

## Attribution request

If you reuse the corpus, please cite both the article and the archive DOI.

## Takedown

If you hold rights in a source abstract and want it removed, contact the corresponding
author; the record will be removed and the affected aggregates recomputed.
