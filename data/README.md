# Data

`benchmark_records.jsonl` — 8,381 records, one JSON object per line, 493 source abstracts
with 17 records each (the original plus 4 instructions x 4 models). Field descriptions are
in `docs/data_dictionary.md`.

`benchmark_records_metadata_only.jsonl` — the same records with the four text fields
removed and `text_withheld = true`. Use this variant if redistribution of the source
abstracts is not permitted; the analysis pipeline reads the full file by default.

## Content warning and terms

The 493 source abstracts are third-party published material, included to make the reported
analyses reproducible. The edited variants were generated for this study. Before
redistributing the full file, confirm that the publisher terms permit it (see
`../LICENSE-DATA.md`).

## Integrity

```bash
python -c "from provenance import corpus; print(corpus.load().shape)"
# (8381, 39)
```

Record-level hashes are stored per record (`input_sha256`, `output_sha256`,
`title_sha256`); the pipeline verifies that every edited record's input hash matches its
source's output hash (see `tests/test_corpus.py`).
