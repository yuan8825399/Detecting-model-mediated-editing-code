# Data dictionary

`benchmark_records.jsonl` contains one JSON object per line; 8,381 records from 493
source abstracts. Each source contributes exactly 17 records: the original plus
4 instructions x 4 models.

| Field | Type | Description |
|---|---|---|
| `record_id` | string | unique record identifier |
| `source_id` | string | source abstract, `china-law-YYYY-NNNNN`; year matches `source_year` |
| `group_id` | string | document group; the unit for all data splits |
| `parent_record_id` | string | populated for edited records; names the source record |
| `corpus_id` | string | `H` for human originals, `E` for edited variants |
| `intervention_present` | 0/1 | 1 for any model-processed record |
| `intervention_level` | 0-4 | 0 original, 1 E1, 2 E2, 3 E3, 4 E4 |
| `operation` | string | `original`, `grammar_correction`, `academic_polish`, `sentence_restructure`, `abstract_rewrite` |
| `provider` | string | `deepseek`, `doubao`, `kimi`, `wenxin`; empty for originals |
| `model_name` | string | API model identifier |
| `model_snapshot` | string | vendor snapshot label |
| `prompt_id` | string | E1-E4 |
| `group_split` | string | `train`, `dev`, `calibration`, `test` |
| `source_year` | string | publication year, 2000-2020 |
| `abstract_raw` / `abstract_clean` | string | text before / after deterministic cleaning |
| `char_count` | int | characters in `abstract_clean` |
| `chinese_ratio` | float | share of CJK characters |
| `normalized_edit_distance` | float | character Levenshtein distance / source length |
| `numeric_retention` | float | share of numeric tokens preserved |
| `quality_status` / `quality_reason` | string | generation-quality gate |
| `input_sha256` / `output_sha256` | string | content hashes of input and output text |
| `title_sha256` | string | hash of the cleaned title |
| `normalization_applied` | bool | whether cleaning altered the record |
| `text_withheld` | bool | true only in the metadata-only variant |

`benchmark_records_metadata_only.jsonl` has the same schema with the four text fields
removed and `text_withheld = true`. Use it when full-text redistribution is not permitted.

## Not recorded

Prompt text, decoding parameters, seeds, request timestamps, retry/error logs,
bibliographic database, search string, screening log. See Table 1 of the manuscript.
