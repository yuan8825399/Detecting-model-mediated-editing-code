# Contributing

Issues and pull requests are welcome, particularly replications on other corpora, models
or languages.

## Adding or changing an analysis

1. Put the script in `analysis/` with the next free numeric prefix and a descriptive
   name (`45_my_analysis.py`).
2. Resolve every path through `provenance.config`; never hard-code an absolute path.
3. Write outputs to `results/` (numbers) or `figures/` (images), with stable filenames.
4. Register the stage in `scripts/run_pipeline.py` (`STAGES`, and `CORE` if it runs in
   under a minute without fitting models).
5. Add the paper element it produces to `docs/results_map.md`.
6. Commit the regenerated outputs alongside the script.

## Style

* Python 3.10+, `ruff check` clean, line length 100.
* Seeded randomness; document the seed in the docstring.
* One analysis per script; print the numbers you write to disk.

## Tests

`pytest -q` must pass. Data-integrity assertions live in `tests/test_corpus.py`; if you add
a derived quantity that the paper cites, add a test for its invariants.

## Reporting a numerical discrepancy

Open an issue with the table or figure number, the value you obtain, the command you ran
and your environment (`pip freeze`). Every number in the paper should be traceable to a
file in `results/`.
