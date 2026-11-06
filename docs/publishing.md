# Publishing this repository on GitHub

The directory is already a git repository with one commit. Three steps remain.

## 1. Create the remote and push

```bash
git remote add origin https://github.com/<owner>/provenance-audit.git
git branch -M main
git push -u origin main
```

If this is a fresh machine, set an identity first (the initial commit used a
placeholder):

```bash
git config user.name  "Your Name"
git config user.email "you@example.com"
git commit --amend --reset-author --no-edit
```

## 2. Fill the placeholders

Search and replace, then commit:

```bash
grep -rn "\[owner\]\|\[Author" README.md CITATION.cff pyproject.toml LICENSE
```

* `README.md` — repository URL, contact
* `CITATION.cff` — authors, affiliations, repository URL
* `pyproject.toml` — `Homepage`, author
* `LICENSE` — copyright holder
* `LICENSE-DATA.md` — contact for takedown requests

## 3. Check what GitHub will show

* The repository is ~25 MB, dominated by `data/benchmark_records.jsonl` (22.8 MB).
  That is below GitHub's 50 MB warning threshold, so no LFS is required. If you later
  add corpora, move the JSONL files to Git LFS:

  ```bash
  git lfs install
  git lfs track "data/*.jsonl"
  git add .gitattributes
  ```

* If the publisher terms do not allow redistributing the abstracts, replace the file
  with the metadata-only variant before pushing:

  ```bash
  git rm --cached data/benchmark_records.jsonl
  echo "data/benchmark_records.jsonl" >> .gitignore
  git add .gitignore data/benchmark_records_metadata_only.jsonl
  git commit -m "Distribute metadata-only corpus"
  ```

  The pipeline then needs `PROVENANCE_DATA_DIR` pointing at the full file, or an
  editorial extension to skip text-dependent stages (see `docs/reproduction.md`).

## 4. Optional: archive with a DOI

GitHub alone is not a citable archive. For a DOI:

1. Run `python scripts/build_release.py` to produce `dist/provenance-audit-v1.0.0.zip`.
2. Upload it to Zenodo (or an institutional repository) with the same metadata as
   `CITATION.cff`.
3. Add the DOI to `CITATION.cff` (`doi:`), to the README badge, and to the paper's data
   availability statement.

## 5. Enable CI

`.github/workflows/ci.yml` runs on push and pull request: it installs the core
dependencies, lints with ruff, runs `pytest`, and executes the source-isolation audit,
asserting 51 protocols and zero document overlap. No secrets or external services are
needed.
