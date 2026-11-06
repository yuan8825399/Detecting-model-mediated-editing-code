"""Check result contracts without fitting models."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "run_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_method_probe_precedes_its_consumers():
    names = [stage[0] for stage in pipeline.STAGES]
    assert names.index("method-probe") < names.index("palcp")
    assert names.index("main") < names.index("tables")
    for name, script, *_ in pipeline.STAGES:
        assert (ROOT / "analysis" / script).is_file(), name


def test_empty_results_fail_before_a_dependent_model_is_fitted(tmp_path):
    env = os.environ.copy()
    env["PROVENANCE_RESULTS_DIR"] = str(tmp_path / "results")
    env["PROVENANCE_PROCESSED_DIR"] = str(tmp_path / "processed")
    env["PROVENANCE_FIGURES_DIR"] = str(tmp_path / "figures")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--only", "palcp"],
        env=env, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 1
    assert "method_probe.json" in completed.stdout
    assert "main_results_base.csv" in completed.stdout
    assert not list((tmp_path / "results").glob("*"))
