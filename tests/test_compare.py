"""Test compare.py handles directory and file inputs correctly."""
import json
import tempfile
from pathlib import Path

from benchmark.compare import load_results


def test_load_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some fake results
        for i, method in enumerate(["ncm", "icarl", "ewc"]):
            data = {"protocol": "B1", "method": method, "final_oa": 0.5 + i * 0.1}
            with open(Path(tmpdir) / f"{method}_B1.json", "w") as f:
                json.dump(data, f)

        results = load_results([tmpdir])
        assert len(results) == 3


def test_load_from_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            with open(Path(tmpdir) / f"test_{i}.json", "w") as f:
                json.dump({"protocol": "B1", "method": f"m{i}", "final_oa": 0.5}, f)

        results = load_results([str(Path(tmpdir) / "*.json")])
        assert len(results) == 3


def test_load_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        results = load_results([tmpdir])
        assert len(results) == 0


def test_load_invalid_json_skipped():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(Path(tmpdir) / "bad.json", "w") as f:
            f.write("not valid json {{{")
        with open(Path(tmpdir) / "good.json", "w") as f:
            json.dump({"protocol": "B1", "method": "test", "final_oa": 0.5}, f)

        results = load_results([tmpdir])
        assert len(results) == 1
