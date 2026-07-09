import os
import subprocess
import sys


def _run_help(cmd):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, *cmd, "--help"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_root_wrappers_help():
    for script in [
        "train.py",
        "train100k.py",
        "evaluate.py",
        "evaluate100k.py",
        "sample.py",
        "plot_results.py",
        "run_all_tests.py",
        "dataset.py",
    ]:
        _run_help([script])


def test_scripts_help():
    for script in [
        "scripts/train.py",
        "scripts/train100k.py",
        "scripts/evaluate.py",
        "scripts/evaluate100k.py",
        "scripts/sample.py",
        "scripts/plot_results.py",
        "scripts/run_all_tests.py",
    ]:
        _run_help([script])
