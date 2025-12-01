import subprocess
import sys
import os


def test_script_runs_on_sample_csv():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script = os.path.join(repo_root, "aviator_prediction.py")
    sample = os.path.join(repo_root, "sample_aviator_data.csv")
    assert os.path.exists(script)
    assert os.path.exists(sample)

    proc = subprocess.run(
        [sys.executable, script, "--data", sample, "--test-size", "0.5", "--seed", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    # Should complete and print metrics
    assert proc.returncode == 0, proc.stdout
    assert "Accuracy:" in proc.stdout