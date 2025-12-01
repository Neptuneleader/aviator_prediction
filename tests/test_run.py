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


def test_model_save_and_predict():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script = os.path.join(repo_root, "aviator_prediction.py")
    sample = os.path.join(repo_root, "sample_aviator_data.csv")
    model_path = os.path.join(repo_root, "model.joblib")

    # Train and save model
    proc_train = subprocess.run(
        [
            sys.executable,
            script,
            "--data",
            sample,
            "--save-model",
            model_path,
            "--seed",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert proc_train.returncode == 0, proc_train.stdout
    assert os.path.exists(model_path)

    # Predict from saved model
    proc_pred = subprocess.run(
        [
            sys.executable,
            script,
            "--data",
            sample,
            "--predict-artifact",
            model_path,
            "--player-id",
            "1001",
            "--bet-amount",
            "50",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert proc_pred.returncode == 0, proc_pred.stdout
    assert "Predicted probability of winning:" in proc_pred.stdout