import argparse
import json
import os

import optuna

from train import parse_args, train

UNFREEZE_RANGE = {"resnet50": (1, 4), "mobilenet": (2, 10)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50", choices=list(UNFREEZE_RANGE))
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--trial_epochs", type=int, default=8)
    p.add_argument("--final_epochs", type=int, default=17)
    p.add_argument("--data_dir", default="../data/processed_224")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int)
    a = p.parse_args()

    head = os.path.join(a.results_dir, f"{a.model}_head.pt")
    if not os.path.exists(head):
        raise SystemExit(f"{head} not found. Run the baseline first: python src/train.py --model {a.model}")
    tune_dir = os.path.join(a.results_dir, "tuning")
    os.makedirs(tune_dir, exist_ok=True)

    def run_args(params, name, epochs, out_dir):
        cmd = [
            "--model", a.model, "--data_dir", a.data_dir, "--out_dir", out_dir, "--name", name,
            "--init_ckpt", head, "--head_epochs", "0", "--epochs", str(epochs),
            "--ft_lr", str(params["ft_lr"]), "--weight_decay", str(params["weight_decay"]),
            "--unfreeze", str(params["unfreeze"]), "--batch_size", str(params["batch_size"]),
            "--num_workers", str(a.num_workers),
        ]
        if a.max_batches:
            cmd += ["--max_batches", str(a.max_batches)]
        return parse_args(cmd)

    def objective(trial):
        low, high = UNFREEZE_RANGE[a.model]
        params = {
            "ft_lr": trial.suggest_float("ft_lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "unfreeze": trial.suggest_int("unfreeze", low, high),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        }
        name = f"{a.model}_trial{trial.number}"
        try:
            return train(run_args(params, name, a.trial_epochs, tune_dir), trial)
        finally:
            for suffix in (".pt", "_last.pt"):
                path = os.path.join(tune_dir, name + suffix)
                if os.path.exists(path):
                    os.remove(path)

    study = optuna.create_study(
        study_name=f"{a.model}_tuning",
        direction="maximize",
        storage=f"sqlite:///{os.path.join(tune_dir, a.model + '_study.db')}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    if done < a.trials:
        study.optimize(objective, n_trials=a.trials - done)

    study.trials_dataframe().to_csv(os.path.join(tune_dir, f"{a.model}_trials.csv"), index=False)
    best = study.best_trial
    with open(os.path.join(tune_dir, f"{a.model}_best_params.json"), "w") as f:
        json.dump({"trial": best.number, "val_acc": best.value, "params": best.params}, f, indent=2)
    print(f"best trial {best.number}: val_acc={best.value:.4f} params={best.params}")

    train(run_args(best.params, f"{a.model}_tuned", a.final_epochs, a.results_dir))


if __name__ == "__main__":
    main()