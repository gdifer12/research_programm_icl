"""Run the Assign and Add seed-2 interventions from the original three training runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy

import torch
import yaml

from icl_lab import analysis, config, engine
from icl_lab.runtime import ROOT, require_local_environment
from icl_lab.storage import code_hash, load_torch, lock, sha, write_json
from icl_lab.tasks.assign_add import Task


PARENTS = ROOT / "artifacts/runs/assign-add-jupyter-20261119"
SERIES = ROOT / "artifacts/runs/assign-add-followup-20261120"
CROSS = SERIES / "cross-model2-stream1"
CONTINUATION = SERIES / "continue-seed2"
SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verify_cross(config_cross):
    weak = PARENTS / "seed-2"
    good = PARENTS / "seed-1"
    c1, c2 = [yaml.safe_load((folder / "config.yaml").read_text()) for folder in (good, weak)]
    check(config_cross == dict(c1, model_seed=2) == dict(c2, train_seed=10001),
          "Expected only model_seed or train_seed to differ from the corresponding parent")
    for folder in (weak, good):
        check(read(folder / "manifest.json")["code_hash"] == code_hash(), "Scientific code changed")
    initial = load_torch(weak / "checkpoints/initial.pt", full=True)
    engine.setup(config_cross, "cpu")
    cross_task, good_task = Task(config_cross), Task(c1)
    original = initial["model"]
    generated = cross_task.build_model().state_dict()
    check(original.keys() == generated.keys() and all(torch.equal(original[k], generated[k]) for k in original),
          "Initial parameters differ from the original weak seed")
    check(torch.equal(cross_task.pair_mask, initial["task_state"]["pair_mask"]), "Arithmetic split changed")
    g_cross = torch.Generator().manual_seed(config_cross["train_seed"])
    g_good = torch.Generator().manual_seed(c1["train_seed"])
    digest = hashlib.sha256()
    for step in range(3):
        left = cross_task.sample(512, g_cross, "train", step)
        right = good_task.sample(512, g_good, "train", step)
        check(all(torch.equal(left[key], right[key]) for key in right), "Training streams do not match")
        for key in ("tokens", "target"):
            digest.update(left[key].numpy().tobytes())
    return {"initial_parameters_equal": len(original), "initial_source_sha256": sha(weak / "checkpoints/initial.pt"),
            "first_three_batches_sha256": digest.hexdigest(), "good_stream_seed": 10001}


@torch.inference_mode()
def mechanism_curves(run, settings, steps):
    probes = runpy.run_path(str(SCRIPTS / "diagnose-assign-binding.py"))
    task = Task(settings)
    model = task.build_model().eval()
    tokens, _, targets = probes["make_challenge"](task, seen=True, n=400)
    examples = {k: v for k, v in probes["make_name_test"](task).items() if k in ((1, "b"), (1, "c"))}
    results = {}
    for step in steps:
        path = run / "checkpoints" / f"step-{step:08d}.pt"
        if run == CONTINUATION and step == 30000:
            path = run / "checkpoints/parent-00030000.pt"
        state = load_torch(path, full=True)
        check(state["step"] == step and torch.equal(task.pair_mask, state["task_state"]["pair_mask"]),
              "Diagnostic checkpoint does not match the task or step")
        model.load_state_dict(state["model"])
        correct = int((probes["predict"](model, tokens[:, 0]) == targets[:, 0]).sum())
        results[str(step)] = {"checkpoint_sha256": sha(path), "name_test": probes["name_scores"](model, examples),
                              "six_familiar": {"n": len(tokens), "correct": correct}}
    write_json(run / "followup-diagnostics.json", results)


def main():
    require_local_environment()
    with lock(SERIES):
        state_file = SERIES / "series-state.json"
        previous = read(state_file) if state_file.exists() else {}
        if previous.get("state") == "completed":
            return
        state = {"started_at": previous.get("started_at") or datetime.now(timezone.utc).isoformat(),
                 "pid": os.getpid(), "code_hash": code_hash(), "results": previous.get("results", {})}

        def update(**fields):
            state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(state_file, state)

        try:
            update(state="running", stage="preparing")
            cross_config = config.load(ROOT / "configs/experiments/assign_add.yaml", seed=2)
            cross_config["train_seed"] = 10001
            check_cross = verify_cross(cross_config)
            if not CONTINUATION.exists():
                runpy.run_path(str(SCRIPTS / "prepare_assign_continuation.py"))["prepare"](CONTINUATION)
            fork = read(CONTINUATION / "provenance/continuation.json")
            check(fork["source_sha256"] == sha(PARENTS / "seed-2/checkpoints/last.pt"), "Parent mismatch")
            fork_config = config.merge(config.load(ROOT / "configs/experiments/assign_add.yaml", seed=2),
                                       {"train": {"steps": 60000}})
            parent_bank = read(PARENTS / "seed-2/banks/manifest.json")["banks"]
            for run, settings in ((CROSS, cross_config), (CONTINUATION, fork_config)):
                target = settings["train"]["steps"]
                status = read(run / "status.json") if (run / "status.json").exists() else {}
                existing = (run / "config.yaml").is_file()
                if existing:
                    check(yaml.safe_load((run / "config.yaml").read_text()) == settings, "Saved config mismatch")
                update(stage="training", run=run.name, target=target, cross_checks=check_cross)
                if status.get("state") != "completed":
                    status = engine.train(settings, run, "mps", resume="last" if existing else None)
                check(status["state"] == "completed" and status["step"] == target, "Training stopped short")
                last = load_torch(run / "checkpoints/last.pt", full=True)
                check(last["skipped"] == 0 and last["attempted"] == last["step"] == target, "Invalid update count")
                bank = read(run / "banks/manifest.json")["banks"]
                check({s:v["sha256"] for s,v in bank.items()} == {s:v["sha256"] for s,v in parent_bank.items()},
                      "Evaluation banks differ from parent")
                if run == CROSS:
                    actual = load_torch(run / "checkpoints/initial.pt", full=True)["model"]
                    source = load_torch(PARENTS / "seed-2/checkpoints/initial.pt", full=True)["model"]
                    good_rng = load_torch(PARENTS / "seed-1/checkpoints/last.pt", full=True)["rng"]["train_generator"]
                    check(all(torch.equal(actual[k], source[k]) for k in source), "On-run initial weights differ")
                    check(torch.equal(last["rng"]["train_generator"], good_rng), "Final training stream mismatch")
                update(stage="evaluation", run=run.name)
                accuracy = {}
                for split in ("id_test", "test"):
                    evaluation = engine.evaluate(run, "final", split, "mps")
                    accuracy[split] = evaluation["model"]["accuracy"]["mean"]
                analysis.analyze(run, "final", "mps")
                steps = (1000, 3000, 5000, 10000, 20000, 30000) if run == CROSS else (30000, 40000, 50000, 60000)
                mechanism_curves(run, settings, steps)
                state["results"][run.name] = {"step": target, "accuracy": accuracy,
                                              "final_sha256": sha(run / "checkpoints/final.pt")}
                update(stage="evaluated")
            update(state="completed", stage="ready-for-notebook-analysis", run=None)
        except BaseException as exc:
            update(state="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
