#!/usr/bin/env python3
"""Export matched HTTP results and verify parity with completed research arms."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="legacy")
    parser.add_argument("--candidate-arm", default="prefill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    experiments = module("experiments", scripts / "jev-experiments.py")
    paired = module("paired", scripts / "jev-paired-report.py")
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    manifests = {name: json.loads((path / "server-manifest.json").read_text())
                 for name, path in paths.items()}
    if any("completed" not in manifest for manifest in manifests.values()):
        raise ValueError("Both HTTP runs must have completed successfully")
    for key in ("task_sha256", "scoring_sha256", "protocol"):
        if manifests["baseline"][key] != manifests["candidate"][key]:
            raise ValueError(f"HTTP protocols differ: {key}")
    if manifests["baseline"]["command"][1:] != manifests["candidate"]["command"][1:]:
        raise ValueError("HTTP server arguments differ beyond the executable")
    command = manifests["baseline"]["command"]
    if int(command[command.index("--seed") + 1]) != args.seed:
        raise ValueError("HTTP seed differs from the requested confirmation seed")
    fixture = Path(manifests["baseline"]["fixture_run"])
    for name, digest in manifests["baseline"]["task_sha256"].items():
        if hashlib.sha256((fixture / "tasks" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen task file changed: {name}")
    scorer_path = fixture / "harness/jevbench/scoring.py"
    if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != manifests["baseline"]["scoring_sha256"]:
        raise ValueError("Frozen scorer changed")
    score = experiments.load_scorer(scorer_path)
    tasks = {task["id"]: task for task in experiments.load_tasks(fixture / "tasks")}
    indexed, exported = {}, {}
    for name, arm in [("baseline", args.baseline_arm), ("candidate", args.candidate_arm)]:
        path = paths[name]
        rows = [json.loads(line) for line in (path / "http/results.jsonl").read_text().splitlines()]
        indexed[name] = {row["task_id"]: row for row in rows}
        if len(rows) != len(tasks) or indexed[name].keys() != tasks.keys():
            raise ValueError(f"Missing, extra, or duplicated HTTP records: {name}")
        confirmation = {task_id: row for (seed, task_id), row in paired.load(args.confirmation, arm).items()
                        if seed == args.seed}
        if confirmation.keys() != tasks.keys():
            raise ValueError(f"Confirmation does not cover the same cases and seed: {arm}")
        mismatches = []
        for row in rows:
            task = tasks[row["task_id"]]
            rescored = score(row.get("probs") or {}, SimpleNamespace(**task))
            if any(rescored[key] != row[key] for key in ("correct", "valid", "predicted")):
                raise ValueError(f"Pinned scorer disagrees: {name}/{task['id']}")
            row.update(arm=name, seed=args.seed, tier=task["tier"],
                       kind=task["question"]["type"], expected=task["expected"])
            if row.get("probs") != confirmation[task["id"]].get("probs"):
                mismatches.append(task["id"])
        summary = experiments.summarize(rows)[0]
        saved_metrics = json.loads((path / "metrics.json").read_text())
        calculated = {key: summary[key] for key in ("total", "correct", "valid")}
        calculated.update(all_p50_s=summary["p50_s"], all_p95_s=summary["p95_s"],
                          valid_p50_s=summary["valid_p50_s"], valid_p95_s=summary["valid_p95_s"])
        if calculated != saved_metrics:
            raise ValueError(f"Saved HTTP metrics disagree with per-case data: {name}")
        exported[name] = dict(source_run=str(path), manifest=manifests[name], summary=summary,
                              confirmation_arm=arm, probability_parity_mismatches=mismatches,
                              per_case=rows)
    pairs = [(row, indexed["candidate"][task_id]) for task_id, row in indexed["baseline"].items()]
    strata = collections.defaultdict(list)
    for a, b in pairs:
        for field in ("tier", "kind", "family"):
            strata[f"{field}:{a[field]}"].append((a, b))
    published = json.loads((scripts.parent / "benchmarks/jevbench/results-2026-09-21.json").read_text())
    report = dict(scope=f"{len(tasks)} frozen public JevBench cases; serial local HTTP; one excluded warmup; not an untouched holdout or official full leaderboard run",
                  confirmation_run=str(args.confirmation), runs=exported,
                  paired=dict(overall=paired.compare(pairs),
                              strata={key: paired.compare(value) for key, value in strata.items()}),
                  published_provenance=published["upstream"],
                  published_public_subset_comparison=published["published_public_subset_comparison"],
                  note="Reference deployments were not rerun. Their latency is not a matched hardware/network comparison. Paired p values are exploratory and unadjusted for selection.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: run["summary"] for name, run in exported.items()}, indent=2))
    if any(run["probability_parity_mismatches"] for run in exported.values()):
        raise SystemExit("Report saved; HTTP/research probability mismatches require investigation")


if __name__ == "__main__":
    main()
