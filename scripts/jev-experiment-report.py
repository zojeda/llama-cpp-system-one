#!/usr/bin/env python3
"""Export compact experiment results from retained raw run directories."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path


def encode_report(report):
    """Keep metadata readable and put each complete case on one JSON line."""
    runs = []
    for run in report["runs"]:
        metadata = {key: value for key, value in run.items() if key != "per_case"}
        prefix = json.dumps(metadata, indent=2, ensure_ascii=False)[:-1].rstrip()
        cases = ",\n".join("    " + json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                            for case in run["per_case"])
        runs.append(prefix + ',\n  "per_case": [\n' + cases + "\n  ]\n}")
    return ('{\n  "scope": ' + json.dumps(report["scope"]) + ',\n  "runs": [\n'
            + ",\n".join(runs) + "\n  ]\n}\n")


def stratified_summary(rows, summarize):
    output = {}
    for field in ("tier", "kind", "family"):
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        output[field] = {value: summarize(group) for value, group in groups.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("experiments", Path(__file__).with_name("jev-experiments.py"))
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)
    exported = []
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
        arm_ids = manifest.get("arm_task_ids", {name: manifest["task_ids"] for name in manifest["plan"]})
        expected = sum(map(len, arm_ids.values())) * len(manifest["seeds"].split(","))
        exported.append({
            "source_run": str(run), "complete": "completed" in manifest,
            "expected_records": expected, "record_count": len(rows),
            "manifest": manifest, "summary": experiments.summarize(rows),
            "strata": stratified_summary(rows, experiments.summarize),
            "per_case": [{key: row.get(key) for key in [
                "task_id", "arm", "seed", "family", "tier", "kind", "expected", "predicted",
                "correct", "valid", "strict_valid", "latency_s", "probs", "read", "error",
            ]} for row in rows],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encode_report({"scope":"Public JevBench cases; in-process latency, not HTTP", "runs":exported}))


if __name__ == "__main__":
    main()
