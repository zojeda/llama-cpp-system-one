#!/usr/bin/env python3
"""Compare complete, matched Jev experiment arms by task ID and seed."""
import argparse
import collections
import json
import math
from pathlib import Path


def load(run, arm):
    manifest = json.loads((run / "manifest.json").read_text())
    if "completed" not in manifest:
        raise ValueError(f"Run is not complete: {run}")
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["arm"] == arm]
    indexed = {(row["seed"], row["task_id"]): row for row in rows}
    expected = len(manifest["arm_task_ids"][arm]) * len(manifest["seeds"].split(","))
    if len(indexed) != expected or len(rows) != expected:
        raise ValueError("Missing or duplicate records")
    return indexed


def compare(pairs):
    fixes = [b["task_id"] for a, b in pairs if b["correct"] and not a["correct"]]
    losses = [b["task_id"] for a, b in pairs if a["correct"] and not b["correct"]]
    discordant = len(fixes) + len(losses)
    # Exact two-sided McNemar test; exploratory, without multiplicity adjustment.
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(len(fixes), len(losses)) + 1)) / 2**discordant)
    trace_keys = ("prompt_token_ids", "initial_canvas", "slot_positions", "candidate_tokens")
    audited = mismatches = 0
    for a, b in pairs:
        ta, tb = a.get("read", {}).get("traces"), b.get("read", {}).get("traces")
        if ta and tb:
            audited += 1
            left = [{k: t[k] for k in trace_keys} for t in ta]
            right = [{k: t[k] for k in trace_keys} for t in tb]
            mismatches += left != right
    return dict(total=len(pairs), reference_correct=sum(a["correct"] for a, _ in pairs),
                candidate_correct=sum(b["correct"] for _, b in pairs),
                reference_valid=sum(a["valid"] for a, _ in pairs),
                candidate_valid=sum(b["valid"] for _, b in pairs),
                delta_percentage_points=100 * (len(fixes) - len(losses)) / len(pairs),
                fixed_ids=fixes, regressed_ids=losses, mcnemar_exact_unadjusted_p=p,
                trace_pairs_audited=audited, trace_pairs_differing=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-arm", required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = load(args.reference_run, args.reference_arm)
    candidate = load(args.candidate_run, args.candidate_arm)
    if reference.keys() != candidate.keys():
        raise ValueError("Compared arms must have identical task IDs and seeds")
    grouped = collections.defaultdict(list)
    for key, a in reference.items():
        grouped[key[0]].append((a, candidate[key]))
    results = {}
    for seed, pairs in grouped.items():
        strata = collections.defaultdict(list)
        for a, b in pairs:
            for field in ("tier", "kind", "family"):
                strata[(field, a[field])].append((a, b))
        results[seed] = dict(overall=compare(pairs), strata={
            f"{field}:{value}": compare(group) for (field, value), group in strata.items()
        })
    output = dict(reference_run=str(args.reference_run), reference_arm=args.reference_arm,
                  candidate_run=str(args.candidate_run), candidate_arm=args.candidate_arm,
                  note="Each seed is reported separately; repeated tasks are not independent samples. P values are exploratory and unadjusted for arm selection or multiple comparisons.",
                  seeds=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
