#!/usr/bin/env python3
"""Benchmark one local server binary with the frozen public JevBench harness.

Run baseline and candidate serially with identical arguments. Starts and stops
only its own server, uses one excluded warmup, and retains all HTTP attempts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture-run", type=Path, required=True,
                        help="Frozen benchmark directory containing rerun.sh, tasks/, and harness/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    endpoint = f"http://127.0.0.1:{args.port}"
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
               "--bind", f"127.0.0.1:{args.port}", "--context-size", str(args.context),
               "--seed", str(args.seed), "--threads", "8", "--batch-size", "512",
               "--model-id", "gemmadiffusion-0.1"]
    if args.flash_attention:
        command.append("--flash-attention")
    task_files = [args.fixture_run / "tasks" / f"{tier}.jsonl" for tier in ("easy", "original", "hard")]
    expected_ids = [json.loads(line)["id"] for path in task_files for line in path.read_text().splitlines()]
    assert len(set(expected_ids)) == len(expected_ids), "Duplicate fixture IDs"
    env = dict(os.environ)
    env.pop("TYPESAFE_API_KEY", None)
    env.pop("DIFFUSION_MMPROJ", None)
    env.update(HSA_ENABLE_DXG_DETECTION="1", RUST_LOG="info", JEVBENCH_ENDPOINT=endpoint,
               JEVBENCH_KEY_ENV="", JEVBENCH_MODEL="gemmadiffusion-0.1")
    env["LD_LIBRARY_PATH"] = "/opt/rocm-7.2.1/lib:" + env.get("LD_LIBRARY_PATH", "")
    runner = Path(__file__).resolve()
    repository = runner.parents[1]
    manifest = dict(started=time.time(), command=command,
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
                    fixture_run=str(args.fixture_run.resolve()),
                    task_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in task_files},
                    scoring_sha256=hashlib.sha256((args.fixture_run / "harness/jevbench/scoring.py").read_bytes()).hexdigest(),
                    protocol="Serial pinned JevBench HTTP adapter; 120-second request timeout; no retries; one excluded warmup; model loading excluded; type-7 percentiles")
    manifest_path = args.output / "server-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / runner.name).write_bytes(runner.read_bytes())
    (args.output / "source.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=repository))
    log_path = args.output / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            # Require this process's readiness message so an occupied port cannot
            # accidentally direct the benchmark to an unrelated service.
            while "System One service is ready" not in log_path.read_text(errors="replace"):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
                time.sleep(1)
            warmup = {"model": "gemmadiffusion-0.1", "state": "A red ball.",
                      "questions": {"decision": {"type": "noul", "instructions": "Is the ball red?"}}}
            request = Request(endpoint + "/v1/systemone", json.dumps(warmup).encode(),
                              {"Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                warmed = json.load(response)
            assert "answers" in warmed, warmed
            (args.output / "excluded-warmup.json").write_text(json.dumps(warmed, indent=2) + "\n")
            subprocess.run(["bash", str(args.fixture_run.resolve() / "rerun.sh"),
                            str(args.output / "http")], env=env, check=True)
            # Exercise the multi-question example and HTTP validation after timing
            # has finished, so these requests cannot affect benchmark warmup.
            with (args.output / "smoke-test.log").open("w") as smoke:
                subprocess.run(["python3", str(Path(__file__).with_name("smoke-test.py")),
                                "--url", endpoint], env=env, stdout=smoke,
                               stderr=subprocess.STDOUT, check=True)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    if server.returncode != 0:
        raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
    rows = [json.loads(line) for line in (args.output / "http/results.jsonl").read_text().splitlines()]
    assert len(rows) == len(expected_ids) and {r["task_id"] for r in rows} == set(expected_ids), "Incomplete or duplicated HTTP benchmark results"
    valid = [row for row in rows if row["valid"]]
    metrics = dict(total=len(rows), correct=sum(row["correct"] for row in rows), valid=len(valid))
    for name, cases in [("all", rows), ("valid", valid)]:
        for label, fraction in [("p50", .5), ("p95", .95)]:
            metrics[f"{name}_{label}_s"] = percentile([row["latency_s"] for row in cases], fraction)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest["completed"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
