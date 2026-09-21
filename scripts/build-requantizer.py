#!/usr/bin/env python3
"""Link a small quantization driver against Cargo's pinned native build."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Optional alternative native research utility")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "crates/llama-diffusion-sys/vendor/llama.cpp"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["c++", "-std=c++17", "-O2", "-I" + str(source / "include"),
               "-I" + str(source / "ggml/include"), str(args.source or root / "scripts/requantize-gguf.cpp"),
               "-o", str(args.output), "-Wl,--start-group"]
    for line in args.link_manifest.read_text().splitlines():
        if line.startswith("cargo::rustc-link-search=native="):
            command.append("-L" + line.split("=", 2)[2])
        elif line.startswith("cargo::rustc-link-lib="):
            linkage, name = line.split("=", 2)[1:]
            command.append(("-l:lib" + name + ".a") if linkage == "static" else "-l" + name)
    command += ["-Wl,--end-group", "-lpthread", "-ldl", "-Wl,-rpath,/opt/rocm-7.2.1/lib"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
