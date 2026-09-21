#!/usr/bin/env python3
"""Inspect GGUF metadata/tensor descriptors without reading tensor payloads."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct


def inspect(path, wanted_tokens, wanted_arrays=()):
    formats = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}
    with path.open("rb") as f:
        def number(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return f.read(number("Q")).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = number("I"), number("Q")
                return [value(subtype) for _ in range(count)]
            return number(formats[kind])

        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF")
        version, tensors, fields = number("I"), number("Q"), number("Q")
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        metadata, tokenizer_hashes, decoded, selected_arrays = {}, {}, {}, {}
        for _ in range(fields):
            key, kind = string(), number("I")
            start = f.tell()
            v = value(kind)
            if key in wanted_arrays:
                selected_arrays[key] = v
            end = f.tell()
            if key.startswith("tokenizer."):
                f.seek(start)
                tokenizer_hashes[key] = hashlib.sha256(f.read(end-start)).hexdigest()
            if key == "tokenizer.ggml.tokens":
                decoded = {str(i): v[i] for i in wanted_tokens if 0 <= i < len(v)}
            if not isinstance(v, list) and key != "tokenizer.chat_template":
                metadata[key] = v
        types = collections.Counter()
        descriptors = []
        for _ in range(tensors):
            name = string()
            dimensions = [number("Q") for _ in range(number("I"))]
            kind, offset = number("I"), number("Q")
            types[kind] += 1
            descriptors.append({"name":name, "shape":dimensions, "type":kind, "offset":offset})
        return dict(model_file=path.name, size_bytes=path.stat().st_size, gguf_version=version,
                    metadata=metadata, tokenizer_value_sha256=tokenizer_hashes, decoded_tokens=decoded,
                    selected_arrays=selected_arrays,
                    tensor_type_counts=dict(types), tensors=descriptors, header_bytes=f.tell())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-ids", default="")
    parser.add_argument("--arrays", default="", help="Comma-separated metadata arrays to retain")
    args = parser.parse_args()
    result = inspect(args.model, [int(i) for i in args.token_ids.split(",") if i], args.arrays.split(","))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
