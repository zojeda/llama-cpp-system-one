# HTTP API

[Back to README](../README.md)

## Start the service

```bash
export TYPESAFE_API_KEY="local-development-key"
cargo run --release --locked -p llama-cpp-system-one -- \
  -m "$DIFFUSION_MODEL" \
  --bind 127.0.0.1:8080 \
  --context-size 4096 --batch-size 512 --seed 42
```

For AMD GPUs, add `--features hip,native` before `--` after following the [ROCm setup](build.md#rocmhip).

If you omit both `--api-key` and `TYPESAFE_API_KEY`, the server disables authentication. A configured key protects `/v1/*`; `/health` remains open. Logs include counts and timing, excluding request state and credentials. Ctrl-C and SIGTERM drain pending requests and release the model.

## Routes

| Route | Purpose |
| --- | --- |
| `POST /v1/systemone` | Evaluate `state`, `model`, and one or more `questions`. |
| `GET /v1/models` | List models with `name`, `description`, and `release_date`. |
| `GET /health` | Check model readiness and worker availability. |

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/system-one.json
```

Run this from the repository root. The [example request](../examples/system-one.json) includes all three question types.

## Questions and answers

| Type | Input | Output |
| --- | --- | --- |
| `noul` | Instructions, with optional `true`/`false` criteria | Probability of yes |
| `choice` | 1 to 128 named options; descriptions may contain JSON or null | Highest-probability label, distribution, entropy confidence |
| `score` | 2 to 10 string rubric levels | Expected zero-based level, legend, distribution, entropy confidence |

Pass `state` and each question's optional `instructions` as strings, objects, or arrays. We serialize structured content as JSON. We preserve question IDs in the response and keep them outside the model prompt. External labels can span multiple tokens; the compiler maps them to one-token answer codes.

See [probability math](inference.md#from-logits-to-answers) for score and confidence calculations. These restricted distributions express preference among your options; they do not reproduce Codiv/OpenJev's calibrated probabilities, uncertainty rereads, training, or throughput.

## Models and clients

The served model ID defaults to `gemmadiffusion-0.1`; change it with `--model-id`. The server accepts `gemmadiffusion-latest`, `openjev-latest`, and `jev-latest` as routing aliases and reports the local model ID in responses. Those aliases do not identify Codiv's hosted model.

For a TypeSafe client, set `TYPESAFE_BASE_URL=http://127.0.0.1:8080` and use a listed model or `jev-latest`. The server runs without a TypeSafe SDK or a connection to Codiv's infrastructure.

With Node.js 20+ and a running service:

```bash
just js-install
just js-example models
just js-example system-one "Steel bars reinforce concrete."
just js-example errors
```

The [JavaScript guide](../examples/javascript/README.md) covers connection settings and npm commands. For these examples, set the client shell's `TYPESAFE_API_KEY` to the server key if authentication is enabled. The benchmark uses separate local and hosted credentials; follow its [runner guide](../benchmarks/system-one/README.md#run).

## Limits

The server accepts `steps=1`, `samples=1`, `think=0`, `sequential=false`, and empty `images`. It returns `422` for unsupported extension values. This version exposes text classification and scoring; it has no free-text generation or image endpoint.

Fit the full canvas within `--batch-size` and prompt plus canvas within `--context-size`. The defaults are 512 and 4096 tokens. Oversized requests return `422`. Increase those limits to accommodate larger requests, with the corresponding memory cost. We chunk prompt prefill and evaluate the canvas in one batch.

The worker handles one request at a time. `--queue-capacity` defaults to eight waiting requests. It skips disconnected queued requests and lets an active native forward finish. Configure TLS, rate limits, accounts, and billing outside this service.

## Errors

Responses include `x-typesafe-request-id`. Validation errors use `{"detail":[{"loc":...,"msg":...,"type":...}]}`. Other errors use `{"detail":{"error_type":...,"message":...}}`.

| Status | Meaning |
| --- | --- |
| 401 / 403 | Incorrect key / missing key with authentication enabled |
| 404 | Unknown model or path |
| 413 | Body exceeds 8 MiB |
| 422 | Invalid input, unsupported extension, or token capacity exceeded |
| 529 | Full queue; `retry-after: 1` accompanies the response |
| 503 | Inference worker unavailable |
| 500 | Native inference or internal response mapping failure |

Protocol references: [System One](https://codiv.ai/docs/api-reference/system-one), [errors](https://codiv.ai/docs/api-reference/errors), [models](https://codiv.ai/docs/api-reference/models), and [confidence](https://codiv.ai/docs/guides/confidence). The integration targets the text contract reviewed on 2026-09-19.
