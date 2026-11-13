# ADR-0004: Context7 MCP, keyless by default

- **Status:** Accepted (2026-11-13)

## Context
Context7 serves current docs for MuJoCo, Gymnasium, SB3, build123d, onnx/onnxruntime and PyTorch. It works without a key, at lower rate limits.

## Decision
`.mcp.json` registers `https://mcp.context7.com/mcp` with the header `CONTEXT7_API_KEY: ${CONTEXT7_API_KEY:-}`. A key can be added later by exporting the variable, with no file change.

## Consequences
If rate limits bite, export a key. If Context7 is unavailable, fall back to the upstream docs via WebFetch.
