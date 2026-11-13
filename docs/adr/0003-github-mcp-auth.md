# ADR-0003: GitHub MCP authenticated with the gh CLI token

- **Status:** Accepted (2026-11-13)

## Context
The official remote GitHub MCP server (`https://api.githubcopilot.com/mcp/`) needs a bearer token. The gh CLI is already logged in as `pao800a`, with scopes `repo, workflow`, which cover repos, PRs, issues and Actions.

## Decision
- `.mcp.json` sends `Authorization: Bearer ${GITHUB_PAT}` and `X-MCP-Toolsets: repos,issues,pull_requests,actions`.
- `GITHUB_PAT` is exported from the shell profile: `export GITHUB_PAT="$(gh auth token)"`. No separate PAT has to be created or rotated.
- `gh` is the fallback for every operation (repo creation, PRs, `gh pr checks`, `gh run view`).

## Consequences
The token's scope is the gh login's scope (wider than a fine-grained PAT). A fine-grained PAT scoped to this repo can replace it at any time without touching `.mcp.json`.
