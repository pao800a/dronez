# ADR-0002: Public GitHub repo `pao800a/dronez`, noreply commit identity

- **Status:** Accepted (2026-11-13)

## Context
CI runs on both `ubuntu-latest` and `windows-latest` and includes a smoke training job. On private repos, Actions minutes are capped and Windows runners count double.

## Decision
- The repository is `github.com/pao800a/dronez`, **public**.
- Commits use a repo-local git identity, `pao800a <73484016+pao800a@users.noreply.github.com>`, so no personal email address appears in a public history.
- Workflow: one branch per phase, conventional commits, a PR per phase, merge on green CI, never force-push `main`.

## Consequences
Standard runners have unlimited Actions minutes. Nothing secret may ever be committed; `.mcp.json` uses `${VAR}` expansion only.
