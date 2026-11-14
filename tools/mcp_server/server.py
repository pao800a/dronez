"""FastMCP (``MCPServer`` in mcp 2.x) stdio server exposing the ``dronez-sim`` tools.

Launch from the repo root: ``uv run python -m tools.mcp_server``.
"""

from __future__ import annotations

from mcp.server import MCPServer
from tools.mcp_server import core

mcp = MCPServer("dronez-sim")

for _fn in (
    core.list_runs,
    core.training_status,
    core.tb_scalars,
    core.list_checkpoints,
    core.eval_summary,
    core.gate_check,
    core.model_info,
    core.sim_rollout,
):
    mcp.tool()(_fn)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
