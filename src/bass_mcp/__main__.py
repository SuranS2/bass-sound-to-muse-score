"""CLI entrypoint for running the bass-sound-to-muse-score MCP server over stdio."""

from __future__ import annotations

from bass_mcp.server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
