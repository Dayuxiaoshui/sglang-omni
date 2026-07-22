# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import Annotated

import typer

from sglang_omni.diagnostics.gpu import (
    collect_gpu_diagnostics,
    render_gpu_diagnostics,
)


def check_gpu(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the diagnostic report as machine-readable JSON.",
        ),
    ] = False,
) -> None:
    """Report GPU mapping, topology, P2P, and installed backends."""

    report = collect_gpu_diagnostics()
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    typer.echo(render_gpu_diagnostics(report))
