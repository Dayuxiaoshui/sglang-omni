# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys


def test_nixl_legacy_import_fallback() -> None:
    script = """
import importlib
import sys
from types import ModuleType

sys.modules["nixl_cu13"] = None
legacy = ModuleType("nixl")
legacy.__path__ = []
api = ModuleType("nixl._api")


class FakeAgent:
    pass


class FakeConfig:
    pass


api.nixl_agent = FakeAgent
api.nixl_agent_config = FakeConfig
sys.modules["nixl"] = legacy
sys.modules["nixl._api"] = api

module = importlib.import_module("sglang_omni.relay.nixl")
assert module.NIXL_AVAILABLE is True
assert module.NixlAgent is FakeAgent
assert module.nixl_agent_config is FakeConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
