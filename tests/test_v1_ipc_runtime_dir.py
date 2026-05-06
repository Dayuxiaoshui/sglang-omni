# SPDX-License-Identifier: Apache-2.0
"""Omni V1 IPC runtime directory lifecycle tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("torch")

from sglang_omni_v1.config.compiler import (
    compile_pipeline,
    compile_pipeline_core,
    create_ipc_runtime_dir,
)
from sglang_omni_v1.config.schema import EndpointsConfig, PipelineConfig, StageConfig


def noop_factory():
    return None


def _make_config(base_path: str) -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                factory="tests.test_v1_ipc_runtime_dir.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(
            scheme="ipc",
            base_path=base_path,
        ),
    )


class TestV1IpcRuntimeDir(unittest.TestCase):
    def test_ipc_runtime_dirs_are_unique_for_same_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            runtime_a = create_ipc_runtime_dir(config)
            runtime_b = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_a)
            self.assertIsNotNone(runtime_b)

            try:
                self.assertNotEqual(runtime_a.path, runtime_b.path)

                _coordinator_a, stages_a, _ = compile_pipeline_core(
                    config,
                    ipc_runtime_dir=runtime_a,
                )
                _coordinator_b, stages_b, _ = compile_pipeline_core(
                    config,
                    ipc_runtime_dir=runtime_b,
                )

                self.assertNotEqual(
                    stages_a[0].control_plane.recv_endpoint,
                    stages_b[0].control_plane.recv_endpoint,
                )
            finally:
                runtime_a.close()
                runtime_b.close()

    def test_compile_pipeline_rejects_unmanaged_ipc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            with self.assertRaisesRegex(ValueError, "does not manage IPC"):
                compile_pipeline(config)

    def test_compile_core_cleans_owned_ipc_dir_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            with patch(
                "sglang_omni_v1.config.compiler._compile_stage",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    compile_pipeline_core(config)

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    def test_caller_owned_ipc_dir_is_not_removed_on_compile_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            with patch(
                "sglang_omni_v1.config.compiler._compile_stage",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    compile_pipeline_core(config, ipc_runtime_dir=runtime_dir)

            self.assertTrue(runtime_path.exists())
            runtime_dir.close()
            self.assertFalse(runtime_path.exists())


class TestV1MultiProcessRunnerIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_mp_runner_cleans_runtime_dir_on_start_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni_v1.pipeline.mp_runner import MultiProcessPipelineRunner

            runner = MultiProcessPipelineRunner(config)

            with patch(
                "sglang_omni_v1.pipeline.mp_runner.Coordinator.start",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    await runner.start()

            self.assertEqual(list(Path(tmp_dir).iterdir()), [])
