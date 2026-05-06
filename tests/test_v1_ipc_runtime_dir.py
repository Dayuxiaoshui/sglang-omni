# SPDX-License-Identifier: Apache-2.0
"""Omni V1 IPC runtime directory lifecycle tests."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

pytest.importorskip("torch")

from sglang_omni_v1.config.compiler import (
    IpcRuntimeDir,
    compile_pipeline,
    compile_pipeline_core,
    create_ipc_runtime_dir,
)
from sglang_omni_v1.config.schema import EndpointsConfig, PipelineConfig, StageConfig


def noop_factory():
    return None


class _FakeControlPlane:
    def __init__(self, recv_endpoint: str):
        self.recv_endpoint = recv_endpoint


class _FakeStage:
    name = "preprocessing"

    def __init__(self, recv_endpoint: str):
        self.control_plane = _FakeControlPlane(recv_endpoint)

    async def run(self) -> None:
        await asyncio.Event().wait()


class _FakeCoordinator:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


def _make_config(base_path: str, *, scheme: str = "ipc") -> PipelineConfig:
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
            scheme=scheme,
            base_path=base_path,
        ),
    )


class TestV1IpcRuntimeDir(unittest.TestCase):
    def test_ipc_runtime_dir_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            runtime_dir.close()
            runtime_dir.close()

            self.assertFalse(runtime_path.exists())

    def test_create_ipc_runtime_dir_returns_none_for_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir, scheme="tcp")

            self.assertIsNone(create_ipc_runtime_dir(config))

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

    def test_compile_core_returns_owned_runtime_dir_for_successful_ipc_compile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)

            _coordinator, stages, runtime_dir = compile_pipeline_core(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path

            try:
                self.assertTrue(runtime_path.exists())
                self.assertIn(
                    str(runtime_path),
                    stages[0].control_plane.recv_endpoint,
                )
            finally:
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

    async def test_mp_runner_cleans_runtime_dir_on_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            from sglang_omni_v1.pipeline.mp_runner import MultiProcessPipelineRunner

            runner = MultiProcessPipelineRunner(config)
            await runner.start(timeout=30.0)

            runtime_path = None
            try:
                runtime_dirs = [
                    path for path in Path(tmp_dir).iterdir() if path.is_dir()
                ]
                self.assertEqual(len(runtime_dirs), 1)
                runtime_path = runtime_dirs[0]
                self.assertTrue(runtime_path.exists())
            finally:
                await runner.stop()

            self.assertIsNotNone(runtime_path)
            self.assertFalse(runtime_path.exists())


class TestV1LauncherIpcCleanup(unittest.IsolatedAsyncioTestCase):
    async def _run_single_process_launcher_with_mocked_server(
        self,
        *,
        config: PipelineConfig,
        runtime_dir: IpcRuntimeDir,
        serve_mock: AsyncMock,
    ) -> tuple[_FakeCoordinator, FastAPI]:
        stage = _FakeStage(f"ipc://{runtime_dir.path}/stage_preprocessing.sock")
        coordinator = _FakeCoordinator()
        app = FastAPI()

        from sglang_omni_v1.serve.launcher import _run_server

        with (
            patch(
                "sglang_omni_v1.serve.launcher._find_available_port",
                return_value=8000,
            ),
            patch(
                "sglang_omni_v1.serve.launcher.compile_pipeline_core",
                return_value=(coordinator, [stage], runtime_dir),
            ) as compile_pipeline_core,
            patch(
                "sglang_omni_v1.serve.launcher.create_app",
                return_value=app,
            ) as create_app,
            patch(
                "sglang_omni_v1.serve.launcher.uvicorn.Server.serve",
                new=serve_mock,
            ),
        ):
            await _run_server(config, port=8000)

        compile_pipeline_core.assert_called_once_with(config)
        create_app.assert_called_once()

        return coordinator, app

    async def test_single_process_launcher_cleans_runtime_dir_on_server_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path
            server_serve = AsyncMock(return_value=None)

            coordinator, app = (
                await self._run_single_process_launcher_with_mocked_server(
                    config=config,
                    runtime_dir=runtime_dir,
                    serve_mock=server_serve,
                )
            )

            self.assertTrue(coordinator.started)
            self.assertTrue(coordinator.stopped)
            server_serve.assert_awaited_once()
            mounted_paths = {route.path for route in app.routes}
            self.assertIn("/start_profile", mounted_paths)
            self.assertIn("/stop_profile", mounted_paths)
            self.assertFalse(runtime_path.exists())

    async def test_single_process_launcher_cleans_runtime_dir_on_server_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config(tmp_dir)
            runtime_dir = create_ipc_runtime_dir(config)
            self.assertIsNotNone(runtime_dir)
            runtime_path = runtime_dir.path
            server_serve = AsyncMock(side_effect=RuntimeError("server failed"))

            with self.assertRaisesRegex(RuntimeError, "server failed"):
                await self._run_single_process_launcher_with_mocked_server(
                    config=config,
                    runtime_dir=runtime_dir,
                    serve_mock=server_serve,
                )

            server_serve.assert_awaited_once()
            self.assertFalse(runtime_path.exists())
