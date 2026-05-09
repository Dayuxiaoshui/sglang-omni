# SPDX-License-Identifier: Apache-2.0
"""SGLang-native encoder worker.

A minimal wrapper around the upstream SGLang model loading + parallel
state init sequence used by ``MMEncoder``. We deliberately do **not**
subclass / instantiate ``MMEncoder`` itself: it bundles ZMQ schedule
sockets, mooncake transfer engines, multimodal cache, image processor
on GPU, and an embedding-to-send queue — every one of those duplicates
work v1 already owns. We just copy the eight init calls so the worker
ends up holding (a) an initialized SGLang TP group and (b) a loaded
upstream encoder-aware model whose ``thinker.get_image_feature`` /
``get_audio_feature`` / ``get_video_feature`` we can call directly.

See sglang-project/sglang-omni#375 design ("SGLangEncoderWorker"
section) for the full rationale.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

import torch

from sglang_omni_v1.scheduling.sglang_backend.encoder_server_args import (
    build_sglang_encoder_server_args,
)

logger = logging.getLogger(__name__)


def _pick_free_port() -> int:
    """Allocate an ephemeral TCP port on the loopback interface.

    Used at ``tp_size == 1`` when the pipeline runner did not pre-allocate
    a per-stage NCCL port (single-rank stages skip the allocator).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


# Keys the worker fills in itself from explicit ``__init__`` kwargs.
# Reaching them through ``server_args_overrides`` would either collide
# with the helper signature or overwrite the worker-managed value.
_WORKER_MANAGED_KEYS: frozenset[str] = frozenset(
    {
        "model_path",
        "tp_size",
        "base_gpu_id",
        "dist_init_addr",
        "dtype",
        "load_format",
    }
)


class SGLangEncoderWorker:
    """Owns SGLang's distributed state and the loaded encoder model.

    Lifecycle:

    1. ``__init__`` — runs once per rank, blocks until the TP NCCL
       group is formed and the upstream model is fully loaded. After
       construction the worker is ready to ``encode_batch``.
    2. ``encode_batch(plan)`` — adapter-driven forward; the adapter
       owns the conversion from a v1 ``BatchPlan`` to upstream
       ``MultimodalDataItem`` and back.

    Args:
        model_path: HF model path. Required.
        tp_rank: Rank of this process within the encoder TP group.
        tp_size: World size of the encoder TP group. Distributed init
            runs unconditionally — even at ``tp_size == 1`` — because
            sglang's TP-parallel layers (``ColumnParallelLinear`` /
            ``RowParallelLinear``) call ``get_tp_group()`` at
            ``__init__`` and that asserts the parallel state has been
            initialized.
        nccl_port: Pre-allocated TCP port for torch.distributed init
            method. ``None`` is allowed only at ``tp_size == 1``; the
            worker falls back to a fresh loopback port.
        dtype: Optional dtype string forwarded to ``ServerArgs``.
        load_format: Optional weight-loader override.
        server_args_overrides: Loader / processor knobs forwarded to
            :func:`build_sglang_encoder_server_args`. Topology, GPU
            placement, and AR-only knobs are rejected — see
            ``_ENCODER_PROTECTED_KEYS`` in
            :mod:`encoder_server_args`.
    """

    def __init__(
        self,
        *,
        model_path: str,
        tp_rank: int,
        tp_size: int,
        nccl_port: int | None,
        dtype: str | None = None,
        load_format: str | None = None,
        server_args_overrides: dict[str, Any] | None = None,
    ) -> None:
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"tp_rank must satisfy 0 <= tp_rank < tp_size, "
                f"got tp_rank={tp_rank} tp_size={tp_size}"
            )

        # Reject worker-managed keys before they reach the helper, so
        # the user gets a clear "set this through the worker's keyword
        # argument" error instead of a generic Python "got multiple
        # values for keyword argument" TypeError.
        overrides = dict(server_args_overrides or {})
        clobbered = sorted(_WORKER_MANAGED_KEYS & overrides.keys())
        if clobbered:
            raise ValueError(
                f"server_args_overrides cannot set worker-managed keys "
                f"{clobbered}. These are derived from StageConfig and "
                f"factory parameters; pass them through StageConfig "
                f"(model_path, tp_size, gpu, dtype, load_format) instead."
            )

        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.is_entry_rank = self.tp_rank == 0

        # GPU placement collapses to the (cuda:0, dist_local_rank=tp_rank)
        # contract once stage_process._prepare_cuda_environment has
        # remapped CUDA_VISIBLE_DEVICES. ``base_gpu_id`` and the worker
        # ``device`` are both always cuda:0 inside the child process,
        # ``dist_local_rank`` is the rank identity (0 in tp_size==1
        # lane, [0, tp_size) when tp_size>1). See #375
        # ("GPU placement across tp_size=1 and tp_size>1 lanes").
        cuda_device = 0
        dist_local_rank = self.tp_rank
        self.device = torch.device(f"cuda:{cuda_device}")

        # SGLang's ``ServerArgs.dist_init_addr`` is ``host:port`` (parsed
        # internally to ``NetworkAddress``); torch's ``init_method`` is
        # the full ``tcp://host:port`` URL. Keep the two forms separate.
        port = nccl_port if nccl_port is not None else _pick_free_port()
        dist_addr = f"127.0.0.1:{port}"
        dist_init_method = f"tcp://{dist_addr}"

        server_args = build_sglang_encoder_server_args(
            model_path=model_path,
            tp_size=self.tp_size,
            base_gpu_id=cuda_device,
            dist_init_addr=dist_addr,
            dtype=dtype,
            load_format=load_format,
            **overrides,
        )
        self.server_args = server_args

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.distributed.parallel_state import (
            get_default_distributed_backend,
            get_tp_group,
        )
        from sglang.srt.managers.io_struct import LoadConfig
        from sglang.srt.model_executor.model_runner import (
            set_global_server_args_for_scheduler,
        )
        from sglang.srt.model_loader import get_model

        try:
            from sglang.srt.managers.io_struct import DeviceConfig
        except ImportError:  # pragma: no cover - older sglang layouts
            from sglang.srt.configs.device_config import DeviceConfig

        set_global_server_args_for_scheduler(server_args)
        self.model_config = ModelConfig.from_server_args(server_args)

        # Match the upstream MMEncoder LoadConfig surface exactly.
        # Some Qwen3-Omni deployments depend on
        # ``model_loader_extra_config`` for FP8 / NVFP4 weights, and the
        # ``remote_instance_weight_loader_*`` triplet wires the
        # remote-streaming weight protocol. Forwarding all six keeps
        # those paths open for users.
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=(
                server_args.remote_instance_weight_loader_seed_instance_ip
            ),
            remote_instance_weight_loader_seed_instance_service_port=(
                server_args.remote_instance_weight_loader_seed_instance_service_port
            ),
            remote_instance_weight_loader_send_weights_group_ports=(
                server_args.remote_instance_weight_loader_send_weights_group_ports
            ),
        )

        torch.cuda.set_device(cuda_device)

        # Always run, including tp_size == 1. ``local_rank`` is the
        # rank identity used for local-master checks
        # (``custom_all_reduce_utils.py`` etc.), not a CUDA device
        # index — device selection is governed by
        # ``SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS`` in the TP > 1 lane
        # and the ``cuda_device == 0`` invariant in both lanes.
        init_distributed_environment(
            backend=get_default_distributed_backend("cuda"),
            world_size=self.tp_size,
            rank=self.tp_rank,
            distributed_init_method=dist_init_method,
            local_rank=dist_local_rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        self.tp_group = get_tp_group()
        try:
            from sglang.srt.distributed.parallel_state import initialize_dp_attention
        except ImportError:  # pragma: no cover - depends on sglang version
            initialize_dp_attention = None
        if initialize_dp_attention is not None:
            try:
                initialize_dp_attention(
                    server_args=server_args, model_config=self.model_config
                )
            except TypeError:
                # Older signature: positional only / different kwargs.
                initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(device="cuda", gpu_id=cuda_device),
        )

        logger.info(
            "SGLangEncoderWorker ready (tp_rank=%d/%d, dist_init=%s)",
            self.tp_rank,
            self.tp_size,
            dist_init_method,
        )

    @torch.no_grad()
    def encode_batch(self, plan: Any) -> Any:
        """Run encoder forward for a :class:`BatchPlan`.

        Delegates the upstream call (e.g.
        ``self.model.thinker.get_image_feature``) to the adapter via
        ``plan.adapter.run_feature(self.model, plan)``. Keeping the
        actual modality routing inside the adapter lets the worker
        stay model-agnostic.
        """
        return plan.adapter.run_feature(self.model, plan)


__all__ = ["SGLangEncoderWorker"]
