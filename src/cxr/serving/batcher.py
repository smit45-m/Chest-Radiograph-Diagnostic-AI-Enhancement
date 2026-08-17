"""Dynamic batching: gather requests for a few milliseconds, then one GPU forward.

This is how we keep single-image latency low while still using the GPU
efficiently when several users hit /predict at once.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _Pending:
    tensor: torch.Tensor
    future: asyncio.Future


class DynamicBatcher:
    def __init__(
        self,
        forward_fn,
        max_batch: int = 8,
        max_delay_ms: float = 12.0,
    ) -> None:
        self.forward_fn = forward_fn
        self.max_batch = max_batch
        self.max_delay = max_delay_ms / 1000.0
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, tensor: torch.Tensor) -> Any:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_Pending(tensor=tensor, future=fut))
        return await fut

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.max_delay
            while len(batch) < self.max_batch:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break
            try:
                stacked = torch.cat([p.tensor for p in batch], dim=0)
                outputs = await asyncio.to_thread(self.forward_fn, stacked)
                for i, pending in enumerate(batch):
                    if not pending.future.done():
                        pending.future.set_result(outputs[i])
            except Exception as exc:  # noqa: BLE001
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
