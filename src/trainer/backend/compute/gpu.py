# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU runtime (queue + submission) and command recording: each recorded batch
becomes one command buffer submitted under the shared queue lock."""

from __future__ import annotations

import contextlib
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import slangpy as spy


@lru_cache(maxsize=None)
def marker_color(name: str) -> tuple[float, float, float]:
    """Run-stable RGB for a debug-marker name (blake2s: builtin hash() is salted per process)."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).digest()
    # Bias into [0.35, 1.0] so colors stay legible (never near-black).
    return tuple(0.35 + (b / 255.0) * 0.65 for b in digest)  # type: ignore[return-value]


class BindingError(TypeError):
    """A dispatch's keyword arguments do not match its entry point's parameters."""


# Reflected parameters per kernel. A ComputeKernel takes no weak reference, so the cache holds the kernel itself
# (and through it its program and device): it is bounded, least recently used out first, and cleared on hot reload.
BINDING_CACHE_SIZE = 1024
_BINDINGS: OrderedDict[int, tuple[frozenset[str], frozenset[str], object]] = OrderedDict()


def clear_binding_cache() -> None:
    """Forget every kernel's reflected parameters (``device.reload_all_programs()`` can change them in place)."""
    _BINDINGS.clear()


def entry_point_bindings(kernel) -> tuple[frozenset[str], frozenset[str]]:
    """(required, bindable) parameter names of the kernel's entry point. System values (scalar or
    vector parameters that take no uniform space, e.g. SV_DispatchThreadID) are neither."""
    cached = _BINDINGS.get(id(kernel))
    if cached is not None and cached[2] is kernel:
        _BINDINGS.move_to_end(id(kernel))
        return cached[0], cached[1]
    ep = kernel.program.layout.entry_points[0]
    bindable = set()
    for i in range(len(ep.parameters)):
        param = ep.parameters[i]
        layout = param.type_layout
        system = (layout.kind in (spy.TypeReflection.Kind.scalar, spy.TypeReflection.Kind.vector)
                  and layout.size == 0)
        if not system:
            bindable.add(param.name)
    cached = (frozenset(bindable), frozenset(bindable), kernel)
    _BINDINGS[id(kernel)] = cached
    _BINDINGS.move_to_end(id(kernel))
    while len(_BINDINGS) > BINDING_CACHE_SIZE:
        _BINDINGS.popitem(last=False)
    return cached[0], cached[1]


def check_bindings(kernel, marker: str, kwargs: dict, *, indirect: bool = False) -> None:
    """Raise BindingError when kwargs miss a parameter or name one the entry point does not take (a system value
    is not one): Slang would leave a missing one zero and silently ignore an extra one. ``thread_count`` sizes a
    direct dispatch; an indirect one reads its size from the argument buffer, so there it is an error too."""
    required, names = entry_point_bindings(kernel)
    given = set(kwargs) - (set() if indirect else {"thread_count"})
    missing, extra = required - given, given - names
    if missing or extra:
        entry = kernel.program.layout.entry_points[0].name
        raise BindingError(f"dispatch {marker!r} ({entry}): "
                           f"missing {sorted(missing)}, not parameters {sorted(extra)}")


@dataclass(frozen=True)
class SubmitHandle:
    """A submitted command buffer; wait on it to bound queue depth without a full device idle."""
    submit_id: int


class CommandBatch:
    """One command buffer's worth of recorded GPU work (wraps a ``spy.CommandEncoder``)."""

    def __init__(self, encoder: spy.CommandEncoder, label: str = "") -> None:
        self._encoder = encoder
        self._label = label

    @property
    def encoder(self) -> spy.CommandEncoder:
        return self._encoder

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        check_bindings(kernel, marker, kwargs)
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        """Record an indirect dispatch; groups_x/y/z are uint32s at ``offset`` in ``args_buf``.

        No barrier before the fetch: sgl's resource tracking covers the ``args_buf`` write.
        """
        check_bindings(kernel, marker, kwargs, indirect=True)
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        pass_enc = self._encoder.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute_indirect(spy.BufferOffsetPair(args_buf.storage, offset))
        finally:
            pass_enc.end()

    @contextlib.contextmanager
    def marker_group(self, label: str):
        self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))
        try:
            yield
        finally:
            self._encoder.pop_debug_group()

    def finish(self):
        return self._encoder.finish()


class GpuRuntime:
    """Owns the device and hands out CommandBatches; submissions serialize under ``gpu_lock``.

    The lock (shared with the viewer; a no-op by default) covers submits and ``wait_for_idle``,
    not single-submission fence waits or the viewer's ``present``.
    """

    def __init__(self, device: spy.Device,
                 gpu_lock: "contextlib.AbstractContextManager | None" = None) -> None:
        self.device = device
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()

    def command_batch(self, label: str) -> CommandBatch:
        return CommandBatch(self.device.create_command_encoder(), label)

    def submit(self, batch: CommandBatch,
               after_submit: Callable[[], None] | None = None) -> SubmitHandle:
        """Submit under the queue lock; ``after_submit`` (viewer present, serialized by sgl)
        runs after unlocking, so a stalled present cannot hold the queue against the trainer.
        """
        cmd = batch.finish()
        with self.gpu_lock:
            submit_id = self.device.submit_command_buffer(cmd)
        if after_submit is not None:
            after_submit()
        return SubmitHandle(submit_id)

    def wait_for_submit(self, handle: SubmitHandle | int) -> None:
        """Wait for one submission's fence; not under ``gpu_lock``, which would serialize
        the viewer and the trainer for the whole GPU wait.
        """
        submit_id = handle.submit_id if isinstance(handle, SubmitHandle) else handle
        self.device.wait_for_submit(submit_id)

    def wait_for_idle(self) -> None:
        """Queue-wide idle wait; a queue operation, so it takes the lock."""
        with self.gpu_lock:
            self.device.wait_for_idle()

    def queue_scope(self):
        """Lock scope for compound queue-adjacent operations (reload/reconfigure)."""
        return self.gpu_lock


class RecordingTransaction:
    """A recorded batch plus host-state commits that run once its submit succeeds."""

    def __init__(self, batch: CommandBatch) -> None:
        self.batch = batch
        self._commits: list[Callable[[], None]] = []

    def after_submit(self, commit: Callable[[], None]) -> None:
        self._commits.append(commit)

    def commit(self) -> None:
        for commit in self._commits:
            commit()


class CommandRecorder:
    """Records GPU work into the active batch, or submits each dispatch on its own.

    Shared by every GPU component, so a training step is one command buffer; ``timer``
    (a ``StepTimer``) is set only while profiling and timestamps every marker group.
    """

    def __init__(self, runtime: GpuRuntime) -> None:
        self.runtime = runtime
        self.batch: CommandBatch | None = None
        self.timer = None
        self.last_submit_id = 0

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch(kernel, marker, **kwargs)
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch(kernel, marker, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    @contextlib.contextmanager
    def marker_group(self, label: str):
        if self.batch is None:
            yield
            return
        batch, timer = self.batch, self.timer
        interval = timer.open(label, batch.encoder) if timer is not None else -1
        with batch.marker_group(label):
            try:
                yield
            finally:
                if timer is not None:
                    timer.close(interval, batch.encoder)

    @contextlib.contextmanager
    def recording(self, label: str):
        """Record into a fresh batch and submit it once, only if the body succeeds."""
        batch = self.runtime.command_batch(label)
        transaction = RecordingTransaction(batch)
        self.batch = batch
        try:
            yield transaction
            self.last_submit_id = self.runtime.submit(batch).submit_id
            transaction.commit()
        finally:
            self.batch = None

    @contextlib.contextmanager
    def attached(self, batch: CommandBatch):
        """Record into a batch the caller owns and submits (the viewer frame)."""
        previous, self.batch = self.batch, batch
        try:
            yield
        finally:
            self.batch = previous

    def upload_texture(self, texture: spy.Texture, pixels) -> None:
        if self.batch is not None:
            self.batch.encoder.upload_texture_data(texture, 0, 0, pixels)
        else:
            texture.copy_from_numpy(pixels)
