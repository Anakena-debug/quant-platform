"""Tests for SpscQueue (S36b D3).

Coverage:

- ``TestConstruction``: maxsize validation.
- ``TestBasicRoundTrip``: put_nowait + get FIFO ordering.
- ``TestBackpressure``: QueueFull at maxsize; drop-newest semantics
  preserved at the exception surface (matches asyncio.Queue).
- ``TestBlockingGet``: get() blocks when empty + wakes on put_nowait.
- ``TestEventRaceSafety``: the empty-check / clear / re-check sequence
  in get() handles the producer-sets-between-check-and-clear race.
- ``TestCancellation``: get() cancellation propagates cleanly.
- ``TestQsizeEmpty``: qsize + empty track the buffer state accurately.
- ``TestSurfaceCompatibility``: SpscQueue can be used where the engine
  uses asyncio.Queue — exception types match, method signatures match.
"""

from __future__ import annotations

import asyncio

import pytest

from quantengine.runtime.streaming.spsc_queue import SpscQueue


class TestConstruction:
    def test_maxsize_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="maxsize"):
            SpscQueue(maxsize=0)

    def test_maxsize_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="maxsize"):
            SpscQueue(maxsize=-1)

    def test_maxsize_one_constructs(self) -> None:
        q: SpscQueue[int] = SpscQueue(maxsize=1)
        assert q.empty()
        assert q.qsize() == 0


class TestBasicRoundTrip:
    def test_put_get_fifo_single_item(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)
            q.put_nowait(42)
            result = await q.get()
            assert result == 42
            assert q.empty()

        asyncio.run(run())

    def test_put_get_fifo_burst(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=100)
            for i in range(50):
                q.put_nowait(i)
            results = [await q.get() for _ in range(50)]
            assert results == list(range(50))
            assert q.empty()

        asyncio.run(run())


class TestBackpressure:
    def test_queue_full_at_maxsize(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=3)
            q.put_nowait(1)
            q.put_nowait(2)
            q.put_nowait(3)
            with pytest.raises(asyncio.QueueFull):
                q.put_nowait(4)
            # Buffer state unchanged after refusal
            assert q.qsize() == 3
            # The "earliest" three are retained; the engine's drop-
            # newest policy lives at the caller (it catches QueueFull
            # and discards the item it tried to push).
            assert await q.get() == 1
            assert await q.get() == 2
            assert await q.get() == 3

        asyncio.run(run())

    def test_queuefull_is_asyncio_queuefull(self) -> None:
        # Surface-compat with asyncio.Queue: engine's existing
        # `except asyncio.QueueFull` clause must still catch us.
        q: SpscQueue[int] = SpscQueue(maxsize=1)
        q.put_nowait(1)
        with pytest.raises(asyncio.QueueFull):
            q.put_nowait(2)


class TestBlockingGet:
    def test_get_blocks_then_wakes(self) -> None:
        async def run() -> None:
            q: SpscQueue[str] = SpscQueue(maxsize=10)

            async def producer() -> None:
                await asyncio.sleep(0.01)  # ensure consumer awaits first
                q.put_nowait("item")

            async def consumer() -> str:
                return await q.get()

            cons_task = asyncio.create_task(consumer())
            prod_task = asyncio.create_task(producer())
            result = await asyncio.wait_for(cons_task, timeout=1.0)
            await prod_task
            assert result == "item"

        asyncio.run(run())

    def test_get_wait_for_timeout(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.05)

        asyncio.run(run())


class TestEventRaceSafety:
    """The producer-sets-between-empty-check-and-clear race.

    Without the double-check in get(), a producer that set the event
    between the consumer's `if self._buf:` check and the consumer's
    `self._event.clear()` would have its signal dropped — the consumer
    would then wait forever on a cleared event with non-empty data.
    """

    def test_interleaved_put_get_does_not_hang(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)

            async def producer() -> None:
                for i in range(100):
                    q.put_nowait(i)
                    # Yield occasionally so the consumer interleaves
                    if i % 10 == 0:
                        await asyncio.sleep(0)

            async def consumer() -> list[int]:
                received: list[int] = []
                while len(received) < 100:
                    item = await q.get()
                    received.append(item)
                return received

            cons_task = asyncio.create_task(consumer())
            prod_task = asyncio.create_task(producer())
            results = await asyncio.wait_for(cons_task, timeout=2.0)
            await prod_task
            assert results == list(range(100))

        asyncio.run(run())

    def test_burst_and_drain_no_lost_signals(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=1000)

            async def producer() -> None:
                # Repeated empty → non-empty → empty cycles
                for batch in range(10):
                    for j in range(10):
                        q.put_nowait(batch * 10 + j)
                    await asyncio.sleep(0)  # let consumer drain

            async def consumer() -> list[int]:
                received: list[int] = []
                while len(received) < 100:
                    received.append(await q.get())
                return received

            cons_task = asyncio.create_task(consumer())
            prod_task = asyncio.create_task(producer())
            results = await asyncio.wait_for(cons_task, timeout=2.0)
            await prod_task
            assert results == list(range(100))

        asyncio.run(run())


class TestCancellation:
    def test_cancelled_get_does_not_corrupt_state(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)

            async def consumer_cancel() -> None:
                task = asyncio.create_task(q.get())
                await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            await consumer_cancel()
            # Queue should still work correctly after cancellation
            q.put_nowait(42)
            assert await q.get() == 42
            assert q.empty()

        asyncio.run(run())


class TestQsizeEmpty:
    def test_qsize_tracks_buffer(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)
            assert q.qsize() == 0
            assert q.empty()
            q.put_nowait(1)
            assert q.qsize() == 1
            assert not q.empty()
            q.put_nowait(2)
            assert q.qsize() == 2
            await q.get()
            assert q.qsize() == 1
            await q.get()
            assert q.qsize() == 0
            assert q.empty()

        asyncio.run(run())


class TestSurfaceCompatibility:
    """SpscQueue must be drop-in compatible with asyncio.Queue's
    put_nowait / get / qsize / empty surface so engine.py's existing
    call sites work without code changes.
    """

    def test_method_set_matches_asyncio_queue_subset(self) -> None:
        q: SpscQueue[int] = SpscQueue(maxsize=10)
        # The exact methods the engine uses:
        assert callable(q.put_nowait)
        assert callable(q.get)
        assert callable(q.qsize)
        assert callable(q.empty)

    def test_get_returns_coroutine(self) -> None:
        async def run() -> None:
            q: SpscQueue[int] = SpscQueue(maxsize=10)
            q.put_nowait(1)
            coro = q.get()
            assert asyncio.iscoroutine(coro)
            assert await coro == 1

        asyncio.run(run())
