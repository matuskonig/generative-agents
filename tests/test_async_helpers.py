import time

import pytest

from generative_agents.async_helpers import Throttler


class TestThrottlerInitialization:
    def test_invalid_rps_zero_raises(self) -> None:
        with pytest.raises(
            AssertionError,
            match="requests_per_second must be positive",
        ):
            Throttler(requests_per_second=0)

    def test_invalid_rps_float_greater_than_one_raises(self) -> None:
        with pytest.raises(
            AssertionError,
            match="only integer is allowed for requests_per_second >= 1",
        ):
            Throttler(requests_per_second=10.5)


class TestThrottlerBasicBehavior:
    @pytest.mark.asyncio
    async def test_single_call_no_throttle(self) -> None:
        throttler = Throttler(requests_per_second=10)
        start = time.monotonic()
        await throttler.throttle()
        elapsed = time.monotonic() - start
        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_multiple_consecutive_calls_throttled(self) -> None:
        rps = 10
        requests = 20
        throttler = Throttler(requests_per_second=rps)
        start = time.monotonic()
        for _ in range(requests):
            await throttler.throttle()
        elapsed = time.monotonic() - start
        minimum_time = 1
        maximum_time = requests / rps
        assert minimum_time <= elapsed <= maximum_time

    @pytest.mark.asyncio
    async def test_sequential_calls_respect_rps(self) -> None:
        rps = 2
        throttler = Throttler(requests_per_second=rps)
        times: list[float] = []

        for _ in range(4):
            await throttler.throttle()
            times.append(time.monotonic())

        assert (times[-1] - times[0]) / len(times) <= rps

    @pytest.mark.asyncio
    async def test_sequential_calls_throttles_correctly(self) -> None:
        rps = 2
        throttler = Throttler(requests_per_second=rps)

        now = time.monotonic()

        await throttler.throttle()
        await throttler.throttle()
        assert time.monotonic() - now < 0.1

        await throttler.throttle()
        await throttler.throttle()
        assert time.monotonic() - now >= 1

        await throttler.throttle()
        await throttler.throttle()
        assert time.monotonic() - now >= 2



    @pytest.mark.asyncio
    async def test_sequential_calls_subsecond_throttle(self) -> None:
        rps = 0.5
        throttler = Throttler(requests_per_second=rps)

        now = time.monotonic()

        await throttler.throttle()
        await throttler.throttle()
        assert time.monotonic() - now >= 2
