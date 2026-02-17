import asyncio
import heapq
import time


class Throttler:
    def __init__(self, requests_per_second: int | float) -> None:
        if requests_per_second >= 1:
            assert isinstance(
                requests_per_second, int
            ), "only integer is allowed for requests_per_second >= 1"
        assert requests_per_second > 0, "requests_per_second must be positive"

        self.requests_per_second = requests_per_second
        self._lock = asyncio.Lock()

        # stores timestamps (in sec) of relevant requests within the time window
        self._heap: list[float] = []

    async def throttle(self) -> None:
        # everything is in seconds

        async with self._lock:
            now = time.monotonic()
            time_window = max(1, 1 / self.requests_per_second)

            # Release unrelevant request past the time window
            while (len(self._heap)) and (now - self._heap[0] > time_window):
                heapq.heappop(self._heap)

            if len(self._heap):
                oldest_request = self._heap[0]

                if (len(self._heap)) >= self.requests_per_second:
                    # wait until eligible for removal
                    time_to_expire = time_window - (now - oldest_request)
                    assert 0 <= time_to_expire <= time_window
                    await asyncio.sleep(time_to_expire)

            heapq.heappush(self._heap, time.monotonic())
