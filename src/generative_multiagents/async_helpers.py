import asyncio
import time
import heapq
from typing import Callable, Awaitable


class Throtler:
    def __init__(self, requests_per_second: int | float):
        self.requests_per_second = requests_per_second
        self.heap: list[float] = []
        self.lock = asyncio.Lock()

    async def throttle(self):
        with await self.lock:
            now = time.time()
            # Clear the heap of requests older than 1 second
            while len(self.heap) and now - self.heap[0] > 1:
                heapq.heappop(self.heap)

            heapq.heappush(self.heap, now)

            oldest_request = self.heap[0]

            # adding the new request to the heap hit the throttle
            if len(self.heap) > self.requests_per_second:
                time_to_expire = 1 - (now - oldest_request)
                assert 0 <= time_to_expire <= 1
                await asyncio.sleep(time_to_expire)

    def __call__(self):
        return self.throttle()


def cached_async_getter[**P, R](func: Callable[P, Awaitable[R]]):
    lock = asyncio.Lock()
    value: R | None

    async def inner(*args: P.args, **kwargs: P.kwargs):
        nonlocal value

        async with lock:
            if value:
                return value
            value = await func(*args, **kwargs)
            return value

    return inner
