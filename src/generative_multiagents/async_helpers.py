import asyncio
import time
import heapq
from typing import Callable, Awaitable, Union, overload, TypeVar, Generic, Type


class Throttler:
    def __init__(self, requests_per_second: int | float):
        self.requests_per_second = requests_per_second
        self.heap: list[float] = []
        self.lock = asyncio.Lock()

    async def throttle(self):
        async with self.lock:
            now = time.monotonic()
            # Clear the heap of requests older than 1 second
            while len(self.heap) and (now - self.heap[0] > 1):
                heapq.heappop(self.heap)

            if len(self.heap):
                oldest_request = self.heap[0]
                # there are at least requests_per_second requests in the last second, thus we need to throttle
                if len(self.heap) >= self.requests_per_second:
                    time_to_expire = 1 - (now - oldest_request)
                    assert 0 <= time_to_expire <= 1
                    await asyncio.sleep(time_to_expire)

            heapq.heappush(self.heap, time.monotonic())

    async def __call__(self):
        await self.throttle()


def memoized_async_function_argless[**P, R](func: Callable[P, Awaitable[R]]):
    lock = asyncio.Lock()
    value: R | None = None

    async def inner(*args: P.args, **kwargs: P.kwargs):
        nonlocal value

        async with lock:
            if value:
                return value
            value = await func(*args, **kwargs)
            return value

    return inner


Instance = TypeVar("Instance")
Value = TypeVar("Value")


class cached_async_property(Generic[Instance, Value]):
    """Decorator for cached async class properties"""

    def __init__(self, func: Callable[[Instance], Awaitable[Value]]):
        self.func = func
        self.__doc__ = func.__doc__
        self.attrname: str | None = None

    def __set_name__(self, owner: Instance, name: str):

        assert self.attrname is None and type(name) == str
        self.attrname = name

    @overload
    def __get__(self, instance: None, owner: Type[Instance]) -> "cached_async_property":
        """Called when an attribute is accessed via class not an instance"""

    @overload
    def __get__(self, instance: Instance, owner: Type[Instance]) -> Awaitable[Value]:
        """Called when an attribute is accessed on an instance variable"""

    def __get__(
        self, instance: Union[Instance, None], owner: Type[Instance]
    ) -> Union[Awaitable[Value], "cached_async_property"]:
        """Full implementation is declared here"""

        if instance is None:
            return self
        assert self.attrname is not None

        cached_property_holder_name = "__" + self.attrname + "__cache_getter"
        if not hasattr(instance, cached_property_holder_name):
            setattr(
                instance,
                cached_property_holder_name,
                memoized_async_function_argless(self.func),
            )

        cached_getter: Callable[[Instance], Awaitable[Value]] = getattr(
            instance, cached_property_holder_name
        )
        result = cached_getter(instance)
        return result
