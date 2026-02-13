import asyncio
import time
import heapq


class Throttler:
    def __init__(self, requests_per_second: int | float):
        self.requests_per_second = requests_per_second
        self.heap: list[float] = []
        self.lock = asyncio.Lock()

    async def throttle(self) -> None:
        async with self.lock:
            now = time.monotonic()
            # Clear the heap of requests older than 1 second
            # TODO: wtf
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

    async def __call__(self) -> None:
        await self.throttle()


# def memoized_async_function_argless[**P, R](func: Callable[P, Awaitable[R]]):
#     lock = asyncio.Lock()
#     value: R | None = None

#     async def inner(*args: P.args, **kwargs: P.kwargs):
#         nonlocal value

#         async with lock:
#             if value:
#                 return value
#             value = await func(*args, **kwargs)
#             return value

#     return inner


# def memoized_async_function[**P, R](
#     func: Callable[P, Awaitable[R]], max_cache_size=128
# ):
#     cache = LRUCache(maxsize=max_cache_size)
#     lock = asyncio.Lock()

#     async def inner(*args: P.args, **kwargs: P.kwargs):
#         key = (args, frozenset(kwargs.items()))
#         async with lock:
#             if key in cache:
#                 return cache[key]
#             value = await func(*args, **kwargs)
#             cache[key] = value
#             return value

#     return inner


# def memoized_async_method[Self, **P, R](
#     self: Self, func: Callable[Concatenate[Self, P], Awaitable[R]], max_cache_size=128
# ):
#     cache = LRUCache(maxsize=max_cache_size)
#     lock = asyncio.Lock()

#     async def inner(*args: P.args, **kwargs: P.kwargs):
#         key = (args, frozenset(kwargs.items()))
#         async with lock:
#             if key in cache:
#                 return cache[key]
#             value = await func(self, *args, **kwargs)
#             cache[key] = value
#             return value

#     return inner


# Instance = TypeVar("Instance")
# Value = TypeVar("Value")
# Args = ParamSpec("Args")


# class cached_async_property(Generic[Instance, Value]):
#     """Decorator for cached async class properties"""

#     def __init__(self, func: Callable[[Instance], Awaitable[Value]]):
#         self.func = func
#         self.__doc__ = func.__doc__
#         self.attrname: str | None = None

#     def __set_name__(self, owner: Instance, name: str):

#         assert self.attrname is None and type(name) == str
#         self.attrname = name

#     @overload
#     def __get__(self, instance: None, owner: Type[Instance]) -> "cached_async_property":
#         """Called when an attribute is accessed via class not an instance"""

#     @overload
#     def __get__(self, instance: Instance, owner: Type[Instance]) -> Awaitable[Value]:
#         """Called when an attribute is accessed on an instance variable"""

#     def __get__(
#         self, instance: Union[Instance, None], owner: Type[Instance]
#     ) -> Union[Awaitable[Value], "cached_async_property"]:
#         """Full implementation is declared here"""

#         if instance is None:
#             return self
#         assert self.attrname is not None

#         cached_property_holder_name = "__" + self.attrname + "__cache_getter"
#         if not hasattr(instance, cached_property_holder_name):
#             setattr(
#                 instance,
#                 cached_property_holder_name,
#                 memoized_async_function_argless(self.func),
#             )

#         cached_getter: Callable[[Instance], Awaitable[Value]] = getattr(
#             instance, cached_property_holder_name
#         )
#         result = cached_getter(instance)
#         return result


# class _cached_async_method(Generic[Instance, Args, Value]):
#     """Decorator for cached async class properties"""

#     def __init__(
#         self,
#         func: Callable[Concatenate[Instance, Args], Awaitable[Value]],
#         max_cache_size: int = 128,
#     ):

#         self.func = func
#         self.__doc__ = func.__doc__
#         self.attrname: str | None = None
#         self.max_cache_size = max_cache_size

#     def __set_name__(self, owner: Instance, name: str):

#         assert self.attrname is None and type(name) == str
#         self.attrname = name

#     @overload
#     def __get__(self, instance: None, owner: Type[Instance]) -> "cached_async_property":
#         """Called when an attribute is accessed via class not an instance"""

#     @overload
#     def __get__(
#         self, instance: Instance, owner: Type[Instance]
#     ) -> Callable[Concatenate[Instance, Args], Awaitable[Value]]:
#         """Called when an attribute is accessed on an instance variable"""

#     def __get__(
#         self, instance: Union[Instance, None], owner: Type[Instance]
#     ) -> Union[
#         Callable[Concatenate[Instance, Args], Awaitable[Value]], "cached_async_property"
#     ]:
#         """Full implementation is declared here"""

#         if instance is None:
#             return self
#         assert self.attrname is not None

#         cached_property_holder_name = "__" + self.attrname + "__cached_method"
#         if not hasattr(instance, cached_property_holder_name):
#             setattr(
#                 instance,
#                 cached_property_holder_name,
#                 memoized_async_method(instance, self.func, self.max_cache_size),
#             )

#         cached_getter: Callable[Concatenate[Instance, Args], Awaitable[Value]] = (
#             getattr(instance, cached_property_holder_name)
#         )
#         return cached_getter


# def cached_async_method2(max_cache_size: int = 128):
#     def inner(func: Callable[Concatenate[Instance, Args], Awaitable[Value]]):
#         return _cached_async_method(func, max_cache_size)

#     return inner


# def cached_async_method(
#     func: Callable[[Instance], Awaitable[Value]] | None = None,
#     max_cache_size: int = 128,
# ):
#     if func:
#         return _cached_async_method(func)
#     else:
#         return lambda func: _cached_async_method(func, max_cache_size)
