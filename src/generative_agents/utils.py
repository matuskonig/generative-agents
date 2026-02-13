from typing import TypeVar, Generic, Callable, Iterator

import contextlib
import contextvars

T = TypeVar("T")


class OverridableContextVar(Generic[T]):
    def __init__(self, var_name: str, default: T):
        self.var = contextvars.ContextVar(var_name, default=default)

    def get(self) -> T:
        return self.var.get()

    def __call__(self) -> T:
        return self.get()

    @contextlib.contextmanager
    def override(self, value: T) -> Iterator[None]:
        token = self.var.set(value)
        try:
            yield None
        finally:
            self.var.reset(token)
