import contextlib
import contextvars


class OverridableContextVar[T]:
    def __init__(self, var_name: str, default: T):
        self.var = contextvars.ContextVar(var_name, default=default)

    def get(self):
        return self.var.get()

    def __call__(self):
        return self.get()

    @contextlib.contextmanager
    def override(self, value: T):
        token = self.var.set(value)
        try:
            yield None
        finally:
            self.var.reset(token)
            return False
