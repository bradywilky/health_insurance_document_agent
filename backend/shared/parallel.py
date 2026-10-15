"""Run model calls side by side, keeping each one inside the caller's trace."""
from concurrent.futures import ThreadPoolExecutor
import contextvars


def parallel_map(function, items, workers):
    """[function(item) or the exception it raised] for each item, in order.

    Each call runs in a copy of the caller's context, so its spans nest under the span open here.
    """
    def attempt(item):
        try:
            return function(item)
        except Exception as exc:
            return exc
    items = list(items)
    if len(items) <= 1 or workers <= 1:
        return [attempt(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, attempt, item) for item in items]
        return [future.result() for future in futures]
