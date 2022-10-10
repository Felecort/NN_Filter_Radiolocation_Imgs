from time import time


def delta_time() -> 'function':
    start = time()

    def wrapper() -> float:
        return time() - start
    return wrapper
