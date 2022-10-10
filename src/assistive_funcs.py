from time import time, sleep


def delta_time() -> 'function':
    start = time()
    def wrapper() -> float:
        return time() - start
    return wrapper
