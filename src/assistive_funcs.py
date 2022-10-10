from time import time, sleep


def delta_time() -> 'function':
    start = time()
    def wrapper() -> float:
        return round((time() - start), 2)
    return wrapper
