import time


def func_time(keyword=""):
    def func(fun):
        def wrapper(*args, **kwargs):
            time_start = time.time()
            res = fun(*args, **kwargs)
            time_end = time.time()
            cost_time = time_end - time_start
            print(fun.__name__ + f" {keyword} costs time: {cost_time}s.")
            return res

        return wrapper

    return func


def singleton(cls, *args, **kw):
    instance = {}

    def _singleton():
        if cls not in instance:
            instance[cls] = cls(*args, **kw)
        return instance[cls]

    return _singleton
