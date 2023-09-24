import functools
import warnings


class UnfinishedAPIWarning(Warning):
    pass


class UntestedAPIWarning(Warning):
    pass


# a decorator to mark a function as untested
def untested(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', UntestedAPIWarning)  # turn off filter
        warnings.warn("Call to untested function {}.".format(func.__name__),
                      category=UntestedAPIWarning,
                      stacklevel=2)
        warnings.simplefilter('default', UntestedAPIWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def stable_api(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        return func(*args, **kwargs)

    return new_func


def unfinished_api(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', UnfinishedAPIWarning)  # turn off filter
        warnings.warn("Call to unfinished api {}.".format(func.__name__),
                      category=UnfinishedAPIWarning,
                      stacklevel=2)
        warnings.simplefilter('default', UnfinishedAPIWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
