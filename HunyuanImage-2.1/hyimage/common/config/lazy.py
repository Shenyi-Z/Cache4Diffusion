import collections.abc as abc
import copy
import pydoc
from typing import Any


class DictConfig(dict):

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'DictConfig' object has no attribute '{item}'")


def locate(name: str) -> Any:
    """
    Locate and return an object using a string like {x.__module__}.{x.__qualname__}.

    Args:
        name:Dotted path to the object

    Returns:
        The located object

    Raises:
        ImportError if the object cannot be found
    """
    return pydoc.locate(name)



class LazyObject:

    def __init__(self, target, **kwargs):
        self._target = target
        self._kwargs = kwargs

    def instantiate(self, **kwargs):
        new_kwargs = copy.deepcopy(self._kwargs)
        new_kwargs.update(kwargs)
        return self._target(**new_kwargs)


class LazyCall:

    def __init__(self, target):
        if not callable(target):
            raise ValueError(f"`target` of LazyCall must be a callable, got {target}")
        self._target = target

    def __call__(self, **kwargs):
        return LazyObject(self._target, **kwargs)


def instantiate(config: LazyObject, **kwargs):
    if config is None:
        return None
    return config.instantiate(**kwargs)

