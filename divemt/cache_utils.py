"""
The hashing idea adapted from https://death.andgravity.com/stable-hashing
https://github.com/lemon24/reader/blob/1efcd38c78f70dcc4e0d279e0fa2a0276749111e/src/reader/_hash_utils.py
"""
import dataclasses
import datetime
import functools
import hashlib
import inspect
import json
import pickle
from collections.abc import Collection
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

_VERSION = 0
_EXCLUDE = "_hash_exclude_"


def _json_dumps(thing: object) -> str:
    return json.dumps(
        thing,
        default=_json_default,  # force formatting-related options to known values
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )


def _json_default(thing: object) -> Any:
    try:
        return _dataclass_dict(thing)
    except TypeError:
        pass
    if isinstance(thing, datetime.datetime):
        return thing.isoformat(timespec="microseconds")
    raise TypeError(f"Object of type {type(thing).__name__} is not JSON serializable")


def _dataclass_dict(thing: object) -> Dict[str, Any]:
    # we could have used dataclasses.asdict()
    # with a dict_factory that drops empty values,
    # but asdict() is recursive and we need to intercept and check
    # the _hash_exclude_ of nested dataclasses;
    # this way, json.dumps() does the recursion instead of asdict()

    # raises TypeError for non-dataclasses
    fields = dataclasses.fields(thing)
    # ... but doesn't for dataclass *types*
    if isinstance(thing, type):
        raise TypeError("got type, expected instance")

    exclude = getattr(thing, _EXCLUDE, ())

    rv = {}
    for field in fields:
        if field.name in exclude:
            continue

        value = getattr(thing, field.name)
        if value is None or not value and isinstance(value, Collection):
            continue

        rv[field.name] = value

    return rv


def calc_obj_hash(obj: object) -> bytes:
    """Calculate hash of a single object"""
    prefix = _VERSION.to_bytes(1, "big")
    hash_object = hashlib.sha256()
    hash_object.update(_json_dumps(obj).encode("utf-8"))
    return prefix + hash_object.digest()


def calc_args_hash(*args: Any, **kwargs: any) -> bytes:
    """Calculate hash of arguments to function"""
    prefix = _VERSION.to_bytes(1, "big")
    hash_object = hashlib.sha256()
    for arg in args:
        if isinstance(arg, (pd.DataFrame, pd.Series)):
            hash_object.update(str(pd.util.hash_pandas_object(arg).sum()).encode("utf-8"))
        else:
            hash_object.update(_json_dumps(arg).encode("utf-8"))
    for key, value in kwargs.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            hash_object.update(key.encode("utf-8") + str(pd.util.hash_pandas_object(value).sum()).encode("utf-8"))
        else:
            hash_object.update(_json_dumps([key, value]).encode("utf-8"))
    return prefix + hash_object.digest()


class CacheDecorator:
    def __init__(self, cache_dir: Optional[Path] = None, version: int = 0):
        self.version = version
        self.cache_dir = cache_dir or Path(".cache")

    @staticmethod
    def _is_bound_method(function: Callable, arg: Any):
        return inspect.ismethod(function) or (hasattr(arg, "__class__") and function.__name__ in dir(arg.__class__))

    def __call__(self, function: Callable) -> Any:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key_args = args[1:] if self._is_bound_method(function, args[0]) else args
            hash_val = calc_args_hash(*cache_key_args, **kwargs)
            cache_file = self.cache_dir / f"{function.__name__}_v{self.version}_{hash_val.hex()}.pkl"

            # TODO: add logging, not printing

            if cache_file.exists():
                print(f"LOADING CACHE: {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            else:
                print(len(args), len(kwargs.items()))
                result = function(*args, **kwargs)
                print(f"CREATE CACHE: {cache_file}")
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                return result

        return wrapper

    def __get__(self, instance, owner):
        """note: adapted from chat-gpt-4 =)"""
        # Support method decorators for class instances
        if instance is None:
            return self

        # Bind the decorated method to the instance
        bound_method = functools.partial(self, instance)

        return bound_method
