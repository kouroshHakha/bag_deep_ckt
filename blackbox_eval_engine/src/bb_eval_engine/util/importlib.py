from typing import Type

import importlib


def import_cls(class_str: str) -> Type:
    """Given a Python class string, convert it to the Python class.

    Parameters
    ----------
    class_str : str
        a Python class string/

    Returns
    -------
    py_class : class
        a Python class.
    """
    sections = class_str.split('.')

    module_str = '.'.join(sections[:-1])
    class_str = sections[-1]
    modul = importlib.import_module(module_str)
    return getattr(modul, class_str)
