import pkgutil
import os
import importlib


def iter_modulename(module_name: str):
    loader = pkgutil.get_loader(module_name)
    dirname = os.path.dirname(loader.path)
    submodules = pkgutil.iter_modules([dirname])
    for subm in submodules:
        yield module_name + '.' + subm.name


def iter_module(module_name: str):
    for submodule_name in iter_modulename(module_name):
        yield importlib.import_module(submodule_name)
