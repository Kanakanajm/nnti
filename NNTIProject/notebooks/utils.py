from copy import deepcopy
from pprint import pprint, pformat


def left_indent(text, indent: str = "\t"):
    return "".join([indent + l for l in text.splitlines(True)])


def pprint_tab(obj, indent: str = "\t"):
    print(left_indent(pformat(obj), indent))


def get_available_objects(data):
    return [obj for obj in dir(data) if not obj.startswith("__")]


def copy(obj):
    return deepcopy(obj)
