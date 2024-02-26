from copy import deepcopy


def get_available_objects(data):
    return [obj for obj in dir(data) if not obj.startswith("__")]


def copy(obj):
    return deepcopy(obj)
