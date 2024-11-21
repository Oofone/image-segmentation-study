from typing import Dict, Any
from json import JSONEncoder, load, dump


class SimpleObjectJsonEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def read_json(path: str, object_callback=None):
    with open(path, 'r') as f:
        return load(f, object_hook=object_callback)


def write_json_objects(data: Dict[Any, Any], path: str):
    with open(path, 'w') as meta_file:
        dump(data, meta_file, cls=SimpleObjectJsonEncoder)
