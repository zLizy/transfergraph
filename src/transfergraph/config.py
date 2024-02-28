import pathlib


def get_root_path_string() -> str:
    return pathlib.Path(__file__).parent.parent.parent.resolve().__str__()
