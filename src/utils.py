import os

current_dir = os.getcwd()
root_dirname = os.path.abspath(os.path.join(current_dir, os.pardir))


def to_path(src):
    return os.path.join(root_dirname, 'public', src)
