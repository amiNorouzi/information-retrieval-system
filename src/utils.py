import os
from os import path

current_dir = os.getcwd()
root_dirname = os.path.abspath(os.path.join(current_dir, os.pardir))
data_path = path.join(root_dirname, 'data')
default_verbs = path.join(data_path, 'verbs.dat')


def to_path(src):
    return os.path.join(root_dirname, 'public', src)
