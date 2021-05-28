import os
import sys


def add_parent_path(level=1):

    script_path = os.path.realpath(sys.argv[0])
    parent_dir = os.path.dirname(script_path)

    for _ in range(level):
        parent_dir = os.path.dirname(parent_dir)

    sys.path.insert(0, parent_dir)


def add_parent_paths(levels=[1,2]):
    for level in levels:
        add_parent_path(level=level)
