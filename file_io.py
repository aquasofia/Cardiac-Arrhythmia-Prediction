
from typing import List
import os


def get_files_from_dir_with_os(dir_name: str) \
        -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)

# EOF
