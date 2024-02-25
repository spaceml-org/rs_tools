import glob, os
from dateutil.parser import parse


def get_list_filenames(data_path: str="./", ext: str="*"):
    """Loads a list of file names within a directory
    """
    pattern = f"*{ext}"
    return sorted(glob.glob(os.path.join(data_path, "**", pattern), recursive=True))
