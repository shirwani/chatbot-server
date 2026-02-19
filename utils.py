import os
import shutil
from functools import wraps
import csv
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
os.environ["DEBUG"] = os.getenv("DEBUG")

def read_from_csv_file_with_header(file):
    """
    Read from csv file and return list() of dict(),
    where keys of doct() are column names from the header line
    """
    with open(file) as f:
        csv_reader = csv.reader(f)
        col_names = next(csv_reader)  # header
        data = list()
        for row in csv_reader:
            d = dict()
            for c in col_names:
                d[c] = row[col_names.index(c)]
            data.append(d)
    return data


def dump_to_json_file(file, data, indent=0):
    """
    Dump data to json file
    """
    with open(file, 'w') as f:
        if indent == 0:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=indent)


def read_from_text_file(file):
    """
    Read the text file as a string
    """
    with open(file, "r") as file:
        contents = file.read()
    return contents


def read_from_json_file(file):
    """
    Read and return contents of json file,
    if file exists
    """
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)


def read_file_as_tuple(file: str) -> tuple[str, ...]:
    """
    Reads a text file and returns its non-empty lines as a tuple of strings.
    Each line in the file is stripped of leading and trailing whitespace, and only non-empty lines are included in the resulting tuple.
    Lines that start with # are ignored as comments.
    """
    with open(file, "r", encoding="utf-8") as f:
        items = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return tuple(items)


def read_file_as_list(file: str) -> list[str]:
    """
    Reads a text file and returns its non-empty lines as a list of strings.
    Each line in the file is stripped of leading and trailing whitespace, and only non-empty lines are included in the resulting list.
    Lines that start with # are ignored as comments.
    """
    with open(file, "r", encoding="utf-8") as f:
        items = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return items


def copy_file(src: str, dst: str):
    """
    Copies a file from the source path to the destination path.

    Args:
        src (str): The path to the source file that needs to be copied.
        dst (str): The path to the destination where the file should be copied.
    """
    try:
        shutil.copy(src, dst)
        print(f"File copied successfully from {src} to {dst}")
    except Exception as e:
        print(f"Error copying file from {src} to {dst}: {e}")


def move_file(src: str, dst: str):
    """
    Moves a file from the source path to the destination path.

    Args:
        src (str): The path to the source file that needs to be moved.
        dst (str): The path to the destination where the file should be moved.
    """
    try:
        #dst = os.path.join(dst, os.path.basename(src))
        print(f"Moving file from {src} to {dst}")
        shutil.move(src, dst)
        print(f"File moved successfully from {src} to {dst}")
    except Exception as e:
        print(f"Error moving file from {src} to {dst}: {e}")


def copy_files_from_directory(source_dir, dest_dir):
    """
    Copies all files from the source directory to the destination directory.

    Args:
        source_dir (str): The path to the source directory containing the files to be copied.
        dest_dir (str): The path to the destination directory where the files should be copied.
    """
    try:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for filename in os.listdir(source_dir):
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(dest_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)
                print(f"File copied successfully from {src_file} to {dst_file}")
    except Exception as e:
        print(f"Error copying files from {source_dir} to {dest_dir}: {e}")



def dbg_print(func):
    """
    A decorator that prints the name of the function being called for debugging purposes.

    Args:
        func:

    Returns:

    """
    debug_enabled = os.getenv("DEBUG") == "1" or os.getenv("DEBUG", "").lower() == "true"

    if not debug_enabled:
        # No-op: return original function
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[DEBUG] Calling: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def get_non_unique_columns(csv_path: str) -> list[str]:
    """Return column names from the CSV whose values are not all unique.

    A column is considered "non-unique" if at least one value in that
    column appears more than once.
    """
    # Reuse existing helper so CSV parsing is consistent across the project
    rows = read_from_csv_file_with_header(csv_path)
    if not rows:
        return []

    # Initialize tracking for seen values and a flag for each column
    first_row = rows[0]
    columns = list(first_row.keys())

    seen_values = {col: set() for col in columns}
    has_duplicate = {col: False for col in columns}

    for row in rows:
        for col in columns:
            val = row.get(col)
            if val in seen_values[col]:
                has_duplicate[col] = True
            else:
                seen_values[col].add(val)

    # Return columns where we detected at least one duplicate
    return [col for col, dup in has_duplicate.items() if dup]

