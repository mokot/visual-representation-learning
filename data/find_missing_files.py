import os
import argparse
from typing import Set
from pathlib import Path


def find_missing_files(root_dir: Path, num_samples: int) -> Set[int]:
    found_files = set()

    # Recursively walk through directories
    for _, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt") and filename[:-3].isdigit():
                found_files.add(int(filename[:-3]))

    expected_files = set(range(num_samples))
    missing_files = expected_files - found_files

    return missing_files


def main():
    parser = argparse.ArgumentParser(
        description="Find missing .pt files in a directory."
    )
    parser.add_argument("path", type=str, help="Root directory to search in")
    parser.add_argument("num_samples", type=int, help="Number of expected samples")
    args = parser.parse_args()

    missing_files = find_missing_files(Path(args.path), args.num_samples)

    if missing_files:
        print("Missing files:", sorted(missing_files))
    else:
        print("No missing files found.")


if __name__ == "__main__":
    main()
