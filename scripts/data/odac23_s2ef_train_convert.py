"""Scripts to pre-process ODAC23 dataset into a HuggingFace dataset."""

import argparse
import glob
from multiprocessing import Pool
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from ocpmodels.datasets import LmdbDataset
from tqdm import tqdm


def process_data(db_paths: List[str], output_dir: str) -> None:
    """Process a chunk of data."""
    for db_path in db_paths:
        dataset_dict: Dict[str, List[Any]] = {
            "input_ids": [],
            "coords": [],
            "forces": [],
            "formation_energy": [],
            "total_energy": [],
            "has_formation_energy": [],
        }
        db = LmdbDataset(config={"src": db_path})
        for i in range(len(db)):
            dataset_dict["input_ids"].append(db[i].atomic_numbers.short())
            dataset_dict["coords"].append(db[i].pos)
            dataset_dict["forces"].append(db[i].force)
            dataset_dict["formation_energy"].append(np.array(db[i].y).astype("float32"))
            dataset_dict["total_energy"].append(np.array(db[i].raw_y).astype("float32"))
            dataset_dict["has_formation_energy"].append(True)
        dataset = Dataset.from_dict(dataset_dict)
        dataset.save_to_disk(f'{output_dir}/{db_path.split("/")[-1].split(".")[-2]}')


def main(args: argparse.Namespace) -> None:
    """Process the ODAC23 dataset."""
    # Process LMDB files
    odac23_dbs = glob.glob(f"{args.input_dir}/*.lmdb")
    odac23_dbs = sorted(odac23_dbs, key=lambda x: int(x.split("/")[-1].split(".")[-2]))

    num_processes = args.num_processes
    chunks = odac23_dbs
    chunk_size = len(chunks) // num_processes
    chunks = [chunks[i : i + chunk_size] for i in range(0, len(chunks), chunk_size)]  # type: ignore

    with Pool(num_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(lambda x: process_data(x, args.output_dir), chunks),  # type: ignore
            total=len(chunks),
        ):
            pass

    # Concatenate processed datasets
    datasets = []
    odac23_s2ef_paths = glob.glob(f"{args.output_dir}/*")
    odac23_s2ef_paths = sorted(odac23_s2ef_paths, key=lambda x: int(x.split("/")[-1]))

    for path in tqdm(odac23_s2ef_paths):
        datasets.append(load_from_disk(path))

    odac23_s2ef = concatenate_datasets(datasets)
    odac23_s2ef.save_to_disk(args.final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ODAC23 dataset into a HuggingFace dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input LMDB files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save intermediate processed datasets",
    )
    parser.add_argument(
        "--final_output",
        type=str,
        required=True,
        help="Path to save the final concatenated dataset",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for parallel processing",
    )

    args = parser.parse_args()
    main(args)
