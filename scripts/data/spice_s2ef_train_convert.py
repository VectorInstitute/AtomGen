"""Convert the SPICE dataset into a HuggingFace dataset."""

import argparse
from typing import Any, Dict, List

import h5py
from datasets import Dataset
from tqdm import tqdm


def process_hdf5_file(file_path: str, output_path: str) -> None:
    """Process the SPICE dataset."""
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        dataset: Dict[str, List[Any]] = {
            "input_ids": [],
            "coords": [],
            "forces": [],
            "formation_energy": [],
            "total_energy": [],
            "has_formation_energy": [],
        }

        for key in tqdm(keys, desc="Processing SPICE dataset"):
            num_conformations = f[key]["conformations"][()].shape[0]
            for i in range(num_conformations):
                dataset["input_ids"].append(
                    f[key]["atomic_numbers"][()].astype("int16")
                )
                dataset["coords"].append(0.529177249 * f[key]["conformations"][()][i])
                dataset["forces"].append(
                    -27.211407953 * f[key]["dft_total_gradient"][()][i]
                )
                dataset["formation_energy"].append(
                    27.211407953 * f[key]["formation_energy"][()][i]
                )
                dataset["total_energy"].append(
                    27.211407953 * f[key]["dft_total_energy"][()][i]
                )
                dataset["has_formation_energy"].append(True)

        hf_dataset = Dataset.from_dict(dataset)
        hf_dataset.save_to_disk(output_path)


def main(args: Any) -> None:
    """Process the SPICE dataset."""
    process_hdf5_file(args.input_file, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess SPICE dataset into a HuggingFace dataset."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input HDF5 file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed dataset",
    )

    args = parser.parse_args()
    main(args)
