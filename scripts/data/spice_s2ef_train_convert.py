import argparse

import h5py
from datasets import Dataset
from tqdm import tqdm


def process_hdf5_file(file_path, output_path):
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        dataset = {
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

        dataset = Dataset.from_dict(dataset)
        dataset.save_to_disk(output_path)


def main(args):
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
