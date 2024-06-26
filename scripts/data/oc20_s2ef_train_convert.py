import argparse
import glob
import lzma
import os
from multiprocessing import Pool

import pandas as pd
from ase import io
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm


def extract_xz(xz_path):
    output_path = xz_path.replace(".xz", "")
    with lzma.open(xz_path) as f:
        with open(output_path, "wb") as out:
            file_content = f.read()
            out.write(file_content)
    return output_path


def extract_txt(txt_path):
    output_path = txt_path.replace(".xz", "")
    with lzma.open(txt_path) as f:
        with open(output_path, "wb") as out:
            file_content = f.read()
            out.write(file_content)
    return output_path


def process_data(xz_txt_paths):
    for xz_txt_path in xz_txt_paths:
        dataset = {
            "input_ids": [],
            "coords": [],
            "forces": [],
            "formation_energy": [],
            "total_energy": [],
            "has_formation_energy": [],
        }
        xz_path, txt_path, output_dir = xz_txt_path
        atoms_path = extract_xz(xz_path)
        energy_path = extract_txt(txt_path)
        meta = pd.read_csv(
            energy_path, names=["system_id", "frame_id", "reference_energy"]
        )
        atoms = io.read(atoms_path, index=":")
        os.remove(atoms_path)
        os.remove(energy_path)

        for i in range(len(atoms)):
            dataset["input_ids"].append(atoms[i].get_atomic_numbers().astype("int16"))
            dataset["coords"].append(atoms[i].get_positions().astype("float32"))
            dataset["forces"].append(
                atoms[i].get_forces(apply_constraint=False).astype("float32")
            )
            dataset["formation_energy"].append(
                atoms[i].get_potential_energy(apply_constraint=False).astype("float32")
                - meta.iloc[i]["reference_energy"].astype("float32")
            )
            dataset["total_energy"].append(
                atoms[i].get_potential_energy(apply_constraint=False).astype("float32")
            )
            dataset["has_formation_energy"].append(True)

        dataset = Dataset.from_dict(dataset)
        dataset.save_to_disk(
            os.path.join(output_dir, f'{xz_path.split("/")[-1].split(".")[0]}')
        )


def main():
    parser = argparse.ArgumentParser(description="Process OC20 S2EF data")
    parser.add_argument(
        "--split", type=str, help='The value to replace "val_id" in the paths'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data2/atomistic_datasets/opencatalyst/s2ef/oc20_s2ef",
        help="Output directory",
    )
    parser.add_argument(
        "--num_processes", type=int, default=11, help="Number of processes to use"
    )
    args = parser.parse_args()

    output_dir = args.output_dir + f"_{args.split}" + "_datasets"

    oc20_s2ef_xz = glob.glob(
        f"/mnt/data2/atomistic_datasets/opencatalyst/s2ef/oc20_s2ef-val-test/{args.split}/s2ef_{args.split}/*.extxyz.xz"
    )
    oc20_s2ef_xz = sorted(
        oc20_s2ef_xz, key=lambda x: int(x.split("/")[-1].split(".")[0])
    )

    oc20_s2ef_txt = glob.glob(
        f"/mnt/data2/atomistic_datasets/opencatalyst/s2ef/oc20_s2ef-val-test/{args.split}/s2ef_{args.split}/*.txt.xz"
    )
    oc20_s2ef_txt = sorted(
        oc20_s2ef_txt, key=lambda x: int(x.split("/")[-1].split(".")[0])
    )

    # Split data into chunks for parallel processing
    chunks = [
        (xz, txt, dir)
        for xz, txt, dir in zip(
            oc20_s2ef_xz, oc20_s2ef_txt, [output_dir] * len(oc20_s2ef_xz)
        )
    ]
    chunk_size = len(chunks) // args.num_processes
    chunks = [chunks[i : i + chunk_size] for i in range(0, len(chunks), chunk_size)]

    # Create a pool of processes
    with Pool(args.num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_data, chunks), total=len(chunks)):
            pass

    datasets = []
    oc20_s2ef_paths = glob.glob(output_dir + "/*")
    oc20_s2ef_paths = sorted(oc20_s2ef_paths, key=lambda x: int(x.split("/")[-1]))
    # load and concatenate the datasets
    for path in tqdm(oc20_s2ef_paths):
        datasets.append(load_from_disk(path))

    oc20_s2ef = concatenate_datasets(datasets)
    oc20_s2ef.save_to_disk(output_dir[:-9])

    os.system(f"rm -rf {output_dir}")


if __name__ == "__main__":
    main()
