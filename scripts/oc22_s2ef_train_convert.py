import argparse
from datasets import Dataset, load_from_disk, concatenate_datasets
from ocpmodels.datasets import LmdbDataset
from tqdm import tqdm
import glob
import numpy as np

def process_data(db_path, output_dir):
    dataset = {"input_ids": [], "coords": [], "forces": [], "formation_energy": [], "total_energy": [], "has_formation_energy": []}
    db = LmdbDataset(config={"src": db_path})
    for j in tqdm(range(len(db)), desc=f"Processing {db_path.split('/')[-1]}"):
        dataset['input_ids'].append(db[j].atomic_numbers.short())
        dataset['coords'].append(db[j].pos)
        dataset['forces'].append(db[j].force)
        dataset['formation_energy'].append(np.array(0).astype("float32"))
        dataset['total_energy'].append(np.array(db[j].y).astype("float32"))
        dataset['has_formation_energy'].append(False)
    dataset = Dataset.from_dict(dataset)
    dataset.save_to_disk(f'{output_dir}/{db_path.split("/")[-1].split(".")[-2]}')

def main(args):
    # Process LMDB files
    oc22_dbs = glob.glob(f'{args.input_dir}/*.lmdb')
    oc22_dbs = sorted(oc22_dbs, key=lambda x: int(x.split('/')[-1].split('.')[-2]))

    for db_path in oc22_dbs:
        process_data(db_path, args.output_dir)

    # Concatenate processed datasets
    datasets = []
    oc22_s2ef_paths = glob.glob(f'{args.output_dir}/*')
    oc22_s2ef_paths = sorted(oc22_s2ef_paths, key=lambda x: int(x.split('/')[-1]))

    for path in tqdm(oc22_s2ef_paths, desc="Loading processed datasets"):
        datasets.append(load_from_disk(path))

    oc22_s2ef = concatenate_datasets(datasets)
    oc22_s2ef.save_to_disk(args.final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess OC22 dataset into a HuggingFace dataset.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input LMDB files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save intermediate processed datasets')
    parser.add_argument('--final_output', type=str, required=True, help='Path to save the final concatenated dataset')
    
    args = parser.parse_args()
    main(args)