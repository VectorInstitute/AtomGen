import argparse
import numpy as np
import json
from pymatgen.core import Structure
from datasets import Dataset
from tqdm import tqdm
from json import JSONDecodeError

def process_json_chunk(chunk):
    json_loaded = json.loads(chunk)
    struct = Structure.from_dict(json_loaded['structure'])
    
    input_ids = np.array(list(struct.atomic_numbers)).astype("int16")
    return {
        'input_ids': input_ids,
        'coords': struct.cart_coords.astype("float32"),
        'forces': np.array(json_loaded['force']).astype("float32"),
        'formation_energy': np.array(json_loaded['ef_per_atom'] * len(input_ids)).astype("float32"),
        'total_energy': np.array(json_loaded['corrected_total_energy']).astype("float32"),
        'has_formation_energy': True
    }

def main(args):
    with open(args.input_file, 'r') as file:
        output = ""
        dataset = {"input_ids": [], "coords": [], "forces": [], "formation_energy": [], "total_energy": [], "has_formation_energy": []}
        num_datasets = 0
        num_samples = 0
        pbar = tqdm(total=args.total_samples)
        read = True
        
        while True:
            try:
                if read:
                    output = output + file.read(int(1e6))
                    read = False
                start = output.find('{"structure"')
                end = output.find('"mp_id"')
                if start != -1 and end != -1:
                    end = output.find('}', end)
                    if end != -1:
                        end += 1
                        num_samples += 1
                        pbar.update(1)
                        
                        chunk_data = process_json_chunk(output[start:end])
                        for key, value in chunk_data.items():
                            dataset[key].append(value)
                        
                        output = output[end:]
                        
                        if num_samples == args.samples_per_dataset:
                            dataset = Dataset.from_dict(dataset)
                            dataset.save_to_disk(f'{args.output_dir}/{num_datasets}')
                            num_datasets += 1
                            num_samples = 0
                            dataset = {"input_ids": [], "coords": [], "forces": [], "formation_energy": [], "total_energy": [], "has_formation_energy": []}
                            pbar.close()
                            pbar = tqdm(total=args.total_samples)
                    else:
                        read = True
                        continue
                else:
                    read = True
                    continue
            except Exception as e:
                print(f"An error occurred: {e}")
                break
        
        pbar.close()
        
        # Save any remaining data
        if num_samples > 0:
            dataset = Dataset.from_dict(dataset)
            dataset.save_to_disk(f'{args.output_dir}/{num_datasets}')
        
        # Concatenate all datasets
        all_datasets = [Dataset.load_from_disk(f'{args.output_dir}/{i}') for i in range(num_datasets + 1)]
        final_dataset = Dataset.concatenate_datasets(all_datasets)
        final_dataset.save_to_disk(args.final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess MPtrj dataset into a HuggingFace dataset.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed datasets')
    parser.add_argument('--final_output', type=str, required=True, help='Path to save the final concatenated dataset')
    parser.add_argument('--total_samples', type=int, default=1580395, help='Total number of samples in the dataset')
    parser.add_argument('--samples_per_dataset', type=int, default=1580394, help='Number of samples per dataset chunk')
    
    args = parser.parse_args()
    main(args)