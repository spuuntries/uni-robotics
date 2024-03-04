import json
import numpy as np
import argparse

def dump_ronin(npy_path, json_path, output_path):
    ronin_data = [pos[:2] for pos in (np.load(npy_path)).tolist()]
    with open(json_path, 'r', encoding='utf-8') as file_B:
        json_data = json.load(file_B)
    
    merged_data = { **json_data,
                   "computed/aligned_pos": ronin_data, 
                   "computed/ronin": ronin_data}
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_path', help='Path to the .npy file.')
    parser.add_argument('json_path', help='Path to the JSON file.')
    parser.add_argument('output', help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    dump_ronin(args.npy_path, args.json_path, args.output)

if __name__ == '__main__':
    main()
