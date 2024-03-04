import json
import argparse

def merge_json_files(file_path_A, file_path_B, output_path):
    with open(file_path_A, 'r', encoding='utf-8') as file_A:
        data_A = json.load(file_A)
    with open(file_path_B, 'r', encoding='utf-8') as file_B:
        data_B = json.load(file_B)
    
    merged_data = {**data_A, **data_B}
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Merge two JSON files.')
    parser.add_argument('file_A', help='Path to the first JSON file.')
    parser.add_argument('file_B', help='Path to the second JSON file.')
    parser.add_argument('output', help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    merge_json_files(args.file_A, args.file_B, args.output)

if __name__ == '__main__':
    main()
