import h5py
import json
import argparse

def create_group_from_json(hdf5_file, json_data, path=None):
    if path is None:
        path = '/'
    for key, value in json_data.items():
        if isinstance(value, dict):
            group = hdf5_file.create_group(path + key)
            create_group_from_json(group, value, path + key + '/')
        else:
            hdf5_file.create_dataset(path + key, data=value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dump JSON data into an HDF5 file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file.')
    parser.add_argument('hdf5_file', type=str, help='Path to the output HDF5 file.')

    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    with h5py.File(args.hdf5_file, 'w') as hdf:
        create_group_from_json(hdf, data)
