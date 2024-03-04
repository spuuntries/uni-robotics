import h5py
import json
import logging
import argparse

class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')  # Decode bytes to string
        return json.JSONEncoder.default(self, obj)

def hdf5_to_json(hdf5_path, json_path):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[()].tolist()
        
        data = {}
        hdf5_file.visititems(visit_func)
        
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, cls=BytesEncoder)

def __main__():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    try:
        hdf5_to_json(args.data, args.data + ".json")
        logging.info(f"Successfully converted {args.data} to JSON.")
    except Exception as e:
        logging.error(f"Failed to convert {args.data} to JSON: {e}")

if __name__ == "__main__":
    __main__()
