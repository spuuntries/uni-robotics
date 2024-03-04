import sys
import h5py
import json
import os
import glob
import argparse
import subprocess
from pathlib import Path
from dumper import create_group_from_json
from conversion import hdf5_to_json
from merger import merge_json_files
from rodump import dump_ronin

# sys.path.append(str(Path(__file__).resolve().parent / 'niloc'))
# sys.path.append(str(Path(__file__).resolve().parent / 'ronin')) 
# sys.path.append(str(Path(__file__).resolve().parent / 'ronin/source'))

parser = argparse.ArgumentParser(description='Preprocess raw IMU via RoNIN and run evaluation on Niloc')
parser.add_argument('imu_files', type=str, help='Path to a directory containing HDF5 files of the IMU.')
parser.add_argument('ronin_path', type=str, help='Path to the RoNIN resnet model file.')
parser.add_argument('output_dir', type=str, help='Path to the output dir.')
parser.add_argument('--keep_inter', action="store_true", help='Keep intermediary conversion steps?')
parser.add_argument('--recompute_ronin', action="store_true", help='Recompute RoNIN?')

args = parser.parse_args()

Path(os.path.join(os.getcwd(), args.output_dir, "ronin_config")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "imu_dumps")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "ronin_dumps")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "ronin_merged")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "ronin_output")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "niloc_input")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(os.getcwd(), args.output_dir, "niloc_output")).mkdir(parents=True, exist_ok=True)

imu_list: list[str] = os.listdir(args.imu_files)
with open(os.path.join(os.getcwd(), args.output_dir, "ronin_config", "filelist.txt"), "w") as f:
    f.write("\n".join(imu_list))

print(imu_list)
def find_python_binary():
    # Check if we are in a virtual environment
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        # Construct the path to the Python binary within the virtual environment
        python_binary = os.path.join(os.getcwd(), virtual_env, 'scripts', 'python')
        if sys.platform.startswith('win'):
            python_binary += '.exe'
        return python_binary
    else:
        # Fallback to the system's default Python binary
        return sys.executable

if not os.path.isdir(os.path.join(os.getcwd(), args.output_dir, "ronin_merged")) or args.recompute_ronin:
    subprocess.run([find_python_binary(), "ronin/source/ronin_resnet.py", "--root_dir", os.path.join(os.getcwd(), args.imu_files), 
              "--test_list", os.path.join(os.getcwd(), args.output_dir, "ronin_config", "filelist.txt"),
              "--out_dir", os.path.join(os.getcwd(), args.output_dir, "ronin_output"),
              "--model_path", args.ronin_path, "--mode", "test"])
    
    for imu in imu_list:
        hdf5_to_json(os.path.join(os.getcwd(), args.imu_files, imu, "data.hdf5"), 
                     os.path.join(os.getcwd(), args.output_dir, "imu_dumps", imu + ".json"))
    print("Dumped HDF5 inputs to JSON")

    ronin_list: list[str] = list(filter(lambda x: "npy" in x, os.listdir(os.path.join(os.getcwd(), args.output_dir, "ronin_output"))))
    for npy in ronin_list:
        dump_ronin(os.path.join(os.getcwd(), args.output_dir, "ronin_output", npy), 
        os.path.join(os.getcwd(), args.output_dir, "imu_dumps", ".".join(npy.split(".")[:-1]).replace("_gsn", "") + ".json"), 
        os.path.join(os.getcwd(), args.output_dir, "ronin_dumps", ".".join(npy.split(".")[:-1]).replace("_gsn", "") + ".dumped.json"))
    print("Dumped RoNIN results into JSON")

    dump_list: list[str] = list(filter(lambda x: "dumped" in x, os.listdir(os.path.join(os.getcwd(), args.output_dir, "ronin_dumps"))))
    for dump in dump_list:
        merge_json_files(os.path.join(os.getcwd(), args.imu_files, ".".join(dump.split(".")[:-2]), "info.json"), 
        os.path.join(os.getcwd(), args.output_dir, "ronin_dumps", dump),
        os.path.join(os.getcwd(), args.output_dir, "ronin_merged", ".".join(dump.split(".")[:-2]) + ".merged.json"))
    print("Merged metadata into JSON")

    merged_list: list[str] = os.listdir(os.path.join(os.getcwd(), args.output_dir, "ronin_merged"))
    for merged in merged_list:
        with open(os.path.join(os.getcwd(), args.output_dir, "ronin_merged", merged), 'r') as f:
            data = json.load(f)

        with h5py.File(os.path.join(os.getcwd(), args.output_dir, 
                                    "niloc_input", ".".join(merged.split(".")[:-1]) + ".hdf5"), 'w') as hdf:
            create_group_from_json(hdf, data)

if not args.keep_inter:
    for f in glob.glob(os.path.join(os.getcwd(), args.output_dir, "imu_dumps") + "/*"):
        os.remove(f)
    for f in glob.glob(os.path.join(os.getcwd(), args.output_dir, "ronin_merged") + "/*"):
        os.remove(f)
    for f in glob.glob(os.path.join(os.getcwd(), args.output_dir, "ronin_dumps") + "/*"):
        os.remove(f)
    for f in glob.glob(os.path.join(os.getcwd(), args.output_dir, "ronin_output") + "/*"):
        os.remove(f)

