from pathlib import Path
from tqdm import tqdm

import numpy as np

import quaternion
import argparse
import torch
import json
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parent / 'niloc'))
sys.path.append(str(Path(__file__).resolve().parent / 'ronin')) 
sys.path.append(str(Path(__file__).resolve().parent / 'ronin/source')) 
# NOTE: Both projects, but especially ronin, does some weird things with its internal imports, 
# janky pathing, so we have to um, append to system path

from torch.utils.data import Dataset, DataLoader
# from niloc.preprocess.real_data.distance_sample import adjust_to_uniform_speed
# from niloc.niloc.network.scheduled_2branch import Scheduled2branchModule
# from niloc.niloc.network.base_models import ScheduledSamplingModule
# from niloc.niloc.evaluate import compute_output_for_trajectory
from ronin.source.data_glob_speed import GlobSpeedSequence
from ronin.source.model_resnet1d import *
from ronin.source.ronin_resnet import  run_test, get_dataset
from ronin.source.data_utils import CompiledSequence, select_orientation_source, load_cached_sequences

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}

class IMUDataDataset(Dataset):
    def __init__(self, features, device):
        self.features = features
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]).to(self.device)

def get_model(arch):
    if arch == 'resnet18':
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
    elif arch == 'resnet101':
        _fc_config['fc_dim'] = 1024
        network = ResNet1D(_input_channel, _output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **_fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    window_size = 200
    
    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    network = get_model(args.arch)
    network.eval()
    network.double()

    with open(args.data) as f:
     data = json.loads(f.read())

    gyro_uncalib = data['synced/gyro_uncalib']
    acce_uncalib = data['synced/acce']
    gyro = gyro_uncalib - np.array(data['imu_init_gyro_bias'])
    acce = np.array(data['imu_acce_scale']) * (acce_uncalib - np.array(data['imu_acce_bias']))

    ori_q = quaternion.from_float_array(np.copy(data['synced/game_rv']))
    init_tango_ori = quaternion.quaternion(*data['pose/tango_ori'][0])
    rot_imu_to_tango = quaternion.quaternion(*data["start_calibration"])
    init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
    ori_q = init_rotor * ori_q

    _fc_config['in_dim'] = window_size // 32 + 1

    _ts = np.copy(data['synced/time'])
    dt = (_ts[window_size:] - _ts[:-window_size])[:, None]
    
    tango_pos = np.copy(data['pose/tango_pos'])
    glob_v = (tango_pos[window_size:] - tango_pos[:-window_size]) / dt

    gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
    acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
    glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
    glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

    start_frame = data["start_frame"]
    
    ts = _ts[start_frame:]
    features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
    features_reshaped = features.reshape(-1,  6,  1)
    orientations = quaternion.as_float_array(ori_q)[start_frame:]
    dataset = IMUDataDataset(features_reshaped, device)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # Assuming features is your NumPy array containing the features
    print("Shape of features array:", features.shape)

    preds = np.concatenate([network(feat.view(feat.size(0),  6, -1).to(device).double()).cpu().detach().numpy()  
                            for feat in tqdm(dataloader)],  
                           axis=0)

    np.save(args.out, preds)
    # with open("niloc/niloc/config/defaults.yaml") as f:
    #     config = yaml.safe_load(f)
    #     print(config)
