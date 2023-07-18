import math
import torch
import random
import scipy.signal
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sin, cos


transform_order = {
    'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
}


def process_stream(data, stream):
    if stream == 'joint':
        return data

    elif stream == 'motion':
        motion = torch.zeros_like(data)
        motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
        return motion

    elif stream == 'bone':
        Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        bone = torch.zeros_like(data)

        for v1, v2 in Bone:
            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

        return bone

    elif stream == '3s':
        joint_data = process_stream(data, stream='joint')
        motion_data = process_stream(data, stream='motion')
        bone_data = process_stream(data, stream='bone')
        return torch.cat((joint_data, motion_data, bone_data), dim=1)

    elif stream == '2s':
        joint_data = process_stream(data, stream='joint')
        bone_data = process_stream(data, stream='bone')
        return torch.cat((joint_data, bone_data), dim=1)

    else:
        raise ValueError


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 0]), R)
    data_numpy = data_numpy.transpose(2, 0, 1)
    return data_numpy


def temporal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V = data_numpy.shape
    padding_len = T // temperal_padding_ratio
    frame_start = np.random.randint(0, padding_len * 2 + 1)
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def random_spatial_flip(seq, p=0.5):
    if random.random() < p:
        # Do the left-right transform C,T,V,M
        index = transform_order['ntu']
        trans_seq = seq[:, :, index]
        return trans_seq
    else:
        return seq


def random_time_flip(seq, p=0.5):
    T = seq.shape[1]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return seq[:, time_range_reverse, :]
    else:
        return seq


def random_rotate(seq):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # # z
        # if axis == 2:
        #     R = np.array([[cos(angle), sin(angle), 0],
        #                   [-sin(angle), cos(angle), 0],
        #                   [0, 0, 1]])
        R = R.T
        seq = torch.from_numpy(seq).repeat(1,1,2)[:,:,:3].numpy()
        temp = np.matmul(seq, R)[:,:,:2]
        return temp

    new_seq = seq.copy()
    # C, T, V, M -> T, V, M, C
    new_seq = np.transpose(new_seq, (1, 2, 0))
    total_axis = [0, 1]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    new_seq = np.transpose(new_seq, (2, 0, 1))

    return new_seq


def gaus_noise(data_numpy, mean=0, std=0.01, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V))
        return temp + noise
    else:
        return data_numpy


def gaus_filter(data_numpy):
    g = GaussianBlurConv(2)
    return g(data_numpy)


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=2, kernel=15, sigma=[0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        # kernel =  kernel.float()
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)  # (3,1,1,5)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        x = torch.from_numpy(x)
        if prob < 0.5:
            x = x.permute(0, 2, 1).unsqueeze(0)  # M,C,V,T
            x = F.conv2d(x, self.weight, padding=(
                0, int((self.kernel - 1) / 2)),   groups=self.channels)
            x = x.squeeze(0).permute(0, -1, -2)  # C,T,V,M

        return x.numpy()


class Zero_out_axis(object):
    def __init__(self, axis=None):
        self.first_axis = axis

    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0, 1)

        temp = data_numpy.copy()
        C, T, V = data_numpy.shape
        x_new = np.zeros((T, V))
        temp[axis_next] = x_new
        return temp


def axis_mask(data_numpy, p=0.5):
    am = Zero_out_axis()
    if random.random() < p:
        return am(data_numpy)
    else:
        return data_numpy


def resample(data_numpy):
    rate = random.randint(7, 13) * 0.1
    num_frames = int(data_numpy.shape[1] * rate)
    data_numpy = scipy.signal.resample(data_numpy, num_frames, axis=1)
    return data_numpy


def filter(data_numpy, p=0.5):
    if random.random() < p:
        return scipy.signal.savgol_filter(data_numpy, 15, 2, axis=1)
    else:
        return data_numpy


def limbs_mask(data_numpy, p=0.5):
    if random.random() < p:
        if random.random() < 0.5:
            # drop right limbs
            drop_idx = [9, 10, 11, 17, 18, 19, 23, 24]
            data_numpy[:, :, drop_idx, :] = 0
        else:
            # drop left limbs
            drop_idx = [5, 6, 7, 13, 14, 15, 21, 22]
            data_numpy[:, :, drop_idx, :] = 0

    return data_numpy


def temporal_shift(data_numpy):
    offset = random.randint(0, data_numpy.shape[1]-1)
    data_numpy = np.concatenate((data_numpy[:, offset:], data_numpy[:, :offset]), 1)
    return data_numpy


if __name__ == '__main__':
    data_seq = np.ones((3, 50, 25, 2))
    data_seq = axis_mask(data_seq)
    print(data_seq.shape)
