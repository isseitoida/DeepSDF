#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import re

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split, split_strain, split_mass, split_volume, split_xdirection, split_ydirection, split_zdirection):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    

    strain_files = []
    for dataset1 in split_strain:
        for class_name1 in split_strain[dataset1]:
            for instance_name1 in split_strain[dataset1][class_name1]:
                instance_filename1 = os.path.join(
                    dataset1, class_name1, instance_name1 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename1)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename1)
                    )
                strain_files += [instance_filename1]
    
    mass_files = []
    for dataset2 in split_mass:
        for class_name2 in split_mass[dataset2]:
            for instance_name2 in split_mass[dataset2][class_name2]:
                instance_filename2 = os.path.join(
                    dataset2, class_name2, instance_name2 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename2)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename2)
                    )
                mass_files += [instance_filename2]
    
    volume_files = []
    for dataset3 in split_volume:
        for class_name3 in split_volume[dataset3]:
            for instance_name3 in split_volume[dataset3][class_name3]:
                instance_filename3 = os.path.join(
                    dataset3, class_name3, instance_name3 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename3)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename3)
                    )
                volume_files += [instance_filename3]

    xdirection_files = []
    for dataset4 in split_xdirection:
        for class_name4 in split_xdirection[dataset4]:
            for instance_name4 in split_xdirection[dataset4][class_name4]:
                instance_filename4 = os.path.join(
                    dataset4, class_name4, instance_name4 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename4)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename4)
                    )
                xdirection_files += [instance_filename4]

    ydirection_files = []
    for dataset5 in split_ydirection:
        for class_name5 in split_ydirection[dataset5]:
            for instance_name5 in split_ydirection[dataset5][class_name5]:
                instance_filename5 = os.path.join(
                    dataset5, class_name5, instance_name5 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename5)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename5)
                    )
                ydirection_files += [instance_filename5]

    zdirection_files = []
    for dataset6 in split_zdirection:
        for class_name6 in split_zdirection[dataset6]:
            for instance_name6 in split_zdirection[dataset6][class_name6]:
                instance_filename6 = os.path.join(
                    dataset6, class_name6, instance_name6 + ""
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename6)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename6)
                    )
                zdirection_files += [instance_filename6]

    return npzfiles, strain_files, mass_files, volume_files, xdirection_files, ydirection_files, zdirection_files


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, strain_energy, mass, volume, xdirection, ydirection, zdirection, subsample=None):
  
    strain_energy_tensor = torch.full((50000, 1), strain_energy)
    mass_tensor = torch.full((50000, 1), mass)
    volume_tensor = torch.full((50000, 1), volume)
    xdirection_tensor = torch.full((50000, 1), xdirection)
    ydirection_tensor = torch.full((50000, 1), ydirection)
    zdirection_tensor = torch.full((50000, 1), zdirection)

    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()


    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)


    samples = torch.cat([sample_pos, sample_neg], 0)
    samples = torch.cat((samples, strain_energy_tensor, mass_tensor, volume_tensor, xdirection_tensor, ydirection_tensor, zdirection_tensor), dim=1)
    samples_np = samples.numpy()

    return samples


def unpack_sdf_samples_from_ram(data, strain_energy, mass, volume, xdirection, ydirection, zdirection, subsample=None):
    strain_energy_tensor = torch.full((30000, 1), strain_energy)
    mass_tensor = torch.full((30000, 1), mass)
    volume_tensor = torch.full((30000, 1), volume)
    xdirection_tensor = torch.full((30000, 1), xdirection)
    ydirection_tensor = torch.full((30000, 1), ydirection)
    zdirection_tensor = torch.full((30000, 1), zdirection)
    if subsample is None:
        return data
    
    # print("unpack_sdf_samples_from_ram",data)
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)
    samples = torch.cat((samples, strain_energy_tensor, mass_tensor, volume_tensor, xdirection_tensor, ydirection_tensor, zdirection_tensor), dim=1)

    return samples

def extract_numbers_from_file(filename):
    with open(filename, 'r') as file:
        # numbers = [int(line.strip()) for line in file]
        numbers = [float(line.strip()) for line in file]
    return numbers


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        split_strain,
        split_mass,
        split_volume,
        split_xdirection,
        split_ydirection,
        split_zdirection,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles, self.strainfiles, self.massfiles, self.volumefiles, self.xdirection, self.ydirection, self.zdirection = get_instance_filenames(data_source, split, split_strain, split_mass, split_volume,\
                                                                                                   split_xdirection, split_ydirection, split_zdirection)

        # print(self.npyfiles)
        # print(self.strainfiles)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram


        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        # print(idx)
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        filename_strain = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.strainfiles[idx]
        )     
        filename_mass = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.massfiles[idx]
        )     
        filename_volume = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.volumefiles[idx]
        )  
        filename_xdirection = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.xdirection[idx]
        )  
        filename_ydirection = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.ydirection[idx]
        )  
        filename_zdirection = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.zdirection[idx]
        )  
        # print(filename)

        numbers_in_filename_strain = extract_numbers_from_file(filename_strain)
        numbers_in_filename_mass = extract_numbers_from_file(filename_mass)
        numbers_in_filename_volume = extract_numbers_from_file(filename_volume)
        numbers_in_filename_xdirection = extract_numbers_from_file(filename_xdirection)
        numbers_in_filename_ydirection = extract_numbers_from_file(filename_ydirection)
        numbers_in_filename_zdirection = extract_numbers_from_file(filename_zdirection)


        if numbers_in_filename_strain:
            strain_energy = numbers_in_filename_strain[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")
        
        if numbers_in_filename_mass:
            mass = numbers_in_filename_mass[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")
        
        if numbers_in_filename_volume:
            volume = numbers_in_filename_volume[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")

        if numbers_in_filename_xdirection:
            xdirection = numbers_in_filename_xdirection[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")

        if numbers_in_filename_ydirection:
            ydirection = numbers_in_filename_ydirection[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")

        if numbers_in_filename_zdirection:
            zdirection = numbers_in_filename_zdirection[0]  # リストの最初の要素を取得
        else:
            print("No numbers found in the file.")

        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], strain_energy, mass, volume, self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, strain_energy, mass, volume, xdirection, ydirection, zdirection, self.subsample), idx
