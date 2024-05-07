#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils


def create_mesh(
    decoder, latent_vec, re_strain, re_mass, re_volume, re_xdirection, re_ydirection, re_zdirection, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 67)########################3

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples[:, 3] = re_strain
    samples[:, 4] = re_mass
    samples[:, 5] = re_volume
    samples[:, 6] = re_xdirection
    samples[:, 7] = re_ydirection
    samples[:, 8] = re_zdirection
    




    # samples[:, 3] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3


    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:66]#################

        
        L = 10
        sample_subset_el = []

        for el in range(0,L):
            val = 2 ** el

            x = np.sin(val * np.pi * sample_subset[:, 0].numpy())
            sample_subset_el.append(x)
            x = np.cos(val * np.pi * sample_subset[:, 0].numpy())
            sample_subset_el.append(x)
            y = np.sin(val * np.pi * sample_subset[:, 1].numpy())
            sample_subset_el.append(y)
            y = np.cos(val * np.pi * sample_subset[:, 1].numpy())
            sample_subset_el.append(y)
            z = np.sin(val * np.pi * sample_subset[:, 2].numpy())
            sample_subset_el.append(z)
            z = np.cos(val * np.pi * sample_subset[:, 2].numpy()) 
            sample_subset_el.append(z)
        
        sample_subset_el = np.array(sample_subset_el)
        sample_subset_el = torch.tensor(sample_subset_el, dtype=torch.float32).T
        strain = sample_subset[:, 3].unsqueeze(1)
        mass = sample_subset[:, 4].unsqueeze(1)
        volume = sample_subset[:, 5].unsqueeze(1)
        xdirection = sample_subset[:, 6].unsqueeze(1)
        ydireciton = sample_subset[:, 7].unsqueeze(1)
        zdirection = sample_subset[:, 8].unsqueeze(1)

        sample_subset_el = torch.cat((sample_subset_el, strain, mass, volume, xdirection, ydireciton, zdirection), dim=1)   



        sample_subset = sample_subset_el.cuda()

        samples[head : min(head + max_batch, num_samples), 66] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 66]
    sdf_values = sdf_values.reshape(N, N, N)
    sdf_numpy = sdf_values.numpy()
    print("sdf_values",sdf_numpy.shape)
    file_path = '.sdf_numpy.npy'
    np.save(file_path, sdf_numpy)
    
    # Save the array to the specified file
    np.save(file_path, sdf_numpy)


    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()



    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]
    print("3")

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    print("4")

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    print("5")

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
