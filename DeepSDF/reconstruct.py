#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    filename_strain,
    filename_mass, 
    filename_volume,
    filename_xdirection, 
    filename_ydirection, 
    filename_zdirection, 
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    print("test_sdf",test_sdf)

    if type(stat) == type(0.1):
        print("latent_mean_std")
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    numbers_in_filename_strain = deep_sdf.data.extract_numbers_from_file(filename_strain)######################
    numbers_in_filename_mass = deep_sdf.data.extract_numbers_from_file(filename_mass)######################
    numbers_in_filename_volume = deep_sdf.data.extract_numbers_from_file(filename_volume)######################
    numbers_in_filename_xdirection = deep_sdf.data.extract_numbers_from_file(filename_xdirection)######################
    numbers_in_filename_ydirection = deep_sdf.data.extract_numbers_from_file(filename_ydirection)######################
    numbers_in_filename_zdirection = deep_sdf.data.extract_numbers_from_file(filename_zdirection)######################

    if numbers_in_filename_strain:
        strain_energy_pack = numbers_in_filename_strain[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", strain_energy_pack)
    else:
        print("No numbers found in the file.")
    
    if numbers_in_filename_mass:
        mass_pack = numbers_in_filename_mass[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", mass_pack)
    else:
        print("No numbers found in the file.")

    if numbers_in_filename_volume:
        volume_pack = numbers_in_filename_volume[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", volume_pack)
    else:
        print("No numbers found in the file.")    

    if numbers_in_filename_xdirection:
        xdirection_pack = numbers_in_filename_xdirection[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", xdirection_pack)
    else:
        print("No numbers found in the file.")  

    if numbers_in_filename_volume:
        ydirection_pack = numbers_in_filename_ydirection[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", ydirection_pack)
    else:
        print("No numbers found in the file.")  

    if numbers_in_filename_volume:
        zdirection_pack = numbers_in_filename_zdirection[0]  # リストの最初の要素を取得
        print("Numbers in the filename:", zdirection_pack)
    else:
        print("No numbers found in the file.")  

    for e in range(num_iterations):#正規分布から再構築する際はいらない



        decoder.eval()

        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, strain_energy_pack, mass_pack, volume_pack, xdirection_pack, ydirection_pack, zdirection_pack, num_samples
        ).cuda()
        print(sdf_data.device)


        print("sdf_data",sdf_data.shape)
        xyz = sdf_data[:, 0:3]
        xyz_debug = xyz
        L = 10
        xyz_el = []

        for el in range(0,L):
                val = 2 ** el

                x = np.sin(val * np.pi * xyz[:, 0].cpu().numpy())
                xyz_el.append(x)
                x = np.cos(val * np.pi * xyz[:, 0].cpu().numpy())
                xyz_el.append(x)
                y = np.sin(val * np.pi * xyz[:, 1].cpu().numpy())
                xyz_el.append(y)
                y = np.cos(val * np.pi * xyz[:, 1].cpu().numpy())
                xyz_el.append(y)
                z = np.sin(val * np.pi * xyz[:, 2].cpu().numpy())
                xyz_el.append(z)
                z = np.cos(val * np.pi * xyz[:, 2].cpu().numpy()) 
                xyz_el.append(z)

        xyz_el = np.array(xyz_el)
        xyz_el = torch.tensor(xyz_el, dtype=torch.float32).T
            #xyz_el = torch.tensor(xyz_el).T
        xyz = xyz_el
        sdf_gt = sdf_data[:, 3].unsqueeze(1)
        strain = sdf_data[:, 4].unsqueeze(1).to("cpu")
        mass = sdf_data[:, 5].unsqueeze(1).to("cpu")
        volume = sdf_data[:, 6].unsqueeze(1).to("cpu")
        xdirection = sdf_data[:, 7].unsqueeze(1).to("cpu")
        ydirection = sdf_data[:, 8].unsqueeze(1).to("cpu")
        zdirection = sdf_data[:, 9].unsqueeze(1).to("cpu")

        print("strain",strain)
        print("mass",mass)
        print("volume",volume)
        print("xdirection", xdirection)
        print("ydirection", ydirection)
        print("zdirection", zdirection)
        xyz = torch.cat((xyz, strain, mass, volume, xdirection, ydirection, zdirection), dim=1)        

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs.cuda(), xyz.cuda()], 1)

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()
    
    # print("sdf",xyz_debug)
    # print("sdf_shape", xyz_debug.shape)

    # counter = 0
    # for row in xyz_debug:
    #     for val in row:
    #         print(val.item(), end=" ")
    #         counter += 1
    #         if counter % 3 == 0:
    #             print()  # Change line
    

    
    latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    print("latents",latent)

    return loss_num, latent, strain_energy_pack, mass_pack, volume_pack, xdirection_pack, ydirection_pack, zdirection_pack


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=8,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--split_strain",
        "-s_strain",
        dest="split_strain_filename",
        required=True,
        help="The split to reconstruct."
    )
    arg_parser.add_argument(
        "--mass",
        "-s_mass",
        dest="split_mass_filename",
        required=True,
        help="The split to reconstruct."
    )
    arg_parser.add_argument(
        "--volume",
        "-s_volume",
        dest="split_volume_filename",
        required=True,
        help="The split to reconstruct."
    )
    arg_parser.add_argument(
        "--xdirection",
        dest="split_xdirection_filename",
        required=True,
        help="The split to reconstruct."
    )
    arg_parser.add_argument(
        "--ydirection",
        dest="split_ydirection_filename",
        required=True,
        help="The split to reconstruct."
    )
    arg_parser.add_argument(
        "--zdirection",
        dest="split_zdirection_filename",
        required=True,
        help="The split to reconstruct."
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var
    print("a")

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)
    
    with open(args.split_strain_filename, "r") as f:
        split_strain = json.load(f)#########################################################

    with open(args.split_mass_filename, "r") as f:
        mass = json.load(f)#########################################################

    with open(args.split_volume_filename, "r") as f:
        volume = json.load(f)#########################################################

    with open(args.split_xdirection_filename, "r") as f:
        xdirection = json.load(f)#########################################################

    with open(args.split_ydirection_filename, "r") as f:
        ydirection = json.load(f)#########################################################

    with open(args.split_zdirection_filename, "r") as f:
        zdirection = json.load(f)#########################################################

    npz_filenames, strain_files, mass_files, volume_files, xdirection_files, ydirection_files, zdirection_files = deep_sdf.data.get_instance_filenames(args.data_source, split, split_strain, mass, volume, xdirection, ydirection, zdirection)##############################

    # random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    
    for ii, npz in enumerate(npz_filenames):

        filename_strain = os.path.join(
            args.data_source, ws.sdf_samples_subdir, strain_files[ii]
        )

        filename_mass = os.path.join(
            args.data_source, ws.sdf_samples_subdir, mass_files[ii]
        )

        filename_volume = os.path.join(
            args.data_source, ws.sdf_samples_subdir, volume_files[ii]
        )

        filename_xdirection = os.path.join(
            args.data_source, ws.sdf_samples_subdir, xdirection_files[ii]
        )

        filename_ydirection = os.path.join(
            args.data_source, ws.sdf_samples_subdir, ydirection_files[ii]
        )

        filename_zdirection = os.path.join(
            args.data_source, ws.sdf_samples_subdir, zdirection_files[ii]
        )      

        print("filename_strain",filename_strain)
        print("filename_mass",filename_mass)
        print("filename_volume",filename_volume)

        print(ii)
        print(npz)

        if "npz" not in npz:
            continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)
        print("full_filename", full_filename)

        logging.debug("loading {}".format(npz))

        data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)
        print("data_sdf", data_sdf)
        print("a")

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent, re_strain, re_mass, re_volume, re_xdirection, re_ydirection, re_zdirection = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                filename_strain,
                filename_mass,
                filename_volume,
                filename_xdirection,
                filename_ydirection,
                filename_zdirection,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=30000,
                lr=5e-3,
                l2reg=True,
            )

            print("latent",latent)

            # print(mesh_filename)
            # print(latent_filename)

            # print("3333333333333333333333", latent.shape)

            # aa = ii + 4501

            # # # 保存先のファイルパス
            # file_path1 = f'latent_{aa}.pth'
            # torch.save(latent, file_path1)
            # latent = torch.load(file_path1)
            # file_path2 = 'latent_before_1002.pth'
            # latent2 = torch.load(file_path2)
            # # # # print("latent1",latent1)
            # # # # print("latent2", latent2)
            # # file_path3 = 'latent_689.pth'
            # # torch.save(latent, file_path3)
            # # latent = torch.load(file_path3)

            # std_dev = 0.1  # 例として適当な標準偏差を設定
            # random_displacement = torch.randn_like(latent) * std_dev
            # latent = latent + random_displacement


            # latent = latent1 + latent2
            # latent = latent / 2.0
            # print("latent", latent)
            # latent変数の値をファイルに保存
            
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            # logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            # logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()
            print("1")

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))
            
        

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        decoder, latent, re_strain, re_mass, re_volume, re_xdirection, re_ydirection, re_zdirection, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
            print(latent_filename)
