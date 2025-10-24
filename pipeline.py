import numpy as np
import re
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from seam_finding_pipeline import *
from utility_functions import *
import torch

from tqdm import tqdm
import pickle
import warnings

warnings.simplefilter("default")


def find_and_save_seams(
    axis,
    ckpt,
    edited_ckpt_path,
    editmode,
    energy_grid_mode,
    n_seams,
    special_value,
    radius,
    center,
    links,
    density_data,
    sh_data,
    background_links,
    background_data,
    basis_type,
    downscale_factor,
    bb,
    box,
    remove_floaters,
    carve,
    draw_seam,
    model,
    filename_prefix,
    use_abs,
):

    if draw_seam and not carve:
        carve = True

    # Get the shape of the scene (x,y,z)
    size_x, size_y, size_z = links.shape

    # Get the  value and index of densest voxel in the model (To have a good value for visualisation)
    max_index = np.argmax(density_data)

    if remove_floaters:
        new_links = links

        if model == "lego":

            # Bottom
            new_links[:, 310:, :] = -1

            # # Top
            new_links[:, :190, :] = -1

            # Front and back
            # :190,320:
            new_links[:190, :, :] = -1
            new_links[320:, :, :] = -1

            # Sides
            new_links[:, :, :150] = -1
            new_links[:, :, 350:] = -1

        elif model == "playground":
            # As seen from the perspective of the first render

            # Bottom
            new_links[:, 380:, :] = -1

            # Top
            new_links[:, :300, :] = -1

            # :100,575:
            # :330,360:
            # Left
            new_links[:100, :, :] = -1

            # Right
            new_links[575:, :, :] = -1

            #
            new_links[:, :, :200] = -1

            # Back
            new_links[:, :, 550:] = -1

            # new_links[:, 340, :] = max_index
            # new_links[:, 300, :] = max_index
        # (1408, 1156, 128)
        elif model == "room":
            new_links[:, :, 120:127] = max_index

        links = np.copy(new_links)

    if bb:
        print("Adding bounding box!")
        links[0, 0, :] = max_index
        links[0, size_y - 1, :] = max_index
        links[size_x - 1, 0, :] = max_index
        links[size_x - 1, size_y - 1, :] = max_index

        links[0, :, 0] = max_index
        links[0, :, size_z - 1] = max_index
        links[size_x - 1, :, 0] = max_index
        links[size_x - 1, :, size_z - 1] = max_index

        links[:, 0, 0] = max_index
        links[:, 0, size_z - 1] = max_index
        links[:, size_y - 1, 0] = max_index
        links[:, size_y - 1, size_z - 1] = max_index

    if box:
        print("Adding box!")
        links[:, :, :] = max_index

    scene_grid = (links >= 0).astype(int)
    marching_cubes(scene_grid, f"COMPLETE-{model}")
    # exit_file()

    # Calculate the energy function over the entire scene
    if carve:
        assert editmode in [0, 1], "editmode is not valid!"
        assert n_seams > 0, "Must be more than 0 seams"
        assert axis in [0, 1, 2], "axis must be 0, 1 or 2!"

        # Get the energy grids at the beginning to allow marking done slices
        energy_grid_x, energy_grid_y, energy_grid_z = get_energy_grid(
            axis, model, density_data, links, sh_data, use_abs, mode=energy_grid_mode
        )

        # plot_energy_grid("playground", 0, energy_grid_x, energy_grid_mode, 50)
        # exit_file()

        # Remove empty space to force seam through bulldozer
        if model == "lego":
            if axis == 0:
                # :190,290:
                energy_grid_x[:200, :, :] = special_value
                energy_grid_x[270:, :, :] = special_value
                energy_grid_y[:200, :, :] = special_value
                energy_grid_y[270:, :, :] = special_value
                energy_grid_z[:200, :, :] = special_value
                energy_grid_z[270:, :, :] = special_value
            elif axis == 1:
                energy_grid_x[:, :190, :] = special_value
                energy_grid_x[:, 310:, :] = special_value
                energy_grid_y[:, :190, :] = special_value
                energy_grid_y[:, 310:, :] = special_value
                energy_grid_z[:, :190, :] = special_value
                energy_grid_z[:, 310:, :] = special_value
            elif axis == 2:
                energy_grid_x[:, :, :150] = special_value
                energy_grid_x[:, :, 350:] = special_value
                energy_grid_y[:, :, :150] = special_value
                energy_grid_y[:, :, 350:] = special_value
                energy_grid_z[:, :, :150] = special_value
                energy_grid_z[:, :, 350:] = special_value

        # Remove empty space to force seam through playground
        elif model == "playground":
            if axis == 0:
                # :100, 575:
                energy_grid_x[:330, :, :] = special_value
                energy_grid_x[360:, :, :] = special_value
                energy_grid_y[:330, :, :] = special_value
                energy_grid_y[360:, :, :] = special_value
                energy_grid_z[:330, :, :] = special_value
                energy_grid_z[360:, :, :] = special_value
            elif axis == 1:
                # :300,
                energy_grid_x[:, :290, :] = special_value
                energy_grid_x[:, 350:, :] = special_value
                energy_grid_y[:, :290, :] = special_value
                energy_grid_y[:, 350:, :] = special_value
                energy_grid_z[:, :290, :] = special_value
                energy_grid_z[:, 350:, :] = special_value
            elif axis == 2:
                energy_grid_x[:, :, :200] = special_value
                energy_grid_x[:, :, 550:] = special_value
                energy_grid_y[:, :, :200] = special_value
                energy_grid_y[:, :, 550:] = special_value
                energy_grid_z[:, :, :200] = special_value
                energy_grid_z[:, :, 550:] = special_value

        # Get the downscaled energy grid
        (
            downscaled_energy_grid_x,
            downscaled_energy_grid_y,
            downscaled_energy_grid_z,
        ) = downscale_energy_grid(
            downscale_factor, energy_grid_x, energy_grid_y, energy_grid_z
        )
        if model == "lego":
            if axis == 0:
                downscaled_energy_grid_x[:24, :, :] = special_value
                downscaled_energy_grid_x[73:, :, :] = special_value
                downscaled_energy_grid_y[:24, :, :] = special_value
                downscaled_energy_grid_y[73:, :, :] = special_value
                downscaled_energy_grid_z[:24, :, :] = special_value
                downscaled_energy_grid_z[73:, :, :] = special_value
            else:
                raise NotImplementedError
        elif model == "playground":
            if axis == 0:
                downscaled_energy_grid_x[:13, :, :] = special_value
                downscaled_energy_grid_x[72:, :, :] = special_value
                downscaled_energy_grid_y[:13, :, :] = special_value
                downscaled_energy_grid_y[72:, :, :] = special_value
                downscaled_energy_grid_z[:13, :, :] = special_value
                downscaled_energy_grid_z[72:, :, :] = special_value
            elif axis == 1:
                downscaled_energy_grid_x[:37, :, :] = special_value
                downscaled_energy_grid_x[43:, :, :] = special_value
                downscaled_energy_grid_y[:37, :, :] = special_value
                downscaled_energy_grid_y[43:, :, :] = special_value
                downscaled_energy_grid_z[:37, :, :] = special_value
                downscaled_energy_grid_z[43:, :, :] = special_value
            elif axis == 2:
                raise NotImplementedError

        # Seam carving loop
        if carve:
            for seam_index in tqdm(range(n_seams), desc="Carving progress"):
                clean_shit()

                # print(f"Working for seam {seam_index}")

                # Find the next seam
                seam = seam_carve(
                    axis,
                    downscaled_energy_grid_x,
                    downscaled_energy_grid_y,
                    downscaled_energy_grid_z,
                    downscale_factor,
                    energy_grid_x,
                    energy_grid_y,
                    energy_grid_z,
                    seam_index,
                    size_x,
                    size_y,
                    size_z,
                    special_value,
                    links,
                    model,
                    filename_prefix,
                )

                if axis == 0:
                    for coord in seam:

                        # Shift the energy grid inwards by one seam
                        x, y, z = coord

                        # Shift the values in the energy grid
                        energy_grid_x[x:-1, y, z] = energy_grid_x[x + 1 :, y, z]
                        energy_grid_y[x:-1, y, z] = energy_grid_y[x + 1 :, y, z]
                        energy_grid_z[x:-1, y, z] = energy_grid_z[x + 1 :, y, z]

                    # If enough seams have been generated, also remove part of the downscaled grid
                    if seam_index + 1 % downscale_factor == 0:
                        downscaled_energy_grid_x[x:-1, y, z] = downscaled_energy_grid_x[
                            x + 1 :, y, z
                        ]
                elif axis == 1:
                    for coord in seam:

                        # Shift the energy grid inwards by one seam
                        x, y, z = coord

                        # Shift the values in the energy grid
                        energy_grid_x[x, y:-1, z] = energy_grid_x[x, y + 1 :, z]
                        energy_grid_y[x, y:-1, z] = energy_grid_y[x, y + 1 :, z]
                        energy_grid_z[x, y:-1, z] = energy_grid_z[x, y + 1 :, z]

                    # If enough seams have been generated, also remove part of the downscaled grid
                    if seam_index + 1 % downscale_factor == 0:
                        downscaled_energy_grid_y[x, y:-1, z] = downscaled_energy_grid_y[
                            x, y + 1 :, z
                        ]
                elif axis == 2:
                    for coord in seam:

                        # Shift the energy grid inwards by one seam
                        x, y, z = coord

                        # Shift the values in the energy grid
                        energy_grid_x[x, y, z:-1] = energy_grid_x[x, y, z + 1 :]
                        energy_grid_y[x, y, z:-1] = energy_grid_y[x, y, z + 1 :]
                        energy_grid_z[x, y, z:-1] = energy_grid_z[x, y, z + 1 :]

                    # If enough seams have been generated, also remove part of the downscaled grid
                    if seam_index + 1 % downscale_factor == 0:
                        downscaled_energy_grid_z[x, y, z:-1] = downscaled_energy_grid_z[
                            x, y, z + 1 :
                        ]
                # Get the new downscaled grid
                (
                    downscaled_energy_grid_x,
                    downscaled_energy_grid_y,
                    downscaled_energy_grid_z,
                ) = downscale_energy_grid(
                    downscale_factor, energy_grid_x, energy_grid_y, energy_grid_z
                )

    print("Finished finding seams!")
    return (
        axis,
        ckpt,
        edited_ckpt_path,
        editmode,
        energy_grid_mode,
        n_seams,
        special_value,
        radius,
        center,
        links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
        max_index,
    )


def pack_and_save_checkpoint(
    radius,
    center,
    links,
    density_data,
    sh_data,
    background_links,
    background_data,
    basis_type,
    edited_ckpt,
    edited_ckpt_path,
    ckpt,
):
    edited_ckpt["radius"] = radius
    edited_ckpt["center"] = center
    edited_ckpt["links"] = links
    edited_ckpt["density_data"] = density_data
    edited_ckpt["sh_data"] = sh_data
    edited_ckpt["basis_type"] = basis_type

    if background_links is not None:
        edited_ckpt["background_links"] = background_links
    if background_data is not None:
        edited_ckpt["background_data"] = background_data

    # print(f"Saving edited checkpoint to {edited_ckpt_path}")
    # Save the edited checkpoint dict as a .npz file and remove any ownership restrictions
    np.savez(edited_ckpt_path, **edited_ckpt)
    os.system(f"chmod -R 777 {edited_ckpt_path}")


# Entrypoint for the code. Change parameters here
def main(
    downscale_factor,
    edit,
    bb,
    box,
    remove_floaters,
    carve,
    draw_seam,
    model,
    axis,
    special_value,
    editmode,
    n_seams,
    stats,
    render,
    background,
    energy_grid_mode,
    filename_prefix,
    use_abs,
):
    assert downscale_factor > 0
    bool_list = [True, False]
    assert bb in bool_list
    assert box in bool_list
    assert remove_floaters in bool_list
    assert carve in bool_list
    assert draw_seam in bool_list

    model_list = ["lego", "m60", "playground", "truck", "room"]
    assert model in model_list, f"model is {model}; should be one of {model_list}"
    assert axis in [0, 1, 2]
    assert editmode in [0, 1]
    assert n_seams >= 0
    assert stats in bool_list
    assert render in bool_list
    assert background in bool_list
    assert energy_grid_mode in ["density", "sh", "both"]

    clean_shit()
    # os.system("clear")  # Clear the terminal
    # print(filename_prefix)

    os.system(f"python misc_testing/testing_file.py {model}")  # Changes camera matrix

    os.environ["MKL_THREADING_LAYER"] = "GNU"  # Fixes an error IDK
    change_permissions("/svox2/")  # Makes files editable in the docker container

    edit_str = "shrink" if editmode == 0 else "stretch"
    print(
        f"General:\t\t Model: {model} / axis: {axis} / energy grid mode: {energy_grid_mode} / downscale_factor: {downscale_factor}"
    )
    print(
        f"Settings:\t\t render: {render} / editmode: {editmode} ({edit_str}) / n_seams: {n_seams} / background: {background}"
    )
    print(
        f"Experiments:\t\t bb: {bb} / box: {box} / remove_floaters: {remove_floaters} / carve: {carve} / draw_seam: {draw_seam}"
    )
    print("\n")

    GPU_available = torch.cuda.is_available()
    # print(f"GPU available: {GPU_available}")

    if not GPU_available:
        print("No GPU found!")
        print("=" * 50)
        print("Run the following command in a new terminal:\n\n")
        print(" sudo systemctl restart docker ")
        print("\n\n")
        print("=" * 50)
        exit_file()

    # Path to original checkpoint
    ckpt_path = "/svox2/custom_code/datasets/checkpoints/"
    if model == "lego":
        ckpt_path += "lego_real_night_hitvbg_dec"
    elif model == "m60":
        ckpt_path += "M60"
    elif model == "playground":
        ckpt_path += "Playground"
    elif model == "truck":
        ckpt_path += "Truck"
    elif model == "room":
        ckpt_path += "room"
    ckpt_path += "/ckpt.npz"

    # Get the checkpoint
    ckpt = dict(np.load(ckpt_path))

    # Path to save the edited checkpoint
    edited_ckpt_path = f"/svox2/custom_code/edited_ckpt_{model}/"
    if not os.path.exists(edited_ckpt_path):
        os.makedirs(edited_ckpt_path, exist_ok=True)
    edited_ckpt_path = edited_ckpt_path + "ckpt.npz"

    # Print the stats of the checkpoint
    if stats:
        print_stats(ckpt)

    """
    SEAM FINDING STARTS HERE
    """
    # This creates saves for the individual seams
    # ['radius', 'center', 'links', 'density_data', 'sh_data', 'background_links', 'background_data', 'basis_type']
    (
        radius,
        center,
        links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
    ) = get_checkpoint_data(ckpt)

    (
        axis,
        ckpt,
        edited_ckpt_path,
        editmode,
        energy_grid_mode,
        n_seams,
        special_value,
        radius,
        center,
        links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
        max_index,
    ) = find_and_save_seams(
        axis,
        ckpt,
        edited_ckpt_path,
        editmode,
        energy_grid_mode,
        n_seams,
        special_value,
        radius,
        center,
        links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
        downscale_factor,
        bb,
        box,
        remove_floaters,
        carve,
        draw_seam,
        model,
        filename_prefix,
        use_abs,
    )

    """
    EDIT SCENE USING THE CALCULATED SEAMS
    """
    # Create a copy to not mess with the original model
    edited_ckpt = ckpt.copy()
    final_links = links.copy()

    # Luckily, we can remove or add with the same calculated seams
    if edit:
        if editmode == 0:
            for n in tqdm(range(n_seams), desc="Editing progress"):
                print(n)
                # Get the filename for the current seam
                filename = f"{filename_prefix}-index{n}.pkl"
                # print(f"Loading {filename}.")
                # Check if the file exists
                seam = load_seam(filename)

                # Edit links! This is where seam insertion/removal happens.
                for coord in seam:
                    # Shrink
                    x, y, z = coord

                    if axis == 0:
                        try:
                            final_links[x:-1, y, z] = final_links[x + 1 :, y, z]
                            final_links[-(n + 1) :, y, z] = -1
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()
                    elif axis == 1:
                        try:
                            final_links[x, y:-1, z] = final_links[x, y + 1 :, z]
                            final_links[x, -(n + 1) :, z] = -1
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()
                    elif axis == 2:
                        try:
                            final_links[x, y, z:-1] = final_links[x:, y, z + 1 :]
                            final_links[x, y, -(n + 1) :] = -1
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()

        elif editmode == 1:
            for n in tqdm(reversed(range(n_seams)), desc="Editing progress"):
                # Get the filename for the current seam
                filename = f"{filename_prefix}-index{n}.pkl"
                # print(f"Loading {filename}.")
                # Check if the file exists
                seam = load_seam(filename)

                # Edit links! This is where seam insertion/removal happens.
                for coord in seam:
                    # Shrink
                    x, y, z = coord

                    if axis == 0:
                        try:
                            final_links[x + 1 :, y, z] = final_links[x:-1, y, z]
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()
                    elif axis == 1:
                        try:
                            final_links[x, y + 1 :, z] = final_links[x, y:-1, z]
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()
                    elif axis == 2:
                        try:
                            final_links[x, y, z + 1 :] = final_links[x, y, z:-1]
                        except IndexError:
                            print("IndexError! (Pipeline.py)")
                            exit_file()

    scene_grid = (final_links >= 0).astype(int)
    marching_cubes(scene_grid, f"EDITED-COMPLETE-{model}")

    if draw_seam:
        print("Drawing seam.")
        filename = f"{filename_prefix}-index{0}.pkl"
        seam = load_seam(filename)
        for coord in seam:
            x, y, z = coord
            final_links[x, y, z] = max_index

    # Add links to the checkpoint and save the edited checkpoint
    pack_and_save_checkpoint(
        radius,
        center,
        final_links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
        edited_ckpt,
        edited_ckpt_path,
        ckpt,
    )

    """
    RENDERING STARTS HERE
    """

    # print(edited_ckpt_path)
    if render:
        render_str = "python /svox2/opt/render_imgs.py "
        render_str_bg = render_str + "--nobg"

        render_str = (
            f"{render_str} {edited_ckpt_path} /svox2/custom_code/datasets/scene_data/"
        )
        render_str_bg = f"{render_str_bg} {edited_ckpt_path} /svox2/custom_code/datasets/scene_data/"
        if model == "lego":
            render_str += "lego_real_night_radial/"
            render_str_bg += "lego_real_night_radial/"
        elif model == "m60":
            render_str += "M60/"
            render_str_bg += "M60/"
        elif model == "playground":
            render_str += "Playground/"
            render_str_bg += "Playground/"
        elif model == "truck":
            render_str += "Truck/"
            render_str_bg += "Truck/"
        elif model == "room":
            render_str += "room/"
        else:
            print("Incorrect model!")
            exit_file()

        # print("Rendering with background.")
        os.system(render_str)
        # print("Rendering without background.")
        if model != "room":
            os.system(render_str_bg)
    return 0


if __name__ == "__main__":
    print("=====\n\nTo run, use\n\npython entrypoint.py\n\n=====")
