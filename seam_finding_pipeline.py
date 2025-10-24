import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from utility_functions import *
from downscaled import *
from fullscale import *

import pickle
import os


# The main seam finding pipeline. Seam downsizing and finding happens here
@print_duration
def seam_carve(
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
):
    clean_shit()

    # This works when not having the special value constraints!
    # condition = ((energy_grid_x > -1) & (energy_grid_x < special_value))
    # indices = np.argwhere(condition)
    # coordinates = [(x, y, z) for (x, y, z) in indices]

    # bulldozer = np.zeros_like(energy_grid_x)
    # for x,y,z in coordinates:
    #     bulldozer[x,y,z] = 1
    # marching_cubes(energy_grid_x,f'BULLDOZER')
    # exit_file()

    downsized_size_x = size_x // downscale_factor
    downsized_size_y = size_y // downscale_factor
    downsized_size_z = size_z // downscale_factor

    # First, find a downscaled seam to determine the bounding box for the actual seam
    g, source, sink, sourceid, sinkid = find_grid_downscaled(
        axis, downsized_size_x, downsized_size_y, downsized_size_z
    )

    filename = f"{filename_prefix}-index{seam_index}-downscaled.pkl"

    # Check if the file exists
    if os.path.exists(filename):
        # Load the output from the file
        with open(filename, "rb") as file:
            left_partition = pickle.load(file)
            # print('Loaded downscaled seam!')
    else:
        # print('No seam saved. Calculating.')
        left_partition, cap = find_seam_downscaled(
            axis,
            downscaled_energy_grid_x,
            downscaled_energy_grid_y,
            downscaled_energy_grid_z,
            downsized_size_x,
            downsized_size_y,
            downsized_size_z,
            g,
            seam_index,
            source,
            sourceid,
            special_value,
            sink,
            sinkid,
        )

        with open(filename, "wb") as file:
            pickle.dump(left_partition, file)

    assert len(left_partition) > 0

    for coord in left_partition:
        if (
            coord[0] >= downsized_size_x
            or coord[1] >= downsized_size_y
            or coord[2] >= downsized_size_z
        ):
            exit_file()

    # print("Downscaling done")
    """
    FULL SEAM STARTS HERE
    """
    # Get the region in which the full seam should be found
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_min_max_coords(left_partition)

    min_x *= downscale_factor
    max_x = (max_x * downscale_factor) + (downscale_factor - 1)

    min_y *= downscale_factor
    max_y = (max_y * downscale_factor) + (downscale_factor - 1)

    min_z *= downscale_factor
    max_z = (max_z * downscale_factor) + (downscale_factor - 1)

    alternate_size = 10

    # Ensure temporal direction is not too big for memory
    if axis == 0:
        diff = min(alternate_size, max_x - min_x)
        max_x = min_x + diff
        grid_sizes = (diff, size_y, size_z)
    elif axis == 1:
        diff = min(alternate_size, max_y - min_y)
        max_y = min_y + diff
        grid_sizes = (size_x, diff, size_z)
    elif axis == 2:
        diff = min(alternate_size, max_z - min_z)
        max_z = min_z + diff
        grid_sizes = (size_x, size_y, diff)

    grid_x, grid_y, grid_z = grid_sizes

    g, source, sink, sourceid, sinkid = find_grid_fullscaled(
        axis, grid_x, grid_y, grid_z, grid_sizes
    )

    # Get the energy grid of slice to be seam carved
    if axis == 0:
        sub_energy_grid_x = energy_grid_x[min_x : min_x + diff, :, :]
        sub_energy_grid_y = energy_grid_y[min_x : min_x + diff, :, :]
        sub_energy_grid_z = energy_grid_z[min_x : min_x + diff, :, :]
    elif axis == 1:
        sub_energy_grid_x = energy_grid_x[:, min_y : min_y + diff, :]
        sub_energy_grid_y = energy_grid_y[:, min_y : min_y + diff, :]
        sub_energy_grid_z = energy_grid_z[:, min_y : min_y + diff, :]
    elif axis == 2:
        sub_energy_grid_x = energy_grid_x[:, :, min_z : min_z + diff]
        sub_energy_grid_y = energy_grid_y[:, :, min_z : min_z + diff]
        sub_energy_grid_z = energy_grid_z[:, :, min_z : min_z + diff]

    filename = f"{filename_prefix}-index{seam_index}.pkl"

    # Check if the file exists
    if os.path.exists(filename):
        # Load the output from the file
        with open(filename, "rb") as file:
            full_seam = pickle.load(file)
            # print(f"Seam exists. Loaded full seam {seam_index} from {filename}!")
    else:
        full_seam = find_seam_fullscaled(
            axis,
            sub_energy_grid_x,
            sub_energy_grid_y,
            sub_energy_grid_z,
            size_x,
            size_y,
            size_z,
            g,
            seam_index,
            source,
            sourceid,
            special_value,
            sink,
            sinkid,
            grid_sizes,
            min_x,
            min_y,
            min_z,
            model,
            filename_prefix,
        )

        # Offset the coords by the slice in the graph
        if axis == 0:
            full_seam = [(node[0] + min_x, node[1], node[2]) for node in full_seam]
            for node in full_seam:
                if node[0] >= size_x:
                    exit_file()
        elif axis == 1:
            full_seam = [(node[0], node[1] + min_y, node[2]) for node in full_seam]
            for node in full_seam:
                if node[1] >= size_y:
                    exit_file()
        elif axis == 2:
            full_seam = [(node[0], node[1], node[2] + min_z) for node in full_seam]
            for node in full_seam:
                if node[2] >= size_z:
                    exit_file()

        # Save the output to the file
        with open(filename, "wb") as file:
            pickle.dump(full_seam, file)
        # print(f"Computed and saved full seam to file for N={seam_index}.")

    # Check the seam for any errors
    # If any errors are found here the program has a bug
    # print(size_x,size_y,size_z)
    for coord in full_seam:
        if coord[0] >= size_x or coord[1] >= size_y or coord[2] >= size_z:
            exit_file()
    return full_seam


if __name__ == "__main__":
    print(
        "\n\n\nThis is a helper file, call 'python pipeline.py' to run the code!\n\n\n"
    )
    exit_file()
