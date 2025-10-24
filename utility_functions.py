import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# marching cubes library
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import binary_erosion, binary_dilation

import os
import time
import math

import torch

import gc
import pickle

import inspect
from datetime import datetime

import cv2


def log_function_call(func):
    def wrapper(*args, **kwargs):
        # Get the current frame (call stack) to find out where the function was called from
        caller_frame = inspect.stack()[
            1
        ]  # [1] gives the caller's frame, [0] is the wrapper itself
        filename = caller_frame.filename
        line_number = caller_frame.lineno

        now = datetime.now()
        # print(
        #     f"Function '{func.__name__}' was called from {filename} at line {line_number} on {now.strftime('%H:%M:%S')}"
        # )
        # Call the original function
        return func(*args, **kwargs)

    return wrapper


def print_duration(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the wrapped function
        end_time = time.time()  # Record the end time
        duration = end_time - start_time
        # print(f"Function {func.__name__} took {duration:.2f} seconds.")
        return result

    return wrapper


# Print the contents of the npz file
def print_stats(ckpt):
    # Converts to decimal point notation for easier number reading
    def decimal(n):
        return re.sub(r"(?<!^)(?=(\d{3})+$)", r".", str(n))

    set, counts = np.unique(ckpt["links"], return_counts=True)

    print("Inactive voxel index (Negative index)")
    print("Key:count")
    log = 1
    for s, c in zip(set, counts):
        if c > 1:
            log += c
            print(s, decimal(c))

    total_voxels = ckpt["links"].shape[0] ** 3
    print("Amount of voxels: {}".format(decimal(total_voxels)))
    print("Amount of inactive voxels: {}".format(decimal(log)))
    print("Amount of active voxels: {}".format(decimal(total_voxels - log)))


@print_duration
def get_checkpoint_data(ckpt):
    """
    Unpacks the checkpoint and returns the data
    """

    radius = ckpt["radius"]
    center = ckpt["center"]
    links = ckpt["links"]
    density_data = ckpt["density_data"]
    sh_data = ckpt["sh_data"]

    basis_type = ckpt["basis_type"]

    # Check if the dataset is a foreground-background model
    if "background_links" in ckpt.keys():
        background_links = ckpt["background_links"]
    else:
        background_links = None
    if "background_data" in ckpt.keys():
        background_data = ckpt["background_data"]
    else:
        background_data = None

    # Set all empty voxels to a single value
    links[links < 0] = -1

    return (
        radius,
        center,
        links,
        density_data,
        sh_data,
        background_links,
        background_data,
        basis_type,
    )


def index_to_coord(index, size_x, size_y, size_z):
    """
    Translates a 1D node index to its respective 3d (x,y,z) coord.
    """

    z = index // (size_x * size_y)
    a = index % (size_x * size_y)
    y = a // size_x
    x = a % size_x
    if x >= size_x or y >= size_y or z >= size_z:
        print(f"{x,y,z} yielded for index {index} and sizes {size_x,size_y,size_z}")
        exit_file()
    return x, y, z


def coord_to_index(coord, size_x, size_y, size_z):
    if coord[0] >= size_x or coord[1] >= size_y or coord[2] >= size_z:
        print(
            f"index {index} yielded for coord {coord} and sizes {size_x,size_y,size_z}"
        )
        exit_file()
    return (coord[2] * size_z * size_y) + (coord[1] * size_x) + coord[0]


@print_duration
def get_energy_grid(axis, model, density_data, links, sh_data, use_abs, mode):
    # Density
    if mode == "density":
        energy_grid = density_data[links]
    elif mode == "sh":
        a = np.linalg.norm(sh_data, axis=1)
        # Create a grid using the indices stored in links
        energy_grid = a[links]
    elif mode == "both":
        # density_data_rescaled = (density_data - np.min(sh_data)) / (
        #     np.max(sh_data) - np.min(sh_data)
        # )

        # combined_data = np.column_stack((sh_data, density_data_rescaled))
        # a = np.linalg.norm(combined_data, axis=1)

        # Multiply the values together
        density_data_rescaled = (density_data - np.min(density_data)) / (
            np.max(density_data) - np.min(density_data)
        )
        a = np.linalg.norm(sh_data, axis=1).reshape(-1, 1) * density_data_rescaled

        # Create a grid using the indices stored in links
        energy_grid = a[links]
    else:
        print(f"Mode {mode} not a valid option!")
        exit_file()

    # Convert to float64 to allow larger weights
    energy_grid = np.squeeze(energy_grid.astype(np.float64))
    assert energy_grid.ndim == 3

    if use_abs:
        # Calculate the gradient in every direction
        energy_grid_x = np.abs(
            np.abs(energy_grid) - np.abs(np.roll(energy_grid, 1, axis=0))
        )
        energy_grid_y = np.abs(
            np.abs(energy_grid) - np.abs(np.roll(energy_grid, 1, axis=1))
        )
        energy_grid_z = np.abs(
            np.abs(energy_grid) - np.abs(np.roll(energy_grid, 1, axis=2))
        )
    else:
        # Calculate the gradient in every direction
        energy_grid_x = energy_grid - np.roll(energy_grid, 1, axis=0)
        energy_grid_y = energy_grid - np.roll(energy_grid, 1, axis=1)
        energy_grid_z = energy_grid - np.roll(energy_grid, 1, axis=2)

    plot_energy_grid(model, 0, energy_grid_x, mode, 50)
    return energy_grid_x, energy_grid_y, energy_grid_z


@print_duration
def downscale_energy_grid(
    downscale_factor, energy_grid_x, energy_grid_y, energy_grid_z
):
    # plot_energy_grid(0,energy_grid_x,'1')
    # Calculate the required values for matrix subsizing
    n = energy_grid_x.shape[0]
    block_size = downscale_factor
    new_size = n // block_size

    reshaped_matrix = energy_grid_x.reshape(
        (new_size, block_size, new_size, block_size, new_size, block_size)
    )
    downscaled_energy_grid_x = np.mean(reshaped_matrix, axis=(1, 3, 5))

    # plot_energy_grid(0,downscaled_energy_grid_x,'2')

    reshaped_matrix = energy_grid_y.reshape(
        (new_size, block_size, new_size, block_size, new_size, block_size)
    )
    downscaled_energy_grid_y = np.mean(reshaped_matrix, axis=(1, 3, 5))

    reshaped_matrix = energy_grid_z.reshape(
        (new_size, block_size, new_size, block_size, new_size, block_size)
    )
    downscaled_energy_grid_z = np.mean(reshaped_matrix, axis=(1, 3, 5))

    # Rescale the matrix to recast to int64 (prevents flat seams)
    flattened_matrix_x = downscaled_energy_grid_x.flatten()
    nonzero_x = flattened_matrix_x[flattened_matrix_x != 0]
    smallest_x = np.abs(np.min(nonzero_x))
    exponent_x = math.ceil(-math.log10(smallest_x))
    factor_x = 10**exponent_x

    # Rescale the matrix to recast to int64 (prevents flat seams)
    flattened_matrix_y = downscaled_energy_grid_y.flatten()
    nonzero_y = flattened_matrix_y[flattened_matrix_y != 0]
    smallest_y = np.abs(np.min(nonzero_y))
    exponent_y = math.ceil(-math.log10(smallest_y))
    factor_y = 10**exponent_y

    # Rescale the matrix to recast to int64 (prevents flat seams)
    flattened_matrix_z = downscaled_energy_grid_z.flatten()
    nonzero_z = flattened_matrix_z[flattened_matrix_z != 0]
    smallest_z = np.abs(np.min(nonzero_z))
    exponent_z = math.ceil(-math.log10(smallest_z))
    factor_z = 10**exponent_z

    downscaled_energy_grid_x = (downscaled_energy_grid_x * factor_x).astype(np.int64)
    downscaled_energy_grid_y = (downscaled_energy_grid_y * factor_y).astype(np.int64)
    downscaled_energy_grid_z = (downscaled_energy_grid_z * factor_z).astype(np.int64)

    return downscaled_energy_grid_x, downscaled_energy_grid_y, downscaled_energy_grid_z


@print_duration
def write_3D_matrix_to_txt(matrix, filename):
    # Ensure the input is a 3D numpy array
    if matrix.ndim != 3:
        raise ValueError("Input matrix must be 3D")

    # Open the file for writing
    with open(filename, "w") as file:
        # Iterate over the 3D array
        for i in range(matrix.shape[0]):
            file.write(f"Slice {i}:\n")
            np.savetxt(file, matrix[i], fmt="%.5f")  # Write the 2D slice
            file.write("\n")  # Add a newline for separation between slices


@print_duration
def print_matrix_stats(matrix):
    # Compute statistics
    mean = np.mean(matrix)
    median = np.median(matrix)
    std_dev = np.std(matrix)
    variance = np.var(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    sum_val = np.sum(matrix)

    # Print statistics
    print("Statistics of the 3D matrix:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")
    print(f"Sum: {sum_val}")
    print(f"dtype: {matrix.dtype}")
    print(f"shape: {matrix.shape}")

    print("\n" * 3)


@print_duration
def get_min_max_coords(coords):
    # Find the bounding box from the seam coordinates
    x_values = [coord[0] for coord in coords]
    y_values = [coord[1] for coord in coords]
    z_values = [coord[2] for coord in coords]

    # Find min and max for x, y, and z
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    min_z = min(z_values)
    max_z = max(z_values)

    result = (min_x, max_x), (min_y, max_y), (min_z, max_z)
    return result


@print_duration
def change_permissions(project_folder):
    os.system(f"chmod -R 777 {project_folder}")
    # print("Changed permissions")


@print_duration
def format_code():
    os.system("black /svox2/ -q")


@print_duration
def clean_shit():
    torch.cuda.empty_cache()
    gc.collect()


def load_seam(filename):
    if os.path.exists(filename):
        # Load the output from the file
        with open(filename, "rb") as file:
            seam = pickle.load(file)
            # print("Loading seam from file")
            return seam
    else:
        print(f"No file named {filename} found.")
        exit_file()


@log_function_call
def exit_file():
    exit()


@log_function_call
def marching_cubes(grid, name, energy_grid=None, convert=False):
    # if os.path.exists(f"obj_files/output-{name}.obj"):
    #     # print('.obj already exists!')
    #     return
    # Switch the axes to allign the bulldozer properly
    # grid = np.transpose(grid_z, (0, 2, 1))[:, :, ::-1]
    # print(f'Generating marching cubes {name}.obj')
    if convert:
        if energy_grid is None:
            print("No energy grid provided!")
            exit_file()
        # print(energy_grid.shape)
        full = np.zeros_like(energy_grid)

        for x, y, z in grid:
            full[x, y, z] = 1
        grid = full

    verts, faces, normals, values = measure.marching_cubes(grid)

    def convert_to_obj(verts, faces, normals, values):
        # print('Writing to .obj!')
        # Export to OBJ

        with open(f"obj_files/{name}.obj", "w") as f:
            if "INCORRECT" not in name:
                # print(f"Writing .obj as {f.name}!")

                # Write vertices
                for vertex in verts:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                # Write normals
                for normal in normals:
                    f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

                # Write faces (OBJ format uses 1-based indexing)
                for face in faces:
                    # Adding 1 to each index to match OBJ format
                    f.write(
                        f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n"
                    )
            else:
                print("INCORRECT IN NAME!")
                exit_file()
        f.close()

    convert_to_obj(verts, faces, normals, values)
    return


@log_function_call
def plot_interactive_plot(left_partition, name, size_x, size_y, size_z):
    import plotly.graph_objects as go

    x, y, z = zip(*left_partition)
    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=5,
                color=z,  # Color by z-coordinate (or any other value)
                colorscale="Viridis",
                opacity=0.8,
            ),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X axis", range=[0, size_x], autorange=False),
            yaxis=dict(title="Y axis", range=[0, size_y], autorange=False),
            zaxis=dict(title="Z axis", range=[0, size_z], autorange=False),
        )
    )

    fig.write_html(f"{name}.html")
    print(f"Writing interactive plot as {name}.html")
    # change_permissions(f"/svox2/custom_code/{name}.html")


@log_function_call
def plot_3d_coords(coords, name, size_x, size_y, size_z, axis):
    print("Starting plot!")
    savepath = f"/svox2/custom_code/images/3dseamplots/Axis={axis}-{name}"

    plt.clf()
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the cubes
    for coord in coords:
        ax.bar3d(coord[0], coord[2], coord[1], 1, 1, 1, color="b", alpha=1)

    # Set custom axis limits
    ax.set_xlim(0, size_x)  # Customize x-axis limits
    ax.set_ylim(0, size_z)  # Customize y-axis limits
    ax.set_zlim(0, size_y)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.view_init(-210, 150, -180)
    plt.savefig(savepath)
    plt.close()
    print(f"plotting now, saving the figure to {savepath}")


@log_function_call
def plot_energy_grid(model, axis, energy_grid, energy_grid_mode, interval):
    filename = f"{model}-{energy_grid_mode}-{axis}"

    vmin = -0.1
    vmax = 0.1

    # if not os.path.exists(f"images/animated_energy_grids/{name}_{axis}.gif"):

    fig, ax = plt.subplots()
    if axis == 0:
        im = ax.imshow(energy_grid[0, :, :], cmap="bwr", vmin=vmin, vmax=vmax)
    elif axis == 1:
        im = ax.imshow(energy_grid[:, 0, :], cmap="bwr", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(energy_grid[:, :, 0], cmap="bwr", vmin=vmin, vmax=vmax)

    title = ax.set_title(f"Axis = {axis}| Frame(0)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy Value")

    # Update function to animate through each slice
    def update(frame):
        if axis == 0:
            im.set_array(energy_grid[frame, :, :])
        elif axis == 1:
            im.set_array(energy_grid[:, frame, :])
        else:
            im.set_array(energy_grid[:, :, frame])
        title.set_text(f"Axis = {axis}| Frame({frame})")
        return im, title

    ani = FuncAnimation(fig, update, frames=energy_grid.shape[axis], interval=interval)

    # print(f"writing to images/animated_energy_grids/{name}_{axis}.gif")
    filename_gif = f"images/animated_energy_grids/{filename}.gif"
    ani.save(filename_gif, writer="pillow")
    print(f"Saved energy grid animation to {filename_gif}.")
    plt.close(fig)

    # if not os.path.exists(f"images/energy_grid_slices/{name}_{axis}.png"):
    n_plot = 5
    begin = 200
    end = 300
    diff = end - begin
    plot_per = diff // n_plot

    fig, axes = plt.subplots(1, n_plot, figsize=(12, 3))
    for i in range(n_plot):
        slice_index = begin + (i * plot_per)
        if axis == 0:
            im = axes[i].imshow(
                energy_grid[slice_index, :, :], cmap="bwr", vmin=vmin, vmax=vmax
            )
        elif axis == 1:
            im = axes[i].imshow(
                energy_grid[:, slice_index, :], cmap="bwr", vmin=vmin, vmax=vmax
            )
        else:
            im = axes[i].imshow(
                energy_grid[:, :, slice_index], cmap="bwr", vmin=vmin, vmax=vmax
            )
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        # axes[i].set_title(f"Slice {slice_index}")
        # fig.colorbar(im, ax=axes[i], orientation='vertical')
    # for ax in axes:
    #     ax.axis("off")
    plt.tight_layout()

    filename = f"images/energy_grid_slices/{filename}_sbs.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved sbs energy grid to {filename}.")
    return


@log_function_call
def plot_energy_slice(matrix_3d, slice_index, axis=0, cmap="viridis"):
    """
    Plots a 2D slice from a 3D matrix of floats.

    Parameters:
        matrix_3d (numpy.ndarray): The 3D matrix of floats.
        slice_index (int): The index of the slice to plot.
        axis (int): The axis along which to slice (0, 1, or 2).
        cmap (str): Colormap to use for the plot (default: 'viridis').
    """
    # Extract the slice along the specified axis
    if axis == 0:
        slice_2d = matrix_3d[slice_index, :, :]
    elif axis == 1:
        slice_2d = matrix_3d[:, slice_index, :]
    elif axis == 2:
        slice_2d = matrix_3d[:, :, slice_index]
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Plot the slice
    plt.imshow(slice_2d, cmap=cmap, interpolation="none")
    plt.colorbar()  # Add a color bar to indicate the scale
    plt.title(f"Slice {slice_index} along axis {axis}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f"images/energy_grids/X.png")


def plot_image_grid(n, m, model):
    """
    Plots an n x m grid of images from the given directory path and optionally saves the plot.

    Parameters:
    path (str): Directory containing images.
    n (int): Number of rows.
    m (int): Number of columns.
    """
    path = "/svox2/custom_code/datasets/scene_data/"
    if model == "lego":
        path += "lego_real_night_radial/images"
    elif model == "m60":
        path += "M60/rbg"
    elif model == "playground":
        path += "Playground/rgb"
    elif model == "truck":
        path += "Truck/rgb"
    else:
        print("Incorrect model!")
        exit_file()

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(path)
        if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif"))
    ]

    # Ensure there are enough images
    num_images = min(len(image_files), n * m)

    if num_images == 0:
        print("No images found in the directory.")
        return

    fig, axes = plt.subplots(n, m, figsize=(m * 3, n * 3))
    fig.subplots_adjust(wspace=0.2)  # Adjust horizontal spacing
    axes = axes.flatten() if n > 1 and m > 1 else [axes]

    for i in range(num_images):
        img_path = os.path.join(path, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        axes[i].imshow(img)
        axes[i].axis("off")
        # axes[i].set_title(os.path.basename(image_files[i]))

    # Hide unused subplots
    for j in range(num_images, n * m):
        axes[j].axis("off")

    plt.tight_layout(pad=0.00)
    filename = f"images/dataset_plots/{model}-sbs{m}x{n}-plot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved sbs dataset to {filename}.")
    return
