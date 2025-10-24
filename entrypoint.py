from pipeline import *
from utility_functions import *
import re
import os

if __name__ == "__main__":
    os.system("clear")
    clean_shit()

    # ==========================================================================================================================

    # Pipeline settings
    model = "playground"  # Model name ["lego", "m60", "playground", "truck", "room"]
    editmode = 1  # Editing mode, 0 = shrink, 1 = stretch

    draw_seam = False  # Draw the seam on the render
    downscale_factor = 8  # Downscaled grid resize factor
    axis = 1  # Slicing axis
    special_value = 1000000000  # Value to fill edges with
    energy_grid_mode = "density"  # Energy function type ['density','sh','both']
    use_abs = True  # Get the absolute value of the energy grid (remove negative values)

    # Alter settings
    bb = False  # Draw bounding box
    box = False  # Draw a box on the bulldozer
    remove_floaters = True  # Remove empty space voxels
    stats = False  # Print scene statistics of the dataset

    # Important settings
    n_seams = 30  # Amount of seams to be calculated
    carve = True  # Calculate slice
    edit = True  # Use the slices to edit the scene
    render = True  # Render the scene
    background = True  # Render background

    # Load the correct seam from the checkpoints
    filename_prefix1 = f"seam_checkpoints/"
    filename_prefix2 = f"model:{model}-axis:{axis}-energy:{energy_grid_mode}-ABS:{use_abs}-RM:{remove_floaters}"
    filename_prefix = filename_prefix1 + filename_prefix2

    if n_seams == 0:
        print("No carving required.")
        carve = False

    # Check if seams need to be calculated (Speeds up rendering without needing to calculate field)
    folder_path = "seam_checkpoints/"
    pattern = re.compile(r"-index(\d+)\.pkl$")
    files = os.listdir(folder_path)
    index_values = []
    for filename in files:
        if filename.startswith(filename_prefix2):
            match = pattern.search(filename)
            if match:
                index_values.append(int(match.group(1)))
    if index_values:
        max_seam_index = max(index_values)
    else:
        max_seam_index = -1

    max_seam_index_str = max_seam_index + 1 if max_seam_index != -1 else 0
    print(f"Available seams: {max_seam_index_str}")

    if max_seam_index + 1 == n_seams or n_seams == 0:
        print("Required seams already present, skipping carving.")
        carve = False

    if not carve and edit:
        if max_seam_index + 1 < n_seams and n_seams != 0:
            print("Not enough seams to edit!")
            exit_file()
    # ==========================================================================================================================

    # n, m = 4, 4
    # plot_image_grid(n, m, "lego")
    # plot_image_grid(n, m, "playground")

    # print(filename_prefix)
    change_permissions("/svox2/")

    main(
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
    )

    change_permissions("/svox2/")
    format_code()
    clean_shit()
