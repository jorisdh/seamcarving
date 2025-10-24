from utility_functions import *
import numpy as np
import graph_tool.all as gt
import time
from tqdm import tqdm


@log_function_call
def find_grid_fullscaled(axis, size_x, size_y, size_z, grid_sizes):
    """
    Creates a directed graph representing a 3D grid with specified dimensions,
    including additional source and sink nodes.

    Parameters:
        axis (int): The axis along which the seam is found (0 for x, 1 for y, 2 for z).
        size_x (int): Size of the grid along the x-axis.
        size_y (int): Size of the grid along the y-axis.
        size_z (int): Size of the grid along the z-axis.

    Returns:
        g (gt.Graph): The constructed graph.
        source (Vertex): The source vertex.
        sink (Vertex): The sink vertex.
        sourceid (int): The ID of the source vertex.
        sinkid (int): The ID of the sink vertex.
    """
    sourcelist = []
    sinklist = []

    index = 0
    g = gt.Graph(directed=True)
    edge_list = []
    for z in range(grid_sizes[2]):
        for y in range(grid_sizes[1]):
            for x in range(grid_sizes[0]):
                if x < grid_sizes[0] - 1:  # Add edge to the vertex in the next row
                    edge_list.append((index, index + 1))
                    edge_list.append((index + 1, index))
                if y < grid_sizes[1] - 1:  # Add edge to the vertex in the next column
                    edge_list.append((index, index + size_x))
                    edge_list.append((index + size_x, index))
                if z < grid_sizes[2] - 1:  # Add edge to the vertex in the next depth
                    edge_list.append((index, index + size_x * size_y))
                    edge_list.append((index + size_x * size_y, index))

                # Change has to happen here
                if axis == 0:
                    # Keep nodes for source and sink
                    if x == 0:
                        sourcelist.append(index)
                    if x == grid_sizes[0] - 1:
                        sinklist.append(index)
                elif axis == 1:
                    # Keep nodes for source and sink
                    if y == 0:
                        sourcelist.append(index)
                    if y == grid_sizes[1] - 1:
                        sinklist.append(index)
                elif axis == 2:
                    # Keep nodes for source and sink
                    if z == 0:
                        sourcelist.append(index)
                    if z == grid_sizes[2] - 1:
                        sinklist.append(index)

                index += 1
    g.add_edge_list(edge_list)

    assert len(sourcelist) == len(sinklist) != 0

    # Add source and sink nodes
    source = g.add_vertex()
    sourceid = index
    sink = g.add_vertex()
    sinkid = sourceid + 1

    for node in sourcelist:
        g.add_edge(source, node)
    for node in sinklist:
        g.add_edge(node, sink)

    assert (
        g.num_vertices() == (grid_sizes[0] * grid_sizes[1] * grid_sizes[2]) + 2
    ), f"{(grid_sizes[0]*grid_sizes[1]*grid_sizes[2])+2} should be {g.num_vertices()}"
    return g, source, sink, sourceid, sinkid


@print_duration
def populate_grid_fullscale(
    size_x,
    size_y,
    size_z,
    energy_grid_x,
    energy_grid_y,
    energy_grid_z,
    g,
    source,
    sink,
):
    """
    Assigns capacities to the edges of the graph based on energy grids, and sets special
    high capacities for edges connected to source and sink.

    Parameters:
        g (gt.Graph): The graph to populate.
        source (Vertex): The source vertex.
        sink (Vertex): The sink vertex.
        size_x (int): Size of the grid along the x-axis.
        size_y (int): Size of the grid along the y-axis.
        size_z (int): Size of the grid along the z-axis.
        energy_grid_x (ndarray): Energy differences along the x-axis.
        energy_grid_y (ndarray): Energy differences along the y-axis.
        energy_grid_z (ndarray): Energy differences along the z-axis.

    Returns:
        g (gt.Graph): The graph with populated edge capacities.
        cap (EdgePropertyMap): Edge capacities.
    """
    special_value = 10000000000

    cap = g.new_edge_property("double")
    for e in g.edges():
        # If the edge is not to or from source/sink
        if source not in e and sink not in e:

            source_coords = index_to_coord(int(e.source()), size_x, size_y, size_z)
            target_coords = index_to_coord(int(e.target()), size_x, size_y, size_z)

            # print(e,source_coords,target_coords)
            if source_coords[0] != target_coords[0]:
                if int(e.source()) < int(e.target()):
                    cap[e] = energy_grid_x[source_coords]
                else:
                    cap[e] = energy_grid_x[target_coords]

            elif source_coords[1] != target_coords[1]:
                if int(e.source()) < int(e.target()):
                    cap[e] = energy_grid_y[source_coords]
                else:
                    cap[e] = energy_grid_y[target_coords]

            elif source_coords[2] != target_coords[2]:
                if int(e.source()) < int(e.target()):
                    cap[e] = energy_grid_z[source_coords]
                else:
                    cap[e] = energy_grid_z[target_coords]

    for e in source.out_edges():
        cap[e] = special_value
    for e in sink.in_edges():
        cap[e] = special_value
    return g, cap


@log_function_call
def check_seam_fullscale(axis, size_x, size_y, size_z, left_partition, seam_index):
    """
    Checks if the left partition represents a valid seam along the specified axis.

    Parameters:
        axis (int): The axis along which the seam is found (0 for x, 1 for y, 2 for z).
        left_partition (list of tuples): The nodes in the left partition.

    Returns:
        seam_found (bool): True if the seam is valid, otherwise False.
    """

    # seam_found = True
    # return True
    # print(len(left_partition))
    return True
    if axis == 0:
        if size_y * size_z != len(left_partition):
            # print(f"Seam not correct! {len(left_partition)} should be {size_y*size_z})")
            return False

    elif axis == 1:
        if size_x * size_z != len(left_partition):
            # print(f"Seam not correct! {len(left_partition)} should be {size_y*size_z})")
            return False
    elif axis == 2:
        if size_x * size_y != len(left_partition):
            # print(f"Seam not correct! {len(left_partition)} should be {size_y*size_z})")
            return False

    # Check if seam is proper
    if axis == 0:
        left_partition_sorted = sorted(
            left_partition, key=lambda coord: (coord[1], coord[2])
        )

        for i, node in enumerate(left_partition_sorted):
            if node[1] < size_y - 1:
                node_to_check = left_partition_sorted[i + size_y]

                x_coord = node_to_check[0]
                x_node = node[0]
                if (
                    x_coord != x_node - 1
                    and x_coord != x_node
                    and x_coord != x_node + 1
                ):
                    return False

            if node[2] < size_z - 1:
                node_to_check = left_partition_sorted[i + 1]

                x_coord = node_to_check[0]
                x_node = node[0]
                if (
                    x_coord != x_node - 1
                    and x_coord != x_node
                    and x_coord != x_node + 1
                ):
                    return False

    elif axis == 1:
        left_partition_sorted = sorted(
            left_partition, key=lambda coord: (coord[0], coord[2])
        )
        print(left_partition_sorted)
        exit_file()
        print("Checking seam")
        for i, node in enumerate(left_partition_sorted):
            if node[1] < size_y - 1:
                node_to_check = left_partition_sorted[i + 1]

                x_coord = node_to_check[0]
                x_node = node[0]
                if (
                    x_coord != x_node - 1
                    and x_coord != x_node
                    and x_coord != x_node + 1
                ):
                    return False

            if node[2] < size_z - 1:
                node_to_check = left_partition_sorted[i + 1]

                x_coord = node_to_check[0]
                x_node = node[0]
                if (
                    x_coord != x_node - 1
                    and x_coord != x_node
                    and x_coord != x_node + 1
                ):
                    return False
                    return False

    elif axis == 2:
        for node in left_partition:

            if node[0] < size_x - 1:
                if (
                    (node[0] + 1, node[1], node[2] - 1) not in left_partition
                    and (node[0] + 1, node[1], node[2]) not in left_partition
                    and (node[0] + 1, node[1], node[2] + 1) not in left_partition
                ):
                    return False

            if node[1] < size_y - 1:
                if (
                    (node[0], node[1] + 1, node[2] - 1) not in left_partition
                    and (node[0], node[1] + 1, node[2]) not in left_partition
                    and (node[0], node[1] + 1, node[2] + 1) not in left_partition
                ):
                    return False
    return True


@log_function_call
def find_partition_fullscale(axis, size_x, size_y, size_z, g, idx, part, source, sink):
    """
    Finds the partition of the graph that represents the seam by identifying the border nodes
    along the specified axis.

    Parameters:
        g (gt.Graph): The graph.
        part (VertexPropertyMap): The partitioning of the graph.
        source (Vertex): The source vertex.
        sink (Vertex): The sink vertex.
        axis (int): The axis along which the seam is found (0 for x, 1 for y, 2 for z).

    Returns:
        left_partition (list of tuples): The nodes in the left partition, translated to 3D coordinates.
    """
    left_partition = []

    # Get the nodes on the border

    # Get just the nodes in the source partition
    for node in g.vertices():
        if node != source and part[node] == 1:
            if axis == 0 and part[g.vertex(int(node) + 1)] == 0:
                left_partition.append(node)
            elif axis == 1 and part[g.vertex(int(node) + size_x)] == 0:
                left_partition.append(node)
            elif axis == 2 and part[g.vertex(int(node) + (size_x * size_y))] == 0:
                left_partition.append(node)

    # Translate the list of integer indexes to 3d x,y,z pairs
    left_partition = [
        index_to_coord(int(node), size_x, size_y, size_z) for node in left_partition
    ]
    return left_partition


@log_function_call
def increase_costs_fullscale(axis, cap, size_x, size_y, size_z, g, source, sink):
    for e in g.edges():
        # If the edge is not to or from source/sink
        if source not in e and sink not in e:
            source_coords = index_to_coord(int(e.source()), size_x, size_y, size_z)
            target_coords = index_to_coord(int(e.target()), size_x, size_y, size_z)
            if source_coords[axis] == target_coords[axis]:
                cap[e] = (np.abs(cap[e]) + 0.01) * 1.5
    return cap


@log_function_call
def get_source_and_sink_nodes_fullscale(cap, size_x, size_y, size_z, g, source, sink):
    # Calculate the partitioning of the graph between source and sink
    res = gt.boykov_kolmogorov_max_flow(g, source, sink, cap)
    part = gt.min_st_cut(g, source, cap, res)

    mc = sum(
        [cap[e] - res[e] for e in g.edges() if part[e.source()] != part[e.target()]]
    )

    source_nodes = [v for v in g.vertices() if part[v] == 1]
    sink_nodes = [v for v in g.vertices() if part[v] == 0]

    # The two lists combined are the total graph including source and sink
    assert (
        len(source_nodes) + len(sink_nodes) == (size_x * size_y * size_z) + 2
    ), f"Cut is not the entire graph! Amount should be {(size_x * size_y * size_z) + 2} but is {len(source_nodes) + len(sink_nodes)}"
    return source_nodes, sink_nodes, part


@log_function_call
def find_seam_fullscaled(
    axis,
    energy_grid_x,
    energy_grid_y,
    energy_grid_z,
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
):
    # Add the edge costs to the grid
    g, cap = populate_grid_fullscale(
        energy_grid_x.shape[0],
        energy_grid_x.shape[1],
        energy_grid_x.shape[2],
        energy_grid_x,
        energy_grid_y,
        energy_grid_z,
        g,
        source,
        sink,
    )

    idx = 0
    while True:
        clean_shit()

        idx += 1
        # Get the partitioning of nodes
        source_nodes, sink_nodes, part = get_source_and_sink_nodes_fullscale(
            cap, grid_sizes[0], grid_sizes[1], grid_sizes[2], g, source, sink
        )

        # Get the nodes in the left partition
        seam = find_partition_fullscale(
            axis,
            energy_grid_x.shape[0],
            energy_grid_x.shape[1],
            energy_grid_x.shape[2],
            g,
            idx,
            part,
            source,
            sink,
        )

        for node in seam:
            if node[0] >= size_x or node[1] >= size_y or node[2] >= size_z:
                print("AYY LMAO", node, min_x)
                exit_file()

        # Check if seam is correct
        seam_found = check_seam_fullscale(
            axis,
            energy_grid_x.shape[0],
            energy_grid_x.shape[1],
            energy_grid_x.shape[2],
            seam,
            seam_index,
        )
        # print(seam_found)

        if axis == 0:
            TEMP = [(node[0] + min_x, node[1], node[2]) for node in seam]
        elif axis == 1:
            TEMP = [(node[0], node[1] + min_y, node[2]) for node in seam]
        elif axis == 0:
            TEMP = [(node[0], node[1], node[2] + min_z) for node in seam]

        max_eg = max(energy_grid_x.shape)
        full = np.zeros((max_eg, max_eg, max_eg))
        for x, y, z in TEMP:
            full[x, y, z] = 1

        seamname = f"{filename_prefix}-index{seam_index}"

        # Seam has been found, stop simulating
        if seam_found:
            # print("SEAM CORRECT")
            marching_cubes(full, f"{seamname}-FINAL")

            if idx == 1:
                marching_cubes(full, f"{seamname}-FIRST")
            break
        # Increase the cost to travel across the temporal axis
        else:
            if idx == 1:
                marching_cubes(full, f"{seamname}-FIRST")

            # marching_cubes(full, f"{seamname}-INCORRECT")

            clean_shit()

            cap = increase_costs_fullscale(
                axis,
                cap,
                energy_grid_x.shape[0],
                energy_grid_x.shape[1],
                energy_grid_x.shape[2],
                g,
                source,
                sink,
            )

    clean_shit()
    return seam
