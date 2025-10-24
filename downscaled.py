from utility_functions import *
import numpy as np
import graph_tool.all as gt
import time


@log_function_call
def find_grid_downscaled(axis, size_x, size_y, size_z):
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

    g = gt.Graph(directed=True)

    index = 0
    edge_list = []
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                if x < size_x - 1:  # Add edge to the vertex in the next row
                    edge_list.append((index, index + 1))
                    edge_list.append((index + 1, index))
                if y < size_y - 1:  # Add edge to the vertex in the next column
                    edge_list.append((index, index + size_x))
                    edge_list.append((index + size_x, index))
                if z < size_z - 1:  # Add edge to the vertex in the next depth
                    edge_list.append((index, index + size_x * size_y))
                    edge_list.append((index + size_x * size_y, index))

                if axis == 0:
                    # Keep nodes for source and sink
                    if x == 0:
                        sourcelist.append(index)
                    if x == size_x - 1:
                        sinklist.append(index)
                elif axis == 1:
                    # Keep nodes for source and sink
                    if y == 0:
                        sourcelist.append(index)
                    if y == size_y - 1:
                        sinklist.append(index)
                elif axis == 2:
                    # Keep nodes for source and sink
                    if z == 0:
                        sourcelist.append(index)
                    if z == size_z - 1:
                        sinklist.append(index)

                index += 1
    g.add_edge_list(edge_list)

    assert len(sourcelist) == len(sinklist) != 0

    # Add source and sink nodes

    source = g.add_vertex()
    sourceid = size_x * size_y * size_z
    sink = g.add_vertex()
    sinkid = sourceid + 1

    for node in sourcelist:
        g.add_edge(source, node)
    for node in sinklist:
        g.add_edge(node, sink)

    return g, source, sink, sourceid, sinkid


@log_function_call
def populate_grid(
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
def check_seam(axis, size_x, size_y, size_z, left_partition):
    """
    Checks if the left partition represents a valid seam along the specified axis.

    Parameters:
        axis (int): The axis along which the seam is found (0 for x, 1 for y, 2 for z).
        left_partition (list of tuples): The nodes in the left partition.

    Returns:
        seam_found (bool): True if the seam is valid, otherwise False.
    """
    seam_found = True

    # Check if seam is proper
    if axis == 0:
        for node in left_partition:
            if node[1] < size_y - 1 and seam_found:
                if (
                    (node[0] - 1, node[1] + 1, node[2]) not in left_partition
                    and (node[0], node[1] + 1, node[2]) not in left_partition
                    and (node[0] + 1, node[1] + 1, node[2]) not in left_partition
                ):
                    seam_found = False
            if node[2] < size_z - 1 and seam_found:
                if (
                    (node[0] - 1, node[1], node[2] + 1) not in left_partition
                    and (node[0], node[1], node[2] + 1) not in left_partition
                    and (node[0] + 1, node[1], node[2] + 1) not in left_partition
                ):
                    seam_found = False

    elif axis == 1:
        for node in left_partition:
            if node[0] < size_x - 1 and seam_found:
                if (
                    (node[0] + 1, node[1] - 1, node[2]) not in left_partition
                    and (node[0] + 1, node[1], node[2]) not in left_partition
                    and (node[0] + 1, node[1] + 1, node[2]) not in left_partition
                ):
                    seam_found = False
                if node[2] < size_z - 1 and seam_found:
                    if (
                        (node[0], node[1] - 1, node[2] + 1) not in left_partition
                        and (node[0], node[1], node[2] + 1) not in left_partition
                        and (node[0], node[1] + 1, node[2] + 1) not in left_partition
                    ):
                        seam_found = False

    elif axis == 2:
        for node in left_partition:
            if node[0] < size_x - 1 and seam_found:
                if (
                    (node[0] + 1, node[1], node[2] - 1) not in left_partition
                    and (node[0] + 1, node[1], node[2]) not in left_partition
                    and (node[0] + 1, node[1], node[2] + 1) not in left_partition
                ):
                    seam_found = False
            if node[1] < size_y - 1 and seam_found:
                if (
                    (node[0], node[1] + 1, node[2] - 1) not in left_partition
                    and (node[0], node[1] + 1, node[2]) not in left_partition
                    and (node[0], node[1] + 1, node[2] + 1) not in left_partition
                ):
                    seam_found = False
    return seam_found


@log_function_call
def find_partition(axis, size_x, size_y, size_z, g, idx, part, source, sink):
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
    # Get just the nodes in the source partition
    for node in g.vertices():
        if node != source and part[node] == 1 and node != sink:
            if axis == 0 and part[g.vertex(int(node) + 1)] == 0:
                left_partition.append(node)
            elif axis == 1 and part[g.vertex(int(node) + size_y)] == 0:
                left_partition.append(node)
            elif axis == 2 and part[g.vertex(int(node) + (size_y * size_z))] == 0:
                left_partition.append(node)

    left_partition = [
        index_to_coord(int(node), size_x, size_y, size_z) for node in left_partition
    ]

    return left_partition


@log_function_call
def increase_costs(axis, cap, size_x, size_y, size_z, g, source, sink):
    for e in g.edges():
        # If the edge is not to or from source/sink
        if source not in e and sink not in e:
            source_coords = index_to_coord(int(e.source()), size_x, size_y, size_z)
            target_coords = index_to_coord(int(e.target()), size_x, size_y, size_z)
            if source_coords[axis] == target_coords[axis]:
                cap[e] = (np.abs(cap[e]) + 0.01) * 2
    return cap


@log_function_call
def get_source_and_sink_nodes(cap, size_x, size_y, size_z, g, source, sink):
    # Calculate the partitioning of the graph between source and sink
    part = gt.min_st_cut(
        g, source, cap, gt.boykov_kolmogorov_max_flow(g, source, sink, cap)
    )

    source_nodes = [v for v in g.vertices() if part[v] == 1]
    sink_nodes = [v for v in g.vertices() if part[v] == 0]

    # source_coords = [index_to_coord(int(node),size_x,size_y,size_z) for node in source_nodes if node != source]
    # plot_interactive_plot(source_coords)
    # exit_file()

    # The two lists combined are the total graph including source and sink
    assert (
        len(source_nodes) + len(sink_nodes) == (size_x * size_y * size_z) + 2
    ), f"Cut is not the entire graph! Amount should be {(size_x * size_y * size_z) + 2} but is {len(source_nodes) + len(sink_nodes)}"
    return source_nodes, sink_nodes, part


@log_function_call
def find_seam_downscaled(
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
):

    # Add the edge costs to the grid
    g, cap = populate_grid(
        downsized_size_x,
        downsized_size_y,
        downsized_size_z,
        downscaled_energy_grid_x,
        downscaled_energy_grid_y,
        downscaled_energy_grid_z,
        g,
        source,
        sink,
    )

    idx = 0
    while True:
        idx += 1
        # Get the partitioning of nodes
        source_nodes, sink_nodes, part = get_source_and_sink_nodes(
            cap, downsized_size_x, downsized_size_y, downsized_size_z, g, source, sink
        )

        # Get the nodes in the left partition
        left_partition = find_partition(
            axis,
            downsized_size_x,
            downsized_size_y,
            downsized_size_z,
            g,
            idx,
            part,
            source,
            sink,
        )

        # print('Creating plot!')
        # plot_interactive_plot(left_partition)

        # Check if seam is correct
        seam_found = check_seam(
            axis, downsized_size_x, downsized_size_y, downsized_size_z, left_partition
        )

        # Seam has been found, stop simulating
        if seam_found:
            break
        # Increase the cost to travel across the temporal axis
        else:
            cap = increase_costs(
                axis,
                cap,
                downsized_size_x,
                downsized_size_y,
                downsized_size_z,
                g,
                source,
                sink,
            )

    return left_partition, cap


if __name__ == "__main__":
    print(
        "\n\n\nThis is a helper file, call 'python pipeline.py' to run the code!\n\n\n"
    )
    exit_file()
