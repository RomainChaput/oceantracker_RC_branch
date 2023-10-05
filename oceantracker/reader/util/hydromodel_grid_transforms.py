import numpy as np
from numba import njit
from oceantracker.interpolator.util.interp_kernals import kernal_linear_interp1D
from copy import copy

@njit
def find_node_with_smallest_top_bot_layer(zlevels,bottom_cell_index,z0):
    # find the  profile with thinest top/bottom layer as fraction of water depth
    # from single time step of zlevels
    z_fractions= np.full_like(zlevels,np.nan,dtype=np.float32)
    node_min= -1
    min_dz= np.inf
    for n in range(zlevels.shape[0]): # loop over nodes
        z_surface = float(zlevels[n, -1])
        z_bottom= float(zlevels[n,bottom_cell_index[n]])
        total_water_depth = abs(z_surface-z_bottom)

        if total_water_depth > .2:
            # Get bottom layer thickness as fraction of water depth
            dz_bot = (zlevels[n, bottom_cell_index[n] + 1] - z_bottom) / total_water_depth

            if  dz_bot < min_dz:
                min_dz = dz_bot
                node_min = n
        for nz in range(bottom_cell_index[n], zlevels.shape[1]):
            if total_water_depth > z0:
                z_fractions[n,nz] = (zlevels[n,nz]-z_bottom)/total_water_depth
            else:
                z_fractions[n, nz] = 0.
    return node_min, z_fractions

@njit()
def  interp_4D_field_to_fixed_sigma_values(zlevel_fractions,bottom_cell_index,sigma,
                                           water_depth,tide,z0,minimum_total_water_depth,
                                           data,out, is_water_velocity):
    # assumes time invariant zlevel_fractions, linear interp
    # set up space
    for nt in range(out.shape[0]):
        for node in range(out.shape[1]):
            nz_bottom =  int(bottom_cell_index[node])
            nz_data = nz_bottom
            # loop over new levels
            for nz in range(sigma.size-1):
                # if the sigma is above next zlevel, move data zlevel index up one
                if sigma[nz] > zlevel_fractions[node, nz_data+1]:
                    nz_data += 1
                    nz_data = min(nz_data, zlevel_fractions.shape[1] - 2)

                # do vertical linear interp
                # get fraction within zlevel layer to use in interp
                dzf= zlevel_fractions[node, nz_data + 1] - zlevel_fractions[node, nz_data]
                if dzf < .001:
                    f = 0.
                    dzf = 0.0
                else:
                    f = (sigma[nz] - zlevel_fractions[node, nz_data])/dzf

                # if in bottom data cell use log interp if water velocity
                # by adjusting f
                if nz_data == nz_bottom and is_water_velocity:
                    # get total water depth
                    twd = abs(tide[nt, node, 0, 0] + water_depth[0, node, 0, 0])
                    if twd < minimum_total_water_depth: twd = minimum_total_water_depth

                    # dz is  bottom layer thickness in metres, in original hydro model data
                    dz =  twd*dzf
                    if dz < z0:
                        f = 0.0
                    else:
                        z0p = z0 / dz
                        f = (np.log(f + z0p) - np.log(z0p)) / (np.log(1. + z0p) - np.log(z0p))

                for m in range(out.shape[3]):
                    out[nt,node,nz,m] = (1-f) *data[nt,node,nz_data,m] + f *data[nt,node,nz_data+1,m]
        # do top value at sigma= 1
            for m in range(out.shape[3]):
                out[nt, node, -1, m] = data[nt, node, -1, m]

            pass

    return out

@njit
def convert_layer_field_to_levels_from_fixed_depth_fractions(data, zfraction_center, zfraction_boundaries):
    # convert values at depth at center of the cell to values on the boundaries between cells baed on fractional layer/boundary depthsz
    # used in FVCOM reader
    data_levels = np.full((data.shape[0],) + (data.shape[1],) + (zfraction_boundaries.shape[0],), 0., dtype=np.float32)

    for nt in range(data.shape[0]):
        for n in range(data.shape[1]):
            for nz in range(1, data.shape[2]):
                # linear interp levels not, first or last boundary
                data_levels[nt, n, nz] = kernal_linear_interp1D(zfraction_center[nz - 1], data[nt, n, nz - 1], zfraction_center[nz], data[nt, n, nz], zfraction_boundaries[nz])

            # extrapolate to top zlevel
            data_levels[nt, n, -1] = kernal_linear_interp1D(zfraction_center[-2], data[nt, n, -2], zfraction_center[-1], data[nt, n, -1], zfraction_boundaries[-1])

            # extrapolate to bottom zlevel
            data_levels[nt, n, 0] = kernal_linear_interp1D(zfraction_center[0], data[nt, n, 0], zfraction_center[1], data[nt, n, 1], zfraction_boundaries[0])

    return data_levels

@njit
def get_node_layer_field_values(data, node_to_tri_map, tri_per_node,cell_center_weights):
    # get nodal values from data in surrounding cells based in distance weighting
    # used in FVCOM reader

    data_nodes = np.full((data.shape[0],) + (len(node_to_tri_map),) +(data.shape[2],) , 0., dtype=np.float32)

    for nt in range(data.shape[0]): # loop over time steps
        # loop over triangles
        for node in range(node_to_tri_map.shape[0]):
            for nz in range(data.shape[2]):
                # loop over cells containing this node
                for m in range(tri_per_node[node]):
                    cell = node_to_tri_map[node, m]
                    data_nodes[nt, node, nz] += data[nt, cell, nz]*cell_center_weights[node, m] # weight this cell value

    return data_nodes



@njit
def convert_layer_field_to_levels_from_depth_fractions_at_each_node(data, zfraction_center, zfraction_boundaries):
    # convert values at depth at center of the cell to values on the boundaries between cells baed on fractional layer/boundary depths
    # used in FVCOM reader
    data_levels = np.full((data.shape[0],) + (data.shape[1],) + (zfraction_boundaries.shape[1],), 0., dtype=np.float32)

    for nt in range(data.shape[0]):
        for n in range(data.shape[1]):
            for nz in range(1,data.shape[2]):
                # linear interp levels not, first or last boundary
                data_levels[nt, n, nz] = kernal_linear_interp1D(zfraction_center[n, nz - 1], data[nt, n, nz - 1], zfraction_center[n, nz], data[nt, n, nz], zfraction_boundaries[n, nz])

            # extrapolate to top zlevel
            data_levels[nt, n, -1] = kernal_linear_interp1D(zfraction_center[n, - 2], data[nt, n, -2], zfraction_center[n, -1], data[nt, n, -1], zfraction_boundaries[n, -1])

            # extrapolate to bottom zlevel
            data_levels[nt, n, 0] = kernal_linear_interp1D(zfraction_center[n, 0], data[nt, n, 0], zfraction_center[n, 1], data[nt, n, 1], zfraction_boundaries[n, 0])

    return data_levels

@njit()
def calculate_cell_center_weights_at_node_locations(x_node, x_cell, node_to_tri_map, tri_per_node):
    # calculate distance weights for values at cell centers, to be used in interploting cell center values to nodal values
    weights= np.full_like(node_to_tri_map, 0.,dtype=np.float32)
    dxy= np.full((2,), 0.,dtype=np.float32)

    for n in range(x_node.shape[0]):
        s= 0.
        n_cells=tri_per_node[n]
        for m in range(n_cells):
            dxy[:] = x_cell[node_to_tri_map[n,m],:2] - x_node[n, :2]
            dist = np.sqrt(dxy[0]**2 + dxy[1]**2)
            weights[n,m] = dist
            s += dist

        # normalize weights
        for m in range(n_cells): weights[n,m]=weights[n,m] /s

    return  weights