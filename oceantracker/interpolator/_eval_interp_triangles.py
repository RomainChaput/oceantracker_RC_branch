from scipy.spatial import cKDTree
import numpy as np

from oceantracker.shared_info import shared_info as si
from oceantracker.util import basic_util
from oceantracker.util.profiling_util import function_profiler
from time import perf_counter
from oceantracker.util import numpy_util
from oceantracker.interpolator.util import  triangle_eval_interp
from oceantracker.particle_properties.util import  particle_operations_util

class EvalInterpTriangles(object):
    def __init__(self, grid, params):
        self.grid, self.params = grid, params
        self.info = {}

    def _time_independent_2D_scalar_field(self, field_instance, current_buffer_steps,
                                          fractional_time_steps, output, active, n_cell=None, bc=None):
        # todo find a better way to get water depth for release points with optional args?
        part_prop = si.class_roles.particle_properties
        nc_cell = part_prop['n_cell'].data if n_cell is None else n_cell
        bc = part_prop['bc_cords'].data if bc is None else bc
        triangle_eval_interp.time_independent_2D_scalar_field(output, field_instance.data,
                                                              self.grid['triangles'], nc_cell, bc, active)

    def _time_independent_2D_vector_field(self, field_instance, current_buffer_steps,
                                          fractional_time_steps, output, active):
        part_prop = si.class_roles.particle_properties
        triangle_eval_interp.time_independent_2D_vector_field(output, field_instance.data,
                                                              self.grid['triangles'],
                                                              part_prop['n_cell'].data, part_prop['bc_cords'].data, active)
    def _time_dependent_2D_scalar_field(self,field_instance,current_buffer_steps,
                                       fractional_time_steps,output, active, n_cell=None, bc=None):
        part_prop= si.class_roles.particle_properties
        #scalar, eg water depth
        # todo find a better way to get tde and water depth for release points
        nc_cell = part_prop['n_cell'].data if n_cell is None else n_cell
        bc = part_prop['bc_cords'].data if bc is None else bc

        triangle_eval_interp.time_dependent_2D_scalar_field(
                            current_buffer_steps, fractional_time_steps,output,
                            field_instance.data,self.grid['triangles'],
                            part_prop['n_cell'].data,  part_prop['bc_cords'].data, active)

    def _time_dependent_2D_vector_field(self, field_instance, current_buffer_steps,
                                        fractional_time_steps, output, active):
        part_prop = si.class_roles.particle_properties
        # vector, eg 2D water velocity
        triangle_eval_interp.time_dependent_2D_vector_field(current_buffer_steps, fractional_time_steps, output,
                                                            field_instance.data, self.grid['triangles'],
                                                            part_prop['n_cell'].data, part_prop['bc_cords'].data, active)

    def _time_dependent_3D_scalar_field(self, field_instance, current_buffer_steps,
                                        fractional_time_steps, output, active):
        grid = self.grid
        part_prop = si.class_roles.particle_properties
        vgt = si.vertical_grid_types
        F_data = field_instance.data

        if si.hindcast_info['vert_grid_type'] in [vgt.Sigma, vgt.Slayer]:
            triangle_eval_interp.time_dependent_3D_scalar_field_data_in_all_layers(
                                        current_buffer_steps, fractional_time_steps,
                                                     F_data ,                                                                                 grid['triangles'],
                                                     part_prop['n_cell'].data, part_prop['bc_cords'].data, part_prop['nz_cell'].data,
                                                     part_prop['z_fraction'].data,
                                                     output, active)
        elif si.hindcast_info['vert_grid_type'] in [vgt.LSC, vgt.Zfixed]:
            triangle_eval_interp.time_dependent_3D_scalar_field_ragged_bottom(current_buffer_steps, fractional_time_steps, F_data,
                                                                         grid['triangles'], grid['bottom_cell_index'],
                                                                         part_prop['n_cell'].data, part_prop['bc_cords'].data, part_prop['nz_cell'].data, part_prop['z_fraction'].data,
                                                                         output, active)
        else:
            raise Exception(f'Unknown vertical  grid type {si.hindcast_info["vert_grid_type"]}')

    def _time_dependent_3D_vector_field(self, field_instance, current_buffer_steps,
                                        fractional_time_steps, output, active):
        ## water velocity is main one
        grid= self.grid
        F_data = field_instance.data
        part_prop = si.class_roles.particle_properties
        vgt = si.vertical_grid_types

        # use z_fractions with log layer near bottom for water velocity
        z_fraction = part_prop['z_fraction_water_velocity'] if field_instance.params['name'] == 'water_velocity' else part_prop['z_fraction']

        if si.hindcast_info['vert_grid_type'] in [vgt.Sigma,vgt.Slayer]:
            # these have spatially uniform and static map of z levels
            triangle_eval_interp.time_dependent_3D_vector_field_data_in_all_layers(current_buffer_steps, fractional_time_steps, F_data, grid['triangles'], part_prop['n_cell'].data, part_prop['bc_cords'].data, part_prop['nz_cell'].data, z_fraction.data,
                                                                                   output, active)

        elif si.hindcast_info['vert_grid_type'] in  [vgt.LSC, vgt.Zfixed]:
            triangle_eval_interp.time_dependent_3D_vector_field_ragged_bottom(current_buffer_steps, fractional_time_steps, F_data,
                                                                              grid['triangles'], grid['bottom_cell_index'],
                                                                              part_prop['n_cell'].data, part_prop['bc_cords'].data, part_prop['nz_cell'].data, part_prop['z_fraction'].data,
                                                                              output, active)
            pass
        else:
            raise Exception(f'Unknown vertical  grid type {si.hindcast_info["vert_grid_type"]}')

# special case give bc and cell, used to evaluate water depth and tide  for checking release points
# interp_named_field_at_given_locations_and_time(self, field_name, x, n_cell=None,bc_cords=None, time_sec= None ):
# todo add??
# pass