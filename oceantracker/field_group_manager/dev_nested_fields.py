from oceantracker.field_group_manager.field_group_manager import FieldGroupManager
from oceantracker.util.parameter_base_class import ParameterBaseClass
from oceantracker.util.parameter_checking import ParamValueChecker as PVC
from oceantracker.field_group_manager.util import  field_group_manager_util
import numpy as np
from oceantracker.util.parameter_util import make_class_instance_from_params
from oceantracker.util.time_util import seconds_to_isostr
from time import  perf_counter
from copy import copy
# run fields nested with outer main readers grid

class DevNestedFields(ParameterBaseClass):
    # build a list of field group managers for outer and nest grids
    # first in list grid is the outer grid

    readers=[] # first is outer grid, nesting the others

    def initial_setup(self):
        si = self.shared_info
        ml = si.msg_logger
        # setup outer grid first


        fgm_outer_grid = make_class_instance_from_params('field_group_manager_outer_grid',
                                dict(class_name='oceantracker.field_group_manager.field_group_manager.FieldGroupManager'),
                                ml,   crumbs='adding outer hydro-grid field manager for nested grid run' )
        fgm_outer_grid.initial_setup()

        # note es to check if all hidcasts have same required info

        has_A_Z_profile = fgm_outer_grid.info['has_A_Z_profile']
        self.hydro_time_step = fgm_outer_grid.get_hydo_model_time_step()
        self.start_time, self.end_time  = fgm_outer_grid.get_hindcast_start_end_times()

        # first grid is outer grid
        self.fgm_hydro_grids = [fgm_outer_grid]

        # add nested grids
        for name, params in si.working_params['nested_reader_builders'].items():
            ml.progress_marker(f'Starting nested grid setup #{len(self.fgm_hydro_grids)}, name= "{name}"')

            t0= perf_counter()
            i =  make_class_instance_from_params(name,
                                                 dict(class_name='oceantracker.field_group_manager.field_group_manager.FieldGroupManager'),
                                                 ml, crumbs=f'adding nested hydro-model field manager #{len(self.fgm_hydro_grids)}')


            i._setup_hydro_reader(params)
            i.set_up_interpolator()


            self.fgm_hydro_grids.append(i)

            # record consitency info
            has_A_Z_profile = has_A_Z_profile and i.info['has_A_Z_profile']
            self.hydro_time_step = min(self.hydro_time_step,i.get_hydo_model_time_step())

            # start and end times
            times= i.get_hindcast_start_end_times()
            self.start_time, self.end_time=  max(self.start_time,times[0]), min(self.end_time,times[1])

            ml.progress_marker(f'Finished nested hydro-model grid setup #{len(self.fgm_hydro_grids)}, name= "{name}", from {seconds_to_isostr(times[0])} to  {seconds_to_isostr(times[1])}', start_time=t0)

        #todo add check on overlaping

        # consistency checks
        # A_Z profile data
        if si.settings['use_A_Z_profile'] and not has_A_Z_profile:
            ml.msg(f'Not all hindcasts have A_Z profile variable, set "use_A_Z_profile" to False, to run with use constant dipersion A_Z',
                   crumbs='Nested reader set up ',fatal_error=True, exit_now=True)

        self.info['has_A_Z_profile'] = has_A_Z_profile # set

        # dry cell flag
        if si.settings['write_dry_cell_flag']:
            ml.msg(f'Cannot write dry cell flag to tracks files for nested grids, disabling dry cell writes',
                   crumbs='Nested reader set up ',  note=True)
            si.settings['write_dry_cell_flag'] = False

        #todo check hindcasts over lap

        pass

    def final_setup(self):
        # do final setup for each grid
        for fgm in self.fgm_hydro_grids:
            fgm.final_setup()




    def update_reader(self, time_sec):

        for fgm in self.fgm_hydro_grids:
            fgm.update_reader(time_sec)


    def get_hydo_model_time_step(self): return self.hydro_time_step # return the smallest time step

    def get_hindcast_start_end_times(self):
        return self.start_time, self.end_time

    def add_part_prop_from_fields_plus_book_keeping(self):
        # only use outer grid to add properties for all readers
        self.fgm_hydro_grids[0].add_part_prop_from_fields_plus_book_keeping()

    def are_points_inside_domain(self,x,include_dry_cells):
        si = self.shared_info
        part_prop = si.classes['particle_properties']
        # set up space
        N= x.shape[0]
        is_inside= np.full((N,),False)
        n_cell    = np.full((N,), si.particle_status_flags['unknown'],dtype=np.int32)
        bc  = np.full((N,3), -1, dtype=np.float64)
        hydro_model_gridID = np.full((N,), -1, dtype=np.int8)

        # look find grid containing points, starting with last nested grid
        # do outer domain last, so do in reverse order
        for n in reversed(range(len(self.fgm_hydro_grids))):
            i = self.fgm_hydro_grids[n]
            index = np.flatnonzero(~is_inside) # those not yet found inside inner grid

            sel, n_cell_n, bc_n, ingore_gridID = i.are_points_inside_domain(x[index,:],include_dry_cells)

            #record those inside this grid
            index = index[sel]
            is_inside[index] = True
            n_cell[index] = n_cell_n[sel]
            bc[index] = bc_n[sel,:]
            hydro_model_gridID[index] = n
            pass

        return is_inside, n_cell, bc, hydro_model_gridID


    def interp_named_field_at_given_locations_and_time(self, field_name, x, time_sec= None, n_cell=None,bc_cords=None, output=None,hydro_model_gridID=None):

        vals= np.full((x.shape[0],), 0., dtype=np.float32)

        # look through grids in reverse to find interpolated values, so use outer grid last
        for n in reversed(range(len(self.fgm_hydro_grids))):
            i = self.fgm_hydro_grids[n]
            sel = hydro_model_gridID ==  n
            vals[sel, ...] = i.interp_named_field_at_given_locations_and_time(field_name, x[sel, :],
                                          time_sec=time_sec, n_cell=n_cell[sel], bc_cords=bc_cords[sel, :], output=output, hydro_model_gridID=hydro_model_gridID[sel])

        return vals

    def setup_time_step(self, time_sec, xq, active, fix_bad=True):
        # loop over hydro grids
        for fgm in self.fgm_hydro_grids:
            fgm.setup_time_step(time_sec, xq, active, fix_bad=fix_bad)




