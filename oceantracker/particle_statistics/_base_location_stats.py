import numpy as np
from oceantracker.util import basic_util, output_util
from oceantracker.util.ncdf_util import NetCDFhandler
from oceantracker.util.parameter_base_class import ParameterBaseClass
from os import  path
from oceantracker.util.parameter_checking import  ParamValueChecker as PVC, ParameterListChecker as PLC, ParameterTimeChecker as PTC
from numba.typed import List as NumbaList
from numba import  njit
from oceantracker.util.numba_util import njitOT
from oceantracker.shared_info import SharedInfo as si

from oceantracker.util import time_util
class _BaseParticleLocationStats(ParameterBaseClass):

    def __init__(self):
        # set up info/attributes
        super().__init__()
        #todo add depth range for count
        self.add_default_params(
                        update_interval =   PVC(60*60.,float,units='sec',
                                       doc_str='Time in seconds between calculating statistics, wil be rounded to be a multiple of the particle tracking time step'),
                        start =  PTC(None,doc_str= 'Start particle counting from this date-time, default is start of model run'),
                        end =  PTC(None,  doc_str='Stop particle counting from this iso date-time, default is end of model run'),
                        duration =  PVC(None, float, min=0.,units='sec',
                                doc_str='How long to do counting after start time, can be used instead of "end" parameter'),

                        role_output_file_tag =            PVC('stats_base',str,doc_str='tag on output file for this class'),
                        write =                       PVC(True,bool,doc_str='Write statistcs to disk'),
                        status_min =  PVC('stationary', str, possible_values= si.particle_status_flags.possible_values(),
                                                     doc_str=' Count only those particles with status >= to this value'),
                        status_max =  PVC('moving', str, possible_values=si.particle_status_flags.possible_values(),
                                        doc_str=' Count only those particles with status  <= to this value'),
                        z_min =  PVC(None, float, doc_str=' Count only those particles with vertical position >=  to this value', units='meters above mean water level, so is < 0 at depth'),
                        z_max =  PVC( None, float,  doc_str='Count only those particles with vertical position <= to this value', units='meters above mean water level, so is < 0 at depth'),
                        water_depth_min =  PVC(None, float, min=0.,doc_str='Count only those particles in water depths greater than this value'),
                        water_depth_max =  PVC(None, float,min=0., doc_str='Count only those particles in water depths less than this value'),
                        particle_property_list = PLC(None, str, make_list_unique=True, doc_str='Create statistics for these named particle properties, list = ["water_depth"], for average of water depth at particle locations inside the counted regions') ,
                        coords_allowed_in_lat_lon_order =  PVC(False, bool,
                            doc_str='Allows points to be given (lat,lon) and order will be swapped before use, only used if hydro-model coords are in degrees '),
                        )
        self.add_default_params(count_start_date= PTC(None,  obsolete=True,  doc_str='Use "start" parameter'),
                                count_end_date= PTC(None,   obsolete=True,  doc_str='Use "end" parameter'))

        self.sum_binned_part_prop = {}
        self.info['output_file'] = None
        self.role_doc('Particle statistics, based on spatial particle counts and particle properties in a grid or within polygons. Statistics are \n * separated by release group \n * can be a time series of statistics or put be in particle age bins.')

    def initial_setup(self):

        ml = si.msg_logger
        info = self.info
        params = self.params

        self.check_part_prop_list()

        info['status_range'] = np.asarray([si.particle_status_flags[params['status_min']], si.particle_status_flags[params['status_max']]])

        #set particle depth and water depth limits for counting particles

        f = 1.0E32
        info['z_range'] = np.asarray([-f, f])
        if params['z_min'] is not None:  info['z_range'][0] = params['z_min']
        if params['z_max'] is not None:  info['z_range'][1] = params['z_max']
        if info['z_range'][0] > info['z_range'][1]:
            ml.msg(f'Require zmin > zmax, (zmin,zmax) =({info["z_range"][0]:.3e}, {info["z_range"][1]:.3e}) ', fatal_error=True, caller=self,
                              hint ='z=0 is mean water level, so z is mostly < 0')

        info['water_depth_range'] = np.asarray([-f, f])
        if params['water_depth_min'] is not None:  info['water_depth_range'][0] = params['water_depth_min']
        if params['water_depth_max'] is not None:  info['water_depth_range'][1] = params['water_depth_max']
        if info['water_depth_range'][0]> info['water_depth_range'][1]:
            ml.msg(f'Require water_depth_min > water_depth_max, (water_depth_min,water_depth_max) =({info["water_depth_range"][0]:.3e}, {info["water_depth_range"][1]:.3e}) ',
                     caller=self,fatal_error=True)

        si.add_scheduler_to_class('count_scheduler', self, start=params['start'], end=params['end'], duration=params['duration'], interval=params['update_interval'], caller=self)
        pass

    def check_part_prop_list(self):

        part_prop = si.roles.particle_properties
        pgm = si.core_roles.particle_group_manager
        names=[]
        for name in self.params['particle_property_list']:

            if not pgm.is_particle_property(name,crumbs=f'Particle Statistic "{self.info["name"]}" >'):
                continue

            if part_prop[name].is_vector():
                si.msg_logger.msg('On the fly statistical Binning of vector particle property  "' + name + '" not yet implemented', warning=True)

            elif part_prop[name].get_dtype() != np.float64:
                si.msg_logger.msg(f'On the fly statistics can currently only track np.float64 particle properties, ignoring property  "{name}", of type "{str(part_prop[name].get_dtype())}"',
                                  warning=True)
            else:
                names.append(name)

        # set params to reduced list
        self.params['particle_property_list'] = names

    def set_up_spatial_bins(self): basic_util.nopass()

    def open_output_file(self):

        if self.params['write']:
            self.info['output_file'] = si.run_info.output_file_base + '_' + self.params['role_output_file_tag']
            self.info['output_file'] += '_' + self.info['name'] + '.nc'
            self.nc = NetCDFhandler(path.join(si.run_info.run_output_dir, self.info['output_file']), 'w')
        else:
            self.nc = None
        self.nWrites = 0

    def set_up_time_bins(self,nc):
        # stats time variables commute to all 	for progressive writing
        nc.add_dimension('time_dim', None)  # unlimited time
        nc.create_a_variable('time', ['time_dim'],  np.float64, description= 'time in seconds')

        # other output common to all types of stats
        nc.create_a_variable('num_released_total', ['time_dim'], np.int32, description='total number released')

    def set_up_part_prop_lists(self):
        # set up list of part prop and sums to enable averaging of particle properties

        part_prop = si.roles.particle_properties
        self.prop_list, self.sum_prop_list = [],[]
        # todo put this in numba uti;, for other classes to use
        names=[]
        for key, prop in self.sum_binned_part_prop.items():
            if part_prop[key].is_vector():
                si.msg_logger.msg('On the fly statistical Binning of vector particle property  "' + key + '" not yet implemented', warning=True)

            elif part_prop[key].get_dtype() != np.float64:
                si.msg_logger.msg(f'On the fly statistics  can currently only track float64 particle properties, ignoring property  "{key}", of type "{str(part_prop[key].get_dtype())}"',
                                  warning=True)

            else:
                names.append(names)
                self.prop_list.append(part_prop[key].data) # must used dataptr here
                self.sum_prop_list.append(self.sum_binned_part_prop[key][:])

        # convert to a numba list
        if len(self.prop_list) ==0:
            # must set yp typed empty lists for numba to have right signatures of numba functions
            # make list the right shape and pop to make it empty
            #todo a cleaner way to do this with NumbaList.empty??
            self.prop_list =  NumbaList([np.empty( (1,))])
            self.prop_list.pop(0)
            self.sum_prop_list = NumbaList([np.empty((1,1,1,1))])
            self.sum_prop_list.pop(0)
        else:
            # otherwise use types of arrays
            self.prop_list = NumbaList(self.prop_list)
            self.sum_prop_list = NumbaList(self.sum_prop_list)
        pass




    # overload this method to subset indicies in out of particles to count
    def select_particles_to_count(self, out):
        return out


    def update(self,n_time_step, time_sec):
        '''do particle counts'''
        part_prop = si.roles.particle_properties
        info = self.info
        self.start_update_timer()


        num_in_buffer = si.core_roles.particle_group_manager.info['particles_in_buffer']

        # first select those to count based on status and z location
        sel = self.sel_status_waterdepth_and_z(part_prop['status'].data, part_prop['x'].data, part_prop['water_depth'].data.ravel(),
                                               info['status_range'], info['z_range'], info['water_depth_range'],
                                               num_in_buffer, self.get_partID_buffer('B1'))

        # any overloaded sub-selection of particles given in child classes
        sel = self.select_particles_to_count(sel)

        #update prop list data, as buffer may have expnaded
        #todo do this only when expansion occurs??
        part_prop = si.roles.particle_properties
        for n, name in enumerate(self.sum_binned_part_prop.keys()):
            self.prop_list[n]= part_prop[name].data

        self.do_counts(n_time_step, time_sec,sel)

        self.write_time_varying_stats(self.nWrites, time_sec)
        self.nWrites += 1

        self.stop_update_timer()

    @staticmethod
    @njitOT
    def sel_status_waterdepth_and_z(status, x, water_depth, status_range, z_range, water_depth_range, num_in_buffer, out):
        n_found = 0
        if x.shape[1] == 3:
            # 3D selection
            for n in range(num_in_buffer):
                if status_range[0] <= status[n] <= status_range[1] and z_range[0] <= x[n, 2] <= z_range[1] and water_depth_range[0] <= water_depth[n] <= water_depth_range[1]:
                    out[n_found] = n
                    n_found += 1
        else:
            # 2D selection
            for n in range(num_in_buffer):
                if status_range[0] <= status[n] <= status_range[1] and water_depth_range[0] <= water_depth[n] <= water_depth_range[1]:
                    out[n_found] = n
                    n_found += 1

        return out[:n_found]

    def write_time_varying_stats(self, n, time):
        # write nth step in file
        fh = self.nc.file_handle
        fh['time'][n] = time
        release_groups = si.roles.release_groups

        # add up number released
        num_released = 0
        for rg in release_groups.values():
            num_released += rg.info['number_released']
        fh['num_released_total'][n] = num_released

        fh['count'][n, ...] = self.count_time_slice[:, ...]
        fh['count_all_particles'][n, ...] = self.count_all_particles_time_slice[:, ...]

        for key, item in self.sum_binned_part_prop.items():
            self.nc.file_handle['sum_' + key][n, ...] = item[:]  # write sums  working in original view

    def info_to_write_at_end(self) : pass

    def close(self):

        nc = self.nc
        # write total released in each release group
        num_released=[]
        for name, i in si.roles.release_groups.items():
            num_released.append(i.info['number_released'])

        if self.params['write']:
            self.info_to_write_at_end()
            nc.write_a_new_variable('number_released_each_release_group', np.asarray(num_released,dtype=np.int64), ['release_group_dim'], description='Total number released in each release group')
            nc.write_global_attribute('total_num_particles_released', si.core_roles.particle_group_manager.info['particles_released'])

            nc.close()
        self.nc = None  # parallel pool cant pickle nc