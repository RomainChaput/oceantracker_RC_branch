import numpy as np
from oceantracker.util.parameter_base_class import ParameterBaseClass
from oceantracker.util import time_util
from oceantracker.util.parameter_checking import ParamValueChecker as PVC, ParameterListChecker as PLC, ParameterCoordsChecker as PCC
from numba import njit
from oceantracker.util.numba_util import njitOT
from oceantracker.common_info_default_param_dict_templates import large_float

class PointRelease(ParameterBaseClass):
    # releases particles at fixed points, inside optional radius
    # add checks to see if points inside domain and dry if released in a radius
    #todo make a parent release base class

    def __init__(self):
        # set up info/attributes
        super().__init__()
        self.add_default_params({
                 'points':          PCC(None,is_required=True, doc_str='A N by 2 or 3 list or numpy array of locations where particles are released. eg for 2D [[25,10],[23,2],....] '),
                 'release_radius':  PVC(0., float, min= 0., doc_str= 'Particles are released from random locations in circle of given radius around each point.'),
                 'pulse_size' :     PVC(1, int, min=1, doc_str= 'Number of particles released in a single pulse, this number is released every release_interval.'),
                 'release_interval':PVC(0., float, min =0.,units='sec', doc_str= 'Time interval between released pulses. To release at only one time use release_interval=0.'),
                 'release_start_date': PVC(None, 'iso8601date', doc_str='Must be an ISO date as string eg. "2017-01-01T00:30:00" '),
                   # to do add ability to release on set dates/times 'release_dates': PLC([], 'iso8601date'),
                 'release_duration': PVC(None, float,min=0.,
                                    doc_str='Time in seconds particles are released for after they start being released, ie releases stop this time after first release.,an alternative to using "release_end_date"' ),
                    'release_end_date': PVC(None, 'iso8601date', doc_str='Date to stop releasing particles, ignored if release_duration give, must be an ISO date as string eg. "2017-01-01T00:30:00" '),
                    'max_age': PVC(None,float,min=1.,
                                        doc_str='Particles older than this age in seconds are culled,ie. status=dead, and removed from computation, very useful in reducing run time'),
                     'user_release_groupID' : PVC(0,int, doc_str= 'User given ID number for this group, held by each particle. This may differ from internally uses release_group_ID.'),
                 'user_release_group_name' : PVC(None,str,doc_str= 'User given name/label to attached to this release groups to make it easier to distinguish.'),
                 'allow_release_in_dry_cells': PVC(False, bool,
                              doc_str='Allow releases in cells which are currently dry, ie. either permanently dry or temporarily dry due to the tide.'),

                 'z_range': PLC([],[float, int], min_length=2, obsolete='use z_min and/or z_max'),
                'z_min': PVC(None, float,doc_str='min/ deepest z value to release for to randomly release in 3D, overrides any given release z value'),
                'z_max': PVC( None,float, doc_str='max/ highest z vale release for to randomly release in 3D, overrides any given release z value'),
                'release_offset_from_surface_or_bottom': PVC(0., [float, int], min=0.,
                                                             doc_str=' 3D release particles at offset from free surface or bottom, if release_at_surface or  release_at_bottom = True', units='m'),
                'release_at_surface': PVC(False, bool, doc_str=' 3D release particles at free surface, ie tide height, with  offset given by release_offset_from_surface_or_bottom param, overrides any given release z value'),
                'release_at_bottom': PVC(False, bool, doc_str=' 3D release particles at bottom, with  offset given by release_offset_from_surface_or_bottom param, overrides any given release z value'),
                #'water_depth_min': PVC(None, float,doc_str='min water depth to release in, useful for releases with a depth rage, eg larvae from inter-tidal shellfish', units='m'),
                #'water_depth_max': PVC(None, float, doc_str='max water depth to release in', units='m'),

#Todo implement release group particle with different parameters, eg { 'oxygen' : {'decay_rate: 0.01, 'initial_value': 5.}
                'max_cycles_to_find_release_points': PVC(200, int, min=100, doc_str='Maximum number of cycles to search for acceptable release points, ie. inside domain, polygon etc '),
                                 })
        self.class_doc(description= 'Release particles at 1 or more given locations. Pulse_size particles are released every release_interval. All these particles are tagged as a single release_group.')


    def initial_setup(self):
        # must be called after unpack_x0
        # tidy up parameters to make them numpy arrays with first dimension equal to number of locations

        info=self.info
        info['points'] =   np.array(self.params['points']).astype(np.float64)

        info['number_released'] = 0 # count of particles released in this group
        info['pulse_count'] = 0
        info['release_type'] = 'point'

    def set_up_release_times(self):
        # get release times based on release_start_date, duration
        params = self.params
        info = self.info
        si = self.shared_info
        ml = self.msg_logger

        hindcast_start, hindcast_end  =  si.classes['field_group_manager'].get_hindcast_start_end_times()

        model_time_step = si.settings['time_step']

        self.info['release_info'] ={'first_release_date': None, 'last_release_date':None,
                                    'last_time_alive':None}
        # short cut
        release_info =self.info['release_info']

        if params['release_start_date'] is None:
            # no user start date so use  model runs' start date
            time_start= hindcast_start if not si.backtracking else hindcast_end
        else:
            # user given start date
            time_start = time_util.isostr_to_seconds(params['release_start_date'])

        # now check if start in range
        n_groups_so_far =len(si.classes['release_groups'])
        if not hindcast_start <= time_start <= hindcast_end:
            si.msg_logger.msg('Release group= ' + str(n_groups_so_far + 1) + ', name= ' + self.info['name'] + ',  parameter release_start_time is ' +
                                    time_util.seconds_to_isostr(time_start)
                              + '  is outside hindcast range ' + time_util.seconds_to_isostr(hindcast_start)
                                    + ' to ' + time_util.seconds_to_isostr(hindcast_end), warning=True)

        # set max age of particles
        release_interval = model_time_step if params['release_interval'] is None else params['release_interval']

        # world out release times
        if release_interval == 0.:
            time_end = time_start
        elif self.params['release_duration'] is not None:
            time_end = time_start + si.model_direction*self.params['release_duration']

        elif self.params['release_end_date'] is not None:
            time_end = time_util.isostr_to_seconds(self.params['release_end_date'])
        else:
            # default is limit of hindcast
            time_end = hindcast_start if si.backtracking else hindcast_end

        # get time steps for release in a dow safe way
        model_time_step = si.settings['time_step']


        # get release times within the hindcast
        if abs(time_end-time_start) < model_time_step:
            # have only one release
            release_info['release_times'] = np.asarray(time_start)
        else:
            release_info['release_times'] = time_start + np.arange(0., abs(time_end-time_start),release_interval )*si.model_direction

        # trim releases to be within hindcast
        sel = np.logical_and( release_info['release_times'] >= hindcast_start,  release_info['release_times']  <= hindcast_end)
        release_info['release_times'] = release_info['release_times'][sel]

        if release_info['release_times'].size ==0:
            ml.msg(f'No release times in range of hydro-model for release_group {info["instanceID"]:2d}, ',
                   fatal_error=True,
                   hint=' Check hydro-model date range and release dates  ')

        # get time steps when released, used to determine when to release
        release_info['release_time_steps'] =  np.round(( release_info['release_times']- hindcast_start)/model_time_step).astype(np.int32)

        # find last time partiles alive
        max_age = 1.0E30 if params['max_age'] is None else params['max_age']
        release_info['last_time_alive'] =  release_info['release_times'][-1] + si.model_direction*max_age
        release_info['last_time_alive'] =  min(max(hindcast_start,release_info['last_time_alive']),hindcast_end) # trim to limits of hind cast

        # useful info
        release_info['first_release_time'] = release_info['release_times'][0]
        release_info['last_release_time'] = release_info['release_times'][-1]

        release_info['release_dates'] = time_util.seconds_to_datetime64(release_info['release_times'])
        release_info['first_release_date'] = time_util.seconds_to_datetime64(release_info['first_release_time'])
        release_info['last_release_date'] = time_util.seconds_to_datetime64(release_info['last_release_time'])

        # index of release the  times to be released next
        release_info['index_of_next_release'] =  0

        if not   hindcast_start <= release_info['first_release_time'] <= hindcast_end :
            ml.msg(f'Release group "{info["name"]}" >  start time {time_util.seconds_to_isostr(release_info["first_release_time"])}  is outside the range of hydro-model times for release_group instance #{info["instanceID"]:2d}, ',
                   fatal_error=True,hint=f' Check release start time is in hydro-model  range of  {time_util.seconds_to_isostr(hindcast_start)}  to {time_util.seconds_to_isostr(hindcast_start)} ')

    def get_release_locations(self, time_sec):
        # set up full set of release locations inside  polygons
        si = self.shared_info
        info= self.info
        params=self.params

        n_required = self.get_number_required()
        release_data =dict(
            x= np.full((0, info['points'].shape[1]), 0.,dtype=np.float64, order='C'),
            n_cell = np.full((0,), -1, dtype=np.int32),
            hydro_model_gridID = np.full((0,), 0, dtype=np.int8),
            bc_cords = np.full((0,3), 0, dtype=np.float32),
            is_inside=np.full((0, ), False, dtype=bool),
            )
        count = 0
        while release_data['x'].shape[0] < n_required:
            # get 2D release candidates
            x0 = self.get_release_location_candidates()

            # get candiate locations inside domain
            rd  = self.check_potential_release_locations_in_bounds(x0)

            # keep those inside:
            for key in release_data.keys():
                rd[key] = np.concatenate((rd[key], rd[key][rd['is_inside'], ...]), axis=0)


            if len(rd) > 0 :
                is_ok = self.filter_release_points(rd,  time_sec= time_sec)
                # if any add to list
                for key in release_data.keys():
                    release_data[key] = np.concatenate((release_data[key], rd[key][is_ok, ...]), axis=0)

            # allow max_cycles_to_find_release_points cycles to find points
            count += 1
            if count > self.params["max_cycles_to_find_release_points"]: break

        if release_data['x'].shape[0] < n_required:
            si.msg_logger.msg(f'Release group-"{self.info["name"]}", only found {release_data["x"].shape[0]} of {n_required} required points inside domain after {self.params["max_cycles_to_find_release_points"]} cycles',
                              warning=True,
                           hint=f'Maybe, release points outside the domain?, or hydro-model grid and release points use different coordinate systems?? or increase parameter  max_cycles_to_find_release_points, current value = {self.params["max_cycles_to_find_release_points"]:3}' )
            n_required = release_data['x'].shape[0] #


        # trim initial location, cell  etc to required number
        for key in release_data.keys():
            release_data[key] = release_data[key][:n_required, ...]

        # add release IDs
        release_data['IDrelease_group'] = self.info['instanceID']
        release_data['IDpulse'] = info['pulse_count']
        info['pulse_count'] += 1
        release_data['user_release_groupID'] = self.params['user_release_groupID']

        info['number_released'] += release_data['x'].shape[0]  # count number released in this group

        # get water depth and tide
        fgm = si.classes['field_group_manager']
        release_data['water_depth'] = fgm.interp_named_field_at_given_locations_and_time(
                                        'water_depth', release_data['x'], time_sec=None,
                                         n_cell=release_data['n_cell'],
                                         bc_cords=release_data['bc_cords'],
                                         hydro_model_gridID=release_data['hydro_model_gridID'])
        release_data['tide'] = fgm.interp_named_field_at_given_locations_and_time(
                                        'tide',  release_data['x'], time_sec=time_sec,
                                          n_cell=release_data['n_cell'],
                                          bc_cords=release_data['bc_cords'],
                                          hydro_model_gridID=release_data['hydro_model_gridID'])
        # if z data not given in 3D
        if si.is3D_run:
            if release_data['x'].shape[1] <= 2:
                # expand x0 to 3D if needed
                release_data['x'] = np.concatenate((release_data['x'], np.zeros((release_data['x'].shape[0], 1), dtype=x0.dtype)), axis=1)

            if params['release_at_surface']:
                release_data['x'][:, 2] = release_data['tide'] - params['release_offset_from_surface_or_bottom']
            elif params['release_at_bottom']:
                release_data['x'][:, 2] = release_data['water_depth'] + params['release_offset_from_surface_or_bottom']

            elif params['z_min'] is not None or  params['z_max'] is not None:
                z_min = -large_float if params['z_min'] is None else params['z_min']
                z_max =  large_float if params['z_max'] is None else params['z_max']

                if z_min > z_max:
                    si.msg_logger.msg(f'Release group-"{self.info["name"]}", zmin >= zmax, (zmin,zmax) =({z_min:.3e}, {z_max:.3e}) ',
                                      hint='z=0 is mean tide level and z < 0 below the mean tide level',
                                      fatal_error=True, exit_now=True)
                release_data['x']  = self.get_z_release_in_depth_range(release_data['x'] ,z_min,z_max,
                                                                   release_data['water_depth'],release_data['tide'])
        return release_data

    @staticmethod
    @njitOT
    def get_z_release_in_depth_range(x,z_min, z_max, water_depth,tide):
        # get random release within z_min and z_max and with water depth and tide

        for n in range(x.shape[0]):
            z1  = max(-water_depth[n], z_min)
            z2  = min(tide[n], z_max)
            x[n, 2] = np.random.uniform(z1, z2, size=1)[0]
        return x

    def get_number_required(self):
        return self.params['pulse_size']*self.info['points'].shape[0]

    def get_release_location_candidates(self):
        si = self.shared_info
        x = np.repeat(self.info['points'], self.params['pulse_size'], axis=0)

        if self.params['release_radius']> 0.:
            rr = abs(float(self.params['release_radius']))
            n = x.shape[0]
            rr = np.repeat(rr, n, axis=0)
            r = np.random.random((n,)) * rr * np.exp(1.0j * np.random.random((n,)) * 2.0 * np.pi)
            r = r.reshape((-1, 1))
            x[:, :2] += np.hstack((np.real(r), np.imag(r)))

        return x


    def filter_release_points(self, release_data, time_sec= None):
        # user can filter release points by inheritance of this class and overriding this method
        # return booleaon of keeped points
        return np.full((release_data['x'].shape[0],), True)

    def check_potential_release_locations_in_bounds(self, x):
        si= self.shared_info
        # use KD tree to find points those inside model domain
        fgm = si.classes['field_group_manager']
        release_info  = fgm.are_points_inside_domain(x, self.params['allow_release_in_dry_cells'])

        return release_info

