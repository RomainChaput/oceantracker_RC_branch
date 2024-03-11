from inspect import signature
import numpy as np
from oceantracker.util import  time_util
class Scheduler(object):
    # set up event shedule based on times since 1/1/1970
    # rounds starts, times and intervals to model time steps,
    # uses times given, otherwise start and interval
    # all times in seconds
    def __init__(self, run_info,hindcast_info,
                 start=None, end=None, duration=None,
                 interval = None, times=None):

        self.interval_rounded_to_time_step = False
        self.times_rounded_to_time_step =  False
        self.start_time_outside_hydro_model_times = False

        interval= abs(interval) # ensure positive

        # deal with None and isodates
        if start is None: start = run_info['start_time']
        if type(start) is str: start = time_util.isostr_to_seconds(start)

        if end is None: end = run_info['end_time']
        if type(end) is str: end = time_util.isostr_to_seconds(end)

        dt = run_info['time_step']
        tol = 0.05

        if times is not None:
            # use times given
            # check if at model time steps
            n = (times - run_info['start_time'])/run_info['time_step']
            times_rounded = run_info['start_time'] + round(n) * run_info['time_step']

            if np.any(np.abs(times-times_rounded)/dt > tol * dt):
                self.times_rounded_to_time_step = True
            self.task_times = times_rounded
        else:
            # make from start time and interval
            if start is None: start = run_info['start_time'] # start ast model sart
            if type(start) == str: start = time_util.isostr_to_seconds(start)

            n =(start- run_info['start_time'])/dt  # number of model steps since the start
            start_rounded = run_info['start_time'] + round(n)*dt
            if  abs(start_rounded-start)/dt > tol:
                self.times_rounded_to_time_step = True
            start = start_rounded

            if not ( hindcast_info['start_time'] <= start  <= hindcast_info['end_time']):
                self.start_time_outside_hydro_model_times = True

            # round interval
            rounded_interval = round(interval/dt)*dt
            if abs(interval-rounded_interval)/dt > tol:
                self.interval_rounded_to_time_step = True
            interval = rounded_interval

            # look at duration from end if given
            if duration is None:
                if end is None: end = run_info['end_time'] # start ast model sart
                if type(end) == str: end = time_util.isostr_to_seconds(end)
                duration = abs(end-start)

            # make even starting
            if interval < .1*dt:
                # if interval is zero
                interval = 0.
                self.task_times= np.asarray([start])
            else:
                interval = max(interval, dt)
                self.task_times = start + np.arange(0,abs(duration),interval)

        # make a task flag for each time step of the model run
        self.task_flag = np.full_like(run_info['times'],False, dtype=bool)
        nt_task = ((self.task_times-run_info['start_time'])/run_info['time_step']).astype(np.int32)

        # now clip times to be within model start and end of run
        sel = np.logical_and(nt_task >= 0, nt_task < self.task_flag.size)
        self.task_flag[nt_task[sel]] = True

        # record info
        self.info= dict(start=self.task_times[0], interval=interval, end=self.task_times[-1],
                        start_date=time_util.seconds_to_isostr(self.task_times[0]),
                        end_date=time_util.seconds_to_isostr(self.task_times[-1]),
                        scheduled_times = self.task_times,
                        )
        pass

    def do_task(self, n_time_step):
        # check if task flag is set
        return self.task_flag[n_time_step]