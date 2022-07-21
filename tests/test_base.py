# Ocean particle tracker in pure python
# Ross Vennell Nov 2018


# profilers
# python -m cProfile
# python -m vmprof  <program.py> <program parameters>
# python -m cProfile -s cumtime
# -m bohrium
import numpy as np

from oceantracker.util import basic_util
from  oceantracker.main import  run
import argparse
import matplotlib.pyplot as plt
from os import path
import glob
from copy import copy
import time
from oceantracker.post_processing.read_output_files import load_output_files


def plot_sample(runCaseInfo, num_to_plot=10 ** 3):
    # plot devation from circle

    data = load_output_files.load_particle_track_vars(runCaseInfo, ['x', 'water_depth', 'time', 'x0', 'ID'])
    grid= load_output_files.load_grid(runCaseInfo)


    nx = data['x'].shape[1]
    sel = np.random.default_rng().choice(nx, size=50, replace=False)

    x = data['x'][:,sel,:]
    t   = (data['time']-data['time'][0])/ 24 / 3600
    x0 = data['x0'][sel,:]

    depth = data['water_depth'][:,sel]

    plt.figure()

    # grid and tracks
    plt.subplot(2, 2, 1)
    plt.plot(x[:, :, 0], x[:, :, 1], linewidth=.5)

    tl=''
    if x.shape[2] ==2:
        tl+='2D'
    else:
        tl += '3D'

    plt.title(tl)

    r0=np.sqrt(x0[:,0]**2+x0[:,1]**2)
    for r in r0:  # use r0 from main code
        plt.gcf().gca().add_artist(plt.Circle((0, 0), radius=r, color='black', fill=False, linestyle='--', linewidth=1))

    particle_plot.draw_base_map(grid)
    plt.axis([-16000, 16000, -16000, 16000])

    ax=plt.subplot(2, 2, 2)
    mag = np.sqrt(x[:, :, 0]**2 + x [:, :, 1]**2)
    mag0 = np.sqrt(x0[ :, 0]**2 + x0[ :, 1  ]**2)
    plt.plot(t, mag - mag0)
    plt.text(0.1, .1, 'deviation from circle, m', transform=ax.transAxes)

# particle speed from change in positions is harsher test
    ax=plt.subplot(2, 2, 3)
    #v=nc.variables['particle_velocity'][:,sel,:]
    v=np.diff(x,axis=0)/(data['time'][1]-data['time'][0])
    vmag = np.sqrt(v[:, :, 0] ** 2 + v[:, :, 1] ** 2)
    plt.plot(t[1:], vmag, linewidth=.5)
    plt.xlabel('days')
    plt.ylabel('speed')
    plt.text(0.1, .9, 'Particle Speed', transform=ax.transAxes)

    ax=plt.subplot(2, 2, 4)

    if v.shape[2] ==2:
        plt.plot(t, depth, linewidth=.5)
        plt.xlabel('days')
        plt.text(0.1, .1, 'water depth', transform = ax.transAxes)
    else:
        plt.plot(t, x[:, :, 2], linewidth=.5)
        plt.xlabel('days')
        plt.ylabel('z')
        plt.text(0.1, .1, 'Particle z', transform=ax.transAxes)

    plt.show()
    a=1



def plot_gridded_stats(log,seq_num, annotate_polygon=True, nfig=1):
    p = log['full_params']['particle_statistics'][seq_num]
    l=  log['info']['particle_statistics'][seq_num]

    if 'time' in p['name']:
        lab ="Time"
    else:
        lab ='Age'

    s = readOutputFiles.read_stats_file(l['file_name'], ['x', 'y', 'count', 'sum_water_depth', 'sum_age', 'sum_age_decay'])

    plt.figure()

    n_groups=s['x'].shape[0]

    # infer polygon file
    if annotate_polygon:
        pfile_mask= l['file_name'].replace('gridded','polygon').split('_0')
        pfile_name=glob.glob(pfile_mask[0]+'*.nc')[0]
        poly = readOutputFiles.read_polygon_stats(pfile_name, var_list=['count', 'sum_water_depth', 'sum_age', 'sum_age_decay'])


    for ng in range(n_groups):
        xi,yi = np.meshgrid(s['x'][ng,:],s['y'][ng,:])
        np1= ng*4  # 4 subplot

        count = np.nansum(s['count'][:, ng, :, :].astype(np.float64),axis=0)
        count[count==0.] = np.nan

        plt.subplot(n_groups, 4, 1 + np1)
        z=copy(count)
        z[z == 0.] = np.nan
        z2=np.full_like(xi,np.nan)
        sel=z >0
        z2[sel]= np.log10(z[sel])
        plt.pcolor(xi, yi, z)
        plt.colorbar()
        plt.title(lab +', Count Log_10',fontsize=6)
        plt.xticks([])
        plt.yticks([])
        if annotate_polygon:
            polygon_count= np.sum(poly['count'][:,ng,:].astype(np.float64),axis=0)
            polygon_anotation(poly,polygon_count)


        plt.subplot(n_groups, 4, 2 + np1)
        z = np.nansum(s['sum_water_depth'][:, ng, :, :], axis=0)/count
        plt.pcolor(xi, yi, z)
        plt.colorbar()
        plt.title('Depth',fontsize=6)
        plt.xticks([])
        plt.yticks([])
        if annotate_polygon:
            polygon_anotation(poly, np.nansum(poly['sum_water_depth'][:, ng, :], axis=0)/polygon_count)

        plt.subplot(n_groups, 4, 3 + np1)
        z = np.nansum(s['sum_age'][:, ng, :, :],axis=0)/count/24/3600
        plt.pcolor(xi, yi, z)
        plt.colorbar()
        plt.title('Age, days',fontsize=6)
        plt.xticks([])
        plt.yticks([])
        if annotate_polygon:
            polygon_anotation(poly, np.nansum(poly['sum_age'][:, ng, :] , axis=0)/polygon_count/24/3600)

        plt.subplot(n_groups, 4, 4 + np1)
        z = np.nanmean(s['sum_age_decay'][:, ng, :, :]/count,axis=0)
        plt.pcolor(xi, yi, z)
        plt.colorbar()
        #plt.text(0, 0, 'decay')
        plt.title('Unit age decay',fontsize=6)
        plt.xticks([])
        plt.yticks([])
        if annotate_polygon:
            polygon_anotation(poly, np.nansum(poly['sum_age_decay'][:, ng, :] , axis=0)/ polygon_count)

def polygon_anotation( poly,data):

    for n,p in enumerate(poly['info']['info']['polygon_list']):
        xy = np.array(p['points'])
        plt.plot(xy[:,0],xy[:,1])
        plt.text(xy[0, 0], xy[0, 1], str(np.round(data[n],2) ),fontsize=8)


def time_check_plot(runCaseInfo):
    data = load_output_files.load_particle_track_vars(runCaseInfo, ['time', 'age', 'ID'])
    nx = data['age'].shape[1]

    sel = np.sort(np.random.default_rng().choice(nx, size=30, replace=False))

    plt.figure()
    plt.subplot(3, 1, 1)

    t = (data['time'] - data['time'][0]) / 24 / 3600

    plt.plot(t,data['time'] )
    plt.title('time, days')

    plt.subplot(3, 1, 2)
    dt =np.diff(data['time'],axis=0)
    plt.plot(t[:-1],dt)
    plt.axis([0,t[-1],  np.nanmean(dt) - 50, np.nanmean(dt) + 50])
    plt.title('time diff, seconds')

    plt.subplot(3, 1, 3)
    plt.plot(t,data['age'][:, sel] / 24 / 3600)

    plt.title('Age')
    plt.show()

def base_param(is3D=False, isBackwards = False):


    # for speed comparions with Scipy make sure AH>0 otherwise its last particle recycling trick helps it

    r0 = np.array([2000., 4000., 8000, 10000])  # no bad starts

    # all of of pulse start at same location
    p0 = [[0, 2000.], [0, 4000.], [0, 8000.], [0, 10000.]]
    poly0 = [[9000., 9000], [10000, 9000], [10000, 10000.]]


    base_case={ 'run_params' :{'write_tracks': True,
                               'duration':10.*24*3600,

                                },

            'solver' : { 'RK_order': 4, 'n_sub_steps': 6 }, # 5min steps to mact OT v01 paper
            'particle_group_manager' : {},
            'particle_release_groups': [
                                        {'points': p0, 'pulse_size': 1, 'release_interval': 3600,'userRelease_groupID':5,
                                                                             'maximum_age' : 7*24*3600, 'user_release_group_name': 'A group',
                                         },
                                       {'class_name': 'oceantracker.particle_release_groups.polygon_release.PolygonRelease',
                                       'points': poly0, 'pulse_size': 1, 'release_interval': 3600,'userRelease_groupID':200,
                                      'maximum_age' : 4*24*3600, 'user_release_group_name': 'B group',
                                            }
                                        ],
            'dispersion': {'A_H': 0.},

            'particle_properties': [ {'class_name': 'oceantracker.particle_properties.age_decay.AgeDecay'},
                                       {'class_name' : 'oceantracker.particle_properties.distance_travelled.DistanceTravelled'}
                                                        ],
            'trajectory_modifiers': [],
            'velocity_modifiers' : []
           }

    outputdir = 'output'
    input_dir =path.normpath(path.join(path.split(__file__)[0],'testData'))

    params={  'shared_params': { 'debug': True,
                'root_output_dir': outputdir,
                                  'output_file_base': 'test_particle',
                                  'backtracking': isBackwards
                                  },
              'reader': {'file_mask' : 'circFlow2D*.nc', 'input_dir': input_dir,
                        'field_variables': {'water_depth': 'depth', 'tide': 'tide',
                                            'water_velocity' : [{'file_var' :'u','components': [0]},{'file_var' :'v','components': [1]}] },
                        'dimension_map': {'node': 'node', 'time': 'time'},
                        'grid_variables': {'time': 'time', 'x': [{'file_var' :'x','components': [0]},{'file_var' :'y','components': [1]}],
                                      'triangles': 'simplex',
                                       },
                         'time_buffer_size': 200,
                         'isodate_of_hindcast_time_zero': '2000-01-01'},

                'base_case_params': base_case
    }

    if is3D:
        # tweak for circle flow 3D
        r=params['reader']
        r['water_velocity_map'].update({ 'w':'w'})
        r['grid_map'].update({'zlevel': 'zlevel'})
        r['dimension_map'].update({'z': 'zlevel'})
        r['file_mask'] = params['reader']['file_mask'].replace('2D', '3D')
        base_case['solver']['screen_output_step_count'] = 1
        base_case['dispersion'].update({'A_H': 0.,'A_V': 0.})
        base_case['velocity_modifiers'].append(
            {'class_name': 'oceantracker.velocity_modifiers.terminal_velocity.AddTerminalVelocity', 'mean': 0*0.001})

    return params


def run_test(case_params={},  is3D=False,isBackwards= False):

    # get and update shared_params
    working_params=base_param(is3D, isBackwards)

    basic_util.deep_dict_update(working_params['base_case_params'],case_params)

    runInfoFile= run(working_params)

    runCaseInfo = load_output_files.load_Run_and_CaseInfo(runInfoFile, ncase=0)

    return runCaseInfo

if __name__ == '__main__':

    # windows/linux  data source

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', nargs='?', const=0, type=int, default=0)
    parser.add_argument('--test', nargs='?', const=None, type=int, default=None,)



    args = parser.parse_args()
    args.parallel= False

    print(args)
    if args.test is None:
        testList=range(3)
    else:
        testList=[args.test]

    t0 = time.time()


    for ntest in testList:

        if ntest==0:
            # zero dispersion test

            runCaseInfo = run_test(case_params={'duration':2*24*3600,'dispersion': {'A_H': 0.}
                                       #'particle_buffer_size': 3000,
                                       #'compact_mode': True
                                       })

            plot_sample(runCaseInfo)
            time_check_plot(runCaseInfo)
        elif ntest == 1:
            runCaseInfo = run_test(isBackwards=True,case_params={'duration': 2 * 24 * 3600, 'dispersion': {'A_H': 0.}
                                                # 'particle_buffer_size': 3000,
                                                # 'compact_mode': True
                                                })
            plot_sample(runCaseInfo)
            time_check_plot(runCaseInfo)

        elif ntest ==2:
            # large dispersion wall bc test
                runCaseInfo = run_test(case_params={ 'duration': 5.*24*3600, 'dispersion': {'A_H': 50.},
                                             'particle_group_manager': {'pulse_size': 2, 'release_interval': 300}})

                plot_sample(runCaseInfo)


        elif ntest==3:
            # plotting dev, test dry cell
            dc = None if 1 == 0   else  'dry_cells'

            params={ 'duration': 6.*24*3600, 'dispersion': {'A_H': 10},
                                                          'particle_group_manager': {'pulse_size': 10 ** 1, 'release_interval': 1*3600},
                                                'reader' : {'grid_map' : {'dry_cells' : dc},'dry_water_depth': 4.}
                                                         }

            runCaseInfo = run_test(args,params)
            plot_tracks.animate_particles(runCaseInfo)


        elif ntest == 4:
            # plotting dev
            runCaseInfo = run_test( {'duration': 5. * 24 * 3600,'dispersion': {'A_H': 0., 'A_V': 0.},
                                        'particle_group_manager': {'pulse_size': 10 ** 1, 'release_interval': 1 * 3600},
                                        'trajectory_modifiers':[{'class_name': 'oceantracker.trajectory_modifiers.resuspension.BasicResuspension'}],
                                        }, is3D=True)

            plot_sample(runCaseInfo)


        elif ntest == 1000:
            # large dispersion wall bc test
            ot = run_test(args,params={'dispersion': {'A_H': 0.}, 'solver': {'duration': 30.}} )

        elif ntest == 2000:
            # speed test external fine grid
            args.doplot= False
            args.size  = 3
            args.file=1
            ot = run_test(args,params={'dispersion': {'A_H': 1.0}, 'solver': {'duration': 1.},
                                       'particle_group_manager':{'pulse_size': 25000, 'release_interval': 0},
                                       'tracks_writer':{'class_name': 'oceantracker.tracks_writer.writerBase.BaseWriter','output_step_count': 100000} # a non-writer
                                       })

        elif ntest== 9999:
            # latest dev block
            args.plot = 0
            args.size = 0
            args.file = 1
            ot = run_test(args, params={'dispersion': {'A_H': 0.},
                                        'particle_group_manager': {'pulse_size': 5, 'release_interval': 300},
                                        # test newreader
                                        'solver' : {'n_sub_steps' :2,'duration': 2.}
                                        })

        print(' Total run time ' + '%5.2f' % (time.time() - t0))

