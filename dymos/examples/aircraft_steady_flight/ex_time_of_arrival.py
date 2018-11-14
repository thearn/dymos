from __future__ import division, print_function, absolute_import

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, IndepVarComp

from dymos import Phase

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
from dymos.utils.lgl import lgl


def ex_aircraft_steady_flight(optimizer='SLSQP', transcription='gauss-lobatto'):
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['dynamic_simul_derivs'] = True
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings["Linesearch tolerance"] = 0.10
        p.driver.opt_settings["Major step limit"] = 0.05
        p.driver.opt_settings['iSumm'] = 6

    num_seg = 15
    seg_ends, _ = lgl(num_seg + 1)

    phase = Phase(transcription,
                  ode_class=AircraftODE,
                  num_segments=num_seg,
                  segment_ends=seg_ends,
                  transcription_order=5,
                  compressed=False)

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
    assumptions.add_output('S', val=427.8, units='m**2')
    assumptions.add_output('mass_empty', val=1.0, units='kg')
    assumptions.add_output('mass_payload', val=1.0, units='kg')

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0),
                           duration_bounds=(300, 3700),
                           duration_ref=3600)

    phase.set_state_options('range', units='NM', fix_initial=True, fix_final=True, scaler=0.001,
                            defect_scaler=1.0E-2)
    phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=False,
                            upper=1.5E5, lower=0.0, scaler=1.0E-5, defect_scaler=1.0E-1)

    phase.add_control('alt', units='kft', opt=True, lower=1.0, upper=50.0,
                      rate_param='climb_rate',
                      rate_continuity=True, rate_continuity_scaler=1.0,
                      rate2_continuity=True, rate2_continuity_scaler=1.0, ref=1.0)

    phase.add_control('mach', units=None, opt=True, lower=0.1, upper=0.9499)

    phase.add_input_parameter('S', units='m**2')
    phase.add_input_parameter('mass_empty', units='kg')
    phase.add_input_parameter('mass_payload', units='kg')

    phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)
    phase.add_path_constraint('alt_rate', units='ft/min', lower=-3000, upper=3000, ref=3000)

    #phase.add_boundary_constraint('time', loc='final', upper=3700.0)
    
    #phase.add_boundary_constraint('mach', loc='final', upper=0.2)
    #phase.add_boundary_constraint('mach', loc='initial', upper=0.2)

    phase.add_boundary_constraint('alt', loc='final', upper=1.5)
    phase.add_boundary_constraint('alt', loc='initial', upper=1.5)

    p.model.connect('assumptions.S', 'phase0.input_parameters:S')
    p.model.connect('assumptions.mass_empty', 'phase0.input_parameters:mass_empty')
    p.model.connect('assumptions.mass_payload', 'phase0.input_parameters:mass_payload')

    # max range: 726.53835558
    
    # min arrival time at 500km: 3609
    
    # min fuel burn to reach 500km leaves: 6267.903 mass_fuel at end
    #   arrival time: 3812.05707505

    # min fuel burn to reach 500 s.t. arrives in less than 3700.
    #   time: 3700.0
    #   fuel remaining: 3761.79

    phase.add_objective('mass_fuel', loc='final', ref=-1.0)

    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 3600.0
    p['phase0.states:range'] = phase.interpolate(ys=(0, 500.0), nodes='state_input')
    p['phase0.states:mass_fuel'] = phase.interpolate(ys=(30000, 0), nodes='state_input')

    p['phase0.controls:mach'][:] = 0.4
    p['phase0.controls:alt'][:] = 1.5

    p['assumptions.S'] = 427.8
    p['assumptions.mass_empty'] = 0.15E6
    p['assumptions.mass_payload'] = 84.02869 * 400

    p.run_driver()

    exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 500), record=False)

    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('alt', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('alt'), 'b-')
    plt.suptitle('altitude vs time')

    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('mach', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('mach'), 'b-')
    plt.suptitle('mach vs time')

    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('range', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('range'), 'b-')
    plt.suptitle('range vs time')

    return p


if __name__ == '__main__':
    ex_aircraft_steady_flight(optimizer='SNOPT', transcription='radau-ps')
