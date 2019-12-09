
import matplotlib.pyplot as plt
import openmdao.api as om
from dymos.examples.plotting import plot_results
import dymos as dm
import numpy as np
#
# Setup and solve the optimal control problem
#
p = om.Problem(model=om.Group())
p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['iSumm'] = 6

p.driver.declare_coloring()

from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

#
# Initialize our Trajectory and Phase
#
traj = dm.Trajectory()

phase = dm.Phase(ode_class=LaunchVehicleODE,
                 ode_init_kwargs={'central_body': 'earth'},
                 transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

traj.add_phase('phase0', phase)
p.model.add_subsystem('traj', traj)

#
# Set the options for the variables
#
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                rate_source='eom.xdot', units='m')
phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                rate_source='eom.ydot', targets=['atmos.y'], units='m')
phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                rate_source='eom.mdot', targets=['eom.m'], units='kg')

phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['eom.theta'])
phase.add_control('thrust', units='N', lower=100, upper=2100000, targets=['eom.thrust'])
#phase.add_design_parameter('thrust', units='N', opt=True, val=2100000.0, targets=['eom.thrust'])

#
# Set the options for our constraints and objective
#
phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
phase.add_boundary_constraint('vy', loc='final', equals=0)

#46k
phase.add_path_constraint('eom.q', upper=40000., scaler=1/35000.)
#p.model.add_constraint('traj.phase0.rhs_disc.const1.c', upper=0.0, scaler=1)

phase.add_objective('time', loc='final', scaler=0.001)

phase.add_timeseries_output('eom.q', 'q_out')


p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
p.driver.opt_settings['Major optimality tolerance'] = 1e-3


p.model.linear_solver = om.DirectSolver()

#
# Setup and set initial values
#
p.setup(check=True)


p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 150.0
p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 1.15E5], nodes='state_input')
p['traj.phase0.states:y'] = phase.interpolate(ys=[0, 1.85E5], nodes='state_input')
p['traj.phase0.states:vx'] = phase.interpolate(ys=[0, 3000.6961], nodes='state_input')
p['traj.phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_input')
p['traj.phase0.states:m'] = phase.interpolate(ys=[117000, 1163], nodes='state_input')
p['traj.phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='control_input')
p['traj.phase0.controls:thrust'] = phase.interpolate(ys=[2000000, 2000000], nodes='control_input')

# p.run_model()
# p.check_partials(compact_print=True, includes=['traj.phases.phase0.rhs_disc.const1'])
# quit()

#
# Solve the Problem
#
p.run_driver()


# Get the explitly simulated results
#
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:y',
               'time (s)', 'altitude (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q_out',
               'time (s)', 'q'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:thrust',
               'time (s)', 'N')],
             title='SSTO',
             p_sol=p, p_sim=exp_out)

plt.show()
