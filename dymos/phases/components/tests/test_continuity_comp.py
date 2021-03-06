from __future__ import print_function, division, absolute_import

import unittest

import itertools
from parameterized import parameterized

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from dymos.phases.grid_data import GridData
from dymos.phases.components.continuity_comp import GaussLobattoContinuityComp, \
    RadauPSContinuityComp, ExplicitContinuityComp
from dymos.phases.options import StateOptionsDictionary, ControlOptionsDictionary


class TestContinuityComp(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['compressed', 'uncompressed'],  # jacobian
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_continuity_comp',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_continuity_comp(self, transcription='gauss-lobatto', compressed='compressed'):

        num_seg = 3

        gd = GridData(num_segments=num_seg,
                      transcription_order=[5, 3, 3],
                      segment_ends=[0.0, 3.0, 10.0, 20],
                      transcription=transcription,
                      compressed=compressed == 'compressed')

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        nn = gd.subset_num_nodes['all']

        ivp.add_output('x', val=np.arange(nn), units='m')
        ivp.add_output('y', val=np.arange(nn), units='m/s')
        ivp.add_output('u', val=np.zeros((nn, 3)), units='deg')
        ivp.add_output('v', val=np.arange(nn), units='N')
        ivp.add_output('u_rate', val=np.zeros((nn, 3)), units='deg/s')
        ivp.add_output('v_rate', val=np.arange(nn), units='N/s')
        ivp.add_output('u_rate2', val=np.zeros((nn, 3)), units='deg/s**2')
        ivp.add_output('v_rate2', val=np.arange(nn), units='N/s**2')
        ivp.add_output('t_duration', val=121.0, units='s')

        self.p.model.add_design_var('x', lower=0, upper=100)

        state_options = {'x': StateOptionsDictionary(),
                         'y': StateOptionsDictionary()}
        control_options = {'u': ControlOptionsDictionary(),
                           'v': ControlOptionsDictionary()}

        state_options['x']['units'] = 'm'
        state_options['y']['units'] = 'm/s'

        control_options['u']['units'] = 'deg'
        control_options['u']['shape'] = (3,)
        control_options['u']['continuity'] = True

        control_options['v']['units'] = 'N'

        if transcription == 'gauss-lobatto':
            cnty_comp = GaussLobattoContinuityComp(grid_data=gd, time_units='s',
                                                   state_options=state_options,
                                                   control_options=control_options)
        elif transcription == 'radau-ps':
            cnty_comp = RadauPSContinuityComp(grid_data=gd, time_units='s',
                                              state_options=state_options,
                                              control_options=control_options)
        else:
            raise ValueError('unrecognized transcription')

        self.p.model.add_subsystem('cnty_comp', subsys=cnty_comp)
        # The sub-indices of state_disc indices that are segment ends
        segment_end_idxs = gd.subset_node_indices['segment_ends']

        self.p.model.connect('x', 'cnty_comp.states:x', src_indices=segment_end_idxs)
        self.p.model.connect('y', 'cnty_comp.states:y', src_indices=segment_end_idxs)

        self.p.model.connect('t_duration', 'cnty_comp.t_duration')

        size_u = nn * np.prod(control_options['u']['shape'])
        src_idxs_u = np.arange(size_u).reshape((nn,) + control_options['u']['shape'])
        src_idxs_u = src_idxs_u[gd.subset_node_indices['segment_ends'], ...]

        size_v = nn * np.prod(control_options['v']['shape'])
        src_idxs_v = np.arange(size_v).reshape((nn,) + control_options['v']['shape'])
        src_idxs_v = src_idxs_v[gd.subset_node_indices['segment_ends'], ...]

        self.p.model.connect('u', 'cnty_comp.controls:u', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('u_rate', 'cnty_comp.control_rates:u_rate', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('u_rate2', 'cnty_comp.control_rates:u_rate2', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('v', 'cnty_comp.controls:v', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.model.connect('v_rate', 'cnty_comp.control_rates:v_rate', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.model.connect('v_rate2', 'cnty_comp.control_rates:v_rate2', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.setup(check=True, force_alloc_complex=True)

        self.p['x'] = np.random.rand(*self.p['x'].shape)
        self.p['y'] = np.random.rand(*self.p['y'].shape)
        self.p['u'] = np.random.rand(*self.p['u'].shape)
        self.p['v'] = np.random.rand(*self.p['v'].shape)
        self.p['u_rate'] = np.random.rand(*self.p['u'].shape)
        self.p['v_rate'] = np.random.rand(*self.p['v'].shape)
        self.p['u_rate2'] = np.random.rand(*self.p['u'].shape)
        self.p['v_rate2'] = np.random.rand(*self.p['v'].shape)

        self.p.run_model()

        for state in ('x', 'y'):
            expected_val = self.p[state][segment_end_idxs, ...][2::2, ...] - \
                self.p[state][segment_end_idxs, ...][1:-1:2, ...]

            assert_rel_error(self,
                             self.p['cnty_comp.defect_states:{0}'.format(state)],
                             expected_val.reshape((num_seg - 1,) + state_options[state]['shape']))

        for ctrl in ('u', 'v'):

            expected_val = self.p[ctrl][segment_end_idxs, ...][2::2, ...] - \
                self.p[ctrl][segment_end_idxs, ...][1:-1:2, ...]

            assert_rel_error(self,
                             self.p['cnty_comp.defect_controls:{0}'.format(ctrl)],
                             expected_val.reshape((num_seg-1,) + control_options[ctrl]['shape']))

        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_explicit_continuity_comp(self, compressed='compressed'):

        num_seg = 3

        gd = GridData(num_segments=num_seg,
                      transcription_order=[5, 3, 3],
                      num_steps_per_segment=4,
                      segment_ends=[0.0, 3.0, 10.0, 20],
                      transcription='explicit',
                      compressed=compressed == 'compressed')

        self.p = Problem(model=Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])

        nn = gd.subset_num_nodes['all']

        for iseg in range(gd.num_segments):
            num_steps = gd.num_steps_per_segment[iseg]
            ivp.add_output('seg_{0}:x'.format(iseg), val=np.arange(num_steps + 1), units='m')
            ivp.add_output('seg_{0}:y'.format(iseg), val=np.arange(num_steps + 1), units='m/s')
        ivp.add_output('u', val=np.zeros((nn, 3)), units='deg')
        ivp.add_output('v', val=np.arange(nn), units='N')
        ivp.add_output('u_rate', val=np.zeros((nn, 3)), units='deg/s')
        ivp.add_output('v_rate', val=np.arange(nn), units='N/s')
        ivp.add_output('u_rate2', val=np.zeros((nn, 3)), units='deg/s**2')
        ivp.add_output('v_rate2', val=np.arange(nn), units='N/s**2')
        ivp.add_output('t_duration', val=121.0, units='s')

        state_options = {'x': StateOptionsDictionary(),
                         'y': StateOptionsDictionary()}
        control_options = {'u': ControlOptionsDictionary(),
                           'v': ControlOptionsDictionary()}

        state_options['x']['units'] = 'm'
        state_options['y']['units'] = 'm/s'

        control_options['u']['units'] = 'deg'
        control_options['u']['shape'] = (3,)
        control_options['u']['continuity'] = True

        control_options['v']['units'] = 'N'

        cnty_comp = ExplicitContinuityComp(grid_data=gd,
                                           shooting='multiple',
                                           time_units='s',
                                           state_options=state_options,
                                           control_options=control_options)

        self.p.model.add_subsystem('cnty_comp', subsys=cnty_comp)
        # The sub-indices of state_disc indices that are segment ends
        segment_end_idxs = gd.subset_node_indices['segment_ends']

        for iseg in range(num_seg):
            for state_name in ['x', 'y']:

                self.p.model.connect('seg_{0}:{1}'.format(iseg, state_name),
                                     'cnty_comp.seg_{0}_states:{1}'.format(iseg, state_name))

        self.p.model.connect('t_duration', 'cnty_comp.t_duration')

        size_u = nn * np.prod(control_options['u']['shape'])
        src_idxs_u = np.arange(size_u).reshape((nn,) + control_options['u']['shape'])
        src_idxs_u = src_idxs_u[gd.subset_node_indices['segment_ends'], ...]

        size_v = nn * np.prod(control_options['v']['shape'])
        src_idxs_v = np.arange(size_v).reshape((nn,) + control_options['v']['shape'])
        src_idxs_v = src_idxs_v[gd.subset_node_indices['segment_ends'], ...]

        self.p.model.connect('u', 'cnty_comp.controls:u', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('u_rate', 'cnty_comp.control_rates:u_rate', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('u_rate2', 'cnty_comp.control_rates:u_rate2', src_indices=src_idxs_u,
                             flat_src_indices=True)

        self.p.model.connect('v', 'cnty_comp.controls:v', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.model.connect('v_rate', 'cnty_comp.control_rates:v_rate', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.model.connect('v_rate2', 'cnty_comp.control_rates:v_rate2', src_indices=src_idxs_v,
                             flat_src_indices=True)

        self.p.setup(check=True, force_alloc_complex=True)

        for iseg in range(num_seg):
            for state_name in ['x', 'y']:
                var = 'seg_{0}:{1}'.format(iseg, state_name)
                self.p[var] = np.random.rand(*self.p[var].shape)
        self.p['u'] = np.random.rand(*self.p['u'].shape)
        self.p['v'] = np.random.rand(*self.p['v'].shape)
        self.p['u_rate'] = np.random.rand(*self.p['u'].shape)
        self.p['v_rate'] = np.random.rand(*self.p['v'].shape)
        self.p['u_rate2'] = np.random.rand(*self.p['u'].shape)
        self.p['v_rate2'] = np.random.rand(*self.p['v'].shape)

        self.p.run_model()

        for state_name in ('x', 'y'):
            for i in range(1, num_seg):
                left_var = 'seg_{0}:{1}'.format(i - 1, state_name)
                left = self.p[left_var][-1, ...]
                right_var = 'seg_{0}:{1}'.format(i, state_name)
                right = self.p[right_var][0, ...]
                expected_val = right - left
                assert_rel_error(self,
                                 self.p['cnty_comp.defect_states:{0}'.format(state_name)][i-1, ...],
                                 expected_val)

        for ctrl in ('u', 'v'):

            expected_val = self.p[ctrl][segment_end_idxs, ...][2::2, ...] - \
                self.p[ctrl][segment_end_idxs, ...][1:-1:2, ...]

            assert_rel_error(self,
                             self.p['cnty_comp.defect_controls:{0}'.format(ctrl)],
                             expected_val.reshape((num_seg-1,) + control_options[ctrl]['shape']))

        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
