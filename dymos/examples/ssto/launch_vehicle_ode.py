from __future__ import print_function, division, absolute_import

import openmdao.api as om

from .log_atmosphere_comp import LogAtmosphereComp
from .launch_vehicle_2d_eom_comp import LaunchVehicle2DEOM
from .aggregator_funcs import aggf
import numpy as np

class ConstraintAggregator(om.ExplicitComponent):
    """
    Transform and aggregate constraints to a single value.
    """
    def initialize(self):
        self.options.declare('aggregator', types=str)
        self.options.declare('rho', default=50.0, types=float)
        self.options.declare('reversed', default=False, types=bool)
        self.options.declare('offset', types=float, default=39500.)
        self.options.declare('scale', types=float, default=35000.)
        self.options.declare('nn', types=int)

    def setup(self):
        agg = self.options['aggregator']

        self.nn = self.options['nn']
        self.offset = self.options['offset']
        self.scale = self.options['scale']

        self.reversed = False
        if self.options['reversed']:
            self.reversed = True
        self.aggf = aggf[agg]

        self.add_input(name='g', val=np.zeros(self.nn))
        self.add_output(name='c', val=0.0)

        self.declare_partials('c', 'g')

    def compute(self, inputs, outputs):
        rho = self.options['rho']
        g = inputs['g']

        g2 = (g - self.offset) / self.scale

        k, dk = self.aggf((g2), rho)
        outputs['c'] = np.sum(k)
        self.dk = dk / self.scale

    def compute_partials(self, inputs, partials):
        g = inputs['g']
        partials['c', 'g'] = self.dk

class LaunchVehicleODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('central_body', values=['earth', 'moon'], default='earth',
                             desc='The central gravitational body for the launch vehicle.')

    def setup(self):
        nn = self.options['num_nodes']
        cb = self.options['central_body']

        if cb == 'earth':
            rho_ref = 1.225
            h_scale = 8.44E3
        elif cb == 'moon':
            rho_ref = 0.0
            h_scale = 1.0
        else:
            raise RuntimeError('Unrecognized value for central_body: {0}'.format(cb))

        self.add_subsystem('atmos',
                           LogAtmosphereComp(num_nodes=nn, rho_ref=rho_ref, h_scale=h_scale))

        self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn, central_body=cb))

        self.connect('atmos.rho', 'eom.rho')

        # self.add_subsystem('const1', ConstraintAggregator(aggregator='PRePU', 
        #                                                   rho=2.0, 
        #                                                   nn=nn))

        # self.add_subsystem('const1', ConstraintAggregator(aggregator='KS', 
        #                                                   rho=175.0, 
        #                                                   nn=nn))

        #self.connect('eom.q', 'const1.g')


