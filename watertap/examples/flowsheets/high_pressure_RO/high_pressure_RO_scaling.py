###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
from pyomo.environ import (ConcreteModel,
                           value,
                           Constraint,
                           Expression,
                           Objective,
                           Param,
                           Var,
                           TransformationFactory,
                           units as pyunits,
                           assert_optimal_termination)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import (solve_indexed_blocks,
                                            propagate_state,
                                            fix_state_vars,
                                            revert_state_vars)
from idaes.generic_models.unit_models import Product, Feed, Mixer, Translator
from idaes.generic_models.unit_models.mixer import MomentumMixingType
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

import watertap.examples.flowsheets.high_pressure_RO.high_pressure_RO as high_pressure_RO

from idaes.surrogate.pysmo import polynomial_regression

def main(case='seawater'):
    m = high_pressure_RO.build(case=case)
    high_pressure_RO.specify_model(m)
    high_pressure_RO.initialize_model(m)

    # simulate and display
    high_pressure_RO.solve(m)
    print('\n***---Simulation results---***')
    high_pressure_RO.display_results(m)

    # optimize and display
    high_pressure_RO.set_up_optimization(m)
    high_pressure_RO.optimize(m)
    print('\n***---Optimization results---***')
    high_pressure_RO.display_results(m)

    print('\n***---Optimization results with scaling surrogate---***')
    # add scaling surrogate
    add_scaling_surrogate(m)
    m.fs.product_recovery.unfix()
    m.fs.gypsum_scaling_index.fix(1)
    high_pressure_RO.optimize(m)
    high_pressure_RO.display_results(m)
    print('---scaling index---')
    print('Gypsum scaling index: %.2f' % m.fs.gypsum_scaling_index.value)

def add_scaling_surrogate(m):
    dir_path = r'C:\Users\timvb\Box\WaterTAP (protected by NDA)\nawi (NDA protected)\OLI (protected data)\feed_cases\\'
    file_name = m.case + '_gypsum_surrogate.pickle'
    gypsum_scaling_index_surrogate = polynomial_regression.PolynomialRegression.pickle_load(
        dir_path + file_name)
    m.fs.gypsum_scaling_index = Var(initialize=1,
                                    bounds=(0, None),
                                    units=pyunits.dimensionless)

    m.fs.product_recovery_indexed = Var([0],
                                        initialize=0.73,
                                        bounds=(0, 1),
                                        units=pyunits.dimensionless)
    m.fs.eq_product_recovery_indexed = Constraint(
        expr=m.fs.product_recovery == m.fs.product_recovery_indexed[0])

    m.fs.eq_gypsum_scaling_index = Constraint(
        expr=(m.fs.gypsum_scaling_index ==
              gypsum_scaling_index_surrogate['model'].generate_expression([m.fs.product_recovery_indexed[0]]))
    )

if __name__ == "__main__":
    main(case='seawater')
    # main(case='brackish_1')
    # main(case='brackish_2')
