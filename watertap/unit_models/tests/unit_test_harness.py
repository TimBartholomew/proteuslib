###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################

import pytest

from pyomo.environ import (Block,
                           assert_optimal_termination)
from pyomo.util.check_units import assert_units_consistent
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              number_variables,
                                              number_total_constraints,
                                              number_unused_variables)
from idaes.core.util import get_solver
from watertap.core.util.initialization import check_constraint_status
from watertap.core.util.report import print_report_differences


# -----------------------------------------------------------------------------
class UnitAttributeError(AttributeError):
    """
    ProteusLib exception for generic attribute errors arising from unit model testing.
    """
    pass


class UnitValueError(ValueError):
    """
    ProteusLib exception for generic value errors arising from unit model testing.
    """
    pass


class UnitRuntimeError(RuntimeError):
    """
    ProteusLib exception for generic runtime errors arising from unit model testing.
    """
    pass


class UnitTestHarness():
    def configure_class(self):
        self.solver = None  # string for solver, if None use WaterTAP default
        self.optarg = None  # dictionary for solver options, if None use WaterTAP default

        self.configure()
        blk = self.unit_model_block

        # attaching objects to model to carry through in pytest frame
        assert not hasattr(blk, '_test_objs')
        blk._test_objs = Block()
        blk._test_objs.solver = self.solver
        blk._test_objs.optarg = self.optarg
        blk._test_objs.stateblock_statistics = self.unit_statistics
        blk._test_objs.unit_report = self.unit_report

    def configure(self):
        """
        Placeholder method to allow user to setup test harness.

        The configure function must set the attributes:

        unit_model: pyomo unit model block (e.g. m.fs.unit), the block should
            have zero degrees of freedom, i.e. fully specified

        unit_statistics: dictionary of model statistics
            {'number_config_args': VALUE,
             'number_variables': VALUE,
             'number_total_constraints': VALUE,
             'number_unused_variables': VALUE}

        unit_solution: dictionary of property values for the specified state variables
            keys = (string name of variable, tuple index), values = value
        """
        pass

    @pytest.fixture(scope='class')
    def frame_unit(self):
        self.configure_class()
        return self.unit_model_block

    @pytest.mark.unit
    def test_unit_statistics(self, frame_unit):
        blk = frame_unit
        stats = blk._test_objs.stateblock_statistics

        if number_variables(blk) != stats['number_variables']:
            raise UnitValueError(
                "The number of variables were {num}, but {num_test} was "
                "expected ".format(
                    num=number_variables(blk),
                    num_test=stats['number_variables']))
        if number_total_constraints(blk) != stats['number_total_constraints']:
            raise UnitValueError(
                "The number of constraints were {num}, but {num_test} was "
                "expected ".format(
                    num=number_total_constraints(blk),
                    num_test=stats['number_total_constraints']))
        if number_unused_variables(blk) != stats['number_unused_variables']:
            raise UnitValueError(
                "The number of unused variables were {num}, but {num_test} was "
                "expected ".format(
                    num=number_unused_variables(blk),
                    num_test=stats['number_unused_variables']))

    @pytest.mark.unit
    def test_units_consistent(self, frame_unit):
        assert_units_consistent(frame_unit)

    @pytest.mark.unit
    def test_dof(self, frame_unit):
        if degrees_of_freedom(frame_unit) != 0:
            raise UnitAttributeError(
                "The unit has {dof} degrees of freedom when 0 is required."
                "".format(dof=degrees_of_freedom(frame_unit)))

    @pytest.mark.component
    def test_initialization(self, frame_unit):
        blk = frame_unit

        # initialize
        blk.initialize(solver=blk._test_objs.solver, optarg=blk._test_objs.optarg)

        # check convergence
        # TODO: update this when IDAES API is updated to return solver status for initialize()
        check_constraint_status(blk)


    @pytest.mark.component
    def test_solve(self, frame_unit, capsys):
        blk = frame_unit

        # solve unit
        if blk._test_objs.solver is None:
            opt = get_solver()
        else:
            opt = get_solver(solver=blk._test_objs.solver, options=blk._test_objs.optarg)
        results = opt.solve(blk)

        # check solve
        assert_optimal_termination(results)

        # capture the report
        blk.report()
        captured = capsys.readouterr()

        # check if report matches
        print_report_differences(blk._test_objs.unit_report, captured.out)
        assert captured.out == blk._test_objs.unit_report
