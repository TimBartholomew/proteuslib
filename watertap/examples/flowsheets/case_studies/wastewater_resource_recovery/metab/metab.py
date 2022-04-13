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
import os
from pyomo.environ import (
    ConcreteModel,
    value,
    TransformationFactory,
    units as pyunits,
    assert_optimal_termination,
)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.generic_models.unit_models import Product
import idaes.core.util.scaling as iscale
from idaes.generic_models.costing import UnitModelCostingBlock

from watertap.core.util.initialization import assert_degrees_of_freedom

from watertap.core.wt_database import Database
import watertap.core.zero_order_properties as prop_ZO
from watertap.unit_models.zero_order import (
    FeedZO,
    MetabZO,
)
from watertap.core.zero_order_costing import ZeroOrderCosting


def main():
    m = build()

    set_operating_conditions(m)
    assert_degrees_of_freedom(m, 0)
    assert_units_consistent(m)

    initialize_system(m)

    results = solve(m)
    display_results(m)

    add_costing(m)
    initialize_costing(m)
    assert_degrees_of_freedom(m, 0)

    # m.fs.costing.metab.bead_cost.fix(0.1)

    results = solve(m)
    display_costing(m)
    return m, results


def build():
    # flowsheet set up
    m = ConcreteModel()
    m.db = Database()

    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.prop = prop_ZO.WaterParameterBlock(
        default={"solute_list": ["cod", "hydrogen", "methane"]}
    )

    # unit models
    m.fs.feed = FeedZO(default={"property_package": m.fs.prop})
    m.fs.metab_hydrogen = MetabZO(default={
        "property_package": m.fs.prop,
        "database": m.db,
        "process_subtype": "hydrogen"},
    )
    m.fs.metab_methane = MetabZO(default={
        "property_package": m.fs.prop,
        "database": m.db,
        "process_subtype": "methane"},
    )
    m.fs.product_hydrogen = Product(default={"property_package": m.fs.prop})
    m.fs.product_methane = Product(default={"property_package": m.fs.prop})
    m.fs.product_H2O = Product(default={"property_package": m.fs.prop})

    # connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.metab_hydrogen.inlet)
    m.fs.s02 = Arc(source=m.fs.metab_hydrogen.treated, destination=m.fs.metab_methane.inlet)
    m.fs.s03 = Arc(source=m.fs.metab_hydrogen.byproduct, destination=m.fs.product_hydrogen.inlet)
    m.fs.s04 = Arc(source=m.fs.metab_methane.byproduct, destination=m.fs.product_methane.inlet)
    m.fs.s05 = Arc(source=m.fs.metab_methane.treated, destination=m.fs.product_H2O.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # scaling
    iscale.calculate_scaling_factors(m)

    return m


def set_operating_conditions(m):
    # ---specifications---
    # feed
    # flow_vol = 0.9224 * pyunits.m**3 / pyunits.s
    # conc_mass_tds = 0.63 * pyunits.kg / pyunits.m**3
    # conc_mass_tss = 0.006525 * pyunits.kg / pyunits.m**3
    # conc_mass_toc = 0.004 * pyunits.kg / pyunits.m**3
    #
    # m.fs.feed.flow_vol[0].fix(flow_vol)
    # m.fs.feed.conc_mass_comp[0, "tds"].fix(conc_mass_tds)
    # m.fs.feed.conc_mass_comp[0, "tss"].fix(conc_mass_tss)
    # m.fs.feed.conc_mass_comp[0, "toc"].fix(conc_mass_toc)
    # solve(m.fs.feed)

    m.fs.feed.properties[0].flow_mass_comp["H2O"].fix(1)
    m.fs.feed.properties[0].flow_mass_comp["cod"].fix(0.01)
    m.fs.feed.properties[0].flow_mass_comp["hydrogen"].fix(1e-8)
    m.fs.feed.properties[0].flow_mass_comp["methane"].fix(1e-8)
    # solve(m.fs.feed)

    # metab_hydrogen
    m.fs.metab_hydrogen.load_parameters_from_database(use_default_removal=True)

    # metab_methane
    m.fs.metab_methane.load_parameters_from_database(use_default_removal=True)


def initialize_system(m):
    seq = SequentialDecomposition()
    seq.options.tear_set = []
    seq.options.iterLim = 1
    seq.run(m, lambda u: u.initialize())


def solve(blk, solver=None, tee=False, check_termination=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)
    return results


def display_results(m):
    unit_list = [
        "feed",
        "metab_hydrogen",
        "metab_methane"
    ]
    for u in unit_list:
        m.fs.component(u).report()


def add_costing(m):
    source_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "metab_global_costing.yaml",
    )
    m.fs.costing = ZeroOrderCosting(
        default={"case_study_definition": source_file}
    )
    # typing aid
    costing_kwargs = {"default": {"flowsheet_costing_block": m.fs.costing}}
    m.fs.metab_hydrogen.costing = UnitModelCostingBlock(**costing_kwargs)
    m.fs.metab_methane.costing = UnitModelCostingBlock(**costing_kwargs)

    m.fs.costing.cost_process()
    m.fs.costing.add_electricity_intensity(m.fs.product_H2O.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product_H2O.properties[0].flow_vol)


def initialize_costing(m):
    m.fs.costing.initialize()


def display_costing(m):
    m.fs.costing.total_capital_cost.display()
    m.fs.costing.total_operating_cost.display()
    m.fs.costing.LCOW.display()

    print("\nUnit Capital Costs\n")
    for u in m.fs.costing._registered_unit_costing:
        print(
            u.name,
            " :   ",
            value(pyunits.convert(u.capital_cost, to_units=pyunits.USD_2018)),
        )

    print("\nUtility Costs\n")
    for f in m.fs.costing.flow_types:
        print(
            f,
            " :   ",
            value(
                pyunits.convert(
                    m.fs.costing.aggregate_flow_costs[f],
                    to_units=pyunits.USD_2018 / pyunits.year,
                )
            ),
        )

    print("")
    total_capital_cost = value(
        pyunits.convert(m.fs.costing.total_capital_cost, to_units=pyunits.MUSD_2018)
    )
    print(f"Total Capital Costs: {total_capital_cost:.2f} M$")
    total_operating_cost = value(
        pyunits.convert(
            m.fs.costing.total_operating_cost, to_units=pyunits.MUSD_2018 / pyunits.year
        )
    )
    print(f"Total Operating Costs: {total_operating_cost:.2f} M$/year")
    electricity_intensity = value(
        pyunits.convert(
            m.fs.costing.electricity_intensity, to_units=pyunits.kWh / pyunits.m**3
        )
    )
    print(f"Electricity Intensity: {electricity_intensity:.4f} kWh/m^3")
    LCOW = value(
        pyunits.convert(m.fs.costing.LCOW, to_units=pyunits.USD_2018 / pyunits.m**3)
    )
    print(f"Levelized Cost of Water: {LCOW:.4f} $/m^3")


if __name__ == "__main__":
    m, results = main()
