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
from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Expression,
    Objective,
    Param,
    TransformationFactory,
    units,
    assert_optimal_termination,
)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import solve_indexed_blocks, propagate_state
from idaes.models.unit_models import Mixer, Separator, Product, Feed
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

import watertap.property_models.seawater_prop_pack as properties
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.costing import WaterTAPCosting


def main():
    # flowsheet setup
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.properties = properties.SeawaterParameterBlock()
    m.fs.costing = WaterTAPCosting()  # costing model

    # unit models
    m.fs.feed = Feed(default={"property_package": m.fs.properties})
    m.fs.pump = Pump(default={"property_package": m.fs.properties})
    m.fs.RO = ReverseOsmosis0D(default={
        "property_package": m.fs.properties,
        "has_pressure_change": True,
        "pressure_change_type": PressureChangeType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "concentration_polarization_type": ConcentrationPolarizationType.calculated})
    m.fs.erd = EnergyRecoveryDevice(default={"property_package": m.fs.properties})
    m.fs.product = Product(default={"property_package": m.fs.properties})
    m.fs.disposal = Product(default={"property_package": m.fs.properties})

    # unit costing - equipment capital and operating costs
    # m.fs.pump.work_mechanical[0].setlb(0)
    m.fs.pump.costing = UnitModelCostingBlock(default={"flowsheet_costing_block": m.fs.costing})
    m.fs.RO.costing = UnitModelCostingBlock(default={"flowsheet_costing_block": m.fs.costing})
    m.fs.erd.costing = UnitModelCostingBlock(default={
        "flowsheet_costing_block": m.fs.costing,
        "costing_method_arguments": {"energy_recovery_device_type": "pressure_exchanger"}})

    # system costing - total investment and operating costs
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)

    # connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.s02 = Arc(source=m.fs.pump.outlet, destination=m.fs.RO.inlet)
    m.fs.s03 = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)
    m.fs.s04 = Arc(source=m.fs.RO.retentate, destination=m.fs.erd.inlet)
    m.fs.s05 = Arc(source=m.fs.erd.outlet, destination=m.fs.disposal.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # scaling
    # set default property values
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "TDS"))
    # set unit model values
    iscale.set_scaling_factor(m.fs.pump.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.erd.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.RO.area, 1e-2)
    # touch properties used in specifying the model
    m.fs.feed.properties[0].flow_vol_phase["Liq"]
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.disposal.properties[0].flow_vol_phase["Liq"]
    m.fs.disposal.properties[0].mass_frac_phase_comp["Liq", "TDS"]
    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)


    # feed, 4 degrees of freedom
    m.fs.feed.properties[0].flow_vol_phase["Liq"].fix(1e-3)                # volumetric flow rate (m3/s)
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].fix(0.035)  # TDS mass fraction (-)
    m.fs.feed.properties[0].pressure.fix(101325)  # pressure (Pa)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # temperature (K)
    # m.fs.feed.properties.calculate_state(
    #     var_args={
    #         ("flow_vol_phase", "Liq"): 1e-3,  # feed volumetric flow rate [m3/s]
    #         ("mass_frac_phase_comp", ("Liq", "TDS")): 0.035,
    #     },  # feed TDS mass fraction [-]
    #     hold_state=True,  # fixes the calculated component mass flow rates
    # )

    # high pressure pump, 2 degrees of freedom
    m.fs.pump.efficiency_pump.fix(0.80)  # pump efficiency (-)
    m.fs.pump.control_volume.properties_out[0].pressure.fix(75e5)  # pump outlet pressure (Pa)

    # RO unit, 7 degrees of freedom
    m.fs.RO.A_comp.fix(4.2e-12)  # membrane water permeability coeff (m/Pa/s)
    m.fs.RO.B_comp.fix(3.5e-8)  # membrane salt permeability coeff (m/s)
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(0.5)  # volumetric recovery (-) *
    m.fs.RO.velocity[0, 0].fix(0.15)  # crossflow velocity (m/s) *
    m.fs.RO.channel_height.fix(1e-3)  # channel height in membrane stage (m)
    m.fs.RO.spacer_porosity.fix(0.97)  # spacer porosity in membrane stage (-)
    m.fs.RO.permeate.pressure[0].fix(101325)  # permeate pressure (Pa)

    # energy recovery device, 2 degrees of freedom
    m.fs.erd.efficiency_pump.fix(0.80)  # erd efficiency (-)
    m.fs.erd.control_volume.properties_out[0].pressure.fix(101325)  # ERD outlet pressure (Pa)

    print("DOF = ", degrees_of_freedom(m))

    # initialize unit by unit
    solver = get_solver()
    solver.solve(m.fs.feed)
    propagate_state(m.fs.s01)
    m.fs.pump.initialize()
    propagate_state(m.fs.s02)
    m.fs.RO.initialize()
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s04)
    m.fs.erd.initialize()
    propagate_state(m.fs.s05)

    # initialize cost
    m.fs.costing.initialize()

    # solve model
    results = solver.solve(m)
    print(results)


if __name__ == "__main__":
    main()
