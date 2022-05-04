from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors
from pyomo.util.check_units import assert_units_consistent
from watertap.core.util.initialization import assert_degrees_of_freedom
import ion_prop_pack as props

def main():
    # create model, flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    # attach property package
    m.fs.properties = props.PropParameterBlock(
        default={'ion_list': ['Na_+', 'Cl_-', 'Mg_2+'],
                 'mw_data': {
                     'H2O': 18.0e-3,
                     'Na_+': 23.0e-3,
                     'Cl_-': 35.5e-3,
                     'Mg_2+': 24.3e-3}
                 }
    )
    # build a state block, must specify a time which by convention for steady state models is just 0
    m.fs.stream = m.fs.properties.build_state_block([0], default={})
    m.fs.stream[0].mass_frac_comp
    m.fs.stream[0].flow_vol
    m.fs.stream[0].conc_mass_comp
    m.fs.stream[0].flow_mol_comp

    m.fs.properties.set_default_scaling("flow_mass_comp", 1, index="H2O")
    m.fs.properties.set_default_scaling("flow_mass_comp", 1e3, index="Mg_2+")
    calculate_scaling_factors(m)
    # m.fs.stream[0].scaling_factor.display()
    # m.fs.stream[0].constraint_transformed_scaling_factor.display()

    m.fs.stream[0].flow_mass_comp["H2O"].fix(1)
    m.fs.stream[0].flow_mass_comp["Na_+"].fix(0.1)
    m.fs.stream[0].flow_mass_comp["Cl_-"].fix(0.15)
    m.fs.stream[0].flow_mass_comp["Mg_2+"].fix(0.01)

    assert_degrees_of_freedom(m, 0)
    assert_units_consistent(m)

    m.fs.stream.initialize()

    m.fs.stream.display()


if __name__ == "__main__":
    main()