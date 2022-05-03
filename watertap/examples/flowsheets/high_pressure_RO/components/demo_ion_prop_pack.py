from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
from idaes.core.util.scaling import calculate_scaling_factors

import ion_prop_pack as props

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

# display the state block, it only has the state variables and they are all unfixed
print('\n---first display---')
m.fs.stream[0].display()

m.fs.properties.set_default_scaling("flow_mass_comp", 1, index="H2O")
m.fs.properties.set_default_scaling("flow_mass_comp", 1e4, index="Mg_2+")

calculate_scaling_factors(m)

m.fs.stream[0].scaling_factor.display()
m.fs.stream[0].constraint_transformed_scaling_factor.display()