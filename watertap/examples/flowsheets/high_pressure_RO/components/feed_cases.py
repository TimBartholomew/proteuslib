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

from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Feed
from idaes.core.util.scaling import calculate_scaling_factors
from watertap.examples.flowsheets.high_pressure_RO.components import ion_prop_pack


def build_prop(m, case="seawater"):
    """
    Builds a property package for the specified case.
    cases include: 'seawater'
    """
    if case in ["seawater", "brackish_1", "brackish_2"]:
        ion_list = ["NAION", "KION", "MGION", "CAION", "CLION", "SO4ION", "HCO3ION"]
        m.fs.prop_feed = ion_prop_pack.PropParameterBlock(
            default={"ion_list": ion_list}
        )
    else:
        raise ValueError(
            "Unexpected feed case {case} provided to build_prop" "".format(case=case)
        )


def build_specified_feed(m, case="seawater"):
    """
    Build a specified feed block for the given case. The state vars are fixed to the standard condition.
    Feed cases include: 'seawater',
    """

    # build property block
    build_prop(m, case=case)

    # build
    m.fs.feed = Feed(default={"property_package": m.fs.prop_feed})

    # specify
    specify_feed(m.fs.feed.properties[0], case=case)


def specify_feed(sb, case="seawater"):
    """
    Fixes the state variables on the stateblock to the feed case.
    Feed cases include: 'seawater'
    """
    sb.pressure.fix(101325)
    sb.temperature.fix(298.15)
    sb.flow_vol.fix(1e-3)

    if case == "seawater":  # Lienhard and Lenntech
        conc_dict = {
            "NAION": 10.556,
            "KION": 0.380,
            "CAION": 0.400,
            "MGION": 1.262,
            "CLION": 18.973,
            "SO4ION": 2.649,
            "HCO3ION": 0.140,
        }
        for (j, val) in conc_dict.items():
            sb.conc_mass_comp[j].fix(val)
    elif case == "brackish_1":  # Lienhard
        conc_dict = {
            "NAION": 0.739,
            "KION": 0.009,
            "CAION": 0.258,
            "MGION": 0.090,
            "CLION": 0.896,
            "SO4ION": 1.011,
            "HCO3ION": 0.385,
        }
        for (j, val) in conc_dict.items():
            sb.conc_mass_comp[j].fix(val)
    elif case == "brackish_2":  # WT3
        conc_dict = {
            "NAION": 0.77,
            "KION": 0.016,
            "CAION": 0.13,
            "MGION": 0.03,
            "CLION": 1.18,
            "SO4ION": 0.23,
            "HCO3ION": 0.292,
        }
        for (j, val) in conc_dict.items():
            sb.conc_mass_comp[j].fix(val)
    else:
        raise ValueError(
            "Unexpected feed case {case} provided to specify_feed" "".format(case=case)
        )
