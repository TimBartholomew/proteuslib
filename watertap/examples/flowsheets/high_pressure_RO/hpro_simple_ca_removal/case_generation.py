import sys
import os
import time

from idaes.core.util import get_solver
from pyomo.environ import Expression, Param, Constraint
from watertap.tools.parameter_sweep import _init_mpi, LinearSample, parameter_sweep
import watertap.examples.flowsheets.high_pressure_RO.hpro_simple_ca_removal.flowsheet as flowsheet


def run_analysis(case, nx, interp_nan_outputs=False):
    sweep_params = {}
    outputs = {}
    optimize_kwargs = {"check_termination": False}

    # build, set, and initialize
    m = flowsheet.build(case=case)
    flowsheet.specify_model(m)
    flowsheet.initialize_model(m)
    # simulate
    flowsheet.solve(m)
    # set up optimize
    flowsheet.set_up_optimization(m)
    flowsheet.optimize(m)

    # set up parameter sweep
    if case == "seawater":
        lb = 0.3
        ub = 0.85
    else:
        lb = 0.3
        ub = 0.95
    sweep_params["System Recovery"] = LinearSample(m.fs.product_recovery, lb, ub, nx)
    sweep_params["Ca Removal"] = LinearSample(
        m.fs.softening.split_fraction[0, "byproduct", "CAION"], 0, 1, nx)

    for j in m.fs.disposal.properties[0].conc_mass_comp:
        outputs[j] = m.fs.disposal.properties[0].conc_mass_comp[j]

    outputs["LCOW"] = m.fs.costing.LCOW

    output_filename = "output/oli_cases/simple_ca_removal" + case + ".csv"

    opt_function = flowsheet.optimize

    global_results = parameter_sweep(
        m,
        sweep_params,
        outputs,
        csv_results_file_name=output_filename,
        optimize_function=opt_function,
        optimize_kwargs=optimize_kwargs,
        debugging_data_dir=os.path.split(output_filename)[0] + "/local",
        interpolate_nan_outputs=interp_nan_outputs,
    )

    return global_results, sweep_params


if __name__ == "__main__":

    # Start MPI communicator
    comm, rank, num_procs = _init_mpi()

    # analysis
    # case = 'seawater'
    case = 'brackish_1'
    # case = "brackish_2"
    nx = 10

    tic = time.time()
    global_results, sweep_params = run_analysis(case, nx)
    print(global_results)
    toc = time.time()

    if rank == 0:
        total_samples = 1

        for k, v in sweep_params.items():
            total_samples *= v.num_samples

        print("Finished case = %s." % case)
        print(
            "Processed %d swept parameters comprising %d total points."
            % (len(sweep_params), total_samples)
        )
        print("Elapsed time = %.1f s." % (toc - tic))
