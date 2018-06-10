#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import itertools
import numpy as np
import pandas as pd
import regression as reg
import sys
import ujson as json
from remote_ancillary import get_stats_to_dict


def remote_0(args):
    input_list = args["input"]
    missing_columns = [
        input_list[site]['missing_columns'] for site in input_list
    ]

    missing_columns = list(
        set(list(itertools.chain.from_iterable(missing_columns))))

    site_ids = list(input_list.keys())
    site_covar_list = [
        '{}_{}'.format('site', label) for index, label in enumerate(site_ids)
        if index
    ]

    output_dict = {
        "missing_columns": missing_columns,
        "site_covar_list": site_covar_list,
        "computation_phase": "remote_0"
    }

    cache_dict = {"missing_columns": missing_columns}

    computation_output_dict = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output_dict)


def remote_1(args):
    input_list = args["input"]
    userID = list(input_list)[0]

    X_labels = input_list[userID]["X_labels"]
    y_labels = input_list[userID]["y_labels"]

    all_local_stats_dicts = [
        input_list[site]["local_stats_list"] for site in input_list
    ]

    XtransposeX_local = [
        np.array(input_list[site]["XtransposeX_local"]) for site in input_list
    ]

    XtransposeX = sum(XtransposeX_local)

    all_lambdas = [input_list[site]["lambda"] for site in input_list]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    XtransposeX = XtransposeX + np.unique(all_lambdas) * np.eye(
        XtransposeX.shape[0])
    XtransposeX_inv = np.linalg.inv(XtransposeX)

    Xtransposey_local = [
        pd.read_json(input_list[site]["Xtransposey_local"], orient='split')
        .values for site in input_list
    ]

    avg_beta_vector = sum(np.nan_to_num(XtransposeX_inv @ Xtransposey_local)).T

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [input_list[site]["count_local"] for site in input_list]

    mean_y_global = np.ma.average(
        np.ma.masked_array(mean_y_local, not np.nonzero(mean_y_local)),
        weights=count_y_local,
        axis=0).filled(0)

    dof_global = np.subtract(
        np.sum(count_y_local, axis=0),
        [len(vec) if np.count_nonzero(vec) else 0 for vec in avg_beta_vector])

    output_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        "y_labels": y_labels,
        "local_stats_dict": all_local_stats_dicts
    }

    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)


def remote_2(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]
    cache_list = args["cache"]

    X_labels = cache_list["X_labels"]
    y_labels = cache_list["y_labels"]
    dof_global = cache_list["dof_global"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    all_local_stats_dicts = cache_list["local_stats_dict"]

    SSE_local = [input_list[site]["SSE_local"] for site in input_list]
    SSE_global = [
        np.sum(list(filter(None, elem)), axis=0) for elem in zip(*SSE_local)
    ]

    SST_local = [input_list[site]["SST_local"] for site in input_list]
    SST_global = [
        np.sum(list(filter(None, elem)), axis=0) for elem in zip(*SST_local)
    ]

    varX_matrix_local = [
        input_list[site]["varX_matrix_local"] for site in input_list
    ]
    varX_matrix_global = [
        np.sum(list(filter(None, elem)), axis=0)
        for elem in zip(*varX_matrix_local)
    ]

    r_squared_global = 1 - np.divide(SSE_global, SST_global)
    MSE = np.divide(SSE_global, dof_global)

    ts_global = []
    ps_global = []

    keys1 = [
        "avg_beta_vector", "r2_global", "ts_global", "ps_global", "dof_global",
        "covariate_labels"
    ]
    global_dict_list = []

    for i in range(len(MSE)):
        if not np.isnan(MSE[i]):
            var_covar_beta_global = MSE[i] * np.linalg.inv(
                varX_matrix_global[i])
            se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
            ts = (avg_beta_vector[i] / se_beta_global).tolist()
            ps = reg.t_to_p(ts, dof_global[i])
            ts_global.append(ts)
            ps_global.append(ps)

            vals = [
                avg_beta_vector[i], r_squared_global[i], ts, ps, dof_global[i],
                X_labels
            ]
            global_dict_list.append(dict(zip(keys1, vals)))
        else:
            global_dict_list.append({})

    # Block of code to print local stats as well
    sites = [site for site in input_list]
    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts)))

    a_dict = [{key: value
               for key, value in zip(sites, stats_dict)}
              for stats_dict in all_local_stats_dicts]

    # Print Everything
    keys2 = ["ROI", "global_stats", "local_stats"]
    dict_list = get_stats_to_dict(keys2, y_labels, global_dict_list, a_dict)

    if args["cache"]["missing_columns"]:
        output_dict = {
            "regressions": dict_list,
            "missing_ROI": args["cache"]["missing_columns"]
        }
    else:
        output_dict = {"regressions": dict_list}

    computation_output = {"output": output_dict, "success": True}

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.list_recursive(parsed_args, 'computation_phase'))

    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_2" in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
