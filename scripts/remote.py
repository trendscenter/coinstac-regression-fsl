#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for decentralized
regression with decentralized statistic calculation
"""
import sys
from itertools import repeat

import numpy as np
import scipy as sp
import ujson as json
import pandas as pd

import regression as reg
from remote_ancillary import *
import jsonpickle


def remote_0(args):
    input_list = args["input"]

    site_ids = list(input_list.keys())
    site_covar_list = [
        '{}_{}'.format('site', label) for index, label in enumerate(site_ids)
        if index
    ]
    site_info = {site: input_list[site]["categorical_dict"] for site in input_list.keys()}
    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    ref_cols = {site: input_list[site]["reference_columns"] for site in input_list.keys()}
    reference_dict = next(iter(ref_cols.values()))

    output_dict = {
        "site_covar_list": site_covar_list,
        "covar_keys": jsonpickle.encode(covar_keys, unpicklable=False),
        "global_unique_count": unique_count,
        "reference_columns": reference_dict,
        "computation_phase": "remote_0"
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}

    return computation_output


def remote_1(args):
    input_list = args["input"]
    userID = list(input_list)[0]

    X_labels = input_list[userID]["X_labels"]
    y_labels = input_list[userID]["y_labels"]

    all_local_stats_dicts = [
        input_list[site]["local_stats_list"] for site in input_list
    ]

    beta_vector_0 = [
        np.array(input_list[site]["XtransposeX_local"]) for site in input_list
    ]

    beta_vector_1 = sum(beta_vector_0)

    all_lambdas = [input_list[site]["lambda"] for site in input_list]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(
        beta_vector_1.shape[0])

    avg_beta_vector = np.matrix.transpose(
        sum([
            np.matmul(sp.linalg.inv(beta_vector_1),
                      input_list[site]["Xtransposey_local"])
            for site in input_list
        ]))

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [
        np.array(input_list[site]["count_local"]) for site in input_list
    ]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local,
                                                           axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

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

    return computation_output


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

    X_labels = args["cache"]["X_labels"]
    y_labels = args["cache"]["y_labels"]
    all_local_stats_dicts = args["cache"]["local_stats_dict"]

    cache_list = args["cache"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = sum(
        [np.array(input_list[site]["SSE_local"]) for site in input_list])
    SST_global = sum(
        [np.array(input_list[site]["SST_local"]) for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)

    ts_global = []
    ps_global = []

    for i in range(len(MSE)):

        var_covar_beta_global = MSE[i] * sp.linalg.inv(varX_matrix_global[i])
        se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
        ts = (avg_beta_vector[i] / se_beta_global).tolist()
        ps = reg.t_to_p(ts, dof_global[i])
        ts_global.append(ts)
        ps_global.append(ps)

    # Block of code to print local stats as well
    sites = [site for site in input_list]

    all_local_stats_dicts = list(map(list, zip(*all_local_stats_dicts)))

    a_dict = [{key: value
               for key, value in zip(sites, stats_dict)}
              for stats_dict in all_local_stats_dicts]

    # Block of code to print just global stats
    keys1 = [
        "Coefficient", "R Squared", "t Stat", "P-value", "Degrees of Freedom",
        "covariate_labels"
    ]
    global_dict_list = get_stats_to_dict(keys1, avg_beta_vector,
                                         r_squared_global, ts_global,
                                         ps_global, dof_global,
                                         repeat(X_labels, len(y_labels)))

    # Print Everything
    keys2 = ["ROI", "global_stats", "local_stats"]
    dict_list = get_stats_to_dict(keys2, y_labels, global_dict_list, a_dict)

    output_dict = {"regressions": dict_list}

    computation_output = {"output": output_dict, "success": True}

    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(reg.list_recursive(PARAM_DICT, "computation_phase"))

    if "local_0" in PHASE_KEY:
        return remote_0(PARAM_DICT)
    elif "local_1" in PHASE_KEY:
        return remote_1(PARAM_DICT)
    elif "local_2" in PHASE_KEY:
        return remote_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")
