#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for decentralized regression
(normal equation) including decentralized statistic calculation
"""
import sys
import warnings

import numpy as np
import pandas as pd
from local_ancillary import (add_site_covariates, ignore_nans,
                             local_stats_to_dict_fsl)
import parsers
import regression as reg
import utils as ut

warnings.simplefilter("ignore")


def local_0(args):
    ut.log(f'\n\nlocal_0() method input: {str(args["input"])} ', args["state"])

    input_list = args["input"]
    lamb = input_list["lambda"]


    categorical_dict = parsers.parse_for_categorical(args)
    
    (X, y) = parsers.fsl_parser(args)
    reference_dict = {}
    if "reference_columns" in input_list:
        reference_dict = dict((k, v.lower()) for k,v in input_list["reference_columns"].items());

    output_dict = {"computation_phase": "local_0", "categorical_dict": categorical_dict, "reference_columns": reference_dict}

    cache_dict = {
        "covariates": X.to_json(orient='split'),
        "dependents": y.to_json(orient='split'),
        "lambda": lamb,
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
    ut.log(f'\nlocal_0() method output: {str(computation_output)} ', args["state"])

    return computation_output


def local_1(args):
    ut.log(f'\nlocal_1() method input: {str(args["input"])} ', args["state"])

    X = pd.read_json(args["cache"]["covariates"], orient='split')
    y = pd.read_json(args["cache"]["dependents"], orient='split')

    dependents=y

    lamb = args["cache"]["lambda"]

    y_labels = list(y.columns)

    #ADDED : PERFORMED DUMMY ENCODING WITH REFERENCE COLUMN VALUES
    encoded_X = parsers.perform_encoding(args, X)

    ut.log(f'\ncalling local_stats_to_dict_fsl() with X: {str(np.shape(encoded_X))} with columns:'
           f' {str(encoded_X.columns.to_list())}', args["state"])
    ut.log(f'\ncalling local_stats_to_dict_fsl() with Y: {str(np.shape(y))}', args["state"])

    t = local_stats_to_dict_fsl(encoded_X, y)
    _, local_stats_list, meanY_vector, lenY_vector = t


    #ADDED: Edited add_site_covariates with dummy encoding as in VBM regression code
    augmented_X = add_site_covariates(args, X)

    X_labels = list(augmented_X.columns)

    #ADDED: added the .astype("float64")
    biased_X = augmented_X.values.astype("float64")


    y = y.values # another hack


    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), y)

    output_dict = {
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "y_labels": y_labels,
        "lambda": lamb,
        "computation_phase": "local_1",
    }


    cache_dict = {
        "covariates": augmented_X.to_json(orient='split'),
        "dependents": dependents.to_json(orient='split')
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
    ut.log(f'\nlocal_1() method output: {str(computation_output)} ', args["state"])

    return computation_output


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local
    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }
    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }
    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    ut.log(f'\nlocal_2() method input: {str(args["input"])} ', args["state"])

    cache_list = args["cache"]
    input_list = args["input"]

    X = pd.read_json(cache_list["covariates"], orient='split')

    y = pd.read_json(cache_list["dependents"], orient='split')

    #ADDED: this code to drop nan's from the covaraites and dependents
    X=X.dropna()
    y=y.dropna()

    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local, SST_local, varX_matrix_local = [], [], []

    for index, column in enumerate(y.columns):
        curr_y = y[column]
        '''
        Removed this code: The following code does not take objects datatype. 
        Removed this code as we dropna() in the above anyways
        X_, y_ = ignore_nans(biased_X, curr_y)
        '''
        SSE_local.append(
            reg.sum_squared_error(biased_X, curr_y, np.array(avg_beta_vector[index])))
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[index]))))

        varX_matrix_local.append(np.dot(biased_X.T, biased_X).tolist())

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local,
        "computation_phase": 'local_2'
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}
    ut.log(f'\nlocal_2() method output: {str(computation_output)} ', args["state"])


    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(reg.list_recursive(PARAM_DICT, "computation_phase"))

    if not PHASE_KEY:
        return local_0(PARAM_DICT)
    elif "remote_0" in PHASE_KEY:
        return local_1(PARAM_DICT)
    elif "remote_1" in PHASE_KEY:
        return local_2(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")
