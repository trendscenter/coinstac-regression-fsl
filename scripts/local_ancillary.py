#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:28:11 2018

@author: Harshvardhan
"""
import warnings
import json

import numpy as np
import pandas as pd
import scipy as sp

import statsmodels.api as sm
from numba import jit, prange
from parsers import *

warnings.simplefilter("ignore")



def mean_and_len_y(y):
    """Caculate the length mean of each y vector"""
    meanY_vector = y.mean(axis=0).tolist()
    lenY_vector = y.count(axis=0).tolist()

    return meanY_vector, lenY_vector


@jit(nopython=True)
def gather_local_stats(X, y):
    """Calculate local statistics"""
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        curr_y = y[:, voxel]
        beta_vector = np.linalg.inv(X.T @ X) @ (X.T @ curr_y)
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate)**2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(X.T @ X)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


def local_stats_to_dict_vbm(X, y):
    """Wrap local statistics into a dictionary to be sent to the remote"""
    X1 = sm.add_constant(X).values.astype('float64')
    y1 = y.values.astype('float64')

    params, sse, tvalues, rsquared, dof_global = gather_local_stats(X1, y1)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    keys = [
        "Coefficient", "Sum Square of Errors", "t Stat", "P-value", "R Squared"
    ]

    values1 = pd.DataFrame(list(
        zip(params.T.tolist(), sse.tolist(), tvalues.T.tolist(),
            pvalues.T.tolist(), rsquared.tolist())),
                           columns=keys)

    local_stats_list = values1.to_dict(orient='records')

    beta_vector = params.T.tolist()

    return beta_vector, local_stats_list


def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_


def local_stats_to_dict_fsl(X, y):
    """Calculate local statistics"""
    y_labels = list(y.columns)

    biased_X = sm.add_constant(X)
    X_labels = list(biased_X.columns)

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []
    meanY_vector, lenY_vector = [], []

    biased_X = biased_X.apply(pd.to_numeric, errors="ignore")
    biased_X = pd.get_dummies(biased_X, drop_first=True)
    biased_X = biased_X * 1

    for column in y.columns:
        curr_y = y[column]

        X_, y_ = ignore_nans(biased_X, curr_y)
        meanY_vector.append(np.mean(y_))
        lenY_vector.append(len(y_))

        # Printing local stats as well
        model = sm.OLS(y_, X_).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared)

    keys = [
        "Coefficient", "Sum Square of Errors", "t Stat", "P-value",
        "R Squared", "covariate_labels"
    ]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_tvalues[index].tolist(), local_pvalues[index].tolist(),
            local_rsquared[index], X_labels
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)

        beta_vector = [l.tolist() for l in local_params]

    return beta_vector, local_stats_list, meanY_vector, lenY_vector


def add_site_covariates_old(args, X):
    """Add site specific columns to the covariate matrix"""
    biased_X = sm.add_constant(X)
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros((np.array(X).shape[0], len(site_covar_list)),
                           dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns
        if args["state"]["clientId"] in col[len('site_'):]
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)

    return augmented_X

#ADDED This function is as in VBM regression
def add_site_covariates(args, X):
    """Add site covariates based on information gathered from all sites"""
    input_ = args["input"]
    all_sites = input_["covar_keys"]
    glob_uniq_ct = input_["global_unique_count"]
    
    reference_col_dict= input_["reference_columns"]

    all_sites = json.loads(all_sites)

    default_col_sortedval_dict = get_default_dummy_encoding_columns(X)

    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
            default_col_sortedval_dict.pop(key)
        else:
            default_col_sortedval_dict[key] = sorted(all_sites[key])[0]
            covar_dict = pd.get_dummies(all_sites[key], prefix=key, drop_first=False)
            X = merging_globals(args, X, covar_dict, all_sites, key)

    X = adjust_dummy_encoding_columns(X, reference_col_dict, default_col_sortedval_dict)

    X = X.dropna(axis=0, how="any")
    biased_X = sm.add_constant(X, has_constant="add")

    return biased_X


def merging_globals(args, X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix"""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X
