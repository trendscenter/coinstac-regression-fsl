#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""
import pandas as pd


def get_stats_to_dict(a, *b):
    df = pd.DataFrame(list(zip(*b)), columns=a)
    dict_list = df.to_dict(orient='records')

    return dict_list

def return_uniques_and_counts(df):
    """Return unique-values of the categorical variables and their counts"""
    keys, count = dict(), dict()
    keys = (
        df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()
    )  # adding all columns
    count = {k: len(v) for k, v in keys.items()}

    return keys, count
