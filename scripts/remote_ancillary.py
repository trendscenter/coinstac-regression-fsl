#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:56:41 2018

@author: Harshvardhan
"""
import pandas as pd
from ancillary import DummyEncodingReferenceOrder

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

def get_dummy_encoding_reference_dict(covar_keys, input_args):
    from collections import Counter

    reference_dict = None
    encoding_type = DummyEncodingReferenceOrder.from_str(input_args["local0"]["dummy_encoding_reference_order"])
    if encoding_type == DummyEncodingReferenceOrder.CUSTOM:
        ref_cols = {site: input_args[site]["reference_columns"] for site in input_args.keys()}
        reference_dict = next(iter(ref_cols.values()))

    else:
        reference_dict = {}

        for column in covar_keys.keys():
            if encoding_type == DummyEncodingReferenceOrder.SORTED_LAST:
                reference_dict[column] = sorted(covar_keys[column], reverse=True)[0]
            elif encoding_type == DummyEncodingReferenceOrder.MOST_FREQUENT:
                merged_freq_dicts={}
                for site in input_args.keys():
                    merged_freq_dicts = dict(Counter(merged_freq_dicts) +
                                             Counter(input_args[site]["categorical_column_frequency_dict"][column]))
                reference_dict[column] = max(merged_freq_dicts, key=merged_freq_dicts.get)

            elif encoding_type == DummyEncodingReferenceOrder.LEAST_FREQUENT:
                merged_freq_dicts={}
                for site in input_args.keys():
                    merged_freq_dicts = dict(Counter(merged_freq_dicts) +
                                             Counter(input_args[site]["categorical_column_frequency_dict"][column]))
                reference_dict[column] = min(merged_freq_dicts, key=merged_freq_dicts.get)

            #else: it means
            # encoding_type == DummyEncodingReferenceOrder.SORTED_FIRST
            # so we do nothing as this is the default behavior in pandas dummy encoding
        #print(f"Reference encoding:{encoding_type.value} \nReference dict: {str(reference_dict)}")
    return reference_dict