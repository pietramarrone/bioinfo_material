import pandas as pd
import numpy as np
import os
import re
import time
from IPython.display import clear_output
from IPython.display import display

standard_formats = {'df': 'pd.DataFrame()', 'dict': 'dict()', 'list': 'list()'}

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


def human_size(nb):
    i = 0
    while nb >= 1024 and i < len(suffixes)-1:
        nb /= 1024.
        i += 1
    f = ('%.2f' % nb).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


report_tuple_mathys = ("PCs", "explained_var", "no_AD_mean", "early_AD_mean", "late_AD_mean",
                       "no_AD_var", "early_AD_var", "late_AD_var",
                       "pval-no_AD-vs-early_AD", "t_stat-no_AD-vs-early_AD",
                       "pval-no_AD-vs-late_AD", "t_stat-no_AD-vs-late_AD",
                       "pval-early_AD-vs-late_AD", "t_stat-early_AD-vs-late_AD",
                       "Amyloid level", "Neuritic Plaque Burden", "Neurofibrillary Tangle Burden",
                       "Tangle Density", "Global Cognitive Function", "Global AD-pathology Burden", "Age")

report_tuple_grubman = ("PCs", "explained_var", "no_AD_mean", "late_AD_mean",
                       "no_AD_var", "late_AD_var",
                       "pval-no_AD-vs-late_AD", "t_stat-no_AD-vs-late_AD", "Age")


def create_empty_report(dataset: str = 'mathys', statistics: str = 't-test'):

    assert dataset.lower() in ['mathys', 'grubman'], "WRONG DATASET NAME!"

    indices = report_tuple_mathys if dataset == 'mathys' else report_tuple_grubman
    if statistics == 'mann-whitney-u':
        indices = list(map(lambda x: x.replace('t_stat', 'u_stat'), list(indices)))
    elif statistics != 't-test':
        raise NotImplementedError("Only mann-whitney U test and two-sample independent t-test implemented")
    return pd.DataFrame(data=None, index=indices)


def find_overlap_in_dataset(data_in: np.ndarray, list_1: pd.Index, list_2: list):
    if not isinstance(list_1, pd.Index):
        raise TypeError("{} is supposed to be of type [pd.Index]. "
                        "Wrong format instead: {}".format(list_1, type(list_1)))
    else:
        overlap = list(set(list_1.tolist()).intersection(list_2))
    found_bool = list_1.isin(overlap)
    return found_bool, data_in.X[:, found_bool]


def exclude_non_variable_features(data_in: np.ndarray):
    var1 = data_in.sum(axis=0) != 0  # genes in adata
    var2 = data_in.sum(axis=1) != 0  # samples in adata
    tmp_adata = data_in[var2, :]
    tmp_adata = tmp_adata[:, var1]
    return var2, tmp_adata


def create_status_bar():
    block, progress, mins = 0, 0, 0
    start = time.time()
    dh = display("       Progress: [{}] {:.1f}% | {:.2f} mins elapsed".format("#" * block + "-" * (20 - block),
                                                                              progress * 100, mins),
                 display_id=True)
    return dh, start


def update_progress(progress, disp_inst, init_time, bar_len: int = 20):
    bar_length = bar_len
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    current_time = time.time()
    elapsed_time = (current_time - init_time)/60.
    # clear_output(wait=True)
    text = "       Progress: [{}] {:.1f}% | {:.2f} mins elapsed".format("#" * block + "-" * (bar_length - block),
                                                                        progress * 100, elapsed_time)
    disp_inst.update(text)














