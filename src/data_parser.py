import pandas as pd
import numpy as np
import os
import re
import time
from IPython.display import clear_output
from IPython.display import display
import bioinfo as mybio

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


def save_loadings(loadings_dfs: dict,
                  annotation_dict: dict,
                  clusters_list: list,
                  subclusters_dict: dict,
                  annotation_label: str, **kwargs):
    """
    Write the loadings from the pathifier to a structured excel file

    Parameters
    ----------
    loadings_dfs: dictionary of dataframes containing loadings
    annotation_dict: dictinary containing the annotations
    clusters_list: list of cell clusters (samples)
    subclusters_dict: dictionary of subtypes within the cell types
    annotation_label: label of the annotation database (kegg and reactome accepted for now, but can be expanded)

    Returns
    -------
    Writes to file
    """

    assert annotation_label in ('kegg', 'reactome')
    out_folder_name = kwargs.get('out_dir', '.')

    if annotation_label == 'kegg':
        test_path = [entry[0] for entry in list(annotation_dict.values())]  # paths are stored per path name not kegg id
    elif annotation_label == 'reactome':
        anno_df = kwargs.get('annotation_dict', None)
        assert anno_df is not None
        test_path = list(map(lambda x: mybio.get_reactome_stable_id_from_path(anno_df, x), annotation_dict))

    for c in clusters_list:
        for subc in subclusters_dict[c]:
            print(f'Saving Loadings for [{subc}]')
            if loadings_dfs[subc]:
                with pd.ExcelWriter(f'./{out_folder_name}/{c}/Loadings_{subc}_{annotation_label}.xlsx',
                                    engine='xlsxwriter') as writer:
                    dh, t_0 = create_status_bar()
                    for i, path in enumerate(loadings_dfs[subc].keys()):
                        if annotation_label == 'kegg':
                            path_idx = test_path.index(path)
                            hsa_path = list(annotation_dict.keys())[path_idx]
                        elif annotation_label == 'reactome':
                            # path_idx = reactome_er_paths.tolist().index(path)
                            # hsa_path = test_path[path_idx]
                            hsa_path = mybio.get_reactome_stable_id_from_path(anno_df, path)

                        var_df = pd.DataFrame(data=np.reshape(loadings_dfs[subc][path]['variance_fraction'], (1, -1)),
                                              columns=['PC_{}'.format(i)
                                                       for i in
                                                       range(1, len(loadings_dfs[subc][path]['variance_fraction']) + 1)],
                                              index=['Explained Variance [Fraction]'])
                        file_header = pd.DataFrame(data=np.reshape([hsa_path, path], (1, -1)),
                                                   columns=['KEGG_ID', 'Pathway_name'])
                        file_header.to_excel(writer, sheet_name=hsa_path, index=False)
                        var_df.to_excel(writer, sheet_name=hsa_path, startrow=3)
                        loadings_dfs[subc][path]['loadings'].index.name = 'Gene Name'
                        loadings_dfs[subc][path]['loadings'].to_excel(writer, sheet_name=hsa_path, startrow=6)
                        loadings_dfs[subc][path]['overall_ranking'].index.name = 'Gene Name'
                        loadings_dfs[subc][path]['overall_ranking'].to_excel(writer, sheet_name=hsa_path,
                                                                             startrow=6, startcol=var_df.shape[1] + 2)

                        update_progress(i / len(loadings_dfs[subc].keys()), disp_inst=dh, init_time=t_0)
                    writer.save()
        update_progress(1, disp_inst=dh, init_time=t_0)










