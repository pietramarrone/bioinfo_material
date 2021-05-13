import os
import pandas as pd
from typing import Union
import numpy as np
from collections import defaultdict
import data_parser as mypars

import seaborn as sns
from matplotlib.colors import rgb2hex, colorConverter

import bioinfo as mybio


clusters = ['Opc', 'In', 'Ex', 'Mic', 'Ast', 'Oli']

cols = ['Healthy vs Early-AD', 'Healthy vs Late-AD', 'Early-AD vs Late-AD (less)', 'Early-AD vs Late-AD (greater)']
ordered_labs = ['Ast_0', 'Ast_1', 'Ast_3', 'Ast_4', 'Ast [cell type]',
                'Ex_0', 'Ex_1', 'Ex_2', 'Ex_3', 'Ex_4', 'Ex_5', 'Ex_6', 'Ex_7',
                'Ex_8', 'Ex_9', 'Ex_10', 'Ex_11', 'Ex_12', 'Ex [cell type]',
                'In_0', 'In_1', 'In_2', 'In_3', 'In_4', 'In_5', 'In_6', 'In_7', 'In [cell type]',
                'Mic_0', 'Mic_1', 'Mic_2', 'Mic_3', 'Mic_4', 'Mic [cell type]',
                'Oli_0', 'Oli_1', 'Oli_3', 'Oli_4', 'Oli [cell type]',
                'Opc_0', 'Opc_1', 'Opc_2', 'Opc_3', 'Opc [cell type]']

ct_lab = ['Ex', 'In', 'Ast', 'Mic', 'Oli', 'Opc']  # cell_types labels


def load_Xwas(dir_xwas: str = None):
    """
    load folder with the lists of genes of interest
    Returns
    -------

    """
    dir_was = dir_xwas if dir_xwas is not None else '/Users/a.possenti/OneDrive - University Of Cambridge/' \
                                                    'Python/Repositories/Gene_Lists'

    nwas = list(map(str.strip, open(f"{dir_was}/Genes_nwas.txt").readlines()))
    gwas = list(map(str.strip, open(f"{dir_was}/Genes_gwas.txt").readlines()))
    pwas = list(map(str.strip, open(f"{dir_was}/Genes_pwas.txt").readlines()))

    gwas = list(set(gwas + ['PSEN1', 'PSEN2', 'ACE', 'ADAMTS1',
                            'IQCK', 'SLC24A4', 'SPI1', 'TXNDC3', 'WWOX']))  # these were most recents addition
    # from published meta-analysis

    return nwas, gwas, pwas



def load_results_from_pathifier(path_to_res: Union[str, None], results_file_generic: str = None):
    """

    Parameters
    ----------
    path_to_res path to the results folder (assuming the folder is in the same location as the src)
    results_file_generic suffix to results file

    Returns
    -------

    """

    input_dirs = path_to_res if path_to_res is not None else os.listdir()
    results_generic_file = '_Data_KEGG.xlsx' if results_file_generic is None else results_file_generic


    results_dict = dict()

    for d in set(input_dirs).intersection(clusters):
        xls = pd.ExcelFile(os.path.join(os.getcwd(), d, ''.join([d, results_file_generic])))
        results_dict[d] = dict()
        for subc in xls.sheet_names:
            results_dict[d][subc] = pd.read_excel(xls, sheet_name=subc, index_col=0)
            results_dict[d][subc] = results_dict[d][subc].apply(pd.to_numeric)

    return results_dict


def populate_df_for_stats(results_dict: dict):
    f"""
    
    Parameters
    ----------
    results_dict: dictionary populated with the function {load_results_from_pathifier.__name__}

    Returns
    -------

    """

    stat_tests = {'Healthy vs Early-AD': 'pval-no_AD-vs-early_AD',
                  'Healthy vs Late-AD': 'pval-no_AD-vs-late_AD',
                  'Early-AD vs Late-AD (less)': 'pval-early_AD-vs-late_AD',
                  'Early-AD vs Late-AD (greater)': 'pval-early_AD-vs-late_AD'}

    kegg_p = mybio.get_keggs(path_or_file='Default')
    kegg_pathways = [kegg_p[k][0] for k in kegg_p]

    significance_dict = dict()
    for s in stat_tests:
        significance_dict[s] = dict()
        for c in clusters:
            significance_dict[s][c] = pd.DataFrame(index=kegg_pathways, columns=results_dict[c].keys(), dtype=np.float)
            for subc in results_dict[c]:
                found_paths = results_dict[c][subc].columns
                significance_dict[s][c].loc[found_paths, subc] = results_dict[c][subc].loc[stat_tests[s], :].values

    return significance_dict


def enumerate_significant_pathways_per_condition(significance_dict: dict, alpha: float = 0.05):
    """

    Parameters
    ----------
    significance_dict: dictionary of dataframes (one key per cluster)
    alpha: significance level

    Returns
    -------
    pd.DataFrame of significant pathways
    """

    complete_df = pd.DataFrame(columns=cols, index=ordered_labs)

    for s in significance_dict:
        for c in significance_dict[s]:
            complete_df.loc[significance_dict[s][c].columns, s] = (significance_dict[s][c] < alpha).sum()

    return complete_df


# class for html visualisation of the clusters of pathways

class Clusters(dict):

    def __init__(self, **kw):

        if 'map_to_id' in kw:
            self.map_to_id = kw.get('map_to_id')
        else:
            self.map_to_id = False
        if self.map_to_id:
            self.map_dict = kw.get('path_mapping')

    def _map_pathname_to_id(self, col):
        self.c_ids = []
        for x in self[col]:
            self.c_ids.append(self.map_dict[x])

    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            col, i = c.split('_')
            hx = rgb2hex(colorConverter.to_rgb(col))
            if self.map_to_id:
                self._map_pathname_to_id(c)
            else:
                self.c_ids = self[c]
            html += '<tr style="border: 0;">' \
                    '<td style="background-color: {0}; ' \
                    'border: 0;">' \
                    '<code style="background-color: {0};">'.format(hx)
            html += f'Module_{int(i) + 1}' + '</code></td>'
            html += '<td style="border: 0"><code>'
            html += ', '.join(self.c_ids) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html


def get_cluster_classes(den, label='ivl', **kwargs):
    """
    **kwargs: map_to_id = True to map the pathway name to its own ID
              [the dict of mapping should be provided in this case]
    """
    cluster_idxs = defaultdict(list)
    # kgg_map = kwargs['path_map']
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = Clusters(**kwargs)
    for idx, (c, l) in enumerate(cluster_idxs.items()):
        i_l = [den[label][i] for i in l]
        cluster_classes[f'{c}_{idx}'] = i_l

    return cluster_classes


def load_mathys_results(mathys_results: str = '../../Supplementary/41586_2019_1195_MOESM4_ESM.xlsx'):


    de_mathys_dfs = dict()
    de_xls = pd.ExcelFile(mathys_results)

    for shname in de_xls.sheet_names:
        if shname in ct_lab:
            print(f"Processing {shname}")
            de_mathys_dfs[shname] = dict()
            tmp = pd.read_excel(de_xls, sheet_name=shname, skiprows=1, true_values='TRUE', false_values='FALSE')

            de_mathys_dfs[shname]['HC-vs-AD'] = tmp.iloc[:, :9].copy(deep=True)
            de_mathys_dfs[shname]['HC-vs-AD'].set_index('Unnamed: 0', inplace=True)
            de_mathys_dfs[shname]['HC-vs-AD'].apply(pd.to_numeric)
            de_mathys_dfs[shname]['HC-vs-AD'].dropna(how='all', axis=0, inplace=True)
            del de_mathys_dfs[shname]['HC-vs-AD'].index.name
            de_mathys_dfs[shname]['HC-vs-AD'].iloc[:, -2:] = de_mathys_dfs[shname]['HC-vs-AD'].iloc[:, -2:].astype(bool)
            last_col = de_mathys_dfs[shname]['HC-vs-AD'].columns[-1]
            de_mathys_dfs[shname]['HC-vs-AD'] = de_mathys_dfs[shname]['HC-vs-AD'][
                de_mathys_dfs[shname]['HC-vs-AD'][last_col] == True]

            de_mathys_dfs[shname]['HC-vs-Early_AD'] = tmp.iloc[:, 11:20].copy(deep=True)
            de_mathys_dfs[shname]['HC-vs-Early_AD'].set_index('Unnamed: 11', inplace=True)
            de_mathys_dfs[shname]['HC-vs-Early_AD'].apply(pd.to_numeric)
            de_mathys_dfs[shname]['HC-vs-Early_AD'].dropna(how='all', axis=0, inplace=True)
            del de_mathys_dfs[shname]['HC-vs-Early_AD'].index.name
            de_mathys_dfs[shname]['HC-vs-Early_AD'].iloc[:, -2:] = de_mathys_dfs[shname]['HC-vs-Early_AD'].iloc[:, -2:].astype(bool)
            last_col = de_mathys_dfs[shname]['HC-vs-Early_AD'].columns[-1]
            de_mathys_dfs[shname]['HC-vs-Early_AD'] = de_mathys_dfs[shname]['HC-vs-Early_AD'][
                de_mathys_dfs[shname]['HC-vs-Early_AD'][last_col] is True]

            de_mathys_dfs[shname]['Early-vs-Late_AD'] = tmp.iloc[:, 22:32].copy(deep=True)
            de_mathys_dfs[shname]['Early-vs-Late_AD'].set_index('Unnamed: 22', inplace=True)
            de_mathys_dfs[shname]['Early-vs-Late_AD'].apply(pd.to_numeric)
            de_mathys_dfs[shname]['Early-vs-Late_AD'].dropna(how='all', axis=0, inplace=True)
            del de_mathys_dfs[shname]['Early-vs-Late_AD'].index.name
            de_mathys_dfs[shname]['Early-vs-Late_AD'].iloc[:, -2:] = de_mathys_dfs[shname]['Early-vs-Late_AD'].iloc[:, -2:].astype(bool)
            last_col = de_mathys_dfs[shname]['Early-vs-Late_AD'].columns[-1]
            de_mathys_dfs[shname]['Early-vs-Late_AD'] = de_mathys_dfs[shname]['Early-vs-Late_AD'][
                de_mathys_dfs[shname]['Early-vs-Late_AD'][last_col] is True]

    return de_mathys_dfs


def load_pathifier_loadings(pathways_dict: dict):

    clusters = ['Opc', 'In', 'Ex', 'Mic', 'Ast', 'Oli']

    loadings_dict = dict()
    for d in set(os.listdir()).intersection(clusters):
        for fname in os.listdir(os.path.join(os.getcwd(), d)):
            if 'Loadings_' in fname:
                print(f'Loading file: {fname}')
                dh, t_0 = mypars.create_status_bar()
                xls = pd.ExcelFile(os.path.join(os.getcwd(), d, fname))
                subc = '_'.join(fname.split('_')[1:3])
                loadings_dict[subc] = dict()
                for i, hsa_path in enumerate(xls.sheet_names):
                    tmp = pd.read_excel(xls, sheet_name=hsa_path, skiprows=6)
                    loadings_dict[subc][pathways_dict[hsa_path][0]] = tmp.iloc[:, -2:]
                    loadings_dict[subc][pathways_dict[hsa_path][0]].set_index(loadings_dict[subc][pathways_dict[hsa_path][0]].columns[0],
                                                                              inplace=True)
                    del loadings_dict[subc][pathways_dict[hsa_path][0]].index.name
                    mypars.update_progress(i / len(xls.sheet_names), disp_inst=dh, init_time=t_0)
                mypars.update_progress(1, disp_inst=dh, init_time=t_0)


