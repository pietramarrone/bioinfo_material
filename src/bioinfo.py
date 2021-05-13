import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import itertools
import pickle as pkl
import re
import urllib.parse
import urllib.request
from typing import List, Union
from itertools import chain
from tqdm import tqdm
import mygene
import collections as cx
import scipy.stats as stats
import statistics as myst
import data_parser as my_pars
from ml_lib.dimensionality_reduction.principal_curve import princurve as my_princ
from ml_lib.dimensionality_reduction import my_PCA
import anndata as andat
import scanpy

from statsmodels.stats import multitest

# This is specific for human genes
from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT
from goatools.go_search import GoSearch
from goatools.associations import read_ncbi_gene2go

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet, cut_tree
from scipy.spatial.distance import pdist



#super_dir = '/Users/a.possenti/OneDrive - University Of Cambridge/Python/Repositories'
#repo_kg = os.path.join(super_dir, 'Kegg_Scraping')
#repo_go = os.path.join(super_dir, 'GO_Scraping')
#repo_taxid = os.path.join(super_dir, 'Tax_Ids')
#repo_reactome = os.path.join(super_dir, 'Reactome')
#NCBI = os.path.join(super_dir, 'NCBI')
#human_NCBI = os.path.join(NCBI, 'Homo Sapiens')

acceppted_KEGG_abb = ('hsa', 'mmu')

ct_lab = ['Ex', 'In', 'Ast', 'Mic', 'Oli', 'Opc']

ncbi_dtypes = {'tax_id': str, 'Org_name': str, 'GeneID': str, 'CurrentID': int, 'Status': str, 'Symbol': str,
               'Aliases': str, 'description': str, 'other_designations': str, 'map_location': str,
               'chromosome': str, 'genomic_nucleotide_accession.version': str,
               'start_position_on_the_genomic_accession': str, 'end_position_on_the_genomic_accession': str,
               'orientation': str, 'exon_count': str, 'OMIM': str}

clinical_readouts = ['Amyloid level', 'Neuritic Plaque Burden',
                     'Neurofibrillary Tangle Burden', 'Tangle Density',
                     'Global Cognitive Function', 'Global AD-pathology Burden', 'Age']


def gene_alias_chainer(s, sep: str = ','):
    return list(chain.from_iterable(s.str.split(sep)))


def load_ncbi_mapping(data_type: str = 'protein_coding', organism: str = 'human', reindex_to_ncbi: bool = True,
                      expand_aliases: bool = False, return_gene_mapping_only: bool = False):
    dt = {'protein_coding': 'ProteinCoding',
          'DNA': 'All',
          'microrna': 'microRNA'}
    org = {'human': 'Homo_sapiens',
           'mouse': 'Mus_musculus',
           'fruit_fly': 'Drosophila_melanogaster'}

    full_mapping = None

    assert data_type in dt.keys(), 'data_type not in list: {DT}'.format(DT=dt.keys())
    assert organism in org.keys(), 'organism not in list: {ORG}'.format(ORG=org.keys())

    tsv_file = 'genes_NCBI_hsa_{DT}.tsv'.format(DT=dt.get(data_type))
    repo = os.path.join(NCBI, org.get(organism), tsv_file)
    df = pd.read_table(filepath_or_buffer=repo, sep='\t',
                       dtype=ncbi_dtypes,
                       usecols=range(17))
    if reindex_to_ncbi:
        df.set_index('GeneID', inplace=True)
    df.sort_index(inplace=True)
    if expand_aliases:
        df['Aliases'].fillna(value='', inplace=True)
        lens = df['Aliases'].str.split(',').map(len)
        full_mapping = pd.DataFrame({'Symbol': np.repeat(df['Symbol'], lens),
                                     'Aliases': gene_alias_chainer(df['Aliases'])})
        if return_gene_mapping_only:
            return full_mapping
    res = df if not expand_aliases else [df, full_mapping]
    return res


def get_reactome(path_or_file='Default', genes_id='ncbi', mask_species=None):
    """
    Parameters
    ----------
    path_or_file: complete path and filename of the repository
    genes_id: id of the genes (can be uniprot or ncbi
    mask_species: can be either
    Returns
    -------
    kegg_p: dict of pathways
            [keys] -> hsa codes of the kegg pathways
            [values] -> pathway name
                     -> list of entrezID
                     -> list of matching gene names
    """
    if path_or_file.lower() == 'default':
        path_or_file = repo_reactome
    else:
        raise FileNotFoundError('Only the default option available for now')

    if genes_id.lower() == 'ncbi':
        f = 'identifier_mapping_files_NCBItoALL.txt'

    elif genes_id.lower() == 'uniprot':
        f = 'identifier_mapping_files_UNIPROTtoALL.txt'
    else:
        raise FileNotFoundError('Please specify either NCBI or UNIPROT')

    df = pd.read_table(os.path.join(path_or_file, f), header=None, dtype='str')
    df.columns = [genes_id.upper() + '_ID', 'Gene_Stable_ID', 'Name_Stable [Location]', 'Pathway_Stable_ID',
                  'URL', 'Pathway', 'Evidence_Code', 'Species']

    if mask_species is None:
        return df
    else:
        return df[df.Species.str.contains(mask_species, case=False)]


def download_kegg(organism_abb: str, write_paths_info: bool = False, store_to_csv: bool = True):
    from Bio.KEGG import REST

    kegg_paths = {}

    assert organism_abb in acceppted_KEGG_abb, "Organism abbrevaition [{ABB}] not in the accepted ones: " \
                                               "{ALL_ABB}".format(ABB=organism_abb, ALL_ABB=acceppted_KEGG_abb)
    org_path = REST.kegg_list("pathway", organism_abb).read()

    for line in org_path.rstrip().split('\n'):
        entry, descript = line.split('\t')
        kegg_paths[entry[5:]] = [descript[:-23]]  # remove first 5 chars "path:" and last 23 chars "organism name"

    if store_to_csv:
        csv_name = "All_KEGGS_{ORG}.csv".format(ORG=organism_abb)
        csv_out = open("{REPO}/{FILENAME}".format(REPO=repo_kg, FILENAME=csv_name),
                       mode='w') if csv_name not in os.listdir("{REPO}".format(REPO=repo_kg)) else None
    else:
        csv_out = None

    if csv_out is not None:
        print("Saving INFO to: {FILENAME}".format(FILENAME=csv_out.name))
    else:
        print("File exists at: {ORG}.\n\t** Exit".format(ORG=repo_kg))
        return

    with tqdm(total=len(kegg_paths.keys()), file=sys.stdout, leave=False) as pbar:
        for kgg in kegg_paths:
            pbar.update(1)
            pbar.write("Processed: {PATHWAY}".format(PATHWAY=kgg))
            tmp_genes = dict()
            pathway_file = REST.kegg_get(kgg).read()
            if write_paths_info:
                fout = open("{REPO}/Paths/{ORG}/{KEGG}_{DESC}.txt".format(REPO=repo_kg, ORG=organism_abb, KEGG=kgg,
                                                                          DESC=kegg_paths[kgg][0].replace(" / ", "_")),
                            mode='w')
            # iterate through each KEGG pathway file, keeping track of which section
            # of the file we're in, only read the gene in each pathway
            current_section = None
            for idx, line in enumerate(pathway_file.rstrip().split("\n")):
                if write_paths_info:
                    fout.write("{}\n".format(line))
                section = line[:12].strip()  # section names are within 12 columns
                if not section == "":
                    current_section = section

                if current_section == "GENE":
                    try:
                        gene_identifiers, gene_description = line[12:].split("; ")
                        gene_id, gene_symbol = gene_identifiers.split()
                    except ValueError:
                        pbar.write("\t** Missing separator in {} at line {}".format(kgg, idx + 1))
                        gene_id = line[12:].split()[0]
                        gene_symbol = "Not Found"

                    if gene_id not in tmp_genes:
                        tmp_genes[gene_id] = []
                    tmp_genes[gene_id].append(gene_symbol)
            kegg_paths[kgg].append(tmp_genes)
            if write_paths_info:
                fout.close()
            if store_to_csv and csv_out is not None:
                csv_out.write("{},{}\n".format(kgg, kegg_paths[kgg][0].replace(",", ';')))
                for gene in kegg_paths[kgg][1]:
                    if len(kegg_paths[kgg][1][gene]) == 1:
                        csv_out.write("{},{}\n".format(gene, kegg_paths[kgg][1][gene][0]))
                    else:
                        csv_out.write("{},{}\n".format(gene, ",".join(kegg_paths[kgg][1][gene])))
    if store_to_csv and csv_out is not None:
        csv_out.close()


def get_keggs(path_or_file: str = 'Default', organism_abb: str = 'hsa', print_list=False):
    """
    Parameters
    ----------
    path_or_file: complete path and filename of the repository
    print_list : if True, prints all hsa and path names
    Returns
    -------
    kegg_p: dict of pathways 
            [keys] -> hsa codes of the kegg pathways
            [values] -> pathway name
                     -> list of entrezID
                     -> list of matching gene names
    """
    kegg_p = dict()
    if path_or_file == 'Default':
        print('Loading keggs from default repositories\n')
        fin = open(file="{REPO}/All_KEGGS_{ORG}.csv".format(REPO=repo_kg, ORG=organism_abb), mode='r')
    else:
        fin = open(file=path_or_file, mode='r')

    cont = fin.readlines()
    cont = [x.strip().split(',') for x in cont]
    fin.close()

    temp_p = None
    # l1 = []
    # l2 = []
    gene_map = dict()
    for idx, entry in enumerate(cont):
        if entry[0].startswith(organism_abb):
            if idx != 0:
                kegg_p[temp_p].append(gene_map)
            temp_p = entry[0]
            # l1 = []
            # l2 = []
            gene_map = dict()
            kegg_p[temp_p] = [entry[1]]
        else:
            gene_map.update({entry[0]: entry[1]})
            # l1.append(entry[0])
            # l2.append(entry[1])
    # kegg_p[temp_p].append(l1)
    # kegg_p[temp_p].append(l2)
    kegg_p[temp_p].append(gene_map)

    if print_list:
        print('KEGG_IDs\tKEGG_names\n')
        for k in kegg_p:
            print("{}\t{}\n".format(k, kegg_p[k][0]))

    return kegg_p


def format_taxids(path_or_file='Default', show_head=True, check_if_exists=True, return_df=False):
    """
    Parameters
    ----------
    path_or_file: complete path and filename of the repository
    show_head: preview of the header
    check_if_exists: if True check if the formatted version of taxids exists
    return_df: if True return a dataframe with the tax ids
    """

    if check_if_exists:
        if 'TaxId_Dataframe.xlsx' in os.listdir(repo_taxid):
            print("File TaxId_Dataframe.xlsx already present in repository.")
            if return_df:
                df = pd.read_excel(repo_taxid + '/TaxId_Dataframe.xlsx', index_col=0)
                df.index = df.index.map(int)
                return df
            return

    if path_or_file == 'Default':
        print('*** Loading keggs from default repository ***')
        # fin = open(file=repo_taxid+"/names.dmp", mode='r')
        fin = repo_taxid + '/names.dmp'
        print(f'Parsing file {fin}')
        df = pd.read_table(fin, header=None, sep='\t\|\t', engine='python', index_col=0, usecols=[0, 1]).drop(1)
    else:
        print('Option to be implemented')
        return

    if show_head:
        print(df.head())

    df.index.name = 'Tax_Id'
    df.columns = ['Name_Txt']
    df['Synonyms'] = df.groupby(df.index)['Name_Txt'].apply(lambda x: pd.Series(list(x)[1:]))
    df = df.groupby(df.index).agg({'Name_Txt': 'first', 'Synonyms': 'first'})
    df.to_excel('TaxId_Dataframe.xlsx', index=True)

    if return_df:
        return df


def check_taxid(id_or_name, df_repo=None, return_syn=False):
    """
    Parameters
    ----------
    id_or_name: can be either int (tax_id) or string (organism name)
    df_repo: data frame containing the repository
    return_syn: if True returns the synonyms of the organism name
    """
    if 'TaxId_Dataframe.xlsx' not in os.listdir(repo_taxid):
        print("Use format_taxids function to generate the repo first")
        return
    if df_repo is None:
        temp = pd.read_excel('TaxId_Dataframe.xlsx', index_col=0)
        temp.index = temp.index.map(int)
    else:
        temp = df_repo
    if isinstance(id_or_name, int):
        name = temp.loc[id_or_name, 'Name_Txt']
        syn = None if not return_syn else temp.loc[id_or_name, 'Synonyms']
        print(f'Tax ID: {id_or_name} - Name: {name} - Synonyms: {syn}")')
    elif isinstance(id_or_name, str):
        print("Results\n-------")
        print(temp.loc[temp.Name_Txt == id_or_name])


def go2geneids(taxid, ns: str = 'BP'):
    go2geneids = read_ncbi_gene2go(repo_go + "/gene2go", taxids=taxid, namespace=ns, go2geneids=True)
    print(f"{len(go2geneids)} GO terms associated with taxonomy ID {taxid} NCBI Entrez GeneIDs")
    return go2geneids


def convert_NCBI_to_GENENAMES(geneids, prt_output: bool = False):
    convert_dict = dict()
    not_found = list()
    for g_id in geneids:
        nt = GENEID2NT.get(g_id, None)
        if nt is not None:
            if prt_output:
                print(f"{nt.Symbol:<10} {nt.description}")
            convert_dict[g_id] = nt.Symbol
        else:
            not_found.append(g_id)

    if prt_output:
        print("GENES not Found:")
        print(*not_found, sep='\n', file=sys.stdout)

    return convert_dict


def genes2term(taxid, term, ns='BP', write_log=True, convert_geneids=True, print_genes: bool = False,
               path_out: str = None):
    AssoGOs = cx.namedtuple('Associations', 'GO_IDs Genes')

    item = go2geneids(taxid)
    srchhelp = GoSearch(repo_go + "/go-basic.obo", go2items=item)

    # Compile search pattern for term
    term_all = re.compile(r"{}".format(term), flags=re.IGNORECASE)
    term_not = re.compile(r"{}.independent".format(term), flags=re.IGNORECASE)

    fname = path_out if path_out is not None else f"{repo_go}/{term}_{taxid}_gos.log"

    log = open(fname, mode="w") if write_log else sys.stdout
    # Search for 'cell cycle' in GO terms
    gos_term_all = srchhelp.get_matching_gos(term_all, prt=log)
    # Find any GOs matching 'cell cycle-independent' (e.g., "lysosome")
    gos_term_not = srchhelp.get_matching_gos(term_not, gos=gos_term_all, prt=log)
    # Remove GO terms that are not "term" GOs
    gos = gos_term_all.difference(gos_term_not)
    # Add children GOs of "term" GOs
    gos_all = srchhelp.add_children_gos(gos)
    # Get Entrez GeneIDs for "term" GOs
    geneids = srchhelp.get_items(gos_all)
    print(f"{len(geneids)} human NCBI Entrez GeneIDs related to '{term}' found.")
    print(f"{len(gos_all)} human GO:Terms related to '{term}' found.")
    if write_log:
        log.close()

    if convert_geneids:
        converted = convert_NCBI_to_GENENAMES(geneids, prt_output=print_genes)
        return AssoGOs(GO_IDs=gos_all, Genes=converted)
    else:
        return AssoGOs(GO_IDs=gos_all, Genes=geneids)


def convert_gene_ids_uniprot_db(query: list, id_from: str, id_to: str, **kw):
    """
    Parameters
    ----------
    query: list of genes to convert
    id_from: string identifier of the label to convert from
    id_to: string identifier of the label to convert to

    to check the available ids for the conversion, check https://www.uniprot.org/help/api_idmapping
    
    Returns
    -------
    a tab (or ',') separated string with the conversion
    """

    url = 'https://www.uniprot.org/uploadlists/'

    sep = kw['sep'] if 'sep' in kw else 'tab'

    in_query = ' '.join(query)
    params = {'from': id_from, 'to': id_to, 'format': sep, 'query': in_query}
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()
    results = pd.DataFrame(data=np.array(response.decode('utf-8').split()).reshape(-1, 2)[1:], columns=[id_from, id_to])
    return results


def convert_my_genes(query: list, scopes: Union[str, list], fields: str = 'all', species: str = 'human',
                     as_df: bool = True, write_out: bool = False):
    """
    Parameters
    ----------
    query: list of genes to be covnerted
    scopes:
    fields: which fields to keep
    species: eg human, mouse
    as_df: whether to return it in dataframe format
    write_out: write output to screen (if not too long)

    Returns
    -------
    an object - df with the conversion
    """
    mg = mygene.MyGeneInfo()
    ginfo = mg.querymany(query, scopes=scopes, fields=fields, species=species, as_dataframe=as_df)

    if write_out and not as_df and len(query) < 10:
        for g in ginfo:
            for k, v in g.items():
                print("- {0: <10}: {1}".format(k, v))
            print()
    elif write_out and as_df:
        print(ginfo.head())

    if as_df:
        return ginfo.drop(columns=['_id', '_score'])


def compare_clusters_composition(df: pd.DataFrame, obs_1: str, anno_1: str, obs_2: str,
                                 anno_2: str, metric: str = 'jaccard'):
    assert metric.lower() == 'jaccard', "Only Jaccard similarity implemented"

    j_coef = None

    if metric.lower() == 'jaccard':
        union = ((df[obs_1] == anno_1) | (df[obs_2] == anno_2)).sum()
        intersection = ((df[obs_1] == anno_1) & (df[obs_2] == anno_2)).sum()
        j_coef = intersection / union

    return (df[obs_1] == anno_1).sum(), (df[obs_2] == anno_2).sum(), j_coef


def get_reactome_genes_from_path(reactome_db: pd.DataFrame, pathway: str,
                                 genes_column: str = 'Name_Stable [Location]'):
    genes = reactome_db[reactome_db.Pathway == pathway][genes_column].str.split(' ', expand=True)[0]
    genes = genes.str.replace(r'\(.*\)', '')
    genes = genes.unique().tolist()
    return list(set(genes))


def get_reactome_stable_id_from_path(reactome_db: pd.DataFrame, pathway: str,
                                     stable_id_col: str = 'Pathway_Stable_ID'):
    stable_id = reactome_db[reactome_db.Pathway == pathway][stable_id_col].unique()[0]

    return stable_id


def get_reactome_pathname_from_stableid(reactome_db: pd.DataFrame, stable_id: str,
                                        pathway_col: str = 'Pathway'):
    pathname = reactome_db[reactome_db.Pathway_Stable_ID == stable_id][pathway_col].unique()[0]

    return pathname


def jaccard_similarity_of_pathways(all_paths: list, pathways_label: str):
    """
    Compute the Jaccard similarity between pathways
    Parameters
    ----------
    all_paths
    pathways_label: either KEGG or Reactome

    Returns
    -------

    """

    tmp_list = all_paths.copy()

    assert pathways_label.lower() in ('kegg', 'reactome')

    loading_function = get_keggs if pathways_label.lower() == 'kegg' else get_reactome
    paths_loaded = loading_function(path_or_file='Default')

    # pathways manually removed for lack of genes in the list
    meta_paths = ['Carbon metabolism', 'Metabolic pathways', '2-Oxocarboxylic acid metabolism',
                  'Endocrine resistance', 'Platinum drug resistance', 'Fatty acid metabolism',
                  'Biosynthesis of amino acids', 'EGFR tyrosine kinase inhibitor resistance',
                  'EGFR tyrosine kinase inhibitor resistance', 'Antifolate resistance']

    tmp_list = [k for k in tmp_list if k not in meta_paths]
    df_keggs = pd.DataFrame(columns=tmp_list, index=tmp_list)

    tmp_kgg = dict(zip(list(itertools.chain(*paths_loaded.values()))[::2], paths_loaded.keys()))
    tmp2_kgg = [tmp_kgg[k] for k in tmp_list]

    for i, k in enumerate(tmp2_kgg):
        try:
            tmp = np.array([myst.jaccard_similarity(paths_loaded[k][1].values(),
                                                    paths_loaded[tmp2_kgg[j]][1].values())
                            for j in range(i, len(tmp_list))])
            df_keggs.iloc[i, i:] = tmp
            df_keggs.iloc[i:, i] = df_keggs.iloc[i, i:]
        except ZeroDivisionError:
            print(paths_loaded[k][0])
            continue


def cluster_pathways(df_paths: pd.DataFrame,
                     metric: str = 'centroid',
                     method: str = 'ward',
                     optimal_order: bool = True,
                     compute_cophenetic_correlation: bool = True, **kwargs):
    df_paths = df_paths.astype(float)  # convert to correct dtype
    mat = (1 - df_paths)  # distance matrix
    dists = squareform(mat.values)  # square form for the distance matrix
    Z = hierarchy.linkage(dists, metric=metric, method=method, optimal_ordering=optimal_order)

    dn = hierarchy.dendrogram(Z, color_threshold=1.3, labels=df_paths.columns, above_threshold_color='#0A0A0A')

    if kwargs.get('plot_dendro', False):

        list_of_paths = kwargs.get('list_of_pathways', None)
        assert list_of_paths is not None
        pathways_of_interest = kwargs.get('pathways_of_interest', None)
        assert pathways_of_interest is not None
        second_temp_list = kwargs.get('temporary_list', None)
        assert second_temp_list is not None

        f, ax = plt.subplots(figsize=(90, 30))

        no_spine = {'left': False, 'bottom': False, 'right': False, 'top': False}
        sns.despine(**no_spine)
        for lab in ax.get_xticklabels():
            list_of_paths.append(int(lab.get_text()))

        ax.set_xticklabels(df_paths.columns[list_of_paths], size=20)
        for i, lab in enumerate(ax.get_xticklabels()):
            if lab.get_text() in list(itertools.chain(*pathways_of_interest.values())):
                if lab.get_text() in second_temp_list.index:
                    ax.get_xticklabels()[i].set_fontweight("bold")
        plt.tight_layout()
        plt.savefig('./Dendrogram.pdf', bbox_inches='tight')

    if compute_cophenetic_correlation:
        coph_correlation, coph_distances = cophenet(Z, pdist(mat.values))
        return dn, coph_correlation, coph_distances
    else:
        return dn


def get_cluster_members(dendrogram, leaves_index: list):
    """

    Parameters
    ----------
    dendrogram: dendrogram class
    leaves_index: indices of the leaves

    Returns
    -------

    """

    clusters_members = dict()
    for i, _ in enumerate(leaves_index):
        if i == 0:
            clusters_members['Cluster_{}'.format(i + 1)] = dendrogram['ivl'][:leaves_index[i]]
        else:
            clusters_members['Cluster_{}'.format(i + 1)] = dendrogram['ivl'][np.sum(leaves_index[: i]):
                                                                             np.sum(leaves_index[: i + 1])]

    return clusters_members


def get_clusters_proportions(dendrogram):
    """
    Get the proportions of clusters
    Parameters
    ----------
    dendrogram: dendrogram of the clusters obtained

    Returns
    -------

    """

    # this is an example for the KEGG pathways
    all_leaves_idxs = np.array([9, 6, 4, 6, 3, 5, 6, 73, 2, 6, 20, 7, 7, 21, 4,
                                13, 5, 24, 11, 18, 9, 12, 6, 7, 14, 10, 16])

    ordered_cols = []
    for col in dendrogram['color_list']:
        if col not in ordered_cols and col != '#0A0A0A':
            ordered_cols.append(col)

    tot, val = 0, []
    for col in ordered_cols:
        val.append(dendrogram['color_list'].count(col))
        tot += val[-1]

    proportion = np.array(val) / tot

    # proportion = all_leaves_idxs / all_leaves_idxs.sum()

    return proportion, ordered_cols


def get_pathways_clusters_enrichment(clusters_membership: dict,
                                     pathways_dict: dict,
                                     print_membership: bool = True):
    """
    measure enrichment of the clusters in specific subsets of pathways
    Parameters
    ----------
    clusters_membership: dictionary of clusters with the pathways belonging to them
    pathways_dict: dictionary of pathways lists (each key is the name of the list)
    print_membership: if True print the results to screen

    Returns
    -------

    """
    df = pd.DataFrame(index=clusters_membership.keys(), columns=['pval'])

    for mm in clusters_membership:
        if print_membership:
            print(mm)
            print('Perturbation', myst.fisher_ex(in_list=pathways_dict['Perturbation'],
                                                 backgr=list(itertools.chain(*clusters_membership.values())),
                                                 bio_path=clusters_membership[mm],
                                                 sides='greater')[1])
            print('Vulnerability', myst.fisher_ex(in_list=pathways_dict['Vulnerability'],
                                                  backgr=list(itertools.chain(*clusters_membership.values())),
                                                  bio_path=clusters_membership[mm],
                                                  sides='greater')[1])
            print('OVERALL', myst.fisher_ex(in_list=list(itertools.chain(*pathways_dict.values())),
                                            backgr=list(itertools.chain(*clusters_membership.values())),
                                            bio_path=clusters_membership[mm],
                                            sides='greater')[1])

        df.loc[mm, 'pval'] = myst.fisher_ex(in_list=list(itertools.chain(*pathways_dict.values())),
                                            backgr=list(itertools.chain(*clusters_membership.values())),
                                            bio_path=clusters_membership[mm],
                                            sides='greater')[1]

    return df


def compute_correlation_across_clusters(results_dict: dict,
                                        paths_of_interest: dict,
                                        **kwargs):
    df_columns = kwargs.get('col_names', ['KEGG pathway', 'clinical trait', 'correlation', 'inTWAS'])

    correlations_df = dict()

    for c in results_dict:
        correlations_df[c] = dict()
        for subc in results_dict[c]:
            isin_my_paths = np.where(results_dict[c][subc].T.index.isin(itertools.chain(*paths_of_interest.values())),
                                     'yes', 'no')
            correlations_df[c][subc] = pd.melt(results_dict[c][subc].T[clinical_readouts])
            correlations_df[c][subc]['inTWAS'] = np.tile(isin_my_paths, 7)
            correlations_df[c][subc].insert(loc=0, column='KEGG pathway',
                                            value=np.tile(results_dict[c][subc].T.index.values, 7),
                                            allow_duplicates=False)
            correlations_df[c][subc].columns = df_columns

    return correlations_df


def get_median_correlation_to_phenotypes(results_dict: dict,
                                         sorted_pathways_df: pd.DataFrame,
                                         paths_of_interest: dict,
                                         **kwargs):
    df_columns = kwargs.get('col_names', ['KEGG pathway', 'clinical trait', 'correlation', 'inTWAS'])
    full_median_corr_dfs = dict()
    median_corr_dfs = dict()

    for c in ct_lab:
        median_corr_dfs[c] = pd.DataFrame(index=sorted_pathways_df.index)
        tmp_list = list()
        tmp_list.append(median_corr_dfs[c])
        for subc in results_dict[c]:
            tmp_list.append(results_dict[c][subc].T[clinical_readouts])
        median_corr_dfs[c] = pd.concat(tmp_list, axis=1, join='inner')

    for c in ct_lab:
        test = median_corr_dfs[c].groupby(median_corr_dfs[c].columns, axis=1).median()[clinical_readouts]
        test['inTWAS'] = np.where(test.index.isin(itertools.chain(*paths_of_interest.values())), 'yes', 'no')
        full_median_corr_dfs[c] = pd.melt(test.iloc[:, :-1])
        full_median_corr_dfs[c]['inTWAS'] = np.tile(test.inTWAS, 7)
        full_median_corr_dfs[c].insert(loc=0, column='KEGG pathway',
                                       value=np.tile(test.index.values, 7),
                                       allow_duplicates=False)
        full_median_corr_dfs[c].columns = df_columns

    return full_median_corr_dfs


def compute_correlations_pvalues(full_median_corr_df: pd.DataFrame,
                                 col_label: str = 'inTWAS',
                                 correct_multitest: bool = True,
                                 alpha: float = 0.05):
    corr_pvals = dict()
    for c in full_median_corr_df:
        corr_pvals[c] = dict()
        for pheno in full_median_corr_df[c]['clinical trait'].unique():
            mask_yes = (full_median_corr_df[c]['clinical trait'] == pheno) & (
                    full_median_corr_df[c][col_label] == 'yes')
            if pheno not in ['Global Cognitive Function']:
                corr_pvals[c][pheno] = min(1.,
                                           stats.mannwhitneyu(full_median_corr_df[c][mask_yes]['correlation'].values,
                                                              full_median_corr_df[c][~mask_yes]['correlation'].values,
                                                              alternative='greater')[1])
            else:
                corr_pvals[c][pheno] = min(1.,
                                           stats.mannwhitneyu(full_median_corr_df[c][mask_yes]['correlation'].values,
                                                              full_median_corr_df[c][~mask_yes]['correlation'].values,
                                                              alternative='less')[1])
    if correct_multitest:
        for c in corr_pvals:
            corr_pvals[c].update(dict(zip(corr_pvals[c].keys(),
                                          multitest.multipletests(list(corr_pvals[c].values()),
                                                                  alpha=alpha, method='bonferroni')[1])))

    return corr_pvals


def significant_paths_to_cluster(significance_dict: dict, alpha: float = 0.05):

    cols = ['Healthy vs Early-AD', 'Healthy vs Late-AD', 'Early-AD vs Late-AD (less)', 'Early-AD vs Late-AD (greater)']
    ordered_main_cells = ['Ex [cell type]', 'In [cell type]', 'Ast [cell type]',
                          'Mic [cell type]', 'Oli [cell type]', 'Opc [cell type]']

    sign_paths_to_cluster_df = pd.DataFrame(columns=cols, index=ordered_main_cells)

    for s in significance_dict:
        for c in significance_dict[s]:
            col_to_keep = sign_paths_to_cluster_df.loc[
                sign_paths_to_cluster_df.index.isin(significance_dict[s][c].columns)].index.to_list()[0]
            paths_to_keep = (significance_dict[s][c] < alpha)[col_to_keep]

            sign_paths_to_cluster_df[s] = sign_paths_to_cluster_df[s].astype('object')

            sign_paths_to_cluster_df.at[col_to_keep, s] = significance_dict[s][c].loc[paths_to_keep.values.flatten()][
                col_to_keep].index.to_list()

    return sign_paths_to_cluster_df


def map_gwas_to_pathways(sign_paths_to_cluster_df: pd.DataFrame,
                         gwas_in_paths: dict):
    gwas_paths_df = pd.DataFrame(index=sign_paths_to_cluster_df.index,
                                 columns=sorted(set(itertools.chain(*list(gwas_in_paths.values())))),
                                 data=np.zeros((sign_paths_to_cluster_df.index.shape[0],
                                               len(set(itertools.chain(*list(gwas_in_paths.values())))))), dtype=int)

    for c in sign_paths_to_cluster_df.index:

        tmp_gwas = set(list(itertools.chain(*list(sign_paths_to_cluster_df.loc[c].values)))).intersection(
            gwas_in_paths.keys())

        if tmp_gwas:
            for t in tmp_gwas:
                print('\t', t, gwas_in_paths[t])
                gwas_paths_df.loc[c, gwas_in_paths[t]] = 1


def check_pathway_family_enrichment_in_cluster(sign_paths_to_cluster_df: pd.DataFrame,
                                               cluster_membership: dict,
                                               correct_multitest: bool = True,
                                               **kwargs):
    dfs_modules_to_paths_sign = dict()
    dfs_modules_to_paths_foldenr = dict()

    for ct in sign_paths_to_cluster_df.index:
        print(ct)
        tmp_ct = ct.split(' ')[0]
        dfs_modules_to_paths_sign[tmp_ct] = pd.DataFrame(columns=sign_paths_to_cluster_df.columns,
                                                         index=list(cluster_membership.keys()))
        dfs_modules_to_paths_foldenr[tmp_ct] = pd.DataFrame(columns=sign_paths_to_cluster_df.columns,
                                                            index=list(cluster_membership.keys()))
        for cond in dfs_modules_to_paths_sign[tmp_ct].columns:
            for mm in cluster_membership:
                if sign_paths_to_cluster_df.loc[ct, cond]:

                    dfs_modules_to_paths_sign[tmp_ct].loc[mm, cond] = \
                    myst.fisher_ex(in_list=sign_paths_to_cluster_df.loc[ct, cond],
                                   backgr=list(itertools.chain(*cluster_membership.values())),
                                   bio_path=cluster_membership[mm], sides='greater')[1]
                    dfs_modules_to_paths_foldenr[tmp_ct].loc[mm, cond] = myst.fold_enrichment(
                        my_list=sign_paths_to_cluster_df.loc[ct, cond],
                        backgr=list(itertools.chain(*cluster_membership.values())),
                        path=cluster_membership[mm])
                else:
                    dfs_modules_to_paths_sign[tmp_ct].loc[mm, cond] = np.nan
                    dfs_modules_to_paths_foldenr[tmp_ct].loc[mm, cond] = np.nan

    for ct in dfs_modules_to_paths_sign:
        dfs_modules_to_paths_sign[ct] = dfs_modules_to_paths_sign[ct].astype(float)

    if correct_multitest:
        method = kwargs.get('method', 'bonferroni')
        for ct in dfs_modules_to_paths_sign:
            for cond in dfs_modules_to_paths_sign[ct]:
                if not all(dfs_modules_to_paths_sign[ct].loc[:, cond].isnull()):
                    dfs_modules_to_paths_sign[ct].loc[:, cond] = multitest.multipletests(dfs_modules_to_paths_sign[ct][cond].values,
                                                                                         method=method)[1]
        for ct in dfs_modules_to_paths_sign:
            dfs_modules_to_paths_sign[ct].fillna(value=1., inplace=True)

    return dfs_modules_to_paths_foldenr, dfs_modules_to_paths_sign


def compute_jaccard_similarity_of_perturbations(sign_paths_to_cluster_df: pd.DataFrame):
    dfs_jacc_pathways = dict()
    for cond in sign_paths_to_cluster_df.columns[:3]:
        print(cond)
        if 'less' in cond:
            c = cond.rsplit(' ', 1)[0]
        else:
            c = cond
        dfs_jacc_pathways[c] = pd.DataFrame(columns=sign_paths_to_cluster_df.index,
                                            index=sign_paths_to_cluster_df.index)
        for i, ct1 in enumerate(sign_paths_to_cluster_df.index):
            for j, ct2 in enumerate(sign_paths_to_cluster_df.index[i:]):
                if 'less' in cond:
                    l1 = sign_paths_to_cluster_df.loc[ct1, cond] + sign_paths_to_cluster_df.loc[
                        ct1, cond.replace('less', 'greater')]
                    l2 = sign_paths_to_cluster_df.loc[ct2, cond] + sign_paths_to_cluster_df.loc[
                        ct2, cond.replace('less', 'greater')]
                else:
                    l1 = sign_paths_to_cluster_df.loc[ct1, cond]
                    l2 = sign_paths_to_cluster_df.loc[ct2, cond]
                if l1 or l2:
                    dfs_jacc_pathways[c].loc[ct1, ct2] = myst.jaccard_similarity(l1, l2)
                else:
                    dfs_jacc_pathways[c].loc[ct1, ct2] = 0.
        dfs_jacc_pathways[c].fillna(0., inplace=True)

    return dfs_jacc_pathways


def map_DEgenes_to_pathways(sign_paths_to_cluster_df: pd.DataFrame,
                            mathys_dict: pd.DataFrame,
                            all_genes_in_pathways: list,
                            path_dict: dict,
                            clusters: list,
                            path_name_to_hsa_map: dict):
    overlap_df = pd.DataFrame(index=clusters, columns=['HC-vs-AD', 'HC-vs-Early_AD', 'Early-vs-Late_AD'])

    map_cond_mathys_to_me = {'HC-vs-AD': ['Healthy vs Early-AD', 'Healthy vs Late-AD'],
                             'HC-vs-Early_AD': ['Healthy vs Early-AD'],
                             'Early-vs-Late_AD': ['Early-AD vs Late-AD (less)', 'Early-AD vs Late-AD (greater)']}

    for c in overlap_df.index:
        print(f'Cluster {c}')
        for cond in overlap_df.columns:
            relevant_genes = set()
            print(f'\tCondition {cond}')
            relevant_paths = sign_paths_to_cluster_df.loc[
                '{} [cell type]'.format(c), map_cond_mathys_to_me[cond]].values
            if isinstance(relevant_paths, list):
                relevant_paths = set(relevant_paths)
            elif isinstance(relevant_paths, np.ndarray):
                relevant_paths = list(set(list(itertools.chain(*relevant_paths))))
            if not relevant_paths:
                print('\t\tSkipping')
                overlap_df.loc[c, cond] = 0.
                continue
            de_mathys_and_found_in_kegg = set(mathys_dict[c][cond].index).intersection(all_genes_in_pathways)
            for p in relevant_paths:
                relevant_genes.update(de_mathys_and_found_in_kegg.intersection(path_dict[path_name_to_hsa_map[p]][1].values()))
            ovlap_of_genes = set(de_mathys_and_found_in_kegg).intersection(relevant_genes)
            overlap_df.loc[c, cond] = len(ovlap_of_genes) / len(de_mathys_and_found_in_kegg)

    return overlap_df


def run_pathifier(data, cell_anno, id_anno, clinical_anno, gene_list, path_name, stat_test: str,
                  pcs=4, destination_folder=None, test_robustness: bool = False, **kwargs):
    """
    gene_list is the old kegg_p[kgg][1].values()
    path_name is kegg_p[kgg][0]
    kegg_p, kgg removed from the function vars
    """
    show_means_in_bplot = False
    loadings_dict = dict()

    pathology_label = 'pathology_state'

    # print(f"\t==> Processing: {path_name}")
    # ovlap = set(data.var.index.tolist()).intersection(kegg_p[kgg][1].values())
    ovlap = set(data.var.index.tolist()).intersection(gene_list)

    if len(ovlap) < 4:
        return None, None
    # retrieve genes of the pathways in the transcriptome
    genes_in, tmp_data = my_pars.find_overlap_in_dataset(data_in=data,
                                                        list_1=data.var.index,
                                                        list_2=gene_list)
    # exclude samples with all zeroes and genes non expressed
    samples_in, tmp_data = my_pars.exclude_non_variable_features(data_in=tmp_data)
    # generate annotation data
    tmp_df = pd.DataFrame(data=tmp_data, index=data.obs.index[samples_in], columns=data.var.index[genes_in])
    path_adata = andat.AnnData(X=tmp_df, obs=data.obs.loc[samples_in, :])

    # select the number of principal components to run the PCA
    comps_to_use = my_PCA.select_number_of_components(data_in=path_adata, comps=50)
    scanpy.pp.pca(path_adata, n_comps=comps_to_use, svd_solver='arpack', return_info=False)

    dims, tot_var = my_princ.get_total_variance_from_dimensions(var_in=path_adata.uns['pca']['variance_ratio'],
                                                                pathway=path_name, minimun_dim=pcs)
    pds_data = np.vstack(([path_adata.obsm['X_pca'][:, i]
                           for i in range(dims)])).T

    pcurve = my_princ.PCurve(x=pds_data, start=None, scale=False, stat_test=stat_test, plot_iter=False)
    pcurve.convert_npar_to_r()
    succeeded = pcurve.get_r_pcurve()

    if not succeeded:
        return None, None

    pcurve.get_centre_of_reference(series=path_adata.obs[pathology_label],
                                   ref_label='no_AD')

    pcurve.compute_pds(method='pathtracer')

    pcurve.separate_pds_by_condition(series=path_adata.obs[pathology_label],
                                     labels=['no_AD', 'early_AD', 'late_AD'])

    pvals, separation_works = pcurve.pds_statistical_test(conditions=['no_AD', 'early_AD', 'late_AD'],
                                                          col_name=path_name, equal_var=False,
                                                          save_dist=True, alpha=0.05, show_means=False,
                                                          path=destination_folder, **kwargs)

    if separation_works:
        # get loadings to measure genes contribution to the PCs
        df_loadings = pd.DataFrame(path_adata.varm['PCs'][:, :dims], index=path_adata.var_names,
                                   columns=['PC_{}'.format(i) for i in range(1, dims + 1)])
        df_rankings = pd.DataFrame((-1 * df_loadings.abs()).values.argsort(0).argsort(0),
                                   index=df_loadings.index, columns=df_loadings.columns)

        df_overall_ranking = pd.DataFrame((df_loadings.abs() *
                                           path_adata.uns['pca']['variance_ratio'][:dims]).sum(axis=1).sort_values(
            ascending=False))
        df_overall_ranking.columns = ['Gene_weight']
        loadings_dict = {'loadings': df_loadings,
                         'rankings': df_rankings,
                         'variance_fraction': path_adata.uns['pca']['variance_ratio'][:dims],
                         'overall_ranking': df_overall_ranking}
        del df_loadings, df_rankings, df_overall_ranking

    corr_df = pcurve.pds_correlation_report(adata=path_adata, cell_data=cell_anno, id_data=id_anno,
                                            clinical_data=clinical_anno, corr_method='spearman',
                                            traits=['amyloid', 'plaq_n', 'nft', 'tangles',
                                                    'cogn_global_lv', 'gpath', 'age_death'],
                                            traits_labels=['Amyloid level', 'Neuritic Plaque Burden',
                                                           'Neurofibrillary Tangle Burden', 'Tangle Density',
                                                           'Global Cognitive Function',
                                                           'Global AD-pathology Burden', 'Age'],
                                            col_name=path_name, average_obs=True)
    var_per_cond = pcurve.get_pds_variance_per_condition(col_name=path_name)
    mean_per_cond = pcurve.get_pds_mean_per_condition(col_name=path_name)
    fin_df = pd.concat([tot_var, mean_per_cond, var_per_cond, pvals, corr_df])
    if test_robustness:
        pcurve.test_dilution_of_the_data(series=path_adata.obs[pathology_label],
                                         ref_label='no_AD', number_of_tests=100)
        return fin_df, loadings_dict, pcurve.dilution
    else:
        return fin_df, loadings_dict
