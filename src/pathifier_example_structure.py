import os
import numpy as np
import scanpy
import pandas as pd
import bioinfo as mybio


import data_parser as mypars

scanpy.settings.verbosity = 1


# marker genes
gene_markers = {'Ex': ['NRGN', 'SLC17A7'], 'In': ['GAD1', 'GAD2'],  # neurons
                'Ast': ['AQP4', 'GFAP'],  # astrocytes
                'Oli': ['MBP', 'MOBP', 'PLP1'], 'Opc': ['VCAN', 'CSPG4', 'PDGFRA'],  # oligodendrocytes and progenitors
                'Mic': ['CSF1R', 'CD74', 'C3'],  # microglia
                'End': ['FLT1', 'CLDN5'],  # endothelial cells
                'Per': 'AMBP'}  # pericytes


# Annotations DBs
kegg_p = mybio.get_keggs(path_or_file='Default')
reactome_df = mybio.get_reactome(path_or_file='default',
                                 genes_id='ncbi',
                                 mask_species='homo sapiens')

reactome_all_path = dict()
for p in reactome_df.Pathway.unique():
    reactome_all_path[p] = reactome_df.loc[reactome_df.Pathway == p]['Name_Stable [Location]'].map(lambda x: x.split(' ')[0]).unique()

# some parameters initialisation
pds_report = dict()
all_robust = dict()
loadings_dfs = dict()

# test dilution to data
test_dilution = True
out_folder_name = None

test_reactome = False
test_kegg = True

assert not np.array([test_reactome, test_kegg]).all()  # test 1 annotation

if test_kegg:
    annotation_label = 'kegg'
    annotations = kegg_p

if test_reactome:
    test_kegg = False
    annotation_label = 'reactome'
    annotations = reactome_all_path

out_folder_name = 'Pathifier_KEGG_nTWAS' if test_kegg else 'Pathifier_Reactome-TEMP' if test_reactome else ''

principal_cs = 20  # for
statistical_test = 'mann-whitney-u'  # could be either t-test or mann-whitney-u (t-test not suitable for us)


main_clusters = ['Ex', 'In', 'Ast', 'Oli', 'Opc', 'Mic']

clusters_and_subc = dict(zip(main_clusters, ['Ex_0', 'In_1', 'Ast_2', 'Oli_0', 'Opc_1', 'Mic_0']))

pca_clusters = {'Ex_0': pd.DataFrame()}

# these are the metadata, cell ids mapping, and clinical traits excel files taken from Hansruedi et al Nature 2019
md_filtered, id_map, clin_traits = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# main script structure
for c in main_clusters:
    # create folder if needed
    print(f"Analysing cell type: [{c}]")
    path = os.path.join(os.getcwd(), out_folder_name, c)
    if not os.path.isdir(path):
        os.makedirs(path)
    for subc in clusters_and_subc[c]:
        all_robust[subc] = dict()
        loadings_dfs[subc] = dict()

        # create subfolder if needed
        print(f"\tAnalysing cluster: [{subc}]")
        subpath = os.path.join(path, subc)
        if not os.path.isdir(subpath):
            os.makedirs(subpath)

        # create pds report
        pds_report[subc] = mypars.create_empty_report(dataset='mathys', statistics=statistical_test)
        dh, t_0 = mypars.create_status_bar()
        for i, k in enumerate(annotations):
            if annotation_label == 'kegg':
                genes = kegg_p[k][1].values()
                pathway = kegg_p[k][0]
                # print(f"\t\tTesting {pathway}")
            elif annotation_label == 'reactome':
                # genes = get_reactome_genes_from_path(reactome_db=reactome_df, pathway=k) #ER pathways
                pathway = k
                genes = annotations[k].tolist()

            if test_dilution:
                res, loadings, robustness = mybio.run_pathifier(data=pca_clusters[subc],
                                                                cell_anno=md_filtered,
                                                                id_anno=id_map,
                                                                clinical_anno=clin_traits,
                                                                gene_list=genes,
                                                                path_name=pathway,
                                                                stat_test=statistical_test,
                                                                destination_folder=subpath,
                                                                pcs=principal_cs, test_robustness=True,
                                                                database_dimension=len(annotations))
                if res is not None:
                    all_robust[subc].update({k: robustness})
            else:

                res, loadings = mybio.run_pathifier(data=pca_clusters[subc],
                                                    cell_anno=md_filtered,
                                                    id_anno=id_map,
                                                    clinical_anno=clin_traits,
                                                    gene_list=genes,
                                                    path_name=pathway,
                                                    stat_test=statistical_test,
                                                    destination_folder=subpath,
                                                    pcs=principal_cs,
                                                    test_robustness=False,
                                                    database_dimension=len(annotations))
            if res is not None:
                pds_report[subc] = pds_report[subc].join(res, how='inner')
                if loadings:
                    loadings_dfs[subc][pathway] = loadings
            mypars.update_progress(i / len(annotations), disp_inst=dh, init_time=t_0)
        mypars.update_progress(1, disp_inst=dh, init_time=t_0)
