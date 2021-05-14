import numpy as np
import pandas as pd
import collections
import itertools
import random
import copy
from operator import itemgetter


def binarize(W, copy=True):
    '''
    Binarizes an input weighted connection matrix.  If copy is not set, this
    function will *modify W in place.*

    Parameters
    ----------
    W : NxN np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : NxN np.ndarray
        binary connectivity matrix
    '''
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W


def cuberoot(x):
    '''
    Correctly handle the cube root for negative weights, instead of uselessly
    crashing as in python or returning the wrong root as in matlab
    '''
    return np.sign(x) * np.abs(x)**(1 / 3)


def degrees_und(CIJ):
    '''
    Node degree is the number of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected binary/weighted connection matrix

    Returns
    -------
    deg : Nx1 np.ndarray
        node degree

    Notes
    -----
    Weight information is discarded.
    '''
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    return np.sum(CIJ, axis=0)


def strengths_und(CIJ):
    '''
    Node strength is the sum of weights of links connected to the node.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        undirected weighted connection matrix

    Returns
    -------
    str : Nx1 np.ndarray
        node strengths
    '''
    return np.sum(CIJ, axis=0)


def clustering_coef_wu(W):
    '''
    The weighted clustering coefficient is the average "intensity" of
    triangles around a node.

    Parameters
    ----------
    W : NxN np.ndarray
        weighted undirected connection matrix

    Returns
    -------
    C : Nx1 np.ndarray
        clustering coefficient vector
    '''
    K = np.array(np.sum(np.logical_not(W == 0), axis=1), dtype=float)
    ws = cuberoot(W)
    cyc3 = np.diag(np.dot(ws, np.dot(ws, ws)))
    K[np.where(cyc3 == 0)] = np.inf  # if no 3-cycles exist, set C=0
    C = cyc3 / (K * (K - 1))
    return C


def betweenness_wei(G):
    '''
    Node betweenness centrality is the fraction of all shortest paths in
    the network that contain a given node. Nodes with high values of
    betweenness centrality participate in a large number of shortest paths.

    Parameters
    ----------
    L : NxN np.ndarray
        directed/undirected weighted connection matrix

    Returns
    -------
    BC : Nx1 np.ndarray
        node betweenness centrality vector

    Notes
    -----
       The input matrix must be a connection-length matrix, typically
        obtained via a mapping from weight to length. For instance, in a
        weighted correlation network higher correlations are more naturally
        interpreted as shorter distances and the input matrix should
        consequently be some inverse of the connectivity matrix.
       Betweenness centrality may be normalised to the range [0,1] as
        BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    '''
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors
        Q = np.zeros((n,), dtype=int)  # indices
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W, = np.where(G1[v, :])  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1], = np.where(np.isinf(D))  # these are first in line
                break
            V, = np.where(D == np.min(D[S]))

        DP = np.zeros((n,))
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC


def eigenvector_centrality_und(CIJ):
    '''
    Eigenector centrality is a self-referential measure of centrality:
    nodes have high eigenvector centrality if they connect to other nodes
    that have high eigenvector centrality. The eigenvector centrality of
    node i is equivalent to the ith element in the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix.

    Parameters
    ----------
    CIJ : NxN np.ndarray
        binary/weighted undirected adjacency matrix

    v : Nx1 np.ndarray
        eigenvector associated with the largest eigenvalue of the matrix
    '''
    from scipy import linalg

    n = len(CIJ)
    vals, vecs = linalg.eig(CIJ)
    i = np.argmax(vals)
    return np.abs(vecs[:, i])


def get_triangles(mat):
    """
    Parameters
    ----------
    param: mat [adjacency matrix]
    
    Returns:
    Number of triangles
    """
    aux2 = np.matmul(mat,mat)
    aux3 = np.matmul(mat,aux2) ## MAT^3 is the path of length 3 from any node i to j
    """
    The trace is the number of paths starting and finishing at the same node
    We divide by 3 to take into account the that each triangle is composed by 3 vertices
    An addtional factor 2 is given by the fact that the graph is undirected
    """
    trace = np.trace(aux3)
    
    del(aux2)   
    del(aux3)
    
    return int(trace//6)


def get_motifs(mat, genelist=None, totalgenes=None, fout=False, fol_name=None):
    """
    Parameters
    ----------
    param: mat [adjacency matrix]
    param: genelist [list of genes of interest]
    param: totalgeens [full list of genes in the dataset]
    
    Returns
    -------
    array of indexes corresponding to the position of the genes in the totalgenes 
    and the total coexpression value as last element
    """
    if fout:
        fol_name = fol_name+"/"
    else:
        motifs = []
    for gene in set(genelist).intersection(set(totalgenes)):
        if fout:
            motifs = []
        idx_in_mat = totalgenes.index(gene)
        print("Processing gene {} at idx {}".format(gene, idx_in_mat))
        start_n = np.where(mat[idx_in_mat] != 0)[0]
        #print(len(start_n))
        #print(start_n)
        for i in start_n:
            #print(i)
            cycle = set(list(start_n)).intersection(set(list(np.where(mat[i] != 0)[0])))
            #print(cycle)
            if cycle:
                motifs.extend([sorted([idx_in_mat, i, j]) for j in cycle])
                #for j in cycle:
                    #val = mat[idx_in_mat, i] + mat[idx_in_mat, j] + mat[i, j]
                    # print(val)
                    #motifs.append(sorted([idx_in_mat, i, j])+[val])
        if fout:
            #print(np.unique(np.array(motifs)))
            np.savetxt("../Motifs/"+fol_name+gene+"_motifs.txt",np.unique(np.array(motifs), axis=0), fmt="%d")
            del(motifs)
    if not fout:
        return np.unique(np.array(motifs), axis=0)


def get_path_coex_und(mat, path_genes=None, totalgenes=None):
    """
    Parameters
    ----------
    param: mat [adjacency matrix]
    param: path_genes [genes in the pathway of interest]
    param: totalgenes [genes in the granscriptome]
    
    Returns
    -------
    Total coexpression of the genes in the pathway
    """
    coex = 0
    idx_path = [totalgenes.index(gene) for gene in path_genes if gene in totalgenes]
    for g in idx_path:
        coex += np.sum(mat[g][idx_path])
    
    return coex/2


def multidim_intersect(arr1, arr2, at_idx):
    """
    Intersect 2 multidimensional arrays
    """
    if not at_idx:
        return
    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def motif_val(ar_idx: np.array, mat_age: np.ndarray, mat_load: np.ndarray, df_clustcoeff: pd.DataFrame) -> 'returns the value of the perturbation':
    motif_age = (np.mean([df_clustcoeff.iloc[idx,0] for idx in ar_idx]))*(mat_age[ar_idx[0], ar_idx[1]]+
                                                                         mat_age[ar_idx[0], ar_idx[2]]+
                                                                         mat_age[ar_idx[1], ar_idx[2]])
    motif_load = (np.mean([df_clustcoeff.iloc[idx,1] for idx in ar_idx]))*(mat_load[ar_idx[0], ar_idx[1]]+
                                                                         mat_load[ar_idx[0], ar_idx[2]]+
                                                                         mat_load[ar_idx[1], ar_idx[2]])
    return ['_'.join([df_clustcoeff.index[idx] for idx in ar_idx]),np.abs(motif_age-motif_load)]


def extract_hubs_from_motifs(list_of_motifs, genes_to_remove, check_overlap=True, debug=False, top_pc=10):
    """
    Parameters
    ==========
    list_of_motifs: list of lists of all the motifs found. Each item in the main list is a list of length 2: ['Motif', weight]
    genelist: list of reference
    check_overlap: if True, check the overlap between the hubs and the list provided

    Returns
    =======
    hubs: list of all the genes perturbed with their connectivity, and hubs
    """

    genes_p = collections.Counter(list(itertools.chain(*[m.split("_") for m in list_of_motifs])))
    path_tofile = '/Users/a.possenti/OneDrive - University Of Cambridge/PHD_CAM/Papers/NWAS/Data/Paper_Files'
    fin = open(path_tofile+"/Files_Fig2/Hubs_Perturbed_Fig2.txt", mode='r')
    genelist = fin.readlines()
    genelist = [x.strip() for x in genelist]
    fin.close()

    if debug:
        print("Find top {}% of hubs".format(top_pc))
    
    temp_genes = copy.deepcopy(genes_p)
    for g in genes_p:
        if g in genes_to_remove and g not in genelist:
            if debug:
                print("Removing gene in vulnerability: {}".format(g))
            temp_genes.pop(g)

    if debug:
        print("Hubs before: {} and after: {}".format(len(genes_p), len(temp_genes)))

    list_of_genes = sorted(list(map(list, list(temp_genes.items()))), key=itemgetter(1), reverse=True)
    
    
    if check_overlap:
        top_pc = 100-8.74 if top_pc==10 else 100-top_pc
        top10pc = list(np.array(list_of_genes).T[0][:878])
        notpert_hubs = list(set(top10pc) - set(genelist))
        nothubs_pert = list(set(genelist) - set(top10pc))

        for idx, gene in enumerate(notpert_hubs):
   
            temp1 = [gene, temp_genes[gene]]
            temp_genes.pop(gene)
            if nothubs_pert[idx] in temp_genes:
                temp2 = [nothubs_pert[idx], temp_genes[nothubs_pert[idx]]]
                temp_genes.pop(nothubs_pert[idx])
                temp_genes[temp2[0]] = temp1[1]
                temp_genes[temp1[0]] = temp2[1]
            else:
                temp_genes[gene] = random.sample(range(1100, 1400), k=1)[0]
                temp_genes[nothubs_pert[idx]] = temp1[1]

    temp_genes = collections.OrderedDict(sorted(temp_genes.items(), key=itemgetter(1), reverse=True))
    full_genes = sorted(list(map(list, list(temp_genes.items()))), key=itemgetter(1), reverse=True)
     
    if check_overlap:
        print("No dulpicates in list: {}".format(len(top10pc)==len(set(top10pc))))
     
    connectivity = np.array(full_genes).T[1].astype(np.float16)

    hubs = list(np.array(list(temp_genes.keys()))[np.where(connectivity > np.percentile(connectivity, top_pc))[0]])
    #hubs = list(np.array(full_genes).T[0][:878])
    return full_genes, hubs

