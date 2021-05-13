import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import sys
import os
import time
from operator import itemgetter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests as mlt_test
import scipy.stats
from scipy.optimize import leastsq
import pandas as pd
import itertools

import data_loader as myload


def ecdf_plot(df, cols_to_plot, grid_plot=None, return_val=False, log_ax=False, save_fig=False, **kwargs):
    """
    Plot the empiric cumulative density function, given a dataframe
    Parameters
    ----------
    df: df/dict
    cols_to_plot: list of cols to be plotted
                  the columns can be either columns of the df or keys of the dict
    return_val: [bool - default False] if True returns the value at which the distribution
                reaches the cutoff
    log_ax: True
    === kwargs ===
        cutoff: [float] threshold cutoff for the ecdf - default 0.95
    """

    # if isinstance(grid_plot, list):
    #     gx, gy = grid_plot[0], grid_plot[1]
    #
    # num_of_figs = int(len()
    # figs = {}
    # axs = {}

    print(f"Number of plots: {len(cols_to_plot)}")

    for idx, col in enumerate(cols_to_plot):

        print(f"Computing ecdf for {col}")

        # print("No grid_plot option!")
        ax=plt.axes()
        #figs[idx] = plt.figure()
        #axs[idx] = figs[idx].add_subplot(1, 1, 1)
        x = np.sort(df[col])
        y = np.arange(1, x.shape[0] + 1) / x.shape[0]
        # create ecdfplot

        #axs[idx].plot(x, y, marker='o', linestyle='')
        ax.scatter(x, y, marker='o')
        # axs[idx].set_xscale('log')
        # add plot labels
        plt.ylabel('ECDF')
        plt.xlabel(col)
        perc_val = x[y <= 0.80].max()
        plt.axhline(0.80, color='black', linestyle='--')
        plt.axvline(perc_val, color='black', label='95th percentile')
        #axs[idx].legend()
        ax.legend()
        #figs[idx].savefig("prova")
        #plt.clf()
        init = time.time()
        plt.savefig(f"prova_{idx}.png")
        print(time.time()-init)
        plt.cla()


def ecdf_plot_1(df, cols_to_plot, grid_plot=False, return_val=False, log_ax=False, save_fig=False, **kwargs):
    """
    Plot the empiric cumulative density function, given a dataframe
    Parameters
    ----------
    df: df/dict
    cols_to_plot: list of cols to be plotted
                  the columns can be either columns of the df or keys of the dict
    return_val: [bool - default False] if True returns the value at which the distribution
                reaches the cutoff
    log_ax: True
    === kwargs ===
        cutoff: [float] threshold cutoff for the ecdf - default 0.95
    """
    if 'cutoff' in kwargs:
        cutoff = kwargs['cutoff']
    else:
        cutoff = 0.95
    if return_val:
        vals_list = []
    if grid_plot:
        plt.figure(figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
        if len(cols_to_plot) > 1:
            x_dim, y_dim = (1, 3) if len(cols_to_plot) <= 3 else (int(len(cols_to_plot)/3), 3)
    else:
        figs = {}
        axs = {}

    print(f"Number of plots: {len(cols_to_plot)}")

    for idx, col in enumerate(cols_to_plot):

        print(f"Computing ecdf for {col}")

        # print("No grid_plot option!")
        figs[idx] = plt.figure(figsize=(10, 5))
        axs[idx] = figs[idx].add_subplot(1, 1, 1)
        # else:
        #     plt.subplot(x_dim, y_dim, idx+1)
        x = np.sort(df[col])
        y = np.arange(1, x.shape[0] + 1) / x.shape[0]
        # create ecdfplot

        axs[idx].plot(x, y, marker='o', linestyle='')
        # else:
        #     plt.plot(x, y, marker='o', linestyle='')
        if log_ax:
            #plt.semilogx(x, y)
            if not grid_plot:
                axs[idx].set_xscale('log')
            else:
                plt.gca().set_xscale("log")
        # add plot labels
        plt.ylabel('ECDF')
        #plt.xlabel(col)
        perc_val = x[y <= cutoff].max()
        plt.axhline(cutoff, color='black', linestyle='--')
        plt.axvline(perc_val, color='black', label='95th percentile')
        plt.legend()
        if return_val:
            vals_list.append(perc_val)
        if save_fig:
            print("Saving")
            figs[idx].savefig(fname=f"ecdf_{col}.pdf", bbox_inches='tight')
            # plt.show()
            plt.clf()
            print("Saved")
    # plt.show()
    plt.close()
    if return_val:
        return vals_list


def multi_lin_reg(y, x):
    """
    Given a train set and its labels, returns the results of a multiple linear regression
    
    Parameters
    ----------
    y = train-set labels
    x = train-set features
    
    Returns
    -------
    results - report table
    """
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results


def evaluate_model_acc(model, test_features, test_labels):
    """
    Given a model, returns its accuracy on the test set (features + labels)
    Parameters
    ----------
    model: model used for the predictions
    test_features: features to test the model on - ndarray
    test_labels: labels of the outcomes - 1darray
    
    Returns:
    Accuracy - in percent
    """ 
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} minutes.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


def fold_enrichment(backgr, my_list, path):
    """
    Parameters
    ----------
    backgr = list of all the genes in the background transcriptome
    my_list = my list of genes
    path = list of genes belonging 
    Return the fold-enrichment, computed as: FE = (m/n)/(M/N)
    Where N = num genes in backgr, M = num genes in path inside backgr, path = genes in my_list belonging to path, 
    my_list = my list to be checked 
    """

    N = len(backgr)
    M = len(set(path).intersection(set(backgr)))
    if M == 0:
        raise ZeroDivisionError
    n = len(my_list)
    m = len(set(path).intersection(set(my_list)))

    if M == 0:
        print("m: {}\nM: {}".format(m, M))

    return (m/n)/(M/N)


def fisher_ex(in_list, backgr, bio_path, sides='two-sided'):
    """
    Parameters
    ----------
    in_list = genes inside my list
    backgr = genes in backgr
    bio_path = genes in bio_path
    sides = ['two-sided', 'less', 'greater']
    
    Returns
    -------
    oddsratio and pvalue of the test
    """

    bio_path = list(set(bio_path).intersection(set(backgr)))
    outset = set(backgr)-set(in_list)

    x = len(set(in_list).intersection(set(bio_path)))
    m_x = len(set(bio_path).intersection(outset))
    k_x = len(in_list) - x
    n_k_x = len(outset - set(bio_path))

    oddsratio, pval = fisher_exact([[x, m_x], [k_x, n_k_x]], sides)

    return oddsratio, pval


def pvals_to_asterisk(val):
    if val > 5e-2:
        return "ns"
    elif val < 1e-4:
        return "****"
    elif val < 1e-3:
        return "***"
    elif val < 1e-2:
        return "**"
    else:
        return "*"


def jaccard_similarity(l1, l2):
    """
    Parameters
    ----------
    l1: list one
    l2: list two
    
    Returns
    -------
    jaccard similarity coefficient [float]. Which is the intersection of l1 and l2 divided by their union
    """
    return len(set(l1).intersection(l2)) / len(set(l1).union(l2))


def kegg_enrichment(genelist, backgr, kegg_paths, gene_id='gene_names',\
                    multitest=[True,'bonferroni'], side='greater', alpha = 0.01, debug=False):
    """
    Parameters
    ----------
    genelist: list of genes to test (can be entrezid or gene official names)
    backgr: list of genes representing the background
    kegg_paths: list of kegg pathways
    multitest: if True, must be a list with the second entry representing the type of correction;
               if False, provide just list with False
    
    Return
    ------
    enriched_keggs: sorted list of pathways enriched [hsaid, kegg_name, fold_enrichment, pvalue, list_of_genes]
    """
    
    enriched_keggs = []

    idx = 2 if gene_id == 'gene_names' else 1

    if debug:
        print("Kegg_id\tPath Name\tTotal genes")
    for kgg in kegg_paths:
        if debug:
            print("{}\t{}\t{}".format(kgg, kegg_paths[kgg][0], len(kegg_paths[kgg][idx]))) 
        ovlap = list(set(genelist).intersection(set(kegg_paths[kgg][idx])))
        if debug:
            print("Overlap: {}".format(ovlap))
        try:
            fe = fold_enrichment(backgr=backgr, my_list=genelist, path=kegg_paths[kgg][idx])
            if debug:
                print("Fold-Enrichment: {}".format(fe))
        except ZeroDivisionError:
            if debug:
                print("No entries of path {}".format(kegg_paths[kgg][0]))
            continue
        _, fishex = fisher_ex(in_list=genelist, backgr=backgr, bio_path=kegg_paths[kgg][2], sides=side)

        if multitest[0]:
            enriched_keggs.append([kgg, kegg_paths[kgg][0], fe, fishex, ','.join(ovlap)])
        elif fishex < alpha:
            enriched_keggs.append([kgg, kegg_paths[kgg][0], fe, fishex, ','.join(ovlap)])

    if not multitest[0]:
         enriched_keggs = sorted(enriched_keggs, key=itemgetter(2), reverse=True)
         return enriched_keggs

    if multitest[0]:
        pvals = np.array(enriched_keggs).T[3].astype(np.float16)
        passed_pvals = mlt_test(pvals, alpha=alpha, method=multitest[1], is_sorted=False, returnsorted=False)[0]
        corrected_pvals = mlt_test(pvals, alpha=alpha, method=multitest[1], is_sorted=False, returnsorted=False)[1]
        print("Total passed tests: {}".format(np.sum(passed_pvals)))
        
        for i, path in enumerate(enriched_keggs):
            path[3] = corrected_pvals[i]
        enriched_keggs = np.array(enriched_keggs)[passed_pvals]
        enriched_keggs = [list(item) for item in enriched_keggs]
        
        for item in enriched_keggs:
            item[2] = np.float16(item[2])
            item[3] = np.float16(item[3])
            item[4] = item[4].split(',')
        
        enriched_keggs = sorted(enriched_keggs, key=itemgetter(2), reverse=True)

        return enriched_keggs


from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.libqsturng import psturng
import warnings


def kw_dunn(groups, to_compare=None, alpha=0.05, method='bonf'):
    """
    Kruskal-Wallis 1-way ANOVA with Dunn's multiple comparison test
    Arguments:
    ---------------
    groups: sequence
        arrays corresponding to k mutually independent samples from
        continuous populations
    to_compare: sequence
        tuples specifying the indices of pairs of groups to compare, e.g.
        [(0, 1), (0, 2)] would compare group 0 with 1 & 2. by default, all
        possible pairwise comparisons between groups are performed.
    alpha: float
        family-wise error rate used for correcting for multiple comparisons
        (see statsmodels.stats.multitest.multipletests for details)
    method: string
        method used to adjust p-values to account for multiple corrections (see
        statsmodels.stats.multitest.multipletests for options)
    Returns:
    ---------------
    H: float
        Kruskal-Wallis H-statistic
    p_omnibus: float
        p-value corresponding to the global null hypothesis that the medians of
        the groups are all equal
    Z_pairs: float array
        Z-scores computed for the absolute difference in mean ranks for each
        pairwise comparison
    p_corrected: float array
        corrected p-values for each pairwise comparison, corresponding to the
        null hypothesis that the pair of groups has equal medians. note that
        these are only meaningful if the global null hypothesis is rejected.
    reject: bool array
        True for pairs where the null hypothesis can be rejected for the given
        alpha
    Reference:
    ---------------
    Gibbons, J. D., & Chakraborti, S. (2011). Nonparametric Statistical
    Inference (5th ed., pp. 353-357). Boca Raton, FL: Chapman & Hall.
    """

    # omnibus test (K-W ANOVA)
    # -------------------------------------------------------------------------

    groups = [np.array(gg) for gg in groups]

    k = len(groups)

    n = np.array([len(gg) for gg in groups])
    if np.any(n < 5):
        warnings.warn("Sample sizes < 5 are not recommended (K-W test assumes "
                      "a chi square distribution)")

    allgroups = np.concatenate(groups)
    N = len(allgroups)
    ranked = stats.rankdata(allgroups)

    # correction factor for ties
    T = stats.tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    # sum of ranks for each group
    j = np.insert(np.cumsum(n), 0, 0)
    R = np.empty(k, dtype=np.float)
    for ii in range(k):
        R[ii] = ranked[j[ii]:j[ii + 1]].sum()

    # the Kruskal-Wallis H-statistic
    H = (12. / (N * (N + 1.))) * ((R ** 2.) / n).sum() - 3 * (N + 1)

    # apply correction factor for ties
    H /= T

    df_omnibus = k - 1
    p_omnibus = stats.distributions.chi2.sf(H, df_omnibus)

    # multiple comparisons
    # -------------------------------------------------------------------------

    # by default we compare every possible pair of groups
    if to_compare is None:
        to_compare = tuple(combinations(range(k), 2))

    ncomp = len(to_compare)

    Z_pairs = np.empty(ncomp, dtype=np.float)
    p_uncorrected = np.empty(ncomp, dtype=np.float)
    Rmean = R / n

    for pp, (ii, jj) in enumerate(to_compare):

        # standardized score
        Zij = (np.abs(Rmean[ii] - Rmean[jj]) /
               np.sqrt((1. / 12.) * N * (N + 1) * (1. / n[ii] + 1. / n[jj])))
        Z_pairs[pp] = Zij

    # corresponding p-values obtained from upper quantiles of the standard
    # normal distribution
    p_uncorrected = stats.norm.sf(Z_pairs) * 2.

    # correction for multiple comparisons
    reject, p_corrected, alphac_sidak, alphac_bonf = multipletests(
        p_uncorrected, method=method
    )

    return H, p_omnibus, Z_pairs, p_corrected, reject


def general_fit_function(function, xdata, ydata, p_guess, yerr=None, xerr=None, p_boundaries=None,
                         local_params_indices=None, local_args_for_global=None, hop_around=True,
                         unbounded_proxy_value=999, add_mean_residual_to_yerr=True, do_data_resampling=False,
                         bootstrap_cycles=None, ci=0.95):
    '''
    return pfit_leastsq,perr_leastsq [, pfit_bootstrap, perr_bootstrap,pCI_down,pCI_up,ci_error_bars ] where the second part is returned only if bootstrap_cycles>0
      only if bootstrap it also computes equivalent error if xerr is given
    p_boundaries should be a list of length p_guess
     containing tuples with the boundaries, for speed reason the exact int value in unbounded_proxy_value should be given for unspecified limit
      e.g. p_boundaries=[(0,unbounded_proxy_value), (unbounded_proxy_value,unbounded_proxy_value),(2,4)]
    for global it is always function(x,p) and one can give local_params_indices to specify which parameters in p_guess
      are to be treated as local, otherwise all parameters are treated as global.
      There are two ways of entering local parameters in p_guess, the reccommended one is to enter one guess for each profile in ydata in the same order
       of the profiles. So if the function takes 2 local parameters these should be [ pl1_y1, pl2_y1, pl1_y2, pl2_y2,...] (y are the various profile)
         in this case the actual number of local parameters (for each profile in ydata) is then estimated from the number of local parameters in p_guess and the number of profiles in ydata.
        The second way is to give only one guess per local parameter (same guess for all y profiles) but it will work only if the number of profiles is greater than the number of local parameters.
      the x-axis may be shared by all functions and ydata can be a list of lists ro a 2D array
    ISSUE for global fit, if local parameters are not the last in parameter_guess will return them in different order with global parameters first and local last
     local_args_for_global is a list of argument that varies between lines in ydata (e.g. concentration of analyte in bli experiment)
     if given it is assumed that the function will be
     function(x,p,args) the number of args will be inferred by the number of lines in ydata vs the length of local_args_for_global so that nargs=len(local_args_for_global)/len(ydata)
     COULD ADD maxfev : int The maximum number of calls to the function. If zero, then 100*(N+1) is the maximum where N is the number of elements in x0.
    '''
    # Leastsq from scipy.optimize assumes that the objective function is based on the
    # difference between some observed target data (ydata) and a (non-linear)
    # function of the parameters `f(xdata, params)` :
    #       errfunc(params) = ydata - f(xdata, params)
    # so that the objective function is :
    #       Min   sum((ydata - f(xdata, params))**2, axis=0)
    #     params

    # print "DEB BB",yerr,ydata
    # print "BOH",yerr.shape,ydata.shape
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    p_guess = np.array(p_guess)
    # there are wierd array with shape (20,0) - if your ydatas don't have same length than ydata.shape will be
    # (nprofiles, ) rather than (nprofiles, profile_len)
    if (len(ydata.shape) > 1 and ydata.shape[0] > 1 and ydata.shape[1] > 1) \
            or local_args_for_global is not None or local_params_indices is not None:
        is_global = True
        if local_params_indices is None:
            npl = 0
        else:
            npl = len(local_params_indices)
        Ndof = len(ydata) * len(ydata[0]) - (len(p_guess) - npl + npl * len(
            ydata))  # approximation exact only if all profiles have same number of points
        # len(ydata)-len(p_guess)
    else:
        glob_loc_separator, num_local_params = None, None
        is_global = False
        Ndof = len(ydata) - len(p_guess)  # degrees of freedom
    if p_boundaries is not None:
        lower_bounds, upper_bounds = map(np.array, zip(*p_boundaries))
        # Below we don't give abs as we want the funciton to be derivable within the p_boundaries
        # args is given here as a proxy because if the fit is not global we give some None args to errfunc
        # (if global args are used and are not None, see if is_global below)
        errfunc = lambda p, x, y, *args: y - function(x, p) + 1e9 * \
                                         ((p > upper_bounds)[np.where(upper_bounds != unbounded_proxy_value)[0]]).sum() \
                                         + 1e9 * ((p < lower_bounds)[np.where(lower_bounds != unbounded_proxy_value)[0]]).sum()
    else:
        errfunc = lambda p, x, y, *args: y - function(x, p)  # this error function ignores the errors on the x and y
    if is_global:  # assumes y is a list of arrays or 2D array and x may or not be the same for all
        if p_boundaries is not None:
            sys.stderr.write("WARNING p_boundaries not yet supported in global fitting\n")
        if local_params_indices is not None:
            local_params_indices = np.array(local_params_indices)
            local_params_guess = p_guess[local_params_indices]
            p_global = np.array([p_guess[j] for j in range(len(p_guess)) if j not in local_params_indices])
            glob_loc_separator = len(p_global)
            p_guess = p_global
            if len(ydata) > len(
                    local_params_guess):  # assumes that only one guess has been given for one local parameters (otherwise a guess may given for its value in each profile)
                for j in range(len(ydata)): p_guess = np.hstack((p_guess, local_params_guess.copy()))
                num_local_params = len(local_params_guess)
            else:
                if len(local_params_guess) % (len(ydata)) != 0:
                    sys.stderr.write(
                        "\n**ERROR** in fit_function global local_params_guess is given but their number is not a multiple of len(ydata) [number of fits %d %d]\n" % (
                        len(local_params_guess), (len(ydata))))
                p_guess = np.hstack((p_guess,
                                     local_params_guess.copy()))  # no for loop has we already have a set of guesses per profile.
                num_local_params = len(local_params_guess) / (len(ydata))
        else:
            glob_loc_separator = len(p_guess)
            num_local_params = 0
        if local_args_for_global is not None and hasattr(local_args_for_global, '__len__'):
            nargs = len(local_args_for_global) / (len(ydata))
            if len(local_args_for_global) % (len(ydata)) != 0:
                sys.stderr.write(
                    "\n**ERROR** in fit_function global local_args_for_global is given but their number is not a multiple of len(ydata) [number of fits]\n")
            # this function is constructed so that the local arguments are already passed to the relevant set of data in the function
            if hasattr(xdata[0], '__len__'):  # x is not the same for all, different x are given for different curves.
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: \
                    np.concatenate(y - np.array([function(x[j], list(p)[:glob_loc_sep] + list(p[glob_loc_sep:])[
                                                                                         num_loc * j:num_loc * (j + 1)],
                                                          *local_args_for_global[j * nargs:(j + 1) * nargs]) for j in
                                                 range(len(y))]))
            else:
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: \
                    np.concatenate(y - np.array([function(x, list(p)[:glob_loc_sep] + list(p[glob_loc_sep:])[
                                                                                      num_loc * j:num_loc * (j + 1)],
                                                          *local_args_for_global[j * nargs:(j + 1) * nargs]) for j in
                                                 range(len(y))]))
        else:
            # maybe the numpy array below needs to be flatten
            if hasattr(xdata[0], '__len__'):
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: \
                    np.concatenate(y - np.array(
                        [function(x[j], list(p)[:glob_loc_sep] + list(p[glob_loc_sep:])[num_loc * j:num_loc * (j + 1)])
                         for j in range(len(y))]))
            else:
                errfunc = lambda p, x, y, glob_loc_sep, num_loc: \
                    np.concatenate(y - np.array(
                        [function(x, list(p)[:glob_loc_sep] + list(p[glob_loc_sep:])[num_loc * j:num_loc * (j + 1)]) for
                         j in range(len(y))]))

    ##################################################
    ## 1. COMPUTE THE FIT AND FIT ERRORS USING leastsq
    ##################################################

    # If using scipy.optimize.leastsq, the covariance returned is the
    # reduced covariance or fractional covariance, as explained
    # here :
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
    # One can multiply it by the reduced chi squared, s_sq, as
    # it is done in the more recenly implemented scipy.curve_fit
    # The errors in the parameters are then the square root of the
    # diagonal elements.
    # print "DEB fit",glob_loc_separator,'p_guess',p_guess,'num_local_params',num_local_params
    if hop_around is not None and len(p_guess) < 4:  # does it on grid around p_guess and save best solution
        if hasattr(hop_around, '__len__'):
            hop_factor, n_hops = hop_around
        else:
            if type(hop_around) is int or type(hop_around) is float:
                hop_factor = hop_around
            else:
                hop_factor = 10.  # one order of magintude more or less
            n_hops = max([20 / len(p_guess), 3])
        grid = cartesian_product(
            *[np.hstack((np.linspace(p / hop_factor, p, n_hops), np.linspace(p, p * hop_factor, n_hops)[1:])) for p in
              p_guess])
        best = [9e99, p_guess]
        for g in grid:
            pfit, pcov, infodict, errmsg, success = \
                leastsq(errfunc, g, args=(xdata, ydata, glob_loc_separator, num_local_params), full_output=1,
                        maxfev=100000)
            s_sq = (np.abs(errfunc(pfit, xdata, ydata, glob_loc_separator, num_local_params))).sum()
            # if all(g==numpy.array(p_guess)):
            #    print "GUESS",(numpy.abs(errfunc(numpy.array(p_guess), xdata, ydata,glob_loc_separator,num_local_params))).sum()
            if s_sq < best[0]:
                best = [s_sq, g, pfit, pcov, infodict, errmsg, success]
            #    print 'hopDEB ',g,s_sq,'BEST',errmsg, success
            # else : print 'hopDEB ',g,s_sq,errmsg, success
        ##
        # print("   fit hop_around [never changes sign or goes to zero from guess] (hop_factor,n_hops)=%s p_guess_"
        #      "input=%s comb_tried=%d" % (str((hop_factor, n_hops)), str(p_guess), len(grid)))
        #s_sq, p_guess, pfit, pcov, infodict, errmsg, success = best
        # print(" best =%s" % (str(p_guess)))
    else:
        pfit, pcov, infodict, errmsg, success = \
            leastsq(errfunc, p_guess, args=(xdata, ydata, glob_loc_separator, num_local_params), full_output=1,
                    maxfev=100000)
    nfev = infodict['nfev']
    # basin hopping??
    # OptimizeResult = scipy.optimize.basinhopping( errfunc, p_guess, args=(xdata, ydata,glob_loc_separator,num_local_params),full_output=1,maxfev=100000)
    # pfit=OptimizeResult.x
    # errmsg=OptimizeResult.message
    # success=OptimizeResult.success
    # nfev=OptimizeResult.nit
    ##
    '''UNCOMMENT FOR DEBUG!!!
    print "DEB fit success,nfev,errmsg:",success,nfev,errmsg
    '''
    s_sq = None
    if Ndof > 0 and pcov is not None:
        s_sq = (errfunc(pfit, xdata, ydata, glob_loc_separator, num_local_params) ** 2).sum() / (
            Ndof)  # almost the reduced Chi2
        pcov = pcov * s_sq

    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)  # perr = np.sqrt(np.diag(pcov)).
        except:
            error.append(0.00)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)
    if bootstrap_cycles == None or bootstrap_cycles == False:
        return pfit_leastsq, perr_leastsq, s_sq


import math
from scipy.optimize import newton
from scipy.special import digamma


def r_derv(r_var, vec):
    ''' Function that represents the derivative of the neg bin likelihood wrt r
    @param r: The value of r in the derivative of the likelihood wrt r
    @param vec: The data vector used in the likelihood
    '''
    #if not r_var or not vec:
    #    raise ValueError("r parameter and data must be specified")

    if r_var <= 0:
        raise ValueError("r must be strictly greater than 0")

    total_sum = 0
    obs_mean = np.mean(vec)  # Save the mean of the data
    n_pop = float(len(vec))  # Save the length of the vector, n_pop

    for obs in vec:
        total_sum += digamma(obs + r_var)

    total_sum -= n_pop*digamma(r_var)
    total_sum += n_pop*math.log(r_var / (r_var + obs_mean))

    return total_sum


def p_equa(r_var, vec):
    ''' Function that represents the equation for p in the neg bin likelihood wrt p
    @param r: The value of r in the derivative of the likelihood wrt p
    @param vec: Te data vector used in the likelihood
    '''
    #if not r_var or not vec:
    #    raise ValueError("r parameter and data must be specified")

    if r_var <= 0:
        raise ValueError("r must be strictly greater than 0")

    data_sum = np.sum(vec)
    n_pop = float(len(vec))
    p_var = 1 - (data_sum / (n_pop * r_var + data_sum))
    return p_var


def neg_bin_fit(vec, init=0.0001):
    ''' Function to fit negative binomial to data
    @param vec: The data vector used to fit the negative binomial distribution
    @param init: Set init to a number close to 0, and you will always converge
    '''
    if not isinstance(vec, (list, np.ndarray)):
        raise ValueError("Provide numpy array")

    est_r = newton(r_derv, init, args=(vec,))
    est_p = p_equa(est_r, vec)
    return est_r, est_p


corr_methods = ['pearson', 'spearman']


def compute_correlation(x, y, method='pearson', nan_policy='ignore', verbose=False):
    """
    Compute either the pearson's or the spearman's correlation.
    @param x, y: arrays of values
    @param method: method to use
    @param nan_policy: ['ignore', 'raise', 'impute'] 
                       - ignore: remove nans and compute correlation on the reduced array
                       - raise: stop calculating and raise error
                       - impute: impute the values according to some other method.
    Return:
    list: [correlation coefficient, pvalue]
    """
    if method.lower() not in corr_methods:
        raise ValueError(f"Method [{method.lower()}] not in options: [{', '.join(corr_methods)}]")
    x, y = np.array(x), np.array(y)
    nans = np.logical_or(np.isnan(x), np.isnan(y))
    if nan_policy == 'raise':
        sys.stderr.write("ERROR: found nan values in the input arrays\n")
        return
    elif nan_policy == 'ignore':
        if verbose:
            sys.stdout.write(f"Array length: {len(x)}. Total NaNs removed: {nans.sum()}\n")
        return scipy.stats.pearsonr(x[~nans], y[~nans]) if method == 'pearson' else scipy.stast.spearman(x[~nans], y[~nans])


def adjust_multiple_test(x):
    return min(0.99, x * 3) if not np.isnan(x) else np.nan
    # return min(0.99, x) if not np.isnan(x) else np.nan


def reverse_pval(x):
    """
    To test opposite hypothesis in the AD staging
    Parameters
    ----------
    x

    Returns
    -------

    """
    return 1 - x if not np.isnan(x) else np.nan


def replace_exact_num(x):
    """
    This is for the Liptak's method - throws error with exactly 1 as p-val
    Parameters
    ----------
    x

    Returns
    -------

    """
    num_to_rep = 0.99999999
    return x if x != 1. else num_to_rep


def put_back_ones(x):
    """
    This is for the Liptak's method - throws error with exactly 1 as p-val
    Parameters
    ----------
    x

    Returns
    -------

    """
    num_to_rep = 0.99999999
    return x if x != num_to_rep else 1.


def multiple_test_with_nans(x, correction_method: str, alpha: float = 0.05):
    """
    makes sure that nan values are remove when correcting for multiple tests
    Parameters
    ----------
    x
    correction_method
    alpha

    Returns
    -------

    """
    res = x
    tmp = x[~np.isnan(x)]
    tmp = mlt_test(tmp, method=correction_method, alpha=alpha)[1]
    res[~np.isnan(x)] = tmp
    return res


def combine_pvals_with_nans(x, combine_method, **kw):
    """
    Meta-analysis of the pvalues
    Parameters
    ----------
    x
    combine_method
    kw

    Returns
    -------

    """
    if combine_method != 'stouffer' and 'weights' in kw:
        raise AttributeError("weights are compatible with stouffer solution only")
    weights = kw.get('weights', None)
    tmp = x[~np.isnan(x)]
    keep_weights = weights[~np.isnan(x)]
    res = stats.combine_pvalues(tmp, combine_method, weights=keep_weights)[1]
    return res


def adjust_stat_significance(significance_dict: dict, print_significant: bool = True):
    """
    Adjust the statistical significance
    Parameters
    ----------
    significance_dict dictionary of (non)significant pathways
    print_significant whether to print to screen the total number of significant pathways

    Returns
    -------
    significance_dict with adjusted values added as additional column for each df

    """
    # these are obtained with Louvain clustering
    subclusters_size = {'Ex': {'Ex_0': 6343, 'Ex_1': 4392, 'Ex_2': 3452, 'Ex_3': 3162, 'Ex_4': 3147, 'Ex_5': 3135,
                               'Ex_6': 2941, 'Ex_7': 2601, 'Ex_8': 1823, 'Ex_9': 1675, 'Ex_10': 1130, 'Ex_11': 710,
                               'Ex_12': 444},
                        'In': {'In_0': 2183, 'In_1': 1586, 'In_2': 1152, 'In_3': 1052, 'In_4': 1028, 'In_5': 1015,
                               'In_6': 764, 'In_7': 412},
                        'Ast': {'Ast_0': 1157, 'Ast_1': 1580, 'Ast_3': 558, 'Ast_4': 96},
                        'Mic': {'Mic_0': 684, 'Mic_1': 518, 'Mic_2': 378, 'Mic_3': 196, 'Mic_4': 143},
                        'Oli': {'Oli_0': 10170, 'Oli_1': 4906, 'Oli_3': 2740, 'Oli_4': 419},
                        'Opc': {'Opc_0': 907, 'Opc_1': 718, 'Opc_2': 545, 'Opc_3': 457}}

    # fdr_tsbky method is ok
    weigh_populations = True

    for s in significance_dict:

        print(f"Condition [{s}]")
        # i should add a 1-pval step for Early-vs-late if not interested in the gradient
        for c in significance_dict[s]:
            if 'greater' in s:
                significance_dict[s][c] = significance_dict[s][c].applymap(reverse_pval)
            if c in ['Ex', 'Mic', 'Opc']:
                significance_dict[s][c] = significance_dict[s][c].applymap(adjust_multiple_test)
            if c in ['Ex', 'In', 'Oli', 'Ast']:
                significance_dict[s][c] = significance_dict[s][c].apply(
                    lambda x: multiple_test_with_nans(x, correction_method='fdr_tsbky'), axis=0)
            significance_dict[s][c] = significance_dict[s][c].applymap(replace_exact_num)
            weights = np.array(list(subclusters_size[c].values())) if weigh_populations else np.repeat(1, len(
                subclusters_size[c].keys()))
            significance_dict[s][c][f'{c} [cell type]'] = significance_dict[s][c][subclusters_size[c].keys()].apply(
                lambda x: combine_pvals_with_nans(x, 'stouffer', weights=weights), axis=1)
            # significance_dict[s][c] = significance_dict[s][c].applymap(put_back_ones)  # for consistency
            if print_significant:
                print(f"\tCell type [{c}] - Significant pathways: \
                      {(significance_dict[s][c][f'{c} [cell type]'] < 0.05).sum()}")

    return significance_dict


def check_pathway_recurrence_in_cluster(significance_dict: dict,
                                        alpha: float = 0.05,
                                        flag_paths_of_interest: bool = True,
                                        correct_mulitple_tests: bool = True,
                                        **kwargs):
    """
    Sort pathways by significance recurrence
    Parameters
    ----------
    significance_dict: dictionary of statistical significance
    alpha: significance value
    flag_paths_of_interest: add a column flagging if the pathway is of interest or not
    correct_mulitple_tests:

    Returns
    -------

    """

    if flag_paths_of_interest:
        paths_of_interest = kwargs.get('paths_of_interest_dict', None)
        assert isinstance(paths_of_interest, dict)

    over_represented_paths = dict()
    pvals_dict = dict()

    for s in significance_dict:
        over_represented_paths[s] = dict()
        for c in significance_dict[s]:
            subset_of_labs = pd.Index(myload.ordered_labs)[pd.Index(myload.ordered_labs).isin(significance_dict[s][c].columns)]
            over_represented_paths[s][c] = (significance_dict[s][c][subset_of_labs].iloc[:, :-1] < alpha).astype(int)
            over_represented_paths[s][c]['occurrences'] = (
                        significance_dict[s][c][subset_of_labs].iloc[:, :-1] < alpha).sum(axis=1)
            over_represented_paths[s][c] = over_represented_paths[s][c].sort_values(by='occurrences', ascending=False)

    for s in over_represented_paths:
        pvals_dict[s] = dict()
        for c in over_represented_paths[s]:
            mask_yes = over_represented_paths[s][c].index.isin(itertools.chain(*paths_of_interest.values()))
            pvals_dict[s][c] = min(1., stats.mannwhitneyu(
                over_represented_paths[s][c].loc[mask_yes, :]['occurrences'].values,
                over_represented_paths[s][c].loc[~mask_yes, :]['occurrences'].values,
                alternative='greater')[1])
            over_represented_paths[s][c]['in_list'] = np.where(mask_yes, 'yes', 'no')
            over_represented_paths[s][c]['in_list'] = pd.Categorical(over_represented_paths[s][c]['in_list'])

    if correct_mulitple_tests:
        multiple_tests_correct = kwargs.get('correction_method', None)
        assert multiple_tests_correct in ('bonferroni', 'bh')
        alpha = kwargs.get('alpha', 0.05)

        for s in pvals_dict:
            pvals_dict[s].update(dict(zip(pvals_dict[s].keys(),
                                          mlt_test(list(pvals_dict[s].values()),
                                                   alpha=alpha,
                                                   method=multiple_tests_correct)[1])))

    return over_represented_paths, pvals_dict


