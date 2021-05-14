import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from sklearn.preprocessing import StandardScaler
import rpy2
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from scipy.optimize import minimize
from scipy import spatial
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import stats
from sklearn.preprocessing import scale
import itertools
import re
import textwrap
import random
from typing import Optional


def pvals_to_asterisk(pval: float):
    assert isinstance(pval, float), "[{PVAL}] incorrect format. Float expected, it's {TYPE}".format(PVAL=pval,
                                                                                                    TYPE=type(pval))
    if pval > 5e-2:
        return "ns"
    elif pval < 1e-4:
        return "****"
    elif pval < 1e-3:
        return "***"
    elif pval < 1e-2:
        return "**"
    else:
        return "*"


def _annotate_pval(pds_df: pd.DataFrame, sign_pval_idxs: np.array, pvals: np.array, axis):
    """
    :param pds_df: dataframe with pathway deregulation scores
    :param sign_pval_idxs: array of indices where significance is met
    :param pvals: asterisks
    :param axis: ax
    :return: maximum y to extend the plotting
    """
    pval_idx_to_boxp_idx = {0: [0, 1], 1: [0, 2], 2: [1, 2]}
    internal_indices = list(sign_pval_idxs)
    if 1 in sign_pval_idxs and (len(sign_pval_idxs) == 3 or (len(sign_pval_idxs) == 2 and 2 in sign_pval_idxs)):
        internal_indices.remove(1)
        internal_indices.append(1)

    y_max_idx = np.argmax([pds_df[pds_df.columns[0]].max(), pds_df[pds_df.columns[1]].max(),
                           pds_df[pds_df.columns[2]].max()])
    y_max = pds_df[pds_df.columns[y_max_idx]].max()

    shift = 0.1 if y_max <= 4 else 0.2 if 4 < y_max < 8 else 0.4
    h = 0.1
    y = 0

    for idx in internal_indices:
        i1, i2 = pval_idx_to_boxp_idx[idx]
        if y == 0:
            if 1 in internal_indices and y_max_idx in [0, 1]:
                y = pds_df[pds_df.columns[y_max_idx]].max() + shift
            elif set([0, 1]) == set(internal_indices) and y_max_idx == 2:
                y = pds_df[pds_df.columns[y_max_idx]].max() + shift
            else:
                y = max(pds_df[pds_df.columns[i1]].max(), pds_df[pds_df.columns[i2]].max()) + shift
        else:
            y = y + shift + h
        axis.plot([i1, i1, i2, i2], [y, y + h, y + h, y], lw=1.5, c='k')
        axis.text((i1 + i2) * .5, y + h, "{}".format(pvals[idx]), ha='center', va='bottom', c='k')


def _annotate_pval_grubman(pds_df: pd.DataFrame, pvals: np.array, axis):
    y_max = max(pds_df.iloc[:, 0].max(), pds_df.iloc[:, 1].max())
    shift = 0.1 if y_max <= 4 else 0.2 if 4 < y_max < 8 else 0.4
    h = 0.1
    y = y_max + shift + h
    axis.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, c='k')
    axis.text((0 + 1) * .5, y + h, "{}".format(pvals[0]), ha='center', va='bottom', c='k')


print(f"Rpy2 version required: [{rpy2.__version__}]")
r_smooth_spline = robj.r['smooth.spline']
r_predict = robj.r['predict']
r_optimize = robj.r['optimize']
r_float = robj.FloatVector
r_pcurve = importr('princurve', on_conflict="warn")
r_cbind = robj.r['cbind']


def get_total_variance_from_dimensions(var_in: pd.Series, pathway: str, minimun_dim: int):
    dimensions = min(minimun_dim, len(var_in))
    while var_in[:dimensions].sum() < 0.6 and dimensions < len(var_in):
        dimensions += 1
    var_df = pd.DataFrame.from_dict({'PCs': int(dimensions), 'explained_var': (var_in[:dimensions] * 100).sum()},
                                    orient='index', columns=[pathway])
    return dimensions, var_df


accepted_stat_tests = ('t-test', 'mann-whitney-u')


class PCurve(object):

    it = 0

    def __init__(self, x: np.ndarray, start: np.ndarray, scale: bool = False,
                 thresh: float = 0.001, max_it: int = 10, stat_test: str = 't-test', plot_iter: bool = False,
                 dataset: str = 'mathys'):
        """
        Compute the principal curve given a set of data in a matrix. It is a non-parametric non-linear regression.
        :param x: matrix with data points
        :param start: either an array with coordinates to start the curve. If [None] use the 1st principal component
        :param scale: if True, scale the values with the std (ddof=1)
        :param thresh: convergence threshold on shortest distance to the curve.
        :param max_it: maximum number of iterations
        :param stat_test: statistical test to assess the difference
        :param plot_iter: plot the iterations
        :return: an object: - matrix correspodning to x, giving their projection onto the curve
                            - an index, such that the curve is smooth
                            - lambda, from each point, its arc-length from the beginning of the curve. The curve is
                              parametrised approximately by arc-length
                            - the sum-of-squared distances from the points to their projections
                            - a flag, stating whether the algorithm converged
                            - number of iterations
        """
        self.data = x
        self.scale = scale

        self.x_star, self.d_mean = self.scale_data()

        self.thresh = thresh
        self.max_iter = max_it

        self.plot = plot_iter
        self.start = start

        self.check_format()  # check the format
        self.pcurve = dict()

        # R specific quantities
        self.r_data = None
        self.r_pc_object = None
        self.projected_points = None
        self.ordered_curve = None

        self.ref_centre = None
        self.pds = None
        self.pds_per_condition = None

        assert stat_test in accepted_stat_tests, 'Error: available options [{}]'.format(*accepted_stat_tests)
        self.stat_test = stat_test

        # test dilution of the data
        self.dilution = []

        assert dataset in ('mathys', 'grubman'), "WRONG DATASET"
        self.dataset = dataset

    def check_format(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Item [{self.data}] must be of type np.ndarray")

        if isinstance(self.start, np.ndarray):
            raise NotImplementedError(f"The option to give a custom [start] has not been implemented yet. "
                                      f"Leave it to [None] and use the 1st principal component.")

    def scale_data(self):
        mean = self.data.mean(axis=0)  # column-wise mean
        normalised = (self.data - mean) / self.data.std(axis=0, ddof=1) if self.scale else self.data - mean
        return normalised, mean

    def convert_npar_to_r(self):
        tmp = [r_float(self.data.T[j]) for j in range(self.data.shape[1])]
        self.r_data = r_cbind(*tmp)

    def get_r_pcurve(self):

        try:
            self.r_pc_object = r_pcurve.principal_curve(self.r_data, stretch=3)
        except rpy2.rinterface.RRuntimeError:
            return False
        self.pcurve = dict(zip(self.r_pc_object.names,
                               map(list, list(self.r_pc_object))))
        self.pcurve = dict(zip(list(self.pcurve.keys()),
                               map(lambda x: np.array(self.pcurve[x]), self.pcurve.keys())))
        order = self.pcurve['ord'] - 1  # R indexing starts from 1
        self.projected_points = np.reshape(np.array(self.pcurve['s']), newshape=(self.data.shape[1], -1)).T
        self.ordered_curve = self.projected_points[order]
        return True

    @staticmethod
    def map_pred_to_dict(r_prediction):
        return dict(zip(r_prediction.names, map(list, list(r_prediction))))

    @staticmethod
    def extend_lambda_range(lam: np.array, fraction: float = 0.5):
        interval = abs(max(lam) - min(lam)) * fraction
        return np.linspace(min(lam)-interval, max(lam)+interval, len(lam))

    def get_centre_of_reference(self, series: pd.Series, ref_label: str, dilution_test: bool = False):
        if not dilution_test:
            avg_ref = self.projected_points[(series == ref_label).values].mean(axis=0)
            centre_projected_on_curve = self.ordered_curve[spatial.KDTree(self.ordered_curve).query(avg_ref)[1]]
            self.ref_centre = centre_projected_on_curve
        else:
            bool_labels = (series == ref_label).values
            trues_idxs = np.where(bool_labels)[0]
            sampled_idx = random.sample(list(trues_idxs), int(trues_idxs.shape[0]/2))
            bool_labels[sampled_idx] = False
            avg_ref = self.projected_points[bool_labels].mean(axis=0)
            centre_projected_on_curve = self.ordered_curve[spatial.KDTree(self.ordered_curve).query(avg_ref)[1]]
            return centre_projected_on_curve

    def compute_pds(self, method='pathifier', dilution_test: bool = False, **kwargs):
        if method == 'pathifier':
            assert dilution_test is False, "Dilution test not implemented for this method yet"
            # TO SPEED UP THE SEARCH BELOW ONE COULD approximate it with np.where and a threshold
            near_neigh_dist = np.sqrt((np.diff(self.ordered_curve, axis=0) ** 2).sum(axis=1))
            near_neigh_dist = np.append(near_neigh_dist, 0.)
            ref_idx = np.where(self.ordered_curve == self.ref_centre)[0][0]
            nn_on_curve_per_pt = [self.ordered_curve[spatial.KDTree(self.ordered_curve).query(pt)[1]]
                                  for pt in self.projected_points]
            idx_on_curve_per_pt = [np.where(self.ordered_curve == pt)[0][0] for pt in nn_on_curve_per_pt]

            self.pds = np.array([near_neigh_dist[i:ref_idx].sum()
                                 if i < ref_idx else near_neigh_dist[ref_idx:i].sum() for i in idx_on_curve_per_pt])
        elif method == 'pathtracer':  # much faster method
            if dilution_test:
                ref_centre = kwargs.get('ref_centre')
                return np.linalg.norm(self.projected_points - ref_centre, axis=1)
            else:
                self.pds = np.linalg.norm(self.projected_points - self.ref_centre, axis=1)

        else:
            raise NotImplementedError

    def separate_pds_by_condition(self, series: pd.Series, labels: list):
        self.pds_per_condition = pd.DataFrame.from_dict(dict(zip(labels,
                                                                 [self.pds[(series == lab).values] for lab in labels])),
                                                        orient='index').T

    def get_pds_variance_per_condition(self, col_name: str, labels: list = None):
        if self.pds_per_condition is None:
            raise ValueError("PDS not computed per each condition yet")
        res = self.pds_per_condition.apply(np.nanvar, axis=0).to_frame()
        res.columns = [col_name]
        res.index = ['no_AD_var', 'early_AD_var', 'late_AD_var'] if labels is None else labels
        return res

    def get_pds_mean_per_condition(self, col_name: str, labels: list = None):
        if self.pds_per_condition is None:
            raise ValueError("PDS not computed per each condition yet")
        res = self.pds_per_condition.apply(np.nanmean, axis=0).to_frame()
        res.columns = [col_name]
        res.index = ['no_AD_mean', 'early_AD_mean', 'late_AD_mean'] if labels is None else labels
        return res

    def test_dilution_of_the_data(self, series: pd.Series, ref_label: str, number_of_tests: int = 10):
        tests = []
        for i in range(number_of_tests):
            ref_centre = self.get_centre_of_reference(series=series, ref_label=ref_label, dilution_test=True)
            tests.append(self.compute_pds(method='pathtracer', dilution_test=True, ref_centre=ref_centre))

        self.dilution = list(map(lambda x: stats.pearsonr(self.pds, x)[0], tests))

    def pds_statistical_test(self, conditions: list, col_name: str, equal_var: bool = True,
                             save_dist: bool = False, alpha: float = 0.05,
                             show_means: bool = False, path: str = None, **kwargs):

        color_map = {'no_AD': '#599B67', 'early_AD': '#E7E35F', 'late_AD': '#EE1B34'}
        pairs = [(x, y) for i, x in enumerate(conditions) for j, y in enumerate(conditions[1:]) if i <= j]
        stat_lab = 't_stat' if self.stat_test == 't-test' else 'u_stat' if self.stat_test == 'mann-whitney-u' else None
        labs = [['-'.join(['pval', x, 'vs', y]), '-'.join([stat_lab, x, 'vs', y])] for (x, y) in pairs]

        labs = list(itertools.chain(*labs))
        pvals = []
        for p in pairs:
            if self.stat_test == 't-test':
                assert (np.array([stats.normaltest(self.pds_per_condition[p[0]].values)[1],
                                 stats.normaltest(self.pds_per_condition[p[1]].values)[1]]) > 0.05).all(), "NOT NORMAL"
                pvals.extend(reversed(list(stats.ttest_ind(self.pds_per_condition[p[0]].values,
                                                           self.pds_per_condition[p[1]].values,
                                                           equal_var=equal_var, nan_policy='omit'))))
            elif self.stat_test == 'mann-whitney-u':
                pvals.extend(reversed(list(stats.mannwhitneyu(self.pds_per_condition[p[0]].dropna().values,
                                                              self.pds_per_condition[p[1]].dropna().values,
                                                              use_continuity=True, alternative='less'))))

        res = dict(zip(labs, pvals))
        res = pd.DataFrame.from_dict(res, orient='index', columns=[col_name])
        mask = False

        if save_dist:

            tests_num = kwargs.get('database_dimension') if 'database_dimension' in kwargs else 1
            pvals = res[res.index.str.contains('pval')]
            t_or_u_stats = res[res.index.str.contains(stat_lab)]
            any_sign = (pvals * len(pvals) * tests_num < alpha).any().values[0]
            if self.stat_test == 't-test':
                all_ts_ltzero = (t_or_u_stats < 0).all().values[0]
                mask = any_sign and all_ts_ltzero
            elif self.stat_test == 'mann-whitney-u':
                if self.dataset == 'grubman':
                    # ordered_medians = self.pds_per_condition.iloc[:, 0].median() \
                    #                  < self.pds_per_condition.iloc[:, 1].median()
                    ordered_medians = True

                else:
                    ordered_medians = self.pds_per_condition.iloc[:, 0].median() \
                                      < self.pds_per_condition.iloc[:, 1].median() \
                                      < self.pds_per_condition.iloc[:, 2].median()
                mask = any_sign and ordered_medians

            pvals = pvals.values.flatten() * len(pvals) * tests_num

            if mask:
                fig, ax = plt.subplots(figsize=(6, 6))
                bplot = sns.boxplot(x="variable", y="value", data=pd.melt(self.pds_per_condition),
                                    showmeans=show_means, notch=True, ax=ax)
                for i in range(0, self.pds_per_condition.shape[1]):
                    mybox = bplot.artists[i]
                    mybox.set_facecolor(color_map[self.pds_per_condition.columns[i]])
                # annotate pvalues on top of the boxplot
                symbolic_pval = list(map(pvals_to_asterisk, pvals))
                if self.dataset == 'mathys':
                    _annotate_pval(self.pds_per_condition, np.where(pvals < alpha)[0], symbolic_pval, axis=ax)
                elif self.dataset == 'grubman':
                    _annotate_pval_grubman(self.pds_per_condition, pvals=symbolic_pval, axis=ax)

                # plt.title(f'P-value: HC_early={pvals[0]:.2e} - HC_late={pvals[1]:.2e} - early_late={pvals[2]:.2e}')
                title_str = textwrap.fill(col_name, 55)
                plt.title(f"{title_str}", size=12)
                plt.xlabel('Condition', size=12)
                plt.ylabel('Pathway Deregulation Score', size=12)
                plt.tight_layout()

                if path is not None:
                    plt.savefig(f"{path}/{col_name.replace('/', '_')}.pdf", bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

        path_separate_hc_from_disease = (res.iloc[res.index.str.contains('pval-no_AD'), :] *
                                         len(pvals) * tests_num < alpha).any().values[0]
        path_separate_hc_from_disease = True # temporary for nTWAS

        return res, path_separate_hc_from_disease

    def correlate_pds_with_trait(self, adata, cells_df,  clinical_df, trait, id_df: Optional[pd.DataFrame],
                                 method='spearman', avg_per_person: bool = False, **kwargs):
        if self.dataset == 'grubman':
            subjects = cells_df[cells_df.index.isin(adata.obs.index)].patient

        else:
            projids = cells_df[cells_df.TAG.isin(adata.obs.index)]['projid']
            proj_to_subj = id_df[id_df.projid.isin(projids.unique())][['projid', 'Subject']].drop_duplicates()
            subjects = projids.map(dict(zip(proj_to_subj.projid.values, proj_to_subj.Subject.values)))

        if not avg_per_person:
            trait_val = subjects.map(dict(zip(clinical_df.Subject.values,
                                              clinical_df[trait].values))).values
            if method == 'spearman':
                return stats.spearmanr(self.pds, trait_val)[0]
            elif method == 'pearson':
                return stats.pearsonr(self.pds, trait_val)[0]
        else:
            assert 'norm_traits' in kwargs, "ERROR: Must provide the normalised traits as a data-frame"
            traits_tmp = kwargs['norm_traits']
            tmp = pd.DataFrame(np.array([self.pds, subjects.values]).T, columns=['pds', 'subject'])
            tmp['pds'] = pd.to_numeric(tmp['pds'])
            tmp = tmp.groupby(['subject']).mean()
            trait_val = traits_tmp.loc[tmp.index, trait]
            if self.dataset == 'grubman':
                unidentified = tmp.index[tmp.index.isin(['AD-un', 'Ct-un'])]
                trait_val = trait_val.drop(unidentified)
                tmp = tmp.drop(unidentified)
            if method == 'spearman':
                return stats.spearmanr(tmp.pds, trait_val, nan_policy='raise')[0]
            elif method == 'pearson':
                return stats.pearsonr(tmp.pds, trait_val, nan_policy='raise')[0]

    def pds_correlation_report(self, adata, cell_data: pd.DataFrame, clinical_data: pd.DataFrame,
                               corr_method: str, traits: list, traits_labels: list, col_name: str,
                               id_data: Optional[pd.DataFrame] = None, average_obs=False, **kwargs):
        if average_obs:
            traits_tmp = clinical_data[['Subject'] + traits].copy(deep=True)
            traits_tmp.set_index(keys='Subject', drop=True, inplace=True)
            scaled_vals = scale(traits_tmp.values, with_mean=True, with_std=True)
            traits_tmp[:] = scaled_vals
        if not average_obs:
            c_vals = list(map(lambda x: self.correlate_pds_with_trait(adata=adata, cells_df=cell_data,
                                                                      id_df=id_data, clinical_df=clinical_data,
                                                                      trait=x, method=corr_method,
                                                                      avg_per_person=average_obs, **kwargs),
                              traits))
        else:
            c_vals = list(map(lambda x: self.correlate_pds_with_trait(adata=adata, cells_df=cell_data,
                                                                      id_df=id_data, clinical_df=clinical_data,
                                                                      trait=x, method=corr_method,
                                                                      avg_per_person=average_obs,
                                                                      norm_traits=traits_tmp, **kwargs),
                              traits))

        phen_dict = dict(zip(traits_labels, c_vals))
        res = pd.DataFrame.from_dict(phen_dict, orient='index', columns=[col_name])
        return res

    def bias_correct(self):
        ones = np.ones(self.data.shape[1])
        sbar = self.data.mean(axis=0)
        # TODO: finish the bias correction curve (https://rdrr.io/cran/princurve/src/R/bias_correct_curve.R)

    def project_onto_curve(self, s: np.ndarray, stretch: float = 1.):
        print(s.shape)
        if stretch > 0.:
            n = int(s.shape[0])
            diff1 = s[0, :] - s[1, :]
            print(diff1)
            diff2 = s[n-1, :] - s[n-2, :]
            s[0, :] = s[0, :] + stretch * diff1
            s[n-1, :] = s[n-1, :] + stretch * diff2
        elif s < 0.:
            raise ValueError("Argument [stretch] should be in the range [0-2]")

        nseg = s.shape[0] - 1
        npts = self.data.shape[0]
        ncols = self.data.shape[1]
        print(nseg)

        if s.shape[1] != ncols:
            raise Warning('[x] and [s] must have an equal number of columns')

        diff = np.empty(shape=(nseg, ncols))
        length = np.empty(nseg)

        #  pre-compute distances between successive points in the curve and the length of each segment

        for i in range(nseg):
            diff[i, :] = s[i+1, :] - s[i, :]
            length[i] = (diff[i, :] ** 2).sum()

            #ll = 0.
            #for k in range(ncols):
            #    value = s[i + 1, k] - s[i, k]
            #    diff[k * nseg + i] = value ## ??
            #    ll += value ** 2
            #length[i] = ll

        new_s = np.empty(shape=(npts, ncols))
        lam = np.empty(ncols)
        dist_ind = np.empty(npts)

        n_test = np.empty(ncols)
        n = np.empty(ncols)
        p = np.empty(ncols)

        for i in range(npts):
            best_lam = -1
            best_di = np.inf
            for k in range(ncols):
                p[k] = self.data[i, k]

            for j in range(nseg):
                t = 0.
                for k in range(ncols):
                    t += diff[j, k] * (p[k] - s[j, k])

                t /= length[j]  # ??
                if t < 0.:
                    t = 0.
                if t > 1.:
                    t = 1.0

                di = 0.
                for k in range(ncols):
                    value = s[j, k] + t * diff[j, k]
                    n_test[k] = value
                    di += (value - p[k]) ** 2

                if di < best_di:
                    best_di = di
                    best_lam = j + .1 + .9 * t
                    for k in range(ncols):
                        n[k] = n_test[k]

            lam[i] = best_lam
            dist_ind[i] = best_di
            for k in range(ncols):
                new_s[k * npts + i] = n[k]

        new_ord = np.argsort(lam)
        dist = np.sum(dist_ind)

        lam[new_ord[0]] = 0
        for i in range(1, len(new_ord)):
            o1 = new_ord[i]
            o0 = new_ord[i-1]
            o0o1 = 0.

            for k in range(ncols):
                val = new_s[o1, k] - new_s[o0, k]
                o0o1 += val ** 2

            lam[o1] = lam[o0] + np.sqrt(o0o1)

        ret = dict()
        ret.setdefault('s', new_s)
        ret.setdefault('order', new_ord + 1)
        ret.setdefault('lambda', lam)
        ret.setdefault('dist_ind', dist_ind)
        ret.setdefault('tot_dist', dist)

        return ret

    def stack_fits_together(self, lam, fs):

        return np.vstack([self.map_pred_to_dict(r_predict(f, r_float(lam)))['y'] for f in fs]).T

    def euclidean_distance_from_curve(self, lam, f1, f2):

        return ((np.vstack((self.map_pred_to_dict(r_predict(f1, r_float(lam)))['y'],
                            self.map_pred_to_dict(r_predict(f2, r_float(lam)))['y'])).T - self.x_star)**2).sum()

    def euclidean_distance_from_curve_v2(self, lam, fs):

        lam_mat = self.stack_fits_together(lam=lam, fs=fs)

        return ((lam_mat - self.x_star)**2).sum()

    def initialise_curve(self):
        if self.start is None:  # scale the values and compute the SVD

            U, s_values, Vh = np.linalg.svd(self.x_star)
            # build the proper S matrix, so that U.dot(S).dot(Vt) = X_star
            S = np.zeros(shape=(U.shape[0], s_values.shape[0]))
            np.fill_diagonal(S, s_values)
            lam = U.T[0] * s_values[0]  # arc-lengths associated with orthogonal projections of data onto curve
            order = np.argsort(U.T[0])
            first_pc = np.outer(lam, Vh[0])  # these are the projections of the points onto the 1st component
            m = (first_pc[-1][1]-first_pc[0][1]) / (first_pc[-1][0]-first_pc[0][0])
            # s = first_pc * self.data.std(axis=0, ddof=1) - (-self.d_mean) if self.scale else first_pc - (-self.d_mean)
            # dist = (s_values ** 2)[-1] * self.data.shape[0]
            s = first_pc
            projection = self.x_star.dot(Vh[0].reshape(self.x_star.shape[1], -1)).dot(Vh[0].reshape(1, -1))
            dist = ((self.x_star - projection) ** 2).sum()
            print(f'Distance {dist}')
            self.pcurve = {'s': s, 'order': order, 'lambda': lam,
                           'tot_dist': dist, 'has_converged': False,
                           'curve_coordinates': None, 'projection': None}

            if self.plot and self.pcurve:

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(self.x_star.T[0], self.x_star.T[1])

                # plot the 1st component
                # x, y = self.pcurve['s'][self.pcurve['order']].T[0], self.pcurve['s'][self.pcurve['order']].T[1]
                # ax.plot(x, y, lw=2)
                rr = np.arange(min(self.x_star.T[0]) - 0.5, max(self.x_star.T[0]) + 0.5, 0.1)
                ax.plot(rr, m * rr, ls='-', lw=2)
                # project points onto the 1st component
                projection = self.x_star.dot(Vh[0].reshape(self.x_star.shape[1], -1)).dot(Vh[0].reshape(1, -1))
                lines = [[tuple(pt), tuple(proj)] for pt, proj in zip(self.x_star, projection)]
                lc = mc.LineCollection(lines, linewidths=1, color='black')
                ax.add_collection(lc)
                ax.set_xlabel('Principal component 1', size=24)
                ax.set_ylabel('Principal component 2', size=24)
                ax.set_xlim((-3, 3))
                ax.set_ylim((-3, 3))
                ax.set_xticklabels(range(-3, 4), size=16)
                ax.set_yticklabels(range(-3, 4), size=16)
                ax.tick_params(direction='out', length=6, width=2, colors='k')
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2)

                xx = np.array(lines).T[0].flatten()
                yy = np.array(lines).T[1].flatten()

                x_min, x_max = min(xx) - 1., max(xx) + 1.
                y_min, y_max = min(yy) - 1., max(yy) + 1.

                lim = max(np.abs([x_min, x_max, y_min, y_max]))
                #ax.set_xlim((-lim, lim))
                #ax.set_ylim((-lim, lim))
                ax.set_xlim((-3, 3))
                ax.set_ylim((-3, 3))
                # plt.show()
                plt.savefig('INITIAL_CURVE.pdf', bbox_inches='tight')

                del xx, yy, x_max, x_min, y_min, y_max, lim

    def test_few_iterations(self, n_iter=3, dof: int = 4):
        cmap = plt.get_cmap('Greys')
        cmaplist = [cmap(i) for i in range(cmap.N)]
        #print(cmaplist)

        color_map = {'no_AD': '#599B67', 'early_AD': '#E7E35F', 'late_AD': '#EE1B34'}

        if self.plot:
            plt.ion()
            fig = plt.figure(figsize=(10, 10))

        for i in range(n_iter):
            r_lam = r_float(self.pcurve['lambda'])
            fit1 = r_smooth_spline(x=r_lam,
                                   y=r_float(self.x_star.T[0]), df=dof)
            fit2 = r_smooth_spline(x=r_lam,
                                   y=r_float(self.x_star.T[1]), df=dof)

            res = minimize(self.euclidean_distance_from_curve,
                           x0=self.extend_lambda_range(self.pcurve['lambda']),
                           args=(fit1, fit2))

            self.pcurve['lambda'] = res.x

            seq_lam = np.linspace(min(self.pcurve['lambda']), max(self.pcurve['lambda']),
                                  num=100)
            self.pcurve['curve_coordinates'] = np.vstack((self.map_pred_to_dict(r_predict(fit1, x=r_float(seq_lam)))['y'],
                                                          self.map_pred_to_dict(r_predict(fit2, x=r_float(seq_lam)))['y'])).T

            self.pcurve['projection'] = np.vstack((self.map_pred_to_dict(r_predict(fit1, r_float(self.pcurve['lambda'])))['y'],
                                                   self.map_pred_to_dict(r_predict(fit2, r_float(self.pcurve['lambda'])))['y'])).T

            segments = [[tuple(pt), tuple(proj)] for pt, proj in zip(self.x_star, self.pcurve['projection'])]

            print(f"Errors [iteration {i}]: {((self.x_star - self.pcurve['projection']) ** 2).sum()}")

            if self.plot:

                if self.pcurve['curve_coordinates'].shape[1] == 2:
                    ax = fig.add_subplot(111)
                    if i == 0:
                        ax.scatter(self.x_star.T[0], self.x_star.T[1], color='black')
                    p = ax.plot(self.pcurve['curve_coordinates'].T[0], self.pcurve['curve_coordinates'].T[1],
                                lw=2, color=cmaplist[(i+1)*10])
                    lc = mc.LineCollection(segments=segments, linewidths=1, ls='--', colors=cmaplist[(i+1)*10])  # colors=p[0].get_color())
                    ax.add_collection(lc)
                    ax.set_xlabel('Principal component 1', size=24)
                    ax.set_ylabel('Principal component 2', size=24)
                    ax.set_xlim((-3, 3))
                    ax.set_ylim((-3, 3))
                    ax.set_xticklabels(range(-3, 4), size=16)
                    ax.set_yticklabels(range(-3, 4), size=16)
                    ax.tick_params(direction='out', length=6, width=2, colors='k')
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(2)
                    plt.savefig('ITERATIONS.pdf', bbox_inches='tight')

                elif self.pcurve['curve_coordinates'].shape[1] == 3:
                    # fig = plt.figure(figsize=(20, 14))
                    ax = fig.add_subplot(111, projection='3d', facecolor='white')
                    if i == 0:
                        ax.scatter(self.x_star.T[0], self.x_star.T[1], self.x_star.T[2], s=30, color='g')
                    p = ax.plot3D(self.pcurve['curve_coordinates'].T[0], self.pcurve['curve_coordinates'].T[1],
                                  self.pcurve['curve_coordinates'].T[2], lw=2)
                    lc = Line3DCollection(segments=segments, linewidths=1, ls='--', colors=p[0].get_color())
                    ax.add_collection(lc)

                if i == 0:
                    ax.scatter(self.x_star.T[0], self.x_star.T[1], color='black')
                # p = ax.plot(coord_to_plot.T[0], coord_to_plot.T[1], lw=2)
                # lc = mc.LineCollection(segments=segments, linewidths=1, ls='--', colors=p[0].get_color())
                # ax.add_collection(lc)
                ax.set_xlim((-3, 3))
                ax.set_ylim((-3, 3))
                fig.canvas.draw()
                fig.canvas.flush_events()

    def reach_convergence(self, dof: int = 8):

        print(f"Starting curve [SSE: {self.pcurve['tot_dist']}]")
        # dist_old = self.pcurve['tot_dist']
        # has_converged = np.abs(dist_old - self.pcurve['tot_dist']) <= self.thresh * dist_old
        has_converged = False

        z1 = None
        coord_to_plot = None

        while not has_converged and self.it < self.max_iter:
            self.it += 1

            print(f"Iteration {self.it} [SSE = {self.pcurve['tot_dist']}]")
            r_lam = r_float(self.pcurve['lambda'])

            fits = [r_smooth_spline(x=r_lam, y=r_float(self.x_star.T[jj]), df=dof)
                    for jj in range(self.data.shape[1])]

            res = minimize(self.euclidean_distance_from_curve_v2,
                           x0=self.extend_lambda_range(self.pcurve['lambda']),
                           args=fits)
            self.pcurve['lambda'] = res.x

            seq_lam = np.linspace(min(self.pcurve['lambda']), max(self.pcurve['lambda']),
                                  num=100)
            coord_to_plot = self.stack_fits_together(lam=seq_lam, fs=fits)

            z1 = self.stack_fits_together(lam=self.pcurve['lambda'], fs=fits)

            # update the distances
            dist_old = self.pcurve['tot_dist']
            self.pcurve['tot_dist'] = ((self.x_star - z1) ** 2).sum()
            has_converged = np.abs(dist_old - self.pcurve['tot_dist']) <= self.thresh * dist_old
            if has_converged:
                print(f"Convergence reached after {self.it} iterations\n"
                      f"Final SSE: {self.pcurve['tot_dist']}")

        self.pcurve['projection'] = z1
        self.pcurve['has_converged'] = has_converged
        self.pcurve['curve_coordinates'] = coord_to_plot

    def plot_final_curve(self):
        color_map = {'no_AD': '#599B67', 'early_AD': '#E7E35F', 'late_AD': '#EE1B34'}

        cols_to_plot = ['#599B67'] * 4 + ['#E7E35F'] * 4 + ['#599B67'] * 2 + ['#EE1B34'] * 3 + ['#E7E35F'] * 9 + ['#EE1B34'] * 9

        segments = [[tuple(pt), tuple(proj)] for pt, proj in zip(self.x_star, self.pcurve['projection'])]
        if self.pcurve['curve_coordinates'].shape[1] == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(self.x_star.T[0], self.x_star.T[1], color=cols_to_plot)
            p = ax.plot(self.pcurve['curve_coordinates'].T[0], self.pcurve['curve_coordinates'].T[1],
                        color='k', lw=2)
            lc = mc.LineCollection(segments=segments, linewidths=1, ls='--', colors='k')
            ax.add_collection(lc)
            ax.set_xlabel('Principal component 1', size=24)
            ax.set_ylabel('Principal component 2', size=24)
            ax.set_xlim((-3, 3))
            ax.set_ylim((-3, 3))
            ax.set_xticklabels(range(-3, 4), size=16)
            ax.set_yticklabels(range(-3, 4), size=16)
            ax.tick_params(direction='out', length=6, width=2, colors='k')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)
            plt.savefig('FINAL_CURVE.pdf', bbox_inches='tight')

        elif self.pcurve['curve_coordinates'].shape[1] == 3:
            fig = plt.figure(figsize=(20, 14))
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
            ax.scatter(self.x_star.T[0], self.x_star.T[1], self.x_star.T[2], s=30, color='g')
            p = ax.plot3D(self.pcurve['curve_coordinates'].T[0], self.pcurve['curve_coordinates'].T[1],
                          self.pcurve['curve_coordinates'].T[2], lw=2)
            lc = Line3DCollection(segments=segments, linewidths=1, ls='--', colors=p[0].get_color())
            ax.add_collection(lc)

        else:
            raise ValueError(f"The dimensionality of the space: [{self.pcurve['curve_coordinates'].shape[1]}]"
                             f"cannot be plotted")

    def converge_to_optimal(self, stretch: float = 0.5):

        print(f"Starting curve --- squared distance: {self.pcurve['tot_dist']}")
        s = np.zeros_like(self.data)
        dist_old = np.var(self.data, axis=0, ddof=1).sum()

        has_converged = np.abs(dist_old - self.pcurve['tot_dist']) <= self.thresh * dist_old

        while not has_converged and self.it < self.max_iter:
            self.it += 1

            for jj in range(self.data.shape[1]):
                order = np.argsort(self.pcurve['lambda'])
                r_lam = r_float(self.pcurve['lambda'][order])
                r_jj = robj.FloatVector(self.data.T[jj][order])
                fit = r_smooth_spline(x=r_lam, y=r_jj, df=5)
                r_pred = r_predict(fit, x=r_lam)
                pred = dict(zip(r_pred.names, map(list, list(r_pred))))['y']
                s[:, jj] = pred

            dist_old = self.pcurve['tot_dist']
            self.pcurve = self.project_onto_curve(s=s, stretch=stretch)

            has_converged = np.abs(dist_old - self.pcurve['tot_dist']) <= self.thresh * dist_old

        return {'s': self.pcurve['s'], 'order': self.pcurve['order'],
                'lambda': self.pcurve['lambda'], 'total distance': self.pcurve['tot_dist'],
                'converged': has_converged, 'number_of_iterations': self.it}
