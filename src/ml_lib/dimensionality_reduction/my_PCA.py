from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA as skPCA


def select_number_of_components(data_in: np.ndarray, comps: int = 50):
    return min(comps, data_in.shape[1] - 1) if data_in.shape[1] - 1 > 4 else 3


class PCA(object):

    def __init__(self, input_df, normalised=True, norm_method='ss', index_name: str = None, include_nans: bool = False):
        """
        Compute PCA on a dataset
        :param input_df: data frame with observations (index) and features (columns)
        :param normalised: if True, there is no need to normalise the data
        :param norm_method: can be either:
                            - 'ss' -> Standard scaling
                            - 'mms' -> MinMax scaling
        :param include_nans: if True mask the nans when calculating both the scaled values and the covariance matrix
        """
        self.data = input_df.copy(deep=True)
        if index_name is not None:
            self.data.index.name = index_name
        self.scaled = normalised
        self.eigen_val, self.eigen_vec = None, None
        self.method = norm_method
        self.w_matrix = None
        self.projected_data = None
        self.nans = include_nans
        self.eigen_pairs = None

    def check_scale(self):
        if not self.scaled:
            raise Warning(f"Data not normalised. Rescale data to avoid artifacts")
        else:
            print("Data already normalised")
            return True

    def normalise_data(self):
        """
        Bring all the features into the same range
        :return: normalised data
        """
        if self.method == 'ss':
            self.method = StandardScaler()
        elif self.method == 'mms':
            self.method = MinMaxScaler()

        self.data.loc[:, :] = self.method.fit_transform(self.data.astype(dtype='float64'))
        self.scaled = True

    def eigen_decompose(self, print_first=3):
        if not self.check_scale():
            self.normalise_data()
        cov_mat = self.data.corr().values if self.nans else np.cov(self.data.T)
        self.eigen_val, self.eigen_vec = np.linalg.eig(cov_mat)
        if print_first > len(self.eigen_val):
            print(f'Variable [{print_first}] bigger than total number of components [{len(self.eigen_val)}].')
        print(f'Eigenvalues\n{self.eigen_val[:min(print_first, len(self.eigen_val))]}')

    def build_matrix(self, n_components: int = 2):
        """
        Build the matrix from the eigenvectors
        :param n_components: number of components to keep
        :return:
        """
        self.eigen_pairs = [(np.abs(self.eigen_val[i]), self.eigen_vec[:, i]) for i in range(len(self.eigen_val))]
        self.eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        self.w_matrix = np.hstack(tuple([self.eigen_pairs[i][1][:, np.newaxis] for i in range(n_components)]))
        print(f'Matrix W:\n{self.w_matrix}')

    def project_data_in_subspace(self):
        if self.nans:
            self.projected_data = pd.DataFrame(data=self.dot_product_with_nan(),
                                               columns=['PC_%d' % s for s in range(1, self.w_matrix.shape[1] + 1)],
                                               index=self.data.index)
            self.projected_data.index.name = self.data.index.name
        else:
            self.projected_data = self.data.dot(self.w_matrix)
            self.projected_data.columns = ['PC_%d' % s for s in range(1, self.projected_data.shape[1] + 1)]
            self.projected_data.index.name = self.data.index.name

    def dot_product_with_nan(self):
        return np.vstack(tup=tuple([np.nansum(self.data.iloc[i, :] * self.w_matrix[:, j])
                                    for j in range(self.w_matrix.shape[1])] for i in range(self.data.shape[0])))

    def show_explained_variance(self, save_img: str = None):
        """
        Plot the explained variance
        :return:
        """
        if self.eigen_val is None:
            self.eigen_decompose()
        var_exp = -np.sort(-self.eigen_val/np.sum(self.eigen_val))
        cum_var_exp = np.cumsum(var_exp)

        fig, ax = plt.subplots(figsize=(12, 8), facecolor='w')
        plt.bar(range(len(self.eigen_val)), var_exp, alpha=0.6, align='center', label='Individual explained variance')
        plt.step(range(len(self.eigen_val)), cum_var_exp, where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component')
        plt.xticks(ticks=list(range(len(self.eigen_val))), labels=np.array(list(range(len(self.eigen_val)))) + 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1.05),
                  ncol=1, fancybox=True, shadow=True, facecolor='white')
        ax.set_facecolor('white')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis='both', which='both', length=5)
        plt.tight_layout()
        if save_img:
            plt.savefig(f'{save_img}', bbox_inches='tight')
        else:
            plt.show()

    def show_projection(self, annotate_pts: bool = False,
                        rep_pt_name: tuple = ('WTX-0{0,4}', ''), save_img: str = None):
        """
        Show original data projected onto the new subspace
        :param rep_pt_name: In case you want to replace or remove use regular expression
        :param annotate_pts: annotate points in the plot
        :param save_img: if True save the image to path, otherwise just show it
        :return:
        """
        if self.w_matrix.shape[1] not in (1, 2, 3):
            raise ValueError("Incorrect number of dimensions")
        print(f'Projecting points onto a {self.w_matrix.shape[1]}D subspace.\n'
              f'*** In case you want to show the projection on a different number of dimensions ***')
        if self.w_matrix.shape[1] == 1:
            raise NotImplementedError("Not yet implemented.")
        elif self.w_matrix.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.scatter(self.projected_data['PC_1'], self.projected_data['PC_2'])
            if annotate_pts:
                for i, txt in enumerate(self.projected_data.index):
                    ax.annotate(re.sub(rep_pt_name[0], rep_pt_name[1], txt),
                                (self.projected_data.iloc[i, 0] + 0.01, self.projected_data.iloc[i, 1] + 0.01),
                                size=10)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.tight_layout()

        elif self.w_matrix.shape[1] == 3:
            fig = plt.figure(figsize=(20, 14))
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
            ax.scatter(self.projected_data['PC_1'], self.projected_data['PC_2'], self.projected_data['PC_3'],
                       s=30, color='g')
            if annotate_pts:
                for i, txt in enumerate(self.projected_data.index):
                    if re.sub(rep_pt_name[0], rep_pt_name[1], txt) in ['Worst', 'Best']:
                        ax.text(self.projected_data.iloc[i, 0] - 0.01, self.projected_data.iloc[i, 1] - 0.01,
                                self.projected_data.iloc[i, 2] - 0.01,
                                s=re.sub(rep_pt_name[0], rep_pt_name[1], txt),
                                size=20)
                    else:
                        ax.text(self.projected_data.iloc[i, 0] - 0.01, self.projected_data.iloc[i, 1] - 0.01,
                                self.projected_data.iloc[i, 2] - 0.01,
                                s=re.sub(rep_pt_name[0], rep_pt_name[1], txt),
                                size=10)

            ax.set_xlabel('Principal Component 1', labelpad=10)
            ax.xaxis.set_label_coords(205, 1.5)
            ax.set_ylabel('Principal Component 2', labelpad=10)
            ax.set_zlabel('Principal Component 3', labelpad=10)
            ax.grid(color='black')
            ax.w_xaxis.line.set_color("black")
            ax.w_yaxis.line.set_color("black")
            ax.w_zaxis.line.set_color("black")
            ax.w_xaxis.gridlines.set_lw(3.0)
            ax.w_yaxis.gridlines.set_lw(3.0)
            ax.w_zaxis.gridlines.set_lw(3.0)

        if save_img:
            plt.savefig(f'{save_img}', bbox_inches='tight')
        else:
            plt.show()

    def show_pcs_corr_to_features(self, method='Spearman', save_img: str = None):
        """
        Show the correlation between each Principal component and the original features
        :param method: 'Spearman' or 'Pearson'
        :param save_img:
        :return:
        """
        sns.set(font_scale=1.4)
        df_complete = self.data.merge(self.projected_data, on=self.projected_data.index.name)
        pc_n = self.projected_data.shape[1]
        plt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        hm = sns.heatmap(df_complete.corr(method=method.lower()).values[-pc_n:, : -pc_n],
                         cbar=True,
                         annot=True,
                         square=True,
                         fmt='.2f',
                         vmin=-1.0,
                         vmax=1.0,
                         annot_kws={'size': 14},
                         yticklabels=df_complete.columns[-pc_n:],
                         xticklabels=df_complete.columns[:-pc_n],
                         cmap=sns.diverging_palette(220, 20, sep=10, as_cmap=True))
        plt.title(f'Assays Correlation ({method.capitalize()})', size=20)
        del df_complete, pc_n
        if save_img:
            plt.savefig(f'{save_img}', dpi=300, bbox_inches='tight')
        else:
            plt.show()


def select_components_above_background(expression_values: np.ndarray, n_permutations: int, path_name: str = 'pathway'):

    pca = skPCA().fit(expression_values.T)
    expr_flat = expression_values.flatten()
    explained_var_df = pd.DataFrame(index=list(range(n_permutations)),
                                    columns=list(range(expression_values.shape[1])))
    for i in range(n_permutations):
        np.random.shuffle(expr_flat)
        expr_permuted = expr_flat.reshape(expression_values.shape[0], expression_values.shape[1])
        pca_permuted = skPCA().fit(expr_permuted.T)
        explained_var_df.loc[i] = pca_permuted.explained_variance_ratio_

    pval = list()

    for j in range(expression_values.shape[1]):
        pval.append(np.sum(explained_var_df.iloc[:, j] >= pca.explaiend_variance_ratio_[j]) / n_permutations)

    n_significant_components = np.where(np.array(pval) >= 0.05)[0][0]
    explained_var_sign_comp = pca.explained_variance_ratio_[0: n_significant_components] * 100
    var_df = pd.DataFrame.from_dict({'PCs': int(n_significant_components), 'explained_var': explained_var_sign_comp},
                                    orient='index',
                                    columns=[path_name])
    return n_significant_components, var_df









