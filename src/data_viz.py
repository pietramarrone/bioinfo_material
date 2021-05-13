import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.colors import rgb2hex, colorConverter
from scipy.cluster.hierarchy import set_link_color_palette
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.patches as mpatches

import textwrap
import pandas as pd
import seaborn as sns
import itertools

import data_loader as myload
import statistics as mystat


sns.set_palette('Set1', n_colors=10, desat=0.95)
palette1 = sns.color_palette()
sns.set_palette('Set2', n_colors=10, desat=0.95)
palette2 = sns.color_palette()
sns.set_palette('Set3', n_colors=10, desat=0.95)
palette3 = sns.color_palette()
palette = list(map(rgb2hex, palette1)) + \
          list(map(rgb2hex, palette2)) + \
          list(map(rgb2hex, palette3)) + \
          ['#3D5366', '#FF5C5C', '#25412E']
palette = set(palette)
set_link_color_palette(list(map(rgb2hex, palette)))

# example for the nTWAS pathways
nTWAS_colors = {'Vulnerability': '#E73936',
                'Perturbation': '#5177B8'}

## abbreviations for plotting
truncated_conditions = {'Healthy vs Early-AD': 'HC vs Early', 'Healthy vs Late-AD': 'HC vs Late',
                        'Early-AD vs Late-AD (up)': 'Early vs Late (up)',
                        'Early-AD vs Late-AD (down)': 'Early vs Late (down)'}

ct_lab = ['Ex', 'In', 'Ast', 'Mic', 'Oli', 'Opc']


def make_tick_labels_invisible(fig):
    """
    The function deletes the tick labels
    :param fig: figure istance
    :return: void
    """
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


def hex_to_rgb(string):
    """
    Gives and hex color and converts it to RGB color
    """
    h = string.lstrip("#")
    rgb_col = list(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    return rgb_col


def hex_to_norm_rgb(string):
    """
    Gives and hex color and converts it to RGB color scaled between [0,1]
    """
    return mpl.colors.to_rgba_array(string)


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


def plot_connectivity(list_of_nodes, intervals=250, plot_type='points', return_vals=False, log_scale=None, fileout=None):
    """
    Parameters
    ==========
    list_of_nodes: list of lists, with names of genes included as 1st entry of the list
    """
    
    if isinstance(list_of_nodes[0], list):
        print("List of motifs is in the correct format")
        edges = np.array(list_of_nodes).T[1].astype(np.uint16)
   
    #print(edges)

    fig, ax = plt.subplots(figsize=(8,6))
    #ax.set_xlim((10, 60000))
    #ax.set_ylim((10, 10000))
    if plot_type == 'hist':
        y,x,_ = plt.hist(edges, bins=intervals)
        if log_scale=='xlog':
            ax.set(xscale='log')
        elif log_scale== 'ylog':
            ax.set(yscale='log')
        elif log_scale == 'both':
            ax.set(xscale='log', yscale='log')

        plt.show()
    else:
        hist_vals = np.histogram(edges, bins=intervals)
        y, x = hist_vals[0], hist_vals[1]
        mid = np.array([(a + b) /2 for a,b in zip(x[:-1], x[1:])])

        if log_scale == 'both':
            mid = mid[y!=0]
            y = y[y!=0]
        print(len(mid[mid<1550]))
        col_plot =  ['#B5B4B4']*len(mid[mid<1450]) +['#212D87']*(len(mid[mid>1450])) 
        # col_plot = ['#B5B4B4']*21+['#212D87']*(len(mid)-21)
        plt.scatter(x=mid, y=y, color=col_plot, s=50)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2, length=6)
        ax.yaxis.set_tick_params(width=2, length=6)
        plt.xlabel("Connectivity", size=26)
        plt.ylabel("Number of genes", size=26)
        if log_scale=='xlog':
            ax.set(xscale='log')
        elif log_scale== 'ylog':
            ax.set(yscale='log')
        elif log_scale == 'both':
            ax.set(xscale='log', yscale='log')
        ax.tick_params(axis='both', labelsize=18)
        plt.tight_layout()
        if fileout:
           plt.savefig("{}.pdf".format(fileout))
           plt.close()
    if return_vals:
        return mid, y


def create_patch(x0y0, width, height):
    return Rectangle(x0y0, width, height)


def plot_volcano(df, fc_lab, pval_lab, sign_thresh, img_title):

    num_down = df[(df[pval_lab] < sign_thresh) & (df[fc_lab] < -1)].shape[0]
    num_up = df[(df[pval_lab] < sign_thresh) & (df[fc_lab] > 1)].shape[0]

    conditions = [(df[pval_lab] < sign_thresh) & (df[fc_lab] < -1),
                  (df[pval_lab] < sign_thresh) & (df[fc_lab] > 1)]
    choices = ['red', 'navy']
    colors = np.select(conditions, choices, default='grey')
    log_thr = -np.log10(sign_thresh)
    # colors = np.where(df[pval_lab] < sign_thresh, 'red', 'grey')
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title(img_title, size=26)
    ax.scatter(df[fc_lab], -np.log10(df[pval_lab]), alpha=0.2, color=colors)
    ax.axvline(x=1, ymin=0, ymax=1,
               color='black', lw=2, alpha=0.5)
    ax.axvline(x=-1, ymin=0, ymax=1,
               color='black', lw=2, alpha=0.5)
    ax.axhline(y=log_thr, xmin=0, xmax=(-1 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
               color='black', lw=2, alpha=0.5)
    ax.axhline(y=log_thr, xmin=(1 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]), xmax=1,
               color='black', lw=2, alpha=0.5)

    rect_down = create_patch(x0y0=(ax.get_xlim()[0], log_thr),
                             width=np.abs(-1 - ax.get_xlim()[0]), height=np.abs(ax.get_ylim()[1] - log_thr))
    rect_up = create_patch(x0y0=(1, log_thr),
                           width=np.abs(ax.get_xlim()[1] - 1), height=np.abs(ax.get_ylim()[1] - log_thr))
    pc = PatchCollection([rect_down, rect_up], facecolor=['red', 'navy'], alpha=0.1,
                         edgecolor=None)
    ax.add_collection(pc)

    y_offset = 2
    ax.annotate(f'#Genes: {num_down}', xy=(ax.get_xlim()[0] + 0.08, log_thr + y_offset), size=16)
    ax.annotate(f'#Genes: {num_up}', xy=(ax.get_xlim()[1] - 2.1, log_thr + y_offset), size=16)
    ax.set_xlabel('$log_2(Fold-Change)$', size=20)
    ax.set_ylabel('-$\log_{10}(%s)$' % str(pval_lab), size=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    plt.tight_layout()
    plt.savefig(f'./{img_title}.pdf', bbox_inches='tight')


def show_first_significant_pathways(df_to_show_early: pd.DataFrame,
                                    df_to_show_late: pd.DataFrame,
                                    pathways_of_interest: dict,
                                    save_plot_path: str = '.'):
    """

    Parameters
    ----------
    df_to_show_early: PATHWAYS of early AD
    df_to_show_late: PATHWAYS of late AD
    pathways_of_interest: dictionary of pathways to highlight
    save_plot_path: path where to save

    Returns
    -------

    """

    # compute max and min pval to show value scale
    vmin = -np.log10(df_to_show_early.values.max() + 1e-20)
    vmax = -np.log10(df_to_show_early.values.min() + 1e-20)
    # my_cmap = ListedColormap(sns.color_palette("Reds").as_hex())

    f, axs = plt.subplots(nrows=2,
                          ncols=2,
                          gridspec_kw={'height_ratios': [1/2, 1/2],
                                       'width_ratios': [0.97, 0.03],
                                       'hspace': 0.1,
                                       'wspace': 0.04},
                          figsize=(12, 13))
    gs = axs[1, 1].get_gridspec()
    for ax in axs[:, -1]:
        ax.remove()
    ax_cbar = f.add_subplot(gs[:, -1])
    ax_cbar.set_aspect(12)

    ax_hclate, _, ax_hcearly, _ = axs.flatten()
    # empty_ax.remove()
    # ax1.get_shared_y_axes().join(ax2,ax3)

    g1 = sns.heatmap(-np.log10(df_to_show_early[:22].T),
                     cmap="Blues",
                     vmin=vmin,
                     vmax=vmax,
                     cbar=False,  # only one color bar for everything
                     linewidth=1,
                     linecolor='black',
                     ax=ax_hcearly)

    g1.set_xticklabels(g1.get_xticklabels(),
                       size=15,
                       rotation=45,
                       ha='right',
                       rotation_mode='anchor')
    y_labels = [l.get_text() if 'cell type' not in l.get_text()
                else l.get_text().replace('cell type', 'broad') for l in g1.get_yticklabels()]

    g1.set_yticklabels(y_labels, size=15)

    for i, lab in enumerate(g1.get_xticklabels()):
        if lab.get_text() in list(itertools.chain(*pathways_of_interest.values())):
            g1.get_xticklabels()[i].set_fontweight("bold")

    g1.set_ylabel('Sub-clusters', size=20)
    g1.set_xlabel('KEGG pathways', size=28)

    g2 = sns.heatmap(-np.log10(df_to_show_late[4:26].T),
                     cmap="Blues",
                     vmin=vmin,
                     vmax=vmax,
                     cbar=True,
                     linewidth=1,
                     linecolor='black',
                     ax=ax_hclate,
                     cbar_ax=ax_cbar,
                     cbar_kws={'label': '-log10(FDR)', 'shrink': 0.4})

    ax_cbar.yaxis.label.set_size(18)
    ax_cbar.set_frame_on(True)
    g2.xaxis.set_ticks_position('top')
    g2.set_ylabel('Sub-clusters', size=20)

    y_labels = [l.get_text() if 'cell type' not in l.get_text()
                else l.get_text().replace('cell type', 'broad') for l in g2.get_yticklabels()]

    g2.set_yticklabels(y_labels, size=15)

    for i, lab in enumerate(g2.get_xticklabels()):
        if lab.get_text() in list(itertools.chain(*pathways_of_interest.values())):
            g2.get_xticklabels()[i].set_fontweight("bold")
    g2.set_xticklabels(g2.get_xticklabels(), size=15,rotation=315, ha='right', rotation_mode='anchor')
    for g in (g1, g2):
        b, t = g.get_ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        g.set_ylim(b, t)  # update the ylim(bottom, top) values

    aspect = np.diff(ax_cbar.get_xlim()) / np.diff(ax_cbar.get_ylim())

    ax_hclate.annotate('Healthy Control vs Late AD',
                       xy=(1, 1),
                       xytext=(-5.25, 13),
                       rotation=90,
                       size=22)

    ax_hclate.annotate('',
                       xy=(-0.1, 0.08),
                       xycoords='axes fraction',
                       xytext=(-0.1, 0.98),
                       arrowprops=dict(arrowstyle="|-|", color='k'))

    ax_hcearly.annotate('Healthy Control vs Early AD',
                        xy=(1, 1),
                        xytext=(-5.25, 13.2),
                        rotation=90,
                        size=22)

    ax_hcearly.annotate('',
                        xy=(-0.1, 0.01),
                        xycoords='axes fraction',
                        xytext=(-0.1, 0.91),
                        arrowprops=dict(arrowstyle="|-|", color='k'))

    plt.savefig(f'{save_plot_path}/Significant_pathways.pdf', bbox_inches='tight')


def plot_enumerated_pathways_per_cluster(complete_df: pd.DataFrame, save_to_path: str = '.'):
    """

    Parameters
    ----------
    complete_df: DataFrame with the significant pathways
    save_to_path: path to plots folder

    Returns
    -------

    """

    vmin = 0
    vmax = complete_df.values.flatten().max()

    b, t = 0, 0
    # my_cmap = ListedColormap(sns.color_palette("Reds").as_hex())
    rows = complete_df.shape[0]
    split_cell_types = [0, 5, 19, 28, 34, 39, 44]

    f, axs = plt.subplots(1, 6,
                          gridspec_kw={'width_ratios': [5 / rows, 14 / rows, 9 / rows, 6 / rows, 5 / rows, 5 / rows],
                                       'wspace': 0.03},
                          figsize=(30, 3))

    axins = inset_axes(axs[2],
                       width="50%",  # width = 50% of parent_bbox width
                       height="8%",  # height : 5%
                       loc='upper center',
                       bbox_to_anchor=(800, 5, 600, 230))

    for i, ax in enumerate(axs.flatten()):
        g = sns.heatmap(complete_df.iloc[split_cell_types[i]:split_cell_types[i + 1], :].T, vmin=vmin,
                        vmax=vmax, cbar=False,
                        linewidth=1, linecolor='w', ax=ax, annot=True, fmt='d',
                        annot_kws={"weight": "bold", "size": 12}, cmap='Blues')
        if i == 0:
            b, t = g.get_ylim()  # gets the values for bottom and top
            b += 0.5  # Add 0.5 to the bottom
            t -= 0.5  # Subtract 0.5 from the top
            g.set_yticklabels(['HC vs Early AD', 'HC vs Late AD',
                               'Early vs Late AD (up)', 'Early vs Late AD (down)'], size=22)

        g.set_ylim(b, t)  # update the ylim(bottom, top) values
        x_labels = [l.get_text() if 'cell type' not in l.get_text() else l.get_text().replace('cell type', 'broad') for
                    l in g.get_xticklabels()]
        g.set_xticklabels(x_labels, size=22, rotation=45, ha='right', rotation_mode='anchor')

        if i != 0:
            g.set_ylabel('')
            g.set_yticks([])

            if i == 2:
                # ax_divider = make_axes_locatable(ax)
                # define size and padding of axes for colorbar
                # cax = ax_divider.append_axes('top', size = '10%')
                # make colorbar for heatmap.
                # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
                cbar = plt.colorbar(axs[2].get_children()[0],
                                    cax=axins,
                                    orientation='horizontal')
                # locate colorbar ticks
                axins.xaxis.set_ticks_position('top')
                cbar.ax.tick_params(labelsize=18)
                cbar.set_label('# significant KEGG pathways', size=24, rotation=0, labelpad=-70)

    f.text(0.5, -0.55, 'Cell types [subclusters]', ha='center', size=40)
    f.text(-0.02, 0.01, 'Comparison', ha='center', size=40, rotation='vertical')
    # plt.annotate('Cell types', xy=(0, 100), xytext=(0.5, 0.01), xycoords='figure fraction', size=26)
    plt.savefig(f'{save_to_path}/Significant_pathways_TOTAL.pdf', bbox_inches='tight')


def arrange_recurrence_for_plotting(over_represented_paths_dict: dict, pathways_of_interest: dict):
    """
    Return dictionary for plotting the over-representation of pathways
    Parameters
    ----------
    over_represented_paths_dict
    pathways_of_interest

    Returns
    -------

    """
    paths_plotting = dict()
    for s in over_represented_paths_dict:
        # one representative for all: Healthy vs Early-AD
        paths_plotting[s] = pd.DataFrame(index=over_represented_paths_dict['Healthy vs Early-AD']['Ex'].index,
                                         columns=myload.ct_lab)
        for c in over_represented_paths_dict[s]:
            paths_plotting[s].loc[:, c] = over_represented_paths_dict[s][c]['occurrences']
        mask_yes = paths_plotting[s].index.isin(itertools.chain(*pathways_of_interest.values()))
        paths_plotting[s]['in_list'] = np.where(mask_yes, 'yes', 'no')
        paths_plotting[s]['in_list'] = pd.Categorical(paths_plotting[s]['in_list'])

    return paths_plotting


def compare_paths_of_interest_recurrence_to_others(paths_recurrence_for_plotting: dict, save_to_path: str = '.'):
    """
    Violin plot of the comparison between pathways of interest and the others
    Parameters
    ----------
    paths_recurrence_for_plotting
    save_to_path

    Returns
    -------

    """

    color_map = ['#6F969B',
                 '#B4041E']
    plot_labels = ['KEGG pathways - all',
                   'KEGG pathways - nTWAS']
    titles_list = ['Healthy vs Early-AD',
                   'Healthy vs Late-AD',
                   'Early- vs Late-AD (upregulated)',
                   'Early- vs Late-AD (downregulated)']

    f, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1],
                                             'wspace': 0.05, 'hspace': 0.1},
                          figsize=(18, 10))

    axes = axs.flatten()
    for i, s in enumerate(paths_recurrence_for_plotting):
        test = pd.melt(paths_recurrence_for_plotting[s].iloc[:, :-1])
        test['inlist'] = np.tile(paths_recurrence_for_plotting[s].iloc[:, -1],
                                 paths_recurrence_for_plotting[s].shape[1] - 1)

        g = sns.violinplot(x='variable', y='value',
                           data=test, hue='inlist',
                           split=False, legend=True,
                           ax=axes[i], palette=color_map, inner='quartile')

        g1 = sns.stripplot(x='variable', y='value',
                           data=test, hue='inlist', dodge=True,
                           alpha=0.9, ax=axes[i], palette=color_map)

        if i == 2:
            axes[i].set_yticks(range(0, 9, 2))
            axes[i].set_ylim(-1.5, 8)
            axes[i].spines['left'].set_bounds(0, 8)
            axes[i].spines['bottom'].set_bounds(0, 5)
        elif i == 3:
            axes[i].set_yticks(range(0, 11, 2))
            axes[i].set_ylim(-1.5, 10)
            axes[i].spines['left'].set_bounds(0, 10)
            axes[i].spines['bottom'].set_bounds(0, 5)
        else:
            axes[i].set_yticks(range(0, 13, 2))
            axes[i].set_ylim(-2, 13)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_bounds(0, 12)
            axes[i].set_xticks([])
            axes[i].set_xlabel('')

        for c in myload.ct_lab:
            if paths_recurrence_for_plotting[s][c] < 0.05:
                x = myload.ct_lab.index(c)
                y = [12, 8, 4.5] if i == 0 else [10.5, 4.5, 4.5] if i == 1 else [6, 4, 3.5, 3.5, 3.5]
                axes[i].plot([x - 0.2, x - 0.2, x + 0.2, x + 0.2],
                             [y[x], y[x] + 0.1, y[x] + 0.1, y[x]],
                             lw=1.5, c='k')
                axes[i].text((x - 0.2 + x + 0.2) * .5,
                             y[x] + 0.1,
                             "{}".format(mystat.pvals_to_asterisk(paths_recurrence_for_plotting[s][c])),
                             ha='center', va='bottom', c='k', size=16)

        axes[i].set_xticklabels(axes[i].get_xticklabels(), size=18)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        # axes[i].spines['left'].set_bounds(-2, 11)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].get_legend().set_visible(False)
        if i < 2:
            axes[i].text(1.2, 13, titles_list[i], size=20)
        elif i == 2:
            axes[i].text(0.6, 7, titles_list[i], size=20)
        else:
            axes[i].text(0.3, 9, titles_list[i], size=20)
        plt.setp(axes[i].get_yticklabels(), fontsize=14)
        plt.setp(axes[i].collections, alpha=.5)

    f.text(0.5, 0.02, 'Cell types', ha='center', size=28)
    f.text(0.06, 0.5, 'KEGG pathway recurrency in sub-clusters', va='center', rotation='vertical', size=28)
    handles, labels = axes[i].get_legend_handles_labels()

    f.legend(handles[2:], plot_labels, loc=(0.40, 0.91), fontsize=18,
             markerscale=2, frameon=False, handletextpad=0.1)
    plt.savefig(f'{save_to_path}/Vulnerability_encoders_recurrent_in_neurons.pdf', bbox_inches='tight')


def heatmap_of_pathways_of_interest_in_cell_clusters(complete_df: pd.DataFrame,
                                                     pathways_df: pd.DataFrame,
                                                     significance_dict: dict,
                                                     pathways_of_interest_dict: dict,
                                                     save_to_path: str = '.'):

    rows = complete_df.shape[0]
    ct_lab_2 = ['Ast', 'Ex', 'In', 'Mic', 'Oli', 'Opc']
    cond_labs = ['Healthy vs Early-AD',
                 'Healthy vs Late-AD',
                 'Early-AD vs Late-AD (up)',
                 'Early-AD vs Late-AD (down)']

    b, t = 0, 0

    vmin = -np.log10(pathways_df.values.max() + 1e-5)
    vmax = -np.log10(pathways_df.values.min() + 1e-5)

    f, axs = plt.subplots(4, 6,
                          gridspec_kw={'width_ratios': [5 / rows, 14 / rows, 9 / rows, 6 / rows, 5 / rows, 5 / rows],
                                       'wspace': 0.03,
                                       'hspace': 0.04},
                          figsize=(15, 26))

    for i, s in enumerate(significance_dict):
        for j, c in enumerate(ct_lab_2):

            tmp_labs = pd.Index(myload.ordered_labs)[pd.Index(myload.ordered_labs).isin(significance_dict[s][c].columns)]
            mat = -np.log10(significance_dict[s][c].loc[itertools.chain(*pathways_of_interest_dict.values())] + 1e-5)
            mat = mat[tmp_labs]

            g = sns.heatmap(mat, cmap="Blues", vmin=vmin, vmax=vmax, cbar=False,
                            linewidth=2, linecolor='black', ax=axs[i, j])  # cbar_ax=ax_cbar,
            # cbar_kws={'label': '-log10(FDR)', 'shrink': 0.4})

            if i == 0:
                b, t = g.get_ylim()  # discover the values for bottom and top
                b += 0.5  # Add 0.5 to the bottom
                t -= 0.5  # Subtract 0.5 from the top

            g.set_yticklabels(g.get_yticklabels(), size=14)

            if i == 3:
                x_labels = [
                    l.get_text() if 'cell type' not in l.get_text() else l.get_text().replace('cell type', 'broad') for
                    l in g.get_xticklabels()]
                g.set_xticklabels(x_labels, size=14, rotation=45, ha='right', rotation_mode='anchor')
            else:
                g.set_xticks([])

            g.set_ylim(b, t)  # update the ylim(bottom, top) values

            if j != 0:
                g.set_ylabel('')
                g.set_yticks([])

            if i == 3 and j == 0:
                axins = inset_axes(axs[-1, 0],
                                   width="50%",  # width = 50% of parent_bbox width
                                   height="8%",  # height : 5%
                                   loc='lower left',
                                   bbox_to_anchor=(-100, 150, 400, 160))
                # ax_divider = make_axes_locatable(ax)
                # define size and padding of axes for colorbar
                # cax = ax_divider.append_axes('top', size = '10%')
                # make colorbar for heatmap.
                # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
                cbar = plt.colorbar(axs[i, j].get_children()[0], cax=axins, orientation='horizontal',
                                    ticks=np.arange(0, 6))
                # locate colorbar ticks
                axins.xaxis.set_ticks_position('bottom')
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label('-log10(FDR)', size=22, rotation=0, labelpad=-60)
            if j == 5:
                axs[i, j].set_ylabel(cond_labs[i], rotation=270, size=25, labelpad=27)
                axs[i, j].yaxis.set_label_position("right")
    f.text(0.5, 0.068, 'Cell types [subclusters]', ha='center', size=30)
    f.text(-0.25, 0.5, 'KEGG pathways [nTWAS]', va='center', rotation='vertical', size=30)

    plt.savefig(f'{save_to_path}/nTWAS_pathways_vulnerability.pdf', bbox_inches='tight')


def plot_dendrogram_of_clusters_and_similarity_matrix(df_jaccard: pd.DataFrame,
                                                      clusters_membership: dict,
                                                      dendrogram,
                                                      linkage_instance,
                                                      ordered_columns: list,
                                                      clusters_proportions: list,
                                                      pathways_of_interest_dict: dict,
                                                      save_to_dir: str):
    sns.set_style("ticks")



    all_leaves_idxs = np.array([9, 6, 4, 6, 3, 5, 6, 73, 2, 6, 20, 7, 7, 21, 4,
                                13, 5, 24, 11, 18, 9, 12, 6, 7, 14, 10, 16])
    
    f, axs = plt.subplots(4, 27, gridspec_kw={'height_ratios': [0.8, 0.1, 0.06, 3],
                                              'width_ratios': clusters_proportions,
                                              'wspace': 0.,
                                              'hspace': 0.04},
                          frameon=False,
                          figsize=(12, 14))

    gs1 = axs[0, -1].get_gridspec()
    for ax in axs[0, :]:
        ax.remove()
    ax1 = f.add_subplot(gs1[0, :-1])

    gs_rect = axs[1, -1].get_gridspec()
    for ax in axs[1, :]:
        ax.remove()
    ax_rect = f.add_subplot(gs1[1, :-1])

    gs_ntwas = axs[2, -1].get_gridspec()
    for ax in axs[2, :]:
        ax.remove()
    ax_ntwas = f.add_subplot(gs1[2, :-1])

    gs2 = axs[3, -1].get_gridspec()
    for ax in axs[3, :]:
        ax.remove()
    ax2 = f.add_subplot(gs2[3, :-1])

    # axs = axs.flatten()
    df_jaccard_sorted = df_jaccard.iloc[dendrogram['leaves'], dendrogram['leaves']]
    hierarchy.dendrogram(linkage_instance,
                         color_threshold=1.3,
                         labels=df_jaccard.columns,
                         above_threshold_color='#0A0A0A',
                         no_labels=True,
                         ax=ax1)

    for s in ax1.spines:
        if s == 'left':
            continue
        ax1.tick_params(axis='y', direction='out', length=5, color='k')
        ax1.spines[s].set_visible(False)
        ax1.spines[s].set_visible(False)
        ax1.spines[s].set_visible(False)
    ax1.set_yticks(np.arange(0, 4, 0.5))
    ax1.set_ylim(0, 3.5)
    ax1.spines['left'].set_bounds(0, 3.5)
    ax1.set_ylabel('Distance', size=14, labelpad=4)

    for s in ax_rect.spines:
        ax_rect.spines[s].set_visible(False)
        ax_rect.axes.get_xaxis().set_ticks([])
        ax_rect.axes.get_yaxis().set_ticks([])
    for i, col in enumerate(ordered_columns):
        if i == 0:
            ax_rect.add_patch(
                matplotlib.patches.Rectangle((0, 0), clusters_proportions[i] + i * 0. / np.sum(all_leaves_idxs),
                                             1,
                                             color=col))
        else:
            ax_rect.add_patch(matplotlib.patches.Rectangle((np.sum(clusters_proportions[:i]), 0),
                                                           clusters_proportions[i], 1, color=col))

    ax_rect.set_ylabel('Clusters', size=12, rotation=0)
    ax_rect.yaxis.set_label_coords(-0.05, 0.2)

    for i, mm in enumerate(clusters_membership):
        found_nTWAS = set(clusters_membership[mm]).intersection(itertools.chain(*pathways_of_interest_dict.values()))
        if found_nTWAS:
            for p in found_nTWAS:
                nTWAS_p_col = nTWAS_colors['Vulnerability'] if p in pathways_of_interest_dict['Vulnerability'] else \
                nTWAS_colors['Perturbation']
                ax_ntwas.axvline(x=list(itertools.chain(*clusters_membership.values())).index(p) / 322,
                                 ymin=0, ymax=1, color=nTWAS_p_col, lw=2)
    for s in ax_ntwas.spines:
        ax_ntwas.axes.get_xaxis().set_ticks([])
        ax_ntwas.axes.get_yaxis().set_ticks([])

    ax_ntwas.set_ylabel('nTWAS', size=12, rotation=0)
    ax_ntwas.yaxis.set_label_coords(-0.046, 0.06)

    vuln_patch = mpatches.Patch(color=nTWAS_colors['Vulnerability'], label=list(nTWAS_colors.keys())[0])
    pert_patch = mpatches.Patch(color=nTWAS_colors['Perturbation'], label=list(nTWAS_colors.keys())[1])

    ax_ntwas_leg = ax_ntwas.legend(handles=[vuln_patch, pert_patch], frameon=False,
                                   fontsize=12, bbox_to_anchor=(0.75, 2.5, 0.4, 0.1),
                                   handlelength=0.2, handleheight=1, labelspacing=0.06)

    sns.heatmap(df_jaccard_sorted,
                cmap='Blues',
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax2)

    axins = inset_axes(ax2,
                       width="2%",  # width = 50% of parent_bbox width
                       height="20%",  # height : 5%
                       loc='upper center',
                       bbox_to_anchor=(460, 5, 600, 650))

    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    cbar = plt.colorbar(ax2.get_children()[0],
                        cax=axins,
                        orientation='vertical')
    # locate colorbar ticks
    axins.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Jaccard Similarity', size=18, rotation=90, labelpad=5)

    ax2.set_ylabel('KEGG pathways', size=30, labelpad=10)
    ax2.set_xlabel('KEGG pathways', size=30, labelpad=10)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_to_dir}/KEGG_cluster_membership_nTWAS.pdf', bbox_inches='tight')



def plot_phenotypes_correlations(full_median_corr_df: pd.DataFrame,
                                 corr_pvalues: dict,
                                 clinical_readouts: list,
                                 save_to_dir: str):
    import textwrap

    color_map = ['#6F969B', '#B4041E']
    plot_labels = ['KEGG pathways - all', 'KEGG pathways - nTWAS']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    axes = axes.flatten()
    for i, c in enumerate(full_median_corr_df):
        sns.violinplot(data=full_median_corr_df[c], x='clinical trait', y='correlation', inner='quartile',
                       hue='inTWAS', dodge=True, split=True, ax=axes[i], palette=color_map, legend=True)
        # sns.stripplot(x='clinical trait', y='correlation', data=full_median_corr_dfs[c], hue='inTWAS',
        #              dodge=True, alpha=0.9, ax=axes[i], palette=color_map)

        axes[i].get_legend().set_visible(False)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].axhline(y=0, xmin=0, xmax=1, color='k', ls='--', alpha=0.3)
        axes[i].set_ylabel(c, size=20)
        axes[i].set_xlabel('')
        indented_clin_traits = map(lambda y: textwrap.fill(y, 20), clinical_readouts)
        for sp in axes[i].spines:
            if sp == 'bottom' and i in [4, 5]:
                axes[i].spines[sp].set_bounds(0, 6)
                axes[i].set_xticklabels(indented_clin_traits, size=18,
                                        rotation=45, rotation_mode='anchor', ha='right')
            elif sp != 'left':
                axes[i].spines[sp].set_visible(False)
                if i not in [4, 5]:
                    axes[i].set_xticklabels('')
                    axes[i].set_xticks([])
        plt.setp(axes[i].get_yticklabels(), fontsize=14)
        plt.setp(axes[i].collections, alpha=.5)

        for pheno in corr_pvalues[c]:
            if corr_pvalues[c][pheno] < 0.05:
                x = list(corr_pvalues[c].keys()).index(pheno)
                y = full_median_corr_df[c][full_median_corr_df[c]['clinical trait'] == pheno]['correlation'].max()

                offset = 0.05 if c != 'Ast' or pheno != 'Global Cognitive Function' else 0.1

                axes[i].text(x,
                             y + offset,
                             "{}".format(mystat.pvals_to_asterisk(corr_pvalues[c][pheno])),
                             ha='center', va='bottom', c='k', size=16)

    fig.text(0.5, -0.04, 'Clinical traits', ha='center', size=28)
    fig.text(0.02, 0.5, 'Spearman\'s correlation [Cell types]', va='center', rotation='vertical', size=28)

    handles, labels = axes[i].get_legend_handles_labels()

    fig.legend(handles[2:], plot_labels, loc=(0.40, 0.92), fontsize=18,
               markerscale=2, frameon=False, handletextpad=0.1)
    plt.savefig(f'{save_to_dir}/nTWAS_correlation_cell_types_v2.pdf', bbox_inches='tight')


def plot_gwas_in_cell_types(gwas_paths_df: pd.DataFrame, save_to_dir: str):

    fig, ax = plt.subplots(figsize=(14, 4))
    g = sns.heatmap(gwas_paths_df, cmap=['#FFFFFF', '#08306B'], cbar=False, linewidth=1, linecolor='k')
    b, t = g.get_ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    g.set_ylim(b, t)  # update the ylim(bottom, top) values
    g.set_yticklabels(['Ex', 'In', 'Ast', 'Mic', 'Oli', 'Opc'], size=14)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', size=14)

    in_gwas = mpatches.Patch(facecolor='#08306B', label='KEGG: mapped', edgecolor='black')
    out_gwas = mpatches.Patch(facecolor='#FFFFFF', label='KEGG: not mapped', edgecolor='black')
    ax_gwas = ax.legend(handles=[in_gwas, out_gwas], frameon=False, ncol=2,
                        fontsize=14, bbox_to_anchor=(0.34, 1.2, 0.4, 0.1),
                        handlelength=2, handleheight=2.5, labelspacing=0.02)

    g.set_xlabel('GWAS genes', size=22)
    g.set_ylabel('Cell types', size=22)
    plt.savefig(f'{save_to_dir}/GWAS_Pathways.pdf', bbox_inches='tight')


def plot_significant_pathways_in_celltypes(sign_paths_to_cluster_df: pd.DataFrame,
                                           clusters_membership_dict: dict,
                                           pathways_of_interest: dict,
                                           proportions: list,
                                           ordered_columns: list,
                                           save_to_dir: str):

    height_ratios = [1.] + ([0.2] + [0.7] * 4) * 6
    dont_keep_ax = [1, 6, 11, 16, 21, 26]
    increment_j = [4, 8, 12, 16, 20]
    row_labels = ['Healthy vs Early-AD', 'Healthy vs Late-AD',
                  'Early-AD vs Late-AD (up)', 'Early-AD vs Late-AD (down)']
    map_labels = {'Ex': 'Excitatory neuron', 'In': 'Inhibitory neuron', 'Ast': 'Astrocyte',
                  'Oli': 'Oligodendrocyte', 'Opc': 'Oligodendrocyte progenitor', 'Mic': 'Microglia'}

    all_leaves_idxs = np.array([9, 6, 4, 6, 3, 5, 6, 73, 2, 6, 20, 7, 7, 21, 4,
                                13, 5, 24, 11, 18, 9, 12, 6, 7, 14, 10, 16])

    f, axs = plt.subplots(31, 27, gridspec_kw={'height_ratios': height_ratios,
                                               'width_ratios': proportions,
                                               'wspace': 0.,
                                               'hspace': 0.},
                          frameon=False, figsize=(14, 11))

    axes = []

    for i in range(axs.shape[0]):
        gs = axs[i, -1].get_gridspec()
        for ax in axs[i, :]:
            ax.remove()
        if i not in dont_keep_ax:
            axes.append(f.add_subplot(gs[i, :-1]))

    j, k = 0, 0

    for i, ax in enumerate(axes):
        if i == 0:
            for s in ax.spines:
                ax.spines[s].set_visible(False)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])
            for i, col in enumerate(ordered_columns):
                if i == 0:
                    ax.add_patch(matplotlib.patches.Rectangle((0, 0), proportions[i] + i * 0. / np.sum(all_leaves_idxs),
                                                              1, color=col))
                else:
                    ax.add_patch(matplotlib.patches.Rectangle((np.sum(proportions[:i]), 0),
                                                              proportions[i], 1, color=col))

            ax.set_ylabel('KEGG clusters', size=15, rotation=0)
            ax.yaxis.set_label_coords(-0.077, 0.35)
            continue

        # for axis in ['top','bottom','left','right']:
        # ax.spines[axis].set_linewidth(30)
        # ax.spines[axis].set_color("gold")
        # ax.spines[axis].set_zorder(0)

        ax.set_ylabel(row_labels[k], size=12, rotation=0)
        if k == 0:
            ax.yaxis.set_label_coords(-0.083, 0.25)
        elif k == 1:
            ax.yaxis.set_label_coords(-0.08, 0.25)
        elif k == 2:
            ax.yaxis.set_label_coords(-0.1, 0.25)
        elif k == 3:
            ax.yaxis.set_label_coords(-0.11, 0.25)
        if k == 1:
            y_shift = -1.98 if 'Oli' in sign_paths_to_cluster_df.index[j] else -1.97 if 'Opc' in \
                                                                                        sign_paths_to_cluster_df.index[
                                                                                            j] else -1.
            x_shift = 1.037 if 'Opc' in sign_paths_to_cluster_df.index[j] else 1.03
            ax.text(x_shift, y_shift, textwrap.fill(map_labels[sign_paths_to_cluster_df.index[j].split(' ')[0]], 15),
                    ha='center', rotation=270, size=13)

        for s in ax.spines:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

        for mm in clusters_membership_dict:
            found_in_condition = set(clusters_membership_dict[mm]).intersection(sign_paths_to_cluster_df.iloc[j, k])
            if found_in_condition:
                for p in found_in_condition:
                    p_col = nTWAS_colors['Vulnerability'] if p in pathways_of_interest['Vulnerability'] \
                        else nTWAS_colors['Perturbation'] if p in pathways_of_interest['Perturbation'] else '#5C5B5D'
                    ax.axvline(x=list(itertools.chain(*clusters_membership_dict.values())).index(p) / 322,
                               ymin=0, ymax=1, color=p_col, lw=2)
        j += 1 if i in increment_j else 0
        k = k + 1 if (k < 3) else 0

    vuln_patch = mpatches.Patch(color=nTWAS_colors['Vulnerability'], label=list(nTWAS_colors.keys())[0])
    pert_patch = mpatches.Patch(color=nTWAS_colors['Perturbation'], label=list(nTWAS_colors.keys())[1])
    other_patch = mpatches.Patch(color='#5C5B5D', label='Other')

    ax_ntwas_leg = axes[-1].legend(handles=[vuln_patch, pert_patch, other_patch], frameon=False,
                                   fontsize=13, bbox_to_anchor=(-0.215, -0.1, 0.4, 0.1), ncol=3,
                                   handlelength=0.15, handleheight=1.3, labelspacing=-0.03,
                                   title='Pathway identity')
    plt.setp(ax_ntwas_leg.get_title(), fontsize=15)
    ax_ntwas_leg._legend_box.sep = 4
    f.text(-0.07, 0.33, 'Pathological states', ha='center', rotation=90, size=30)
    f.text(0.52, 0.075, 'KEGG pathway', ha='center', rotation=0, size=30)
    plt.savefig(f'{save_to_dir}/KEGG_clusters_comparison.pdf', bbox_inches='tight')


def plot_clusters_enrichment(dfs_modules_to_paths_sign: pd.DataFrame,
                             ordered_columns: list,
                             df_paths_significance: pd.DataFrame,
                             save_to_dir: str = '.'):

    f, axs = plt.subplots(1, 10, gridspec_kw={'width_ratios': [1 / 27, 1 / 54] * 2 + [4 / 27] * 6, 'wspace': 0.},
                          frameon=False, figsize=(12, 12))

    vmin = 0
    vmax = 8

    for i in range(axs.T.shape[0]):
        if i == 1 or i == 3:
            axs[i].remove()

    for i, ax in enumerate(axs.flatten()):
        if i == 0:
            for j, col in enumerate(reversed(ordered_columns)):
                if j == 0:
                    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1 / len(ordered_columns), color=col))
                else:
                    ax.add_patch(matplotlib.patches.Rectangle((0, j / len(ordered_columns)),
                                                              1, (j + 1) / len(ordered_columns), color=col))
            ax.set_yticklabels('')
            ax.set_yticks([])
            ax.set_xticklabels('')
            ax.set_xticks([])
            continue
        if i == 1 or i == 3:
            continue

        if i == 2:
            mat = -np.log10((df_paths_significance.loc[:, 'pval'] + 1e-10).values.astype(float)).reshape((1, -1)).T
            sns.heatmap(mat, vmin=vmin, vmax=vmax, cmap='Greys', cbar=False, lw=1, linecolor='black', ax=ax)

            b, t = ax.get_ylim()  # discover the values for bottom and top
            b += 0.5  # Add 0.5 to the bottom
            t -= 0.5  # Subtract 0.5 from the top
            ax.set_ylim([b, t])
            ax.set_ylabel('KEGG cluster', size=30)
            ax.yaxis.set_label_coords(-1.5, 0.5)
            ax.set_yticklabels('')
            ax.set_yticks([])
            ax.set_xlabel('nTWAS enriched', size=14, rotation=45, rotation_mode='anchor', ha='right')
            ax.set_xticklabels('')
            continue
        mat = -np.log10(dfs_modules_to_paths_sign[list(dfs_modules_to_paths_sign.keys())[i - 4]] + 1e-5)
        sns.heatmap(mat, vmin=vmin, vmax=vmax, cmap='Greys', cbar=False, lw=1, linecolor='black', ax=ax)

        ax.set_ylim([b, t])
        ax.set_yticklabels('')
        ax.set_yticks([])
        ax.set_xticklabels(truncated_conditions.values(), size=13, rotation=45,
                           rotation_mode='anchor', ha='right')
        for sp, spine in ax.spines.items():
            spine.set_visible(True)
            ax.spines[sp].set_linewidth(5)

        ax.set_xlabel(list(dfs_modules_to_paths_sign.keys())[i - 4], size=18)
        ax.xaxis.set_label_coords(0.5, 1.01)
        ax.xaxis.set_label_position('top')
    axins = inset_axes(axs[-1],
                       width="2%",  # width = 50% of parent_bbox width
                       height="20%",  # height : 5%
                       loc='upper center',
                       bbox_to_anchor=(450, 5, 700, 750))

    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    cbar = plt.colorbar(axs[-1].get_children()[0], cax=axins, orientation='vertical')
    # locate colorbar ticks
    axins.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('-log10(FDR)', size=18, rotation=90, labelpad=8)
    f.text(0.55, -0.022, 'Pathological state', ha='center', rotation=0, size=30)
    plt.savefig(f'{save_to_dir}/Pathway_enrichment.pdf', bbox_inches='tight')


def plot_overlap_between_perturbations_in_cell_types(dfs_jaccard_pathways: pd.DataFrame,
                                                     save_to_dir: str = '.'):
    import re
    f, ax = plt.subplots(1, 3, figsize=(12, 4))

    for i, cond in enumerate(dfs_jaccard_pathways):
        mask = np.zeros_like(dfs_jaccard_pathways[cond], dtype=np.bool)
        mask[np.tril_indices_from(mask)] = True
        mat = np.round(dfs_jaccard_pathways[cond], 2)
        sns.heatmap(mat, mask=mask, vmin=0, vmax=0.51, cmap='Blues', lw=1, annot=True, cbar=False, ax=ax[i])
        if i == 0:
            b, t = ax[i].get_ylim()
            b += 0.5
            t -= 0.5
        ax[i].set_ylim([b, t])

        # print(mat.index.str.lsplit(' ', -1))
        xlabs = [''] + mat.index.str.replace('\[cell type\]', '').to_list()[1:]
        ax[i].set_xticklabels(xlabs, size=14, rotation=0)
        ax[i].xaxis.tick_top()

        ylabs = mat.index.str.replace('\[cell type\]', '').to_list()[:-1] + ['']
        ax[i].set_yticklabels(ylabs, size=14, rotation=0)
        ax[i].yaxis.tick_right()
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')
        ax[i].set_xlabel(cond, size=20)
        ax[i].xaxis.set_label_coords(0.59, 0.)
    axins = inset_axes(ax[-1],
                       width="1.5%",  # width = 50% of parent_bbox width
                       height="80%",  # height : 5%
                       loc='upper center',
                       bbox_to_anchor=(430, 80, 800, 150))

    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    cbar = plt.colorbar(ax[-1].get_children()[0], cax=axins, orientation='vertical')
    # locate colorbar ticks
    axins.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Jaccard Similarity', size=18, rotation=90, labelpad=8)
    f.text(0.11, 0.31, 'KEGG pathways\noverlap', ha='center', rotation=90, size=20)
    plt.savefig(f'{save_to_dir}/KEGG_pathways_overlap.pdf', bbox_inches='tight')


def plot_DEgenes_overlap(overlap_df: pd.DataFrame, save_to_dir: str = '.'):

    f, ax = plt.subplots(figsize=(12, 6))
    ylabs = ['Healthy vs AD', 'Healthy vs Early AD', 'Early vs Late AD']

    sns.heatmap(overlap_df.astype(float).T[ct_lab] * 100, cmap='Reds', annot=True,
                annot_kws={'fontsize': 15, 'fontstyle': 'oblique', 'fontweight': 'heavy'}, fmt='.2f', cbar=False, lw=1,
                linecolor='k', ax=ax)
    b, t = ax.get_ylim()
    b += 0.5
    t -= 0.5
    ax.set_ylim(b, t)
    ax.set_yticklabels(ylabs, rotation=0, size=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, size=18)

    axins = inset_axes(ax,
                       width="2%",  # width = 50% of parent_bbox width
                       height="90%",  # height : 5%
                       loc='upper center',
                       bbox_to_anchor=(430, 80, 750, 250))

    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    cbar = plt.colorbar(ax.get_children()[0], cax=axins, orientation='vertical')
    # locate colorbar ticks
    axins.xaxis.set_ticks_position('bottom')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('DE genes found [%]', size=22, rotation=90, labelpad=8)
    f.text(0.20, 0.95,
           'Overlap between disregulated KEGG pathways [our method]\n\t\tand DE genes [$\it{Mathys\ et\ al.\ 2019}$]',
           size=20)
    plt.savefig(f'{save_to_dir}/DE_genes_mapped_KEGG.pdf', bbox_inches='tight')
