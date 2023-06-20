import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
import multiprocessing
from multiprocessing import Pool, freeze_support
from functools import partial
from itertools import product
import anndata
import statsmodels.stats.multitest as sms
import warnings
from pathlib import Path
import glob
import seaborn as sns
import csv
import subprocess

def run_in_background(command, stdoutfile, stderrfile="",
                      force=True, wait=False, quiet=False):
    """Run a system command 
    :param command: A string with the command to run
    :param stdoutfile: A string with file name where stdout will be written
    :param stderrfile: A string with file name where stderr will be written. If "", then combine stdout/stderr to stdoutfile
    :param force: If False, do not run command if stdoutfile already exists. If True, command will be run and stdoutfile will be overwritten (if it already exists)
    :param wait: If True, then function will not return until command is finished running. Otherwise, the function will issue the command to run in the background and then return.
    :param quiet: If False, print contents of stdoutfile and stderrfile to console after command is finished (only if wait=True)
    :return: None
    """
    if force or not os.path.exists(stdoutfile):
        command = 'bash -c \'' + command + '\' > ' + stdoutfile
        if stderrfile:
            command = command + ' 2>' + stderrfile
        else:
            command = command + ' 2>&1'
        if not wait:
            command = command + ' &'
        print('calling ' + command + '\n')
        os.system(command)
    else:
        wait=True
    if wait:
        if not quiet:
            print("Output from stdout file " + stdoutfile)
            get_ipython().system('cat {stdoutfile}')
        if stderrfile and path.exists(stderrfile):
            print("Output from stderr file " + stderrfile)
            get_ipython().system('cat {stderrfile}')
            
CM_DIVERGING = plt.cm.RdBu_r

def hexcolor_to_circos(col, alpha=1):
    """Convert hex color string to circos-formatted string
    :param color: i.e., "#F9D318"
    :param alpha: transparency from 0-1 (0 = fully transparent)
    :return: string like "249,211,24,0.5"
    """
    if alpha is None:
        return(','.join(str(int(y*255)) for y in matplotlib.colors.to_rgb(col)))
    return(','.join(str(int(y*255)) for y in matplotlib.colors.to_rgb(col)) + ','+str(alpha))

def rgb_to_hex(rgb):
    """Convert rgb tuple to hex format
    :param rgb: a 3-tuple like (1,0,0)
    :return: A hex like "#FF0000"
    """
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def cm_to_circos(vals, vmin, vmax, colormap=CM_DIVERGING, alpha=1):
    """Convert numeric array to list of circos-formatted colors
    :param vals: array/list of values
    :param vmin: minimum value on color scale
    :param vmax: maximum value on color scale
    :param colormap: matplotlib colormap to use
    :param alpha: transparency parameter (0-1)
    :return: a list of color strings to used in circos plot
    """
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    co = mapper.to_rgba(vals)
    if alpha is None:
        return [f'{int(x[0]*255)},{int(x[1]*255)},{int(x[2]*255)}' for x in co]    
    return [f'{int(x[0]*255)},{int(x[1]*255)},{int(x[2]*255)},{alpha}' for x in co]

def cm_to_color(val, vmin, vmax, colormap=CM_DIVERGING, alpha=1):
    """Convert numeric value to color based on colormap
    :param val: single numeric value
    :param vmin: minimum value on color scale
    :param vmax: maximum value on color scale
    :param colormap: matplotlib colormap to use
    :alpha: transparency parameter (0-1)
    :return: a 4-tuple color (r,g,b,a), all values between 0-1
    """
    x = cm_to_circos([val], vmin, vmax, colormap, alpha)[0].split(',')
    return((int(x[0])/255, int(x[1])/255,int(x[2])/255,float(x[3])))

def makeTransparent(color, alpha):
    """Add transparency to a color string
    :param color: A string representing a color (i.e., "gray" or "#FF0000")
    :param alpha: Transparency value (0-1)
    :return: A 4-tuple color (r,g,b,a), all values between 0-1
    """
    val = matplotlib.colors.to_rgba(color)
    return((val[0], val[1], val[2], alpha))

            
def read_mast_results(filename, reverse=False, sortBy='scaled_rank_score', quiet=False):
    """Load results created by R/MAST_wrapper.R
    :param filename: The filename containing the results
    :param reverse: If True, flip direction of fold-change parameters
    :param sortBy: parameter to sort results by, should be a column of the results
    :param quiet: If True, suppress verbose output
    :return: data frame with columns including gene,log2FC,p,bonferroni,fdr,rank_score,scaled_rank_score
    """
    if not quiet:
        print("Reading " + filename)
    mastResults = pd.read_csv(filename)
    mastResults.rename(index=str, columns={"primerid": "gene", "coef": "log2FC", 'Pr(>Chisq)':'p',
                                          'Pr..Chisq.':'p'}, inplace=True, errors='ignore')
    #mastResults.log2FC[np.isnan(mastResults.log2FC)] = np.nanmax(mastResults.log2FC)
    mastResults.drop(['Unnamed: 0'], axis=1, inplace=True)
    mastResults.set_index('gene', drop=True, inplace=True)
    mastResults['bonferroni'] = mastResults['p']*mastResults.shape[0]
    mastResults.loc[mastResults.p==0,'p'] = 1e-240
    mastResults.loc[mastResults.fdr==0,'fdr'] = 1e-240
    mastResults.loc[mastResults.bonferroni==0,'bonferroni'] = 1e-240
    mastResults.loc[mastResults.bonferroni > 1,'bonferroni'] = 1
    mastResults['rank_score'] = -10*np.log10(mastResults['bonferroni'])*np.sign(mastResults['log2FC'])
    mastResults['FC'] = 2.0**mastResults['log2FC']
    mastResults['scaled_rank_score'] = mastResults['rank_score']*np.abs(mastResults['log2FC'])
    mastResults['abs_scaled_rank_score'] = np.abs(mastResults['scaled_rank_score'])
    if reverse:
        mastResults['log2FC'] = -mastResults['log2FC']
        mastResults['FC'] = 1.0/mastResults['FC']
        mastResults['ci.hi'] = -mastResults['ci.hi']
        mastResults['ci.lo'] = -mastResults['ci.lo']
        mastResults['rank_score'] = -mastResults['rank_score']
        mastResults['scaled_rank_score'] = -mastResults['scaled_rank_score']
    mastResults = mastResults.sort_values(by=sortBy, ascending=False)
    return(mastResults)


def make_volcano_plot(df,
                      xcol='log2FC',
                      ycol='fdr',
                      label_col='index', 
                      title='MAST volcano plot',
                      xlabel='$log_2(FC)$',
                      ylabel='$-log{10}(p_{adj})$',
                      plot_outfile=None, 
                      max_num_label=15,
                      arrows=True,
                      label_pval_cutoff=0.05,
                      fontsize=12,
                      s=5, label_filter=None,
                      label_sort_col='abs_scaled_rank_score',
                      label_sort_ascending=False,
                      show=True):
    """Volcano plot
    :param df: pandas dataframe with columns label_col (for gene name), xcol, ycol, label_sort_col
    :param xcol: column of df to plot along x-axis
    :param ycol: column of df to plot along y-axis
    :param label_col: column used for labels ('index' implies df.index)
    :param title: title for plot
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :param plot_outfile: If not None, will store image in this file
    :param max_num_label: Maximum number of points on the plot to label
    :param arrows: True/False whether to draw labels connecting gene names to points
    :param label_pval_cutoff: Do not label any genes with ycol >= label_pval_cutoff
    :param fontsize: font size for gene labels
    :param s: point size
    :param label_filter: If not None, a list of genes that may be labelled.
    :param label_sort_col: The column of df used to determine top genes to be labelled
    :param label_sort_ascending: If False, then genes with highest label_sort_col will be labelled. If True, genes with lowest label_sort_col will be labelled.
    :param show: If True, call plt.show() at end of function
    :return: None
    """

    print("ycol:", ycol)
    # Identify significant genes to highlightlabelle
    
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    if (np.sum(y==0) > 0):
        y[y==0] = np.min(y[y!=0])/2
    y = -np.log10(df[ycol].to_numpy())
    
    if max_num_label > 0:
        if label_filter is not None:
            if label_col == 'index':
                label_filter = list(set(label_filter).intersection(set(df.index)))
                label_df = df.loc[label_filter].copy()
            else:
                label_df = df.loc[df[label_col].isin(label_filter)].copy()
        else:
            label_df = df.copy()
        label_df['x'] = x;
        label_df['y'] = y
        label_df = label_df.sort_values(label_sort_col, ascending=label_sort_ascending)
        label_df = label_df.loc[label_df[ycol] < label_pval_cutoff]
        if label_df.shape[0] > max_num_label:
            label_df = label_df.iloc[:max_num_label]
        max_num_label = label_df.shape[0]
        f = [x in label_df.index for x in df.index]
    else:
        f = df[ycol] < label_pval_cutoff
    sig_ind = np.where(f)
    
    # Make Volcano plot
    plt.scatter(x, y, s=s, c='k')
    plt.scatter(x[sig_ind], y[sig_ind], s=s, c='r')
    plt.title(title, fontsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.xlabel(xlabel, size=14, weight='normal')
    plt.ylabel(ylabel, size=14, weight='normal')
    maxX = np.nanmax(np.abs(x))
    maxY = np.nanmax(y)
    plt.xlim(-maxX, maxX)
    plt.ylim(-1, maxY)
#    plt.grid(b=None)
#    sns.despine()
    ax=plt.gca()
    ax.grid(False)
    
    if label_col == 'index':
        z = sorted(zip(y[f], x[f], df.index[f]), reverse=True)
    else:
        z = sorted(zip(y[f], x[f], df[label_col][f]), reverse=True)
    
    if max_num_label > 0:
        texts = []
        for i in z:
            texts.append(ax.text(i[1], i[0], i[2], fontsize=fontsize))
        if (arrows):
            niter=adjust_text(texts, x=x, y=y, 
                              precision=0.001,
                              arrowprops=dict(arrowstyle='-|>', color='gray', lw=0.5))
        else:
            niter = adjust_text(texts, x=x, y=y, force_text=0.05)

    
    # SAVE FIGURE
    if plot_outfile is not None:
        d = os.path.dirname(plot_outfile)
        if not os.path.exists(d):
            os.makedirs(d)
        plt.savefig(plot_outfile, bbox_inches='tight', dpi=300)
        print("Wrote " + plot_outfile)
    if show:
        plt.show()


# after reading in results from population/interaction test, add any untested genes with p-value 1 and logFC 0
# and remove any genes that were tested that are not in genes
def __set_genes(df, genes):
    """Internal function used by read_contactTracing_results. For efficiency purposes, 
       we only run MAST on subset of genes that are expressed in all four categories of cells,
       since otherwise the p-value for interaction coefficient is guaranteed to be 1.
       This function adds the skipped genes back to the results after MAST is run.
    """
    if genes is None:
        return(df)
    df = df.loc[df['primerid'].isin(genes)]
    newgenes = list(set(genes).difference(set(df['primerid'])))
    if len(newgenes) == 0:
        return(df)
    newrow = df.iloc[[0 for x in range(len(newgenes))]].copy()
    newrow['primerid'] = newgenes
    for x in newrow.columns:
        if 'pval' in x:
            newrow[x] = 1
        if 'coef' in x:
            newrow[x] = 0
        if x.startswith('ci.'):
            newrow[x] = 0
    df = pd.concat([df, newrow]).reset_index(drop=True)
    return(df)
    
  

def read_contactTracing_results(celltype_target, ct_outdir, targettest=None, inttest=None,
                                cond1='lowCIN', cond2='highCIN', genes=None):
    """Read contactTracing results for single target and/or interaction test. This function is usually called in parallel by read_all_contactTracing_results.

    :param celltype_target: A doublet (cellType, target) = celltype_target (target is receptor gene being tested in cell type celltype)
    :param ct_outdir: contactTracing output directory; assumes result file is found in 
        f'{ct_outdir}/{cellType}/{celltype_target}[1]
    :param targettest: If the target test was run, this should be 'population_test'. Usually this is not run and the value should be None.
    :param inttest: The name of the interaction test, usually f'interaction_{cond1}_vs_{cond2}'
    :param cond1: condition1 name
    :param cond2: condition2 name
    :param genes: full list of genes tested
    :return: An AnnData object with columns=genes, a single row representing this cell_type/receptor combination, and layers for each statistic read in from contactTracing results
    """
    celltype, target=celltype_target
    targettest_stats = ['coef_clusterTRUE', 'pval', 'ci.hi_clusterTRUE', 'ci.lo_clusterTRUE']
    inttest_stats = [f'coef_cluster_{cond2}TRUE',
                     f'coef_cluster_{cond1}TRUE',
                     f'coef_condition{cond1}',
                     f'coef_condition{cond2}',
                     f'coef_{cond1}TRUE',
                     f'coef_{cond2}TRUE',
                     'coef_clusterTRUE',
                     'pval']
#                     f'ci.hi_cluster_{cond2}TRUE',
#                     f'ci.lo_cluster_{cond2}TRUE']
    targettest_stats_namemap = {'coef_clusterTRUE':'log2FC',
                             'ci.hi_clusterTRUE':'log2FC.highCI',
                             'ci.lo_clusterTRUE':'log2FC.lowCI',
                            'pval':'pval.orig'}
    inttest_stats_namemap = {f'coef_cluster_{cond2}TRUE':f'coef_{cond2}_cluster',
                             f'coef_cluster_{cond1}TRUE':f'coef_{cond1}_cluster',
                             f'ci.hi_cluster_{cond2}TRUE':'i1.highCI',
                             f'ci.hi_cluster_{cond1}TRUE':'i1.highCI',
                             f'ci.lo_cluster_{cond2}TRUE':'i1.lowCI',
                             f'ci.lo_cluster_{cond1}TRUE':'i1.lowCI',
                             f'coef_condition{cond1}':f'coef_{cond1}',
                             f'coef_condition{cond2}':f'coef_{cond2}',
                             f'coef_{cond1}TRUE':f'coef_{cond1}',
                             f'coef_{cond2}TRUE':f'coef_{cond2}',
                             f'ci.hi_{cond1}TRUE':f'coef_{cond1}.highCI',
                             f'ci.lo_{cond1}TRUE':f'coef_{cond1}.lowCI',
                             f'ci.hi_{cond2}TRUE':f'coef_{cond2}.highCI',
                             f'ci.lo_{cond2}TRUE':f'coef_{cond2}.lowCI',
                             'coef_clusterTRUE':'coef_cluster',
                             'ci.hi_clusterTURE':'coef_cluster.highCI',
                             'ci.lo_clusterTRUE':'coef_cluster.lowCI',
                             'pval':'pval'}
    g0 = target.replace(' ', '_').replace('/','_')
    ctdir=celltype.replace(' ', '_').replace('/','_')
    currOutDir = f'{ct_outdir}/{ctdir}'
    t = targettest
    filename = f'{currOutDir}/{g0}/{t}.txt'.replace(' ','_')
    rv = None
    if os.path.exists(filename):
        tmp = pd.read_csv(filename, sep='\t')
        if genes is not None:
            tmp = __set_genes(tmp, genes)
        f = tmp['primerid'] == g0
        if sum(f) > 0:
            tmp.loc[f,'pval'] = 1    
        rv = anndata.AnnData(X=np.array(tmp['coef_clusterTRUE']).reshape(1,-1),
               obs=pd.DataFrame({'cell type':[celltype],
                                'receptor':[target]}, index=['0']),
               var=pd.DataFrame(tmp['primerid']).set_index('primerid'))
        for stat in targettest_stats:
            if stat not in tmp.columns:
                raise Exception("Error; " + stat + "not in population test output")
            if stat in targettest_stats_namemap:
                statname=targettest_stats_namemap[stat]
            else:
                statname=stat
            if statname in rv.layers:
                raise Exception("Error; " + statname + " added multiple times")
            rv.layers[statname] = np.array(tmp[stat]).reshape(1, -1)
            rv.layers['fdr'] = np.array(sms.multipletests(tmp['pval'], method='fdr_bh')[1]).reshape(1, -1)

    tmp=None
    if inttest is not None:
        t=inttest
        filename = f'{currOutDir}/{g0}/{t}.txt'.replace(' ','_')
        if os.path.exists(filename):
            tmp = pd.read_csv(filename, sep='\t')
            if genes is not None:
                tmp = __set_genes(tmp, genes)
            if rv is None:
                emptymat = np.empty((1,tmp.shape[0]))
                emptymat[:] = np.nan
                rv = anndata.AnnData(X=np.float32(emptymat), obs=pd.DataFrame({'cell type':[celltype],
                                                                   'receptor':[target]}, index=['0']),
                                     var=pd.DataFrame(tmp['primerid']).set_index('primerid'))
            tmp = tmp.set_index('primerid').loc[rv.var.index]
            for stat in inttest_stats:
                if stat not in tmp.columns:
                    continue  # just read the ones we find
                    raise Exception("Error; " + stat + "not in interaction test output file=" + filename)
                if stat in inttest_stats_namemap:
                    statname=inttest_stats_namemap[stat]
                else:
                    statname=stat
                if statname in rv.layers:
                    raise Exception("Error; " + statname + " added multiple times")
                rv.layers[statname] = np.array(tmp[stat]).reshape(1, -1)
            rv.layers['fdr.i1'] = np.array(sms.multipletests(tmp['pval'], method='fdr_bh')[1]).reshape(1, -1)
    return(rv)



# in parallel, read contactTracing results for all cellTypes * all targets, return an anndata structure
# with combined results
def read_all_contactTracing_results(cellTypes, targets, ct_outdir, targettest=None, inttest=None,
                                    cond1='lowCIN', cond2='highCIN', ncore=1, genes=None):
    """Read contactTracing results for all cellType/target combinations.

    :param cellTypes: A list of strings, giving the cell type names where contactTracing was run
    :param targets: A list of target genes (receptors) where contactTracing was run
    :param ct_outdir: contactTracing output directory; results for each cellType/target combination should be found in f'{ct_outdir}/{cellType}/{target}
    :param targettest: If the target test was run, this should be 'population_test'. Usually this is not run and the value should be None.
    :param inttest: The name of the interaction test, usually f'interaction_{cond1}_vs_{cond2}'
    :param cond1: condition1 name
    :param cond2: condition2 name
    :param ncore: The number of cores to use to read results in parallel
    :param genes: full list of genes tested for downstream transcriptional effects (often adata.var.index or list of HVGs)
    :return: An AnnData object with columns=genes, and a row for each cell_type/receptor combination, and layers for each statistic read in from contactTracing results
    """
    p = multiprocessing.Pool(processes=ncore)
    tmpad = p.map(partial(read_contactTracing_results, ct_outdir=ct_outdir, targettest=targettest, inttest=inttest, cond1=cond1, cond2=cond2, genes=genes), 
                  list(product(cellTypes, targets)))
    p.close()
    idx=0
    tmpad2 = []
    for ad in tmpad:
        if ad is None:
            continue
        ad.obs.index = [str(idx)]
        idx=idx+1
        tmpad2.append(ad)
    print("Done reading files, " + str(idx) +  " found of " + str(len(tmpad)) + " combinations. concatenating.")
    deg = anndata.concat(tmpad2, join="outer")
    print("Done")
    return(deg)



def estimate_contactTracing_coef(deg_idx, degobs, adata, condition1, condition2, layers, genes):
    """Estimate logFC and/or interaction statistics from contactTracing. In some cases, MAST returns
    a significant p-value for a parameter, but the actual parameter estimate is NA. This function uses
    counts of expressed/non-expressed genes in each condition to produce a simple estimate for these
    parameters.  This function computes parameters for a single cellType/receptor, and is usually 
    called in parallel by estimate_contactTracing_coefs.
    :param deg_idx: The index of the degobs object, indicating the relevant cellType/receptor.
    :param degobs: The 'obs' matrix of the contactTracing AnnData results structure
    :param adata: An AnnData object representin single-cell data, which was used as input to contactTracing
    :param condition1: condition1 (a string), same as used for contactTracing
    :param condition2: condition2 (a string), same as used for contactTracing
    :return: A dictionary, with one element per parameter, each element is an array of parameters
    """
    receptor = degobs.loc[deg_idx,'receptor']
    cellType = degobs.loc[deg_idx,'cell type']
    receptorIdx = np.where(adata.var.index == receptor)[0][0]
    ctf = np.array(adata.obs['cell type'] == cellType)
    cond1 = np.array(adata.obs.loc[ctf, 'condition'] == condition1)
    cond2 = np.array(adata.obs.loc[ctf, 'condition'] == condition2)
    geneOn = np.array(adata.layers['logX'][ctf,receptorIdx] > 0).reshape(-1)
    geneOff = np.array(adata.layers['logX'][ctf,receptorIdx] == 0).reshape(-1)
    logX = np.array(adata[:,genes].layers['logX'][ctf,:])
    rv = {}
    rv['deg_idx'] = deg_idx
    if 'coef_cluster' in layers:
        rv['coef_cluster'] = (np.nan_to_num(np.mean(logX[geneOn], axis=0) -
                                            np.mean(logX[~geneOn], axis=0)))
    if f'coef_{condition1}' in layers:
        rv[f'coef_{condition1}'] = (np.nan_to_num(np.mean(logX[cond1], axis=0) -
                                                 np.mean(logX[cond2], axis=0)))
    if f'coef_{condition2}' in layers:
        rv[f'coef_{condition2}'] = (np.nan_to_num(np.mean(logX[cond2], axis=0) -
                                                 np.mean(logX[cond1], axis=0)))
    if f'coef_{condition1}_cluster' in layers:
        rv[f'coef_{condition1}_cluster'] = (np.nan_to_num(np.mean(logX[cond1 & geneOn], axis=0) -
                                                         np.mean(logX[cond2 & geneOn], axis=0) -
                                                         np.mean(logX[cond1 & (~geneOn)], axis=0) +
                                                         np.mean(logX[cond2 & (~geneOn)], axis=0)))
    if f'coef_{condition2}_cluster' in layers:
        rv[f'coef_{condition2}_cluster'] = (np.nan_to_num(np.mean(logX[cond2 & geneOn], axis=0) -
                                                         np.mean(logX[cond1 & geneOn], axis=0) -
                                                         np.mean(logX[cond2 & (~geneOn)], axis=0) +
                                                         np.mean(logX[cond1 & (~geneOn)], axis=0)))
    return(rv)



def estimate_contactTracing_coefs(deg, adata, condition1, condition2, ncores=1, chunksize=50):
    """Estimate logFC and/or interaction statistics from contactTracing. In some cases, MAST returns
    a significant p-value for a parameter, but the actual parameter estimate is NA. This function uses
    counts of expressed/non-expressed genes in each condition to produce a simple estimate for these
    parameters.   
    :param deg: The AnnData object returned by read_all_contactTracing_results
    :param adata: An AnnData object representin single-cell data, which was used as input to contactTracing
    :param condition1: condition1 (a string), same as used for contactTracing
    :param condition2: condition2 (a string), same as used for contactTracing
    :param ncores: Number of cores to use to parallelize computations
    :param chunksize: How many jobs to give to each core at time. Each job is a receptor/celltype combination (one row of deg).
    :return: None, but new layers are added to deg, with the suffix '_est', representing estimated parameters
    """

    possible_layers = ['coef_cluster',
                       f'coef_{condition1}', f'coef_{condition2}',
                       f'coef_{condition1}_cluster', f'coef_{condition2}_cluster']
    layers = []
    newlayers = {}
    for layer in possible_layers:
        if layer in deg.layers:
            newlayers[f'{layer}_est'] = deg.layers[layer].copy()
            layers.append(layer)
    tenpercent=np.ceil(deg.obs.shape[0]/10)
    nextprint=tenpercent
    numdone=0
    with multiprocessing.Pool(processes=ncores) as p:
        func = partial(estimate_contactTracing_coef,
                       degobs=deg.obs,
                       adata=adata,
                       condition1=condition1,
                       condition2=condition2,
                       layers=layers,
                       genes=deg.var.index)
        for tmp in p.imap_unordered(func, deg.obs.index, chunksize=50):
            deg_idx = tmp['deg_idx']
            idx = np.where((deg.obs.index == deg_idx))[0][0]
            numdone=numdone+1
            if numdone > nextprint:
                print(f'Done {numdone} out of {deg.obs.shape[0]} ({(numdone/deg.obs.shape[0]*100):0.1f}%)')
                nextprint = nextprint + tenpercent
            for l in layers:
                newlayers[f'{l}_est'][idx,:] = tmp[l]
    for l in newlayers.keys():
        deg.layers[l] = newlayers[l]
                

def gsea_linear_scale(data: pd.Series) -> pd.Series:
        """scale input vector to interval [-1, 1] using a linear scaling
        :return correlations: pd.Series, data scaled to the interval [-1, 1]
        """
        data = data.copy()
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0) / 2
        data -= 1
        return data

def gsea_logistic_scale(data: pd.Series) -> pd.Series:
        """scale input vector to interval [-1, 1] using a sigmoid scaling
        :return correlations: pd.Series, data scaled to the interval [-1, 1]
        """
        return pd.Series((expit(data.values) * 2) - 1, index=data.index)

    
def run_gsea(rank, output_root, gmtfile, fdr_cutoff=0.25, label='GSEA run', force=False, wait=False,
             return_command_only=False, readonly=False, gene_map=None):
    """Run GSEA
    :param rank: pd.Series object with score for each gene
    :param output_root: Directory output
    :param gmtfile: Path to GMT file
    :param fdr_cutoff: If returning results, only return results with fdr < fdr_cutoff
    :param label: Name to use (results will be in f'{output_dir}/{label}.csv'. Spaces will be replaced by underscores.
    :param force: If True, run GSEA even if results file already exists.
    :param wait: If True, wait for GSEA to finish running, and return results. Otherwise, start GSEA running, and return None.
    :param return_command_only: If True, return the command to run GSEA on command-line. Otherwise, run the command from the function.
    :param readonly: If True, do not run the command, just read results (from previously run commands).
    :param gene_map: If not None, this should be a dictionary used to convert gene names in rank to gene names in the GMT file. (i.e. to convert between different species gene names). If gene_map is given, it assumes that we are converting to human gene names (all caps) and converts all gene names to all caps, unless the gene mapping is specified in gene_map. If gene_map is None, the given gene names in rank are used and assumed to match with those in the GMT file.
    :return: data frame with GSEA Results
    """
    label = label.replace(" ", "_")
    
    # look to see if output folder already exists
    if return_command_only and not force:
        f = glob.glob(output_root + "/**/*gsea_report*pos*tsv", recursive=True)
        if len(f):
            return None
    if not readonly:
        Path(output_root).mkdir(parents=True, exist_ok=True)
    rnkFile=f'{output_root}/input.rnk'
    if rank is not None:
        if gene_map is not None:
            rank = pd.DataFrame(rank)
            rank['hgene'] = [('' if i.upper() in gene_map.values() else i.upper()) 
                             if i not in gene_map else gene_map[i] for i in rank.index]
            droprows = (rank['hgene'] == '')
            print(f'Dropping {sum(droprows)} results with no human gene mapping')
            rank = rank.loc[~droprows]
            rank.set_index('hgene', inplace=True)
        else:
            rank.index = [i.upper() for i in rank.index]
        rank.to_csv(rnkFile, sep='\t', header=False, index=True)
    cmd=f'/opt/GSEA_Linux_4.3.2/gsea-cli.sh GSEAPreranked -rnk {rnkFile} -gmx {gmtfile} -collapse No_Collapse -mode Max_probe -norm meandiv'
    cmd += f' -nperm 10000 -scoring_scheme weighted -rpt_label {label} -create_svgs true -include_only_symbols true'
    cmd += f' -make_sets true -plot_top_x 20 -rnd_seed 888 -set_max 1500 -set_min 1 -zip_report false -out {output_root}'
    if return_command_only:
        return(cmd)
    stdoutfile=f'{output_root}/{label}_stdout.txt'
    if not readonly:
        run_in_background(cmd,f'{output_root}/{label}_stdout.txt',  wait=wait, force=force, quiet=True)
    # recover information from run
    f = glob.glob(output_root + "/**/*gsea_report*pos*tsv", recursive=True)
    #print(f, output_root + "/**/*gsea_report*pos*tsv")
    if f is None or len(f) == 0:
        raise RuntimeError(
            'seqc.JavaGSEA was not able to recover the output of the Java '
            'executable. This likely represents a bug.')
    f.sort(key=os.path.getmtime)
    f = f[-1]
    f2 = f.replace('_pos_', '_neg_')
    if not os.path.exists(f2):
        raise RuntimeError("Error finding neg result file for GSEA " + f2)
    names = ['name', 'size', 'es', 'nes', 'p', 'fdr_q', 'fwer_p', 'rank_at_max', 'leading_edge']
    print(f'Reading positive GSEA results rom {f}')
    pos = pd.read_csv(f, sep='\t', infer_datetime_format=False, parse_dates=False).iloc[:, :-1]
    pos.drop(['GS<br> follow link to MSigDB', 'GS DETAILS'], axis=1, inplace=True)
    
    neg = pd.read_csv(f2, sep='\t', infer_datetime_format=False, parse_dates=False).iloc[:, :-1]
    print(f'Reading negative GSEA results rom {f2}')
    neg.drop(['GS<br> follow link to MSigDB', 'GS DETAILS'], axis=1, inplace=True)
    pos.columns, neg.columns = names, names
    pos['direction'] = 'pos'
    neg['direction'] = 'neg'

    aa = pos.sort_values('fdr_q', ascending=True).fillna(0)
    bb = neg.sort_values('fdr_q', ascending=True).fillna(0)
    aa = aa[aa['fdr_q'] <= fdr_cutoff]
    bb = bb[bb['fdr_q'] <= fdr_cutoff]
    outcsv = f'{output_root}/{label}.csv'.replace(' ', '_')
    aa.to_csv(outcsv, sep='\t', index=False)
    with open(outcsv, 'a') as f:
        f.write('\n')
    bb.to_csv(outcsv, sep='\t', index=False, mode='a')
    rv = pd.concat([pos, neg])
    f = rv['nes'] == '---'
    rv.loc[f,'nes'] = 0
    rv['nes'] = rv['nes'].astype(np.float64)
    return(rv)



def plot_gsea_results(gr, fdr_cutoff=0.25, plot_outfile=None, title='', remove_strings=None):
    """GSEA plot
    :param gr: GSEA results (data frame returned by run_gsea function)
    :param fdr_cutoff: Only plot results with fdr < fdr_cutoff
    :param plot_outfile: If given, save plot to this file
    :param title: Title for plot
    :param remove_strings: can be a list of strings, any gene set names that contain any of these strins will not be plotted.
    """
    if gr.shape[0] == 0:
        print("No data")
        return
    deg_gsea_results = gr.copy()
    f = (deg_gsea_results['fdr_q'] < fdr_cutoff)
    if remove_strings is not None:
        for rstr in remove_strings:
            f = f & (~deg_gsea_results['name'].str.contains(rstr))
    if sum(f) == 0:
        print("No data after filtering")
        return
    tmp = deg_gsea_results.loc[f]
    tmp = tmp.sort_values('nes', ascending=False)
    tmp = tmp.reset_index()

    if sum(tmp.fdr_q != 0) ==0:
        fdrmin = fdr_cutoff/10
    else:
        fdrmin = np.min(np.array([x for x in tmp.fdr_q if x!=0]))
    tmp.loc[tmp.fdr_q==0,'fdr_q'] = fdrmin

    tmp.fdr_q = tmp.fdr_q.apply(lambda x: -np.log10(x))
    sns.set_style('white')

    tmp['fdr_q_dir'] = [tmp.fdr_q.values[x]*np.sign(tmp.nes.values[x]) for x in range(len(tmp))]

    height=8
    if tmp.shape[0] > 20:
        height = 8*tmp.shape[0]/20
    if height > 20:
        height = 20
    fontscale = 1
    if tmp.shape[0] > 40:
        fontscale = 0.8
    if tmp.shape[0] > 60:
        fontscale = 0.6
    if tmp.shape[0] > 80:
        fontscale = 0.4
    sns.set(font_scale = fontscale)
    
    fig, ax = plt.subplots(figsize=(10,height))

    plot = plt.scatter(x=tmp['name'], y=tmp['nes'], c=tmp['nes'], 
                       cmap='RdBu_r', vmin=-max(abs(tmp.nes)), vmax=max(abs(tmp.nes)))

    plt.clf()
    plt.colorbar(plot)#, label='NES')

    cmap = plt.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=-max(abs(tmp.nes)), vmax=max(abs(tmp.nes)))

    colors = [matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(x) 
              for x in tmp.sort_values(by=['fdr_q_dir','nes'], ascending=False).nes]

    g = sns.barplot(data=tmp, 
                x='fdr_q_dir',
                y='name',
                order=tmp.sort_values(by=['fdr_q_dir','nes'], ascending=False).name)

    for i,thisbar in enumerate(g.patches):
        # Set a different hatch for each bar
        #thisbar.set_color(colors[i])
        thisbar.set_alpha(1)
        thisbar.set_edgecolor('black')
        thisbar.set_linewidth(0.5)
        thisbar.set_color(colors[i])

    #g.set_ylabel('n passengers')
    #g.legend_.remove()

    plt.xlabel('-log10(FDR) * sign(NES)')
    plt.title(title + f' FDR < {fdr_cutoff:0.2f}')


    if plot_outfile is not None:
        plt.savefig(plot_outfile,
                    dpi=400,
                    bbox_inches='tight',
                    transparent=True)
    sns.set(font_scale = 1)


def __make_circos_conf_file(outdir, target_stats, heatmap_plots, histogram_plots,
                            cellType_order=[], cellType_labels=True, label_size=40,
                            label_parallel='yes'):

    f = open(f'{outdir}/circos.conf', 'w')
    cellTypes = target_stats['cell type'].unique()
    celltype_list=[]
    for c in cellType_order:
        if c in cellTypes:
            celltype_list.append(c)
    for c in cellTypes:
        if c not in cellType_order:
            celltype_list.append(c)
    celltype_list = [x.replace(' ', '_').replace('/','_').replace('\n', '_').lower() for x in celltype_list]

    f.write('karyotype = karyotype.txt\n')
    f.write('chromosomes_order = ' + ','.join(celltype_list) + '\n')
    f.write('<ideogram>\n')
    f.write('<spacing>\n')
    f.write('  default = 5u\n')
    f.write('  break = 5u\n')
    f.write('  axis_break = yes\n')
    f.write('  axis_break_style = 2\n')
    f.write('  axis_break_at_edge = no\n')
    f.write('  <break_style 1>\n')
    f.write('      stroke_color = black\n')
    f.write('      fill_color = blue\n')
    f.write('      thickness = 0.25r\n')
    f.write('      stroke_thickness = 2\n')
    f.write('  </break_style>\n')
    f.write('  <break_style 2>\n')
    f.write('      stroke_color = black\n')
    f.write('      thickness = 1.5r\n')
    f.write('      stroke_thickness = 3\n')
    f.write('  </break_style>\n')
    f.write('</spacing>\n\n')

    f.write('radius = 0.7r\n')
    f.write('thickness = 0.04r\n')
    f.write('fill = yes\n')
    f.write('stroke_color = dgrey\n')
    f.write('stroke_thickness = 5p\n')
    if cellType_labels:
        f.write('show_label = yes\n')
    else:
        f.write('show_label = no\n')
    f.write('label_font = default\n')
    f.write('label_radius = 1.05r\n')
    f.write(f'label_size = {label_size}\n')
    f.write(f'label_parallel = {label_parallel}\n')
    f.write('label_with_tag = no\n')
    f.write('label_format = eval(replace(replace(var(label), "ligand", "\\nligand"), " receptor", "\\nreceptor"))\n')
    f.write('label_center = yes\n')
    f.write('</ideogram>\n\n')
    f.write('<plots>\n')
    rstart =1
    width=0.05
    
    for p in heatmap_plots:
        f.write('<plot>\n')
        f.write('  show=yes\n')
        f.write('  type = heatmap\n')
        f.write('  file = ' + p + '.txt\n')
        f.write(f'  r1 = {(rstart-0.00001):0.5f}r\n')
        f.write(f'  r0 = {(rstart - width):0.5f}r\n')
        rstart = rstart - width
        minval = np.nanmin(target_stats[p])
        maxval = np.nanmax(target_stats[p])
        f.write(f'  min = {minval}\n')
        f.write(f'  max = {maxval}\n')
        f.write(f'</plot>\n\n')

    width = 0.1
    for p in histogram_plots:
        f.write('<plot>\n')
        f.write('  show=yes\n')
        f.write('  type = histogram\n')
        f.write('  file = ' + p + '.txt\n')
        f.write(f'  r1 = {(rstart-0.00001):0.5f}r\n')
        f.write(f'  r0 = {(rstart - width):0.5f}r\n')
        rstart = rstart - width
        f.write('  fill_color = black\n')
        f.write('  orientation = in\n')
        f.write(f'</plot>\n\n')

    width = 0.3
    f.write('<plot>\n')
    f.write('  show = yes\n')
    f.write('  type = text\n')
    f.write('  file = labels.txt\n')
    f.write(f'  r1 = {rstart:0.5f}r\n')
    f.write(f'  r0 = {(rstart - width):0.5f}r\n')
    rstart = rstart -  width
    f.write('  label_font = default\n')
    f.write('  label_size = 20\n')
    f.write('  show_links = yes\n')
    f.write('  link_dims = 0p,8p,5p,8p,0p\n')
    f.write('  link_thickness = 1p\n')
    f.write('  link_color = gray\n')
    f.write('  padding = 3p\n')
    f.write('  rpadding = 0p\n')
    f.write('  label_snuggle = yes\n')
    f.write('  max_snuggle_distance = 2r\n')
    f.write('  snuggle_sampling = 3\n')
    f.write('  snuggle_tolerance = 0.25r\n')
    f.write('  snuggle_refine = no\n')
    f.write('  snuggle_link_overlap_test = no\n')
    f.write('  snuggle_link_overlap_tolerance = 2p\n')
    f.write('</plot>\n')
    f.write('</plots>\n\n')

    f.write('<links>\n')
    f.write('<link>\n')
    f.write('  show = yes\n')
    f.write('  file = links.txt\n')
    f.write('  color = 0,0,0,0.1\n')
    f.write(f'  radius = {rstart:0.5f}r\n')
    f.write('  ribbon = no\n')
    f.write('  bezier_radius = 0.1r\n')
    f.write('</link>\n')
    f.write('</links>\n\n')
    
    f.write('<image>\n')
    f.write('<<include image.generic.conf>>\n')
    f.write('background = white\n')
    f.write('angle_offset=-115\n')
    f.write('</image>\n')

    f.write('<<include etc/colors_fonts_patterns.conf>>\n')
    f.write('<<include etc/housekeeping.conf>>\n')

    f.close()
    print(f'Wrote {outdir}/circos.conf')


#  if celltype specified, only draw links and labels for that celltype
# if inttype specified, only draw links and labels for that interactino type (ligand or receptor). inttype only applies if
#   celltype is specified too
# use ext='adj'
# label_only: if specified only label genes in this list AND genes that interact with those genes
# kary_deg_col and kary_int_col specify columns to use for numDEG/numInt for the purposes of defining the
# karyotype - this is used for imposing links from different analyses on the same layout. If they are none,
# then use numDEG_col and numInt_col
def make_circos_plot(interactions,
                     target_stats,
                     outdir,
                     numSigI1_stat='numSigI1_fdr05',
                     ligand_deg_logfc_col='log2FC',
                     ligand_deg_pval_col='fdr',
                     links_min_numSigI1=1,
                     links_max_ligand_fdr=0.05,
                     links_min_ligand_absLog2FC=0,
                     order_col='cell_type_dc1_norm',
                     heatmap_plots=['cell_type_dc1_norm'],
                     histogram_plots=['numSigI1_fdr05'],
                     cellType_order=[],
                     bigGenes=None,
                     bigFontSize=24,
                     max_thickness=25,
                     max_numSigI1=None,
                     boldLigand=True,
                     boldReceptor=False,
                     boldGenes=None,
                     boldCellType=None,
                     log2FC_vmax=0.2,
                     colorMap=None,
                     cellType_labels=True,
                     cellType_filter=None,
                     cellType_filter_receptor=None,
                     cellType_filter_ligand=None,
                     cleanCellTypes=True,
                     title=None,
                     titleSize=60,
                     labelSize=40,
                     labelParallel='yes'):
    """Circos plot
    :param interactions: Data frame with all ligand/receptor interactions, should have a column 'receptor' and a column 'ligand'
    :param target_stats: Data frame with row for every receptor/ligand x cellType combination, should have a column 'target' giving the gene name, 'cellType' with the cell type, columns 'receptor' and 'ligand' that are True/False depending on whether the row represents a ligand or receptor (they can both be True if target is both). Should also have a column for all statistics relevant for plotting (numSigI1_stat, any entry in heatmap_plots or histogram_plots, ligand_deg_logfc_col, ligand_Deg_pval_col).
    :param outdir: output directory where circos plots and files will be written
    :param numSigI1_stat: The name of the column in target_stats that decribes the size of the interaction effect in receptors (it may be undefined for ligands). 
    :params ligand_deg_logfc_col: The column in target_stats that defines the log2FC of the ligand in the relevant cell type. It does not need to be defined for receptors.
    :params ligand_deg_pval_col: The column in target_stats that gives the p-value (or adjusted p-value) that the ligand is differentially expressed between conditions in the relevant cell type
    :param links_min_numSigI1: The miniumum numSigI1_stat value a receptor needs for a link to be drawn
    :param links_max_ligand_fdr: The maximum fdr statistic that a ligand needs for a link to be drawn
    :param links_min_ligand_absLog2FC: The minimum abs(log2FC) that a ligand needs for a link to be drawn
    :param order_col: The column in target_stats used to order ligands/receptors within a cell type
    :param heatmap_plots: The statistics in target_stats that should be drawn in concentric heatmaps along the outside of the plot
    :param histogram_plots: The statistics in target_stats that should be drawn as concentric histograms along the circos plot
    :param cellType_order: The order that cell types should be drawn around the plot. If an empty list or any cell types are not in this list, the order will be arbitrary.
    :param bigGenes: Genes that should be labeled with a large font, if links are drawn to them
    :param bigFontSize: The font size to use for bigGenes
    :param max_thickness: The maximum thickness of a link
    :param max_numSigI1: If given, then any links with numSigI1 > max_numSigI1 will be drawn with the same maximum thickness
    :param boldLigand: If True, then the font for ligands will be bold
    :param boldReceptor: If True, then the font for receptors will be bold
    :param boldGenes: This can be a list of genes which should be labelled with bold font
    :param boldCellType: If given, only use bold font for this cell type
    :param log2FC_vmax: The maximum value for ligand color scale. Also implies lo2FC_vmin=-log2FC_vmax.
    :param colorMap: Dictionary defining colors to use for each cell type
    :param cellType_labels: If False, do not draw labels for each cell type.
    :param cellType_filter: If not None, only show links connected to this cell type. Can be single cell type (string) or list of cellTypes.
    :param cellType_filter_receptor: If not None, show only links where receptor is this cell type. Can be single cell type (string) or list of cellTypes.
    :param cellType_filter_ligand: If not None, show only links where ligand is this cell type. Can be single cell type (string) or list of cellTypes.
    :param title: If not None, add this title to image
    :param titleSize: pointsize to use for title
    :param labelSize: pointsize for cell type labels
    :param labelParallel: should cell type labels be parallel to circle? Should be 'yes' (default) or 'no'
    :param cleanCellTypes: If True, remove cell types from plot that do not have any links
    :return: Within outdir, there should be a file circos.png, as well as summary files circle_plot_tabular.tsv and links_tabular.tsv which describe the plot and links.
    """
    target_stats.loc[target_stats['receptor'],'type'] = 'receptor'
    target_stats.loc[target_stats['ligand'],'type'] = 'ligand'
    target_stats['type'] = pd.Categorical(target_stats['type'], categories=['receptor', 'ligand', 'both'])
    target_stats.loc[(target_stats['ligand']) & (target_stats['receptor']),'type'] = 'both'
    f = np.isnan(target_stats[order_col])
    print(f'Removing {sum(f)} rows of target_stats that have {order_col}=NaN')
    target_stats = target_stats.loc[~f]
    degl = target_stats.loc[target_stats['ligand']].copy()
    degr = target_stats.loc[target_stats['receptor']].copy()
    print(f' num_ligand = {degl.shape[0]}')
    print(f' num_receptor = {degr.shape[0]}')
    
    degMetaAll = degr.merge(interactions, left_on='target', right_on='receptor', suffixes=('_receptor', ''), how='left').merge(
        degl, left_on='ligand', right_on='target', suffixes=('', '_ligand'), how='left').drop(
        columns=['receptor_receptor', 'ligand_receptor', 'ligand_ligand', 'receptor_ligand'])
    
    degMetaAll['labeled0'] = (((degMetaAll[f'{ligand_deg_pval_col}_ligand'] < links_max_ligand_fdr) & 
                               (np.abs(degMetaAll[f'{ligand_deg_logfc_col}_ligand']) >= links_min_ligand_absLog2FC)) &
                               ((degMetaAll[numSigI1_stat] >= links_min_numSigI1)))

    # remove cell types that don't have any labeled interactions
    labf = degMetaAll['labeled0']
    allCellTypes = set(target_stats['cell type'])
    if cleanCellTypes:
        keepCellTypes = set(degMetaAll.loc[labf,'cell type']).union(set(degMetaAll.loc[labf,'cell type_ligand']))
    else:
        keepCellTypes = allCellTypes
    removeCellTypes = allCellTypes.difference(keepCellTypes)
    if (len(removeCellTypes) > 0 and cleanCellTypes):
        removeStr=','.join(list(removeCellTypes))
        print(f'Removing {len(removeCellTypes)} cell Types that have no labels: {removeStr}')
    target_stats = target_stats.loc[target_stats['cell type'].isin(list(keepCellTypes))]
    target_stats['chr'] = target_stats['cell type'].astype(str)
    print("target_stats.shape = " + str(target_stats.shape))
    if target_stats.shape[0] == 0:
        return
    allCellTypes = list(set(target_stats['cell type']))
    if colorMap is None:
        colors = iter(plt.cm.tab20(np.linspace(0, 1, len(allCellTypes))))
        colorMap = {}
        for cellType in list(allCellTypes):
            colorMap[cellType] =  rgb_to_hex(next(colors))
    else:
        numMissing=sum([x not in colorMap for x in allCellTypes])
        if numMissing > 0:
            colors = iter(plt.cm.tab20(np.linspace(0, 1, numMissing)))
            for cellType in allCellTypes:
                if cellType not in colorMap:
                    print(f'No color assigned for {cellType}, assigning random color')
                    colorMap[cellType] = rgb_to_hex(next(colors))
            
 
    # karyotype file
    kary = pd.DataFrame(target_stats['chr'].value_counts()).rename(columns={'chr':'end'})
    kary['cell type'] = list(kary.index)
    kary['type'] = 'interaction'
    kary['col1'] = 'chr'   # defines this entry as a 'chromosome'
    kary['parent'] = '-'    # chromosomes do not have parents
    kary['label'] = kary.index
    kary['id'] = [x.replace(' ', '_').replace('/','_').replace('\n', '_').lower() for x in kary['label']]
    kary['start'] = 0
    kary['color'] = [hexcolor_to_circos(colorMap[i], alpha=None) for i in kary['label']]
    for chrom in target_stats['chr'].value_counts().index:
        startIdx=0
        f = (target_stats['chr'] == chrom)
        tmp = target_stats.loc[f,order_col].copy().argsort()
        ranks = np.empty_like(tmp)
        ranks[tmp] = np.arange(sum(f))
        target_stats.loc[f,'start'] = ranks.astype(int) + startIdx
        startIdx = sum(f)
        print(chrom,startIdx)
    target_stats['start'] = target_stats['start'].astype(int)
    target_stats['end'] = target_stats['start']+1
    target_stats['end'] = target_stats['start'] + 1
    target_stats['parent'] = [x.replace(' ','_').replace('/', '_').lower() for x in target_stats['chr']]
    target_stats['parent'] = [x.replace(' ','_').replace('/', '_').lower() for x in target_stats['chr']]
    target_stats['label'] = target_stats['target']
    target_stats['color'] = [hexcolor_to_circos(colorMap[i], alpha=None) for i in target_stats['cell type']]
    target_stats['id'] = [x.replace(' ','_').replace('/', '_').lower() for x in target_stats['cell type'].astype(str) +
                         '_' + target_stats['type'].astype(str) + '_' + target_stats['target'].astype(str)]
    target_stats['col1'] = 'band'
    karyCols = ['col1', 'parent', 'id', 'label', 'start', 'end','color']
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    pd.concat([kary[karyCols], target_stats[karyCols]]).sort_values(['parent', 'start']).to_csv(
        f'{outdir}/karyotype.txt', sep='\t', index=False, header=False)
    print(f'Wrote karyotype file with {kary.shape[0]} chromosomes and {target_stats.shape[0]} genes')

    # make links structure
    degl = target_stats.loc[(target_stats['ligand']) &
                            (np.abs(target_stats[ligand_deg_logfc_col]) > links_min_ligand_absLog2FC) &
                            (target_stats[ligand_deg_pval_col] < links_max_ligand_fdr)].copy()
    degr = target_stats.loc[(target_stats['receptor']) &
                            (target_stats[numSigI1_stat] >= links_min_numSigI1)].copy()
    links = degr.merge(interactions, left_on='target', right_on='receptor', suffixes=('_receptor', ''), how='left').merge(
        degl, left_on='ligand', right_on='target', how='left').drop(
        columns=['receptor_x', 'ligand_x', 'receptor_y', 'ligand_y'])

    f = np.isnan(links['start_y'])
    links = links.loc[~f]
    for col in ['start_x', 'end_x', 'start_y', 'end_y']:
        links[col] = links[col].astype(np.int64)

    if cellType_filter is not None:
        if type(cellType_filter) == str:
            cellType_filter=[cellType_filter]
        links = links.loc[(links['cell type_x'].isin(cellType_filter)) |
                          (links['cell type_y'].isin(cellType_filter))]

    if cellType_filter_receptor is not None:
        if type(cellType_filter_receptor) == str:
            cellType_filter_receptor=[cellType_filter_receptor]
        links = links.loc[links['cell type_x'].isin(cellType_filter_receptor)]

    if cellType_filter_ligand is not None:
        if type(cellType_filter_ligand) == str:
            cellType_filter_ligand=[cellType_filter_ligand]
        links = links.loc[links['cell type_y'].isin(cellType_filter_ligand)]
        
    print('Number of links: :', links.shape[0])

    # labels filter
#    lf1 = (np.abs(links[f'{ligand_deg_logfc_col}_y']) > links_min_ligand_absLog2FC)
#    if links_min_ligand_fdr is not None:
#        lf1 = (lf1) & (links[f'{ligand_deg_pval_col}_y'] < links_min_ligand_fdr)
#    rf1 = (links[f'{numSigI1_stat}_x'] >= links_min_numSigI1)
#    f = (lf1) & (rf1)

 #   labelsReceptor = links.loc[f,['parent_x', 'start_x', 'end_x', 'label_x']]
    labelsReceptor = links[['parent_x', 'start_x', 'end_x', 'label_x']].rename(columns={'parent_x':'parent','start_x':'start','end_x':'end','label_x':'label'})
    labelsReceptor['color'] = 'color=greys-9-seq-6'
    if boldReceptor:
        if boldCellType is not None:
            fbold = (labelsReceptor['parent'] == boldCellType.lower().replace(' ', '_').replace('/','_'))
        else:
            fbold = (labelsReceptor['parent'] != '')
        if boldGenes is not None:
            fbold = (fbold) & (labelsReceptor['label'].isin(list(boldGenes)))
        if bigGenes is not None:
            fbig = (labelsReceptor['label'].isin(list(bigGenes)))
        else:
            fbig = (labelsReceptor['label'].isin([]))
        labelsReceptor.loc[(fbig) & (~fbold),'color'] = f'color=black,label_size={bigFontSize}'
        labelsReceptor.loc[(~fbig)  & (fbold),'color'] =  'color=black,label_font=semibold'
        labelsReceptor.loc[(fbig) & (fbold),'color'] = f'color=black,label_font=semibold,label_size={bigFontSize}'

    labelsLigand  = links[['parent_y','start_y','end_y','label_y']].copy()
    labelsLigand['color'] = 'color=black'

    labelsLigand.rename(columns={'parent_y':'parent','start_y':'start','end_y':'end','label_y':'label'}, inplace=True)
    if boldLigand:
        if boldCellType is not None:
            fbold = (labelsLigand['parent'] == boldCellType.lower().replace(' ', '_').replace('/','_'))
        else:
            fbold = (labelsLigand['parent'] != '')
        if boldGenes is not None:
            fbold = (fbold) & (labelsLigand['label'].isin(list(boldGenes)))
        if bigGenes is not None:
            fbig = (labelsReceptor['label'].isin(list(bigGenes)))
        else:
            fbig = (labelsReceptor['label'].isin([]))
        labelsLigand.loc[(fbig) & (~fbold),'color'] = f'color=black,label_size={bigFontSize}'
        labelsLigand.loc[(~fbig)  & (fbold),'color'] =  'color=black,label_font=semibold'
        labelsLigand.loc[(fbig) & (fbold),'color'] = f'color=black,label_font=semibold,label_size={bigFontSize}'

    labels = pd.concat([labelsLigand, labelsReceptor]).copy()
    labels = labels[~labels.duplicated(subset=['parent', 'start', 'end'])]
    labels['label'] = [x.replace(' complex','').replace(':','_') for x in labels['label']]
    labels.to_csv(f'{outdir}/labels.txt', sep='\t', index=False, header=False)
    
    # write links file
    links['newScore2'] = links[f'{ligand_deg_logfc_col}_y']
    if links_max_ligand_fdr is not None:
        links.loc[links[f'{ligand_deg_pval_col}_y'] >= links_max_ligand_fdr, 'newScore2'] = 0
    links = links.loc[links['newScore2'] != 0]
    maxDEG = np.max(links[f'{numSigI1_stat}_x'])

    if max_numSigI1 is not None:
        maxDEG = max_numSigI1
    links['thickness'] = (links[f'{numSigI1_stat}_x']/maxDEG*max_thickness+1).apply(np.int64)
    links.loc[links['thickness'] > max_thickness, 'thickness'] = int(max_thickness)
    links['color'] = cm_to_circos(links['newScore2'], colormap=plt.cm.bwr, vmin=-log2FC_vmax, vmax=log2FC_vmax , alpha=0.5)
    links['z'] = (links['newScore2']/np.max(np.abs(links['newScore2']))*1000+1).apply(np.int64)
    links['z'] =  links['z'] - np.min(links['z'])
    links['format'] = 'color=' + links['color'].astype(str) + ',fill_color=' + links['color'].astype(str) + ',thickness=' + links['thickness'].astype(str) + ',z=' + links['z'].astype(str)
    links[['parent_x', 'start_x', 'end_x', 'parent_y', 'start_y', 'end_y', 'format']].to_csv(
        f'{outdir}/links.txt', sep='\t', index=False, header=False)
    # tabular links summary

    linkscols0 = (['target_x', 'cell type_x'] + [f'{x}_x' for x in heatmap_plots + histogram_plots] +
                 ['target_y', 'cell type_y'] + [f'{x}_y' for x in heatmap_plots + histogram_plots] +
                 [f'{numSigI1_stat}_x', f'{ligand_deg_pval_col}_x', f'{ligand_deg_logfc_col}_x',
                  f'{numSigI1_stat}_y', f'{ligand_deg_pval_col}_y', f'{ligand_deg_logfc_col}_y'])
    # remove any duplicates
    linkscols = []
    [linkscols.append(x) for x in linkscols0 if x not in linkscols]
    renamedict = {'target_x':'receptor','target_y':'ligand'}
    for c in linkscols:
        if c == 'target_x' or c == 'target_y':
            continue
        renamedict[c] = c.replace('_x','_receptor').replace('_y','_ligand') 
    links.loc[f & (links['newScore2'] != 0), linkscols].rename(columns=renamedict).to_csv(
        f'{outdir}/links_tabular.tsv', sep='\t', index=False)
    
    ######## plots
    for c in heatmap_plots + histogram_plots:
        if c in target_stats.columns:
            target_stats.loc[~pd.isna(target_stats[c]),['parent','start','end',c]].to_csv(f'{outdir}/{c}.txt', sep='\t', quoting=csv.QUOTE_NONE, quotechar='',escapechar='',
                                                                                       index=False, header=False)

    #numDEG, numSigI1 (colored by celltype)
    for c in [numSigI1_stat]:
        tmp = target_stats[['parent', 'start', 'end', c]].copy()
        tmp['format'] = 'fill_color=' + target_stats['color'] 
        tmp.to_csv(f'{outdir}/{c}.txt', sep='\t', index=False, header=False)

    # tab-delimited summary
    cols=['cell type', 'target', 'type', ligand_deg_logfc_col, ligand_deg_pval_col, numSigI1_stat]
    target_stats[cols].to_csv(f'{outdir}/circle_plot_tabular.tsv', sep='\t', index=False)

    __make_circos_conf_file(outdir, target_stats, heatmap_plots, histogram_plots, cellType_order, cellType_labels, label_size=labelSize, label_parallel=labelParallel)
    cmd=f'cd {outdir} && /opt/circos-0.69-9/bin/circos -debug_group textplace -conf circos.conf > circos_stdout.txt 2>&1'
    os.system(cmd)

    placed=0
    not_placed=0
    f = open(f'{outdir}/circos_stdout.txt')
    for l in f:
        if 'not_placed' in l:
            not_placed = not_placed+1
        elif 'placed' in l:
            placed = placed+1
    print(f'Done making circos plot {outdir}/circos.png')

    if title is not None:
        (width,height) =subprocess.run(['identify', f'{outdir}/circos.png'], stdout=subprocess.PIPE).stdout.decode('utf-8').split(' ')[2].split('x')
        width=int(width)
        height=int(height)
#        os.system(f'mv {outdir}/circos.png {outdir}/circos.1.png')
        hpos=int(width/2 - len(title)*titleSize/4.5)
        print(f'width={width} len(title)={len(title)} hpos={hpos}')
        vpos=int(height/20)
        cmd=f'convert {outdir}/circos.png -pointsize {titleSize} -fill black -annotate +{hpos}+{vpos} \'{title}\' {outdir}/circos.2.png'
        print(cmd)
        os.system(cmd)
        os.system(f'rm -f {outdir}/circos.svg')
        os.system(f'mv -f {outdir}/circos.2.png {outdir}/circos.png')
    
    if not_placed > 0:
        print(f'WARNING: not all labels could be placed. placed={placed} notplaced={not_placed}')
    else:
        print(f'Placed {placed} labels')

    return("Done")
