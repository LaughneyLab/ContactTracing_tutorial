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

## run command in background. If wait=False, run in background and return immediately.
## If stderrfile='', stderr goes to same place as stdout
## If force=False and stdoutfile already exists, do not run again
def run_in_background(command, stdoutfile, stderrfile="",
                      force=True, quiet=False, wait=False):
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

def color_to_str(col, alpha=1):
    if alpha is None:
        return(','.join(str(int(y*255)) for y in matplotlib.colors.to_rgb(col)))
    return(','.join(str(int(y*255)) for y in matplotlib.colors.to_rgb(col)) + ','+str(alpha))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def cm_to_str(val, vmin, vmax, colormap=CM_DIVERGING, alpha=1, circos=False):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    co = mapper.to_rgba(val)
    if alpha is None:
        return [f'{int(x[0]*255)},{int(x[1]*255)},{int(x[2]*255)}' for x in co]    
    if circos:
        alpha = 1-alpha
    return [f'{int(x[0]*255)},{int(x[1]*255)},{int(x[2]*255)},{alpha}' for x in co]

def cm_to_color(val, vmin, vmax, colormap=CM_DIVERGING, alpha=1):
    x = cm_to_str([val], vmin, vmax, colormap, alpha)[0].split(',')
    return((int(x[0])/255, int(x[1])/255,int(x[2])/255,float(x[3])))

def makeTransparent(color, alpha):
    val = matplotlib.colors.to_rgba(color)
    return((val[0], val[1], val[2], alpha))

            
def read_mast_results(filename, filter_genes=None, filter_label='_filtered', reverse=False,
                      sortBy='scaled_rank_score', quiet=False):
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
    if filter_genes is not None:
        mastResults = mastResults.loc[[g for g in mastResults.index if g in filter_genes]]
        filename=os.path.splitext(filename)[0] + filter_label + '.csv'
        mastResults[['p', 'log2FC', 'FC', 'fdr', 'ci.hi','ci.lo','bonferroni','rank_score', 'scaled_rank_score']].to_csv(filename)
    return(mastResults)


def make_volcano_plot(df, title='MAST volcano plot', plot_outfile = None, 
                      max_num_label=15, arrows=True,
                      label_pval_cutoff=0.05,
                      fontsize=12,
                      ycol='fdr',
                      xcol='log2FC',
                      xlabel='$log_2(FC)$',
                      ylabel='$-log{10}(p_{adj})$',
                     label_col='index', s=5, label_filter=None,
                     label_sort_col='abs_scaled_rank_score',
                     label_sort_ascending=False):
    print("ycol:", ycol)
    # Identify significant genes to highlightlabelle
    
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    if (np.sum(y==0) > 0):
        y[y==0] = np.min(y[y!=0])/2
    y = -np.log10(df[ycol].to_numpy())
    
    if max_num_label > 0:
        if label_filter is not None:
            label_df = df.loc[label_filter].copy()
        else:
            label_df = df.copy()
        label_df['x'] = x;
        label_df['y'] = y
        #f = ~((df[genecol].str.lower().str.startswith('mt-')) |
        #      (df[genecol].str.lower().str.startswith('rps')) |
        #      (df[genecol].str.lower().str.startswith('rpl')))
        #label_df = label_df.loc[f].sort_values(label_sort_col, ascending=label_sort_ascending)
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
    plt.show()


# after reading in results from population/interaction test, add any untested genes with p-value 1 and logFC 0
# and remove any genes that were tested that are not in genes
def set_genes(df, genes):
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
    
  

def read_contactTracing_results(celltype_target, ct_outdir, poptest, inttest=None,
                                cond1='lowCIN', cond2='highCIN', genes=None):  
    celltype, target=celltype_target
    poptest_stats = ['coef_clusterTRUE', 'pval', 'ci.hi_clusterTRUE', 'ci.lo_clusterTRUE']
    inttest_stats = [f'coef_cluster_{cond2}TRUE',
                     f'coef_cluster_{cond1}TRUE',
                     f'coef_condition{cond1}',
                     f'coef_condition{cond2}',
                     'coef_clusterTRUE',
                     'pval']
#                     f'ci.hi_cluster_{cond2}TRUE',
#                     f'ci.lo_cluster_{cond2}TRUE']
    poptest_stats_namemap = {'coef_clusterTRUE':'log2FC',
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
                             'coef_clusterTRUE':'coef_cluster',
                             'pval':'pval'}
    g0 = target.replace(' ', '_').replace('/','_')
    ctdir=celltype.replace(' ', '_').replace('/','_')
    currOutDir = f'{ct_outdir}/{ctdir}'
    t = poptest
    filename = f'{currOutDir}/{g0}/{t}.txt'.replace(' ','_')
    rv = None
    if os.path.exists(filename):
        tmp = pd.read_csv(filename, sep='\t')
        if genes is not None:
            tmp = set_genes(tmp, genes)
        f = tmp['primerid'] == g0
        if sum(f) > 0:
            tmp.loc[f,'pval'] = 1    
        rv = anndata.AnnData(X=np.array(tmp['coef_clusterTRUE']).reshape(1,-1),
               obs=pd.DataFrame({'cell type':[celltype],
                                'receptor':[target]}, index=['0']),
               var=pd.DataFrame(tmp['primerid']).set_index('primerid'))
        for stat in poptest_stats:
            if stat not in tmp.columns:
                raise Exception("Error; " + stat + "not in population test output")
            if stat in poptest_stats_namemap:
                statname=poptest_stats_namemap[stat]
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
                tmp = set_genes(tmp, genes)
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
def read_all_contactTracing_results(cellTypes, targets, ct_outdir, poptest=None, inttest=None,
                                    cond1='lowCIN', cond2='highCIN', ncore=1, genes=None):
    p = multiprocessing.Pool(processes=ncore)
    tmpad = p.map(partial(read_contactTracing_results, ct_outdir=ct_outdir, poptest=poptest, inttest=inttest, cond1=cond1, cond2=cond2, genes=genes), 
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


def make_circos_conf_file(outdir, target_stats, heatmap_plots, histogram_plots,
                           cellType_order=[], cellType_labels=True):

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
    f.write('label_radius = 1.1r\n')
    f.write('label_size = 40\n')
    f.write('label_parallel = yes\n')
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
    print(f'Wrote {outdir}')

        
    


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
                     ligand_deg_logfc_col='log2FC',
                     ligand_deg_pval_col='fdr',
                     boldLigand=True,
                     boldReceptor=False,
                     boldGenes=None,
                     boldCellType=None,
                     log2FC_vmax=0.2,
                     colorMap=None,
                     cellType_labels=True):

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
    keepCellTypes = set(degMetaAll.loc[labf,'cell type']).union(set(degMetaAll.loc[labf,'cell type_ligand']))
    removeCellTypes = allCellTypes.difference(keepCellTypes)
    if (len(removeCellTypes) > 0):
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
    kary['color'] = [color_to_str(colorMap[i], alpha=None) for i in kary['label']]
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
    target_stats['color'] = [color_to_str(colorMap[i], alpha=None) for i in target_stats['cell type']]
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
    links['color'] = cm_to_str(links['newScore2'], colormap=plt.cm.bwr, vmin=-log2FC_vmax, vmax=log2FC_vmax , alpha=0.5)
    links['z'] = (links['newScore2']/np.max(np.abs(links['newScore2']))*1000+1).apply(np.int64)
    links['z'] =  links['z'] - np.min(links['z'])
    links['format'] = 'color=' + links['color'] + ',fill_color=' + links['color'] + ',thickness=' + links['thickness'].astype(str) + ',z=' + links['z'].astype(str)
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

    make_circos_conf_file(outdir, target_stats, heatmap_plots, histogram_plots, cellType_order, cellType_labels)
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
    print('Done making circos plot {outdir}/circos.png')
    if not_placed > 0:
        print(f'WARNING: not all labels could be placed. placed={placed} notplaced={not_placed}')
    else:
        print(f'Placed {placed} labels')

    return("Done")
