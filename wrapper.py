import argparse,os,shelve
from multiprocessing import Pool, cpu_count
import pandas as pd
from function import *
from functools import partial
from time import time

if __name__=='__main__':
    #g=shelve.open('data/SS3_cast_UMIs_concat_histogram')['histogram']
    #parallel_likelihood(g[0], psim_path='data/one_gene_two_state_library_grid', repeat=0, shape=[60, 60, 60],
    #                    downsample=str(1.0), percentage=False, optimize=True, probability=False,
    #                    save_path='data/SS3_cast_UMIs_concat_histogram_infer')
    parser = argparse.ArgumentParser('--desrciption: input for pipeline')
    parser.add_argument('--pexp',help='shelve file which store distribution to be inferred',nargs='?',type=str,default='')
    parser.add_argument('--counts',help='csv file which store specific expression count matrix',nargs='?',type=str,default='')
    parser.add_argument('--psim',help='library to used for inference',nargs='?',type=str,default='one_gene_two_state_library_grid')
    parser.add_argument('--percentage',help='whether to use of percentage gene on in CME model,default False',type=bool,nargs='?',default=False)
    parser.add_argument('--probability',help='whether to use probability as stopping criterion in profile likelihood,default False',type=bool,nargs='?',default=False)
    parser.add_argument('--parallel',help='number of nodes/cores to run parallel,0 for running on all available cores by default',type=int,nargs='?',default=0)
    parser.add_argument('--repeat',help='number of sampling repeats to run',type=int,nargs='?',default=0)
    parser.add_argument('--cutoff',help='stopping criterion for profile likelihood confidence interval, default 95%',type=float,nargs='?',default=0.95)
    parser.add_argument('--downsample',help='capture rate used for downsampling',type=str,default='1.0')
    parser.add_argument('--shape',help='grid shape of the library parameter space',type=list,default=[60,60,60])
    parser.add_argument('--optimize',help='whether or not to use optimization for Maximum likelihood inference and profile likelihood, default is True',nargs='?',default=True,type=bool)
    parser.add_argument('--sense',help='whether or not to compute sensitivity in the library,default is False',type=bool,default=False)
    parser.add_argument('--transition', help='whether or not to keep transition matrix in the library,default is False',
                        type=bool, default=False)
    args=parser.parse_args()
    if os.path.exists(args.pexp):
        print('directly loading histogram from file')
        pexp=shelve.open(args.pexp)['histogram']
        print(len(pexp))
        save_name=args.pexp[:-10]
    elif os.path.exists(args.counts):
        print('generating histogram from count matrix')
        count=pd.read_csv(args.counts,sep=',',index_col=0,header=0)
        save_name = args.counts.split('.')[0]
        hist_name=save_name+'_histogram'
        g = shelve.open(hist_name)
        pexp=[]
        for i in range(count.shape[0]):
            temp=count.iloc[i][count.iloc[i].notna()]
            nanmax=max(int(np.nanmax(temp)),1)
            hist=np.histogram(temp,bins=nanmax+1,range=(0,nanmax+1))[0]
            hist=hist/hist.sum()
            pexp.append([hist,temp.shape[0],count.index[i]])
        g['histogram']=pexp
        g.close()
    else:
        print('no relevant file to process, please provide --pexp or --counts in code for histogram file or expression count file')
        exit()
    try:
        flag=False
        max_count = 0
        for i in range(len(pexp)):
            max_count = int(max(max_count, pexp[i][0].shape[0]))
        psim=shelve.open(args.psim)
        psim=psim['downsample_'+args.downsample]
        print('library loaded')
        if max_count>psim.shape[1]:
            flag=True
    except:
        flag=True
    if flag:
        print('error in loading simulated library, either library file is not provided or data has larger mRNA count than library, a new library will be generated')
        max_count=max(298,max_count)
        start=time()
        from grid_library import *
        generate_library(args.shape,args.percentage,args.sense,args.transition,path=os.path.dirname(save_name)+'/one_gene_two_state_library_grid',ksyn_max=np.log10(max_count)-0.2)
        print('Generated/loaded new library in {0:.2f} minutes'.format((time()-start)/60))
    pool=Pool(cpu_count()-1)
    with pool:
        result=list(tqdm(pool.imap(partial(parallel_likelihood, psim_path=args.psim, repeat=args.repeat,shape=args.shape,downsample=args.downsample,percentage=args.percentage, optimize=args.optimize, probability=args.probability, save_path=save_name+'_infer'),pexp), total=len(pexp)))
    pool.close()






