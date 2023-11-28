import argparse,os,shelve,platform,subprocess,sys
from multiprocessing import Pool,cpu_count
try:
    import pandas as pd
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
try:
    import numpy as np
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.23.0'])
from functools import partial
from time import time
if __name__=='__main__':
    #g=shelve.open('data/SS3_cast_UMIs_concat_histogram')['histogram']
    #parallel_likelihood(g[0], psim_path='data/one_gene_two_state_library_grid', repeat=0, shape=[60, 60, 60],
    #                    downsample=str(1.0), percentage=False, optimize=True, probability=False,
    #                    save_path='data/SS3_cast_UMIs_concat_histogram_infer')
    parser = argparse.ArgumentParser('--desrciption: input for pipeline')
    parser.add_argument('--pexp',help='shelve file which store distribution to be inferred',nargs='?',type=str,default='')
    parser.add_argument('--counts',help='csv file which store specific expression count matrix,row as gene label, column as cell sample',nargs='?',type=str,default='')
    parser.add_argument('--transpose',help='csv file to be transpose, meaning data is in row as cell label, column as gene label',nargs='?', type=bool,default=False)
    parser.add_argument('--psim',help='library to used for inference',nargs='?',type=str,default='one_gene_two_state_library_grid')
    parser.add_argument('--percentage',help='whether to use of percentage gene on in CME model,default False',type=bool,nargs='?',default=False)
    parser.add_argument('--probability',help='whether to use probability as stopping criterion in profile likelihood,default False',type=bool,nargs='?',default=False)
    parser.add_argument('--parallel',help='number of nodes/cores to run parallel,1000 for running on all available cores by default',type=int,nargs='?',default=1000)
    parser.add_argument('--repeat',help='number of sampling repeats to run',type=int,nargs='?',default=0)
    parser.add_argument('--cutoff',help='stopping criterion for profile likelihood confidence interval, default 95%',type=float,nargs='?',default=0.95)
    parser.add_argument('--downsample',help='capture rate used for downsampling',type=str,default='1.0')
    parser.add_argument('--shape',help='grid shape of the library parameter space',type=list,default=[60,60,60])
    parser.add_argument('--cell', help='cell number when sampling in repeat', type=int, default=1)
    parser.add_argument('--optimize',help='whether or not to use optimization for Maximum likelihood inference and profile likelihood, default is True',nargs='?',default=True,type=bool)
    parser.add_argument('--sense',help='whether or not to compute sensitivity in the library,default is False',type=bool,default=False)
    parser.add_argument('--transition', help='whether or not to keep transition matrix in the library,default is False',
                        type=bool, default=False)
    parser.add_argument('--index',help='index in histogram shelve list',type=int,default=0)
    parser.add_argument('--end',help='number of distributions to run for this batch',type=int,nargs='?',default=0)
    parser.add_argument('--loop',help='whether to run in for loop,default is 0',type=int,default=0)
    parser.add_argument('--self',help='whether to self infer, default is 0', type=int, default=0)
    parser.add_argument('--maxcount',help='max_mrna number for library, if downsample exists, this will be extrapolated for downsample,  default is 298',type=int,default=298)
    args=parser.parse_args()
    max_count=-1
    if args.self:
        if args.end==0:
            try:
                args.end=shelve.open(args.psim,'r')['parameter'].shape[0]
            except:
                print('please provide a valid library directory or a working library')
                exit()
        missing=[]
        g=shelve.open(args.psim+'_infer_batch_'+str(args.index),writeback=True)
        keys=list(g.keys())
        for i in range(args.index,args.index+args.end):
            if os.path.exists(args.psim+'_infer_'+str(i)+'.dat'):
                h=shelve.open(args.psim+'_infer_'+str(i))
                g[str(i)]=h[str(i)]
                h.close()
                os.remove(args.psim+'_infer_'+str(i)+'.dat')
                os.remove(args.psim + '_infer_' + str(i) + '.dir')
                os.remove(args.psim + '_infer_' + str(i) + '.bak')
                continue
            elif str(i) in keys:
                continue
            else:
                missing.append(i)
        g.close()
        if len(missing)==0:
            exit()
        g=shelve.open(args.psim)
        pexp=g['downsample_'+str(args.downsample)].tocsr()[missing].todense()
        g.close()
        pexp=np.clip(pexp,a_min=10**-11,a_max=1)
        pexp=pexp/(np.sum(pexp,axis=1)[:,None])
        save_name=args.psim
        pexp=zip(pexp,[100000]*pexp.shape[0],missing)
    else:
        if os.path.exists(args.pexp+'.dir'):
            try:
                if args.end == 0:
                    tmp_pexp = shelve.open(args.pexp)['histogram']
                    args.end = len(tmp_pexp)
                else:
                    tmp_pexp = shelve.open(args.pexp)['histogram'][args.index:args.index+args.end]
                print('directly loading specific histogram from file')
            except:
                print('error in loading distribution')
                exit()
            save_name=args.pexp[:-10]
            missing = []
            pexp=[]
            g = shelve.open(save_name + '_infer_batch_' + str(args.index), writeback=True)
            keys = list(g.keys())
            for i in range(args.end):
                if os.path.exists(save_name + '_infer_' + tmp_pexp[i][-1] + '.dat'):
                    h = shelve.open(save_name + '_infer_' + tmp_pexp[i][-1])
                    g[tmp_pexp[i][-1]] = h[tmp_pexp[i][-1]]
                    h.close()
                    os.remove(save_name + '_infer_' + tmp_pexp[i][-1] + '.dat')
                    os.remove(save_name + '_infer_' + tmp_pexp[i][-1] + '.dir')
                    os.remove(save_name + '_infer_' + tmp_pexp[i][-1] + '.bak')
                    continue
                elif tmp_pexp[i][-1] in keys:
                    continue
                else:
                    pexp.append(tmp_pexp[i])
                    missing.append(tmp_pexp[i][-1])
            g.close()
            if len(missing) == 0:
                exit()
        elif os.path.exists(args.counts):
            count=pd.read_csv(args.counts,sep=',',index_col=0,header=0)
            if args.transpose==True:
                count=count.T
            save_name = os.path.join(os.path.dirname(args.counts),os.path.basename(args.counts)[:-4])
            hist_name=save_name+'_histogram'
            print(hist_name)
            g = shelve.open(hist_name)
            pexp=[]
            larger_than_maxcount=[]
            for i in range(count.shape[0]):
                temp=count.iloc[i][count.iloc[i].notna()]
                nanmax=max(int(np.nanmax(temp)),1)
                hist=np.histogram(temp,bins=nanmax+1,range=(0,nanmax+1))[0]
                hist=hist/hist.sum()
                if nanmax >args.maxcount:
                    larger_than_maxcount.append([hist,temp.shape[0],str(count.index[i])])
                    continue
                pexp.append([hist,temp.shape[0],str(count.index[i])])
                max_count=max(max_count,nanmax)
            g['histogram']=pexp
            if len(larger_than_maxcount)>0:
                g['larger_than_'+str(args.maxcount)]=larger_than_maxcount
            g.close()
            print('saved mrna max_count: {}, number of genes larger than {} mrna: {}, number of genes less than {} mrna: {}'.format(max_count,args.maxcount,len(larger_than_maxcount),args.maxcount,len(pexp)))
            print('saved distribution as shelve file')
            #subprocess.Popen(['python', 'wrapper.py', '--counts', 'nandor_data/dko.norm.csv', '--downsample', '0.3', '--maxcount','1000'])
            exit()
        else:
            print('no relevant file to process, please provide --pexp or --counts in code for histogram file or expression count file')
            exit()
        try:
            flag=False
            if max_count == -1:
                for i in range(len(pexp)):
                    max_count = int(max(max_count, pexp[i][0].shape[0]))
            print('max_count: {}, guess original max_count by {} capture rate: {}'.format(max_count,float(args.downsample),int(max_count/float(args.downsample))))
            max_count=int(max(max_count,args.maxcount)/float(args.downsample))
            psim=shelve.open(args.psim)
            psim=psim['downsample_1.0']
            print(psim.shape,max_count)
            if max_count>psim.shape[1]:
                flag=True
                max_count = int(max(args.maxcount, int(max_count)) / float(args.downsample))
            else:
                print('library loaded')
        except:
            start=time()
            from grid_library import *
            if max_count<298:
                generate_library(args.shape,args.percentage,args.sense,args.transition,path=args.psim,ksyn_max=np.log10(args.maxcount)-0.2)
                print('Generated/loaded library in {0:.2f} minutes'.format((time()-start)/60))
            else:
                flag=True
                pass
        if flag:
            print('error in loading simulated library, either library file is not provided or data has larger mRNA count than library, a new library will be generated')
            max_count=max(298,max_count)
            print(max_count)
            start=time()
            from grid_library import *
            if len(args.pexp)>1:
                args.psim=os.path.join(os.path.dirname(args.pexp),'one_gene_two_state_library_grid_'+str(max_count))
            elif len(args.counts)>1:
                args.psim=os.path.join(os.path.dirname(args.counts),'one_gene_two_state_library_grid_'+str(max_count))
            ksyn_max=np.log10(max_count)
            ksyn_max=ksyn_max-0.4/ksyn_max
            generate_library(args.shape,args.percentage,args.sense,args.transition,path=args.psim,ksyn_max=ksyn_max)
            print('Generated/loaded new library in {0:.2f} minutes'.format((time()-start)/60))
    from function import *
    if args.loop:
        for i in range(len(missing)):
            print(missing[i])
            parallel_likelihood(pexp[i],psim_path=args.psim, repeat=args.repeat,shape=args.shape,downsample=args.downsample,percentage=args.percentage,optimize=args.optimize,probability=args.probability, save_path=save_name+'_infer',cell_number=args.cell,self_infer=args.self)
    else:
        if platform.system()=='Linux':
            number_available_core=min(len(os.sched_getaffinity(0)),args.parallel)
        else:
            number_available_core=min(cpu_count()-1,args.parallel)
        pool=Pool(number_available_core)
        total_number=len(pexp)
        with pool:
            result=list(tqdm(pool.imap(partial(parallel_likelihood, psim_path=args.psim, repeat=args.repeat,shape=args.shape,downsample=args.downsample,percentage=args.percentage, optimize=args.optimize, probability=args.probability, save_path=save_name+'_infer',cell_number=args.cell,self_infer=args.self),pexp), total=total_number))
        pool.close()
    g=shelve.open(save_name+'_infer_batch_'+str(args.index),writeback=True)
    for i in range(len(missing)):
        name=str(pexp[i][2])
        if os.path.exists(save_name+'_infer_'+name+'.dat'):
            tmp=shelve.open(save_name+'_infer_'+name)[name]
            g[name]=tmp
            try:
                os.remove(save_name+'_infer_'+name+'.dir')
                os.remove(save_name+'_infer_'+name+'.dat')
                os.remove(save_name+'_infer_'+name+'.bak')
            except:
               print('error in saving file of {}'.format(name))
    g.close()







