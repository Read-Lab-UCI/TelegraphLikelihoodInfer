from function import *
#from pytorch_test import block_two_state_cme
import os,shelve,sparse,argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import cpu_count,Pool
from functools import partial
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors

def generate_library(shape,percentage=False,sense=False,transition=False,path='one_gene_two_state_library_grid',ksyn_max=2.3):
    print(path)
    f = shelve.open(path)
    keys = list(f.keys())
    f.close()
    if 'batch0' not in keys and 'downsample_1.0' not in keys:
        print(ksyn_max)
        unbinding = np.linspace(-3, 3, shape[0])
        binding = np.linspace(-3, 3,shape[1])
        ksyn = np.linspace(-0.3, ksyn_max,shape[2])
        parameter = np.zeros((shape[0] * shape[1] * shape[2], 5))
        parameter[:, 0] = 0
        parameter[:, -1] = 1
        parameter[:, 1] = np.repeat(ksyn, repeats=shape[1] * shape[2])
        parameter[:, 2] = np.tile(np.repeat(unbinding, repeats=shape[0]), reps=shape[2])
        parameter[:, 3] = np.tile(binding, reps=shape[0] * shape[1])
        parameter[:, 1:-1] = 10 ** parameter[:, 1:-1]
        batch = [0]
        max_state_tx=int(10**(ksyn_max+0.4/ksyn_max))
        print(max_state_tx)
        max_state_tx=int(np.ceil((max_state_tx-50)/40))*40+50
        max_state = list(np.arange(50,max_state_tx)[::40])
        for i in max_state:
            batch.append(np.argmax(parameter[batch[-1]:, 1] > i) + batch[-1])
        batch.append(parameter.shape[0])
        batch=np.unique(batch)
        max_state.append(max_state[-1]+40)
        start = time()
        batch_id = 0
        batch_name = 'batch0'
        save_path = path
        error = []
        for i in tqdm(range(1, len(batch))):
            start2 = time()
            test = block_two_state_cme(parameter[batch[i - 1]:batch[i]], save_path, percentage=percentage, sense=sense,
                                       keep_transition=transition, batch=batch_id)
            # error.append(test.error)
            # test = block_four_state_cme(param_values[:70000], percentage=True, sense=True,keep_transition=False)
            batch_id = test.batch
            print('{} large system of size {} solved in {}s'.format(batch[i] - batch[i - 1], max_state[i - 1],
                                                                    time() - start2))
        print('total time:', time() - start)
    if 'downsample_1.0' not in keys:
        f = shelve.open(path)
        keys = list(f.keys())
        num_parameter = 0
        for i in keys:
            num_parameter += f[i].n
        parameter = np.empty((num_parameter, 5))
        distribution = sparse.zeros((1, max_state[-1]+10))
        v0 = sparse.zeros((1, max_state[-1]+10))
        va = sparse.zeros((1, max_state[-1]+10))
        error = np.zeros(num_parameter)
        if 'S' in f[keys[0]].__dict__:
            S = np.zeros((5, num_parameter, max))
        start = 0
        for i in keys:
            end = start + f[i].n
            temp = sparse.concatenate([f[i].v0, sparse.zeros((f[i].n, max_state[-1]+10 - f[i].v0.shape[1]))], axis=1)
            v0 = sparse.concatenate([v0, temp], axis=0)
            temp = sparse.concatenate([f[i].va, sparse.zeros((f[i].n, max_state[-1]+10 - f[i].va.shape[1]))], axis=1)
            va = sparse.concatenate([va, temp], axis=0)
            error[start:end] = f[i].error
            temp = sparse.concatenate([f[i].distribution, sparse.zeros((f[i].n, max_state[-1]+10 - f[i].distribution.shape[1]))],
                                      axis=1)
            distribution = sparse.concatenate([distribution, temp], axis=0)
            parameter[start:end, 0] = f[i].k_off
            parameter[start:end, 1] = f[i].k_on
            parameter[start:end, 2] = f[i].f
            parameter[start:end, 3] = f[i].h
            parameter[start:end, 4] = f[i].kd
            try:
                S[5, start:end, max_state[-1]+10] = f[i].S
            except:
                pass
            start = end
            del f[i]
        distribution = distribution[1:, :]
        distribution = distribution.to_scipy_sparse().tocoo()
        distribution = filter_sparse_data(distribution)
        f['parameter'] = parameter
        f['error'] = error
        f['downsample_1.0'] = distribution
        f['va'] = va[1:, :]
        f['v0'] = v0[1:, :]
        f['n'] = num_parameter
        try:
            f['S'] = S
        except:
            pass
        for i in ['+', '0', '-']:
            if i == '+':
                selection = np.where(f['parameter'][:, 1] > 1.5)[0]
            elif i == '0':
                selection = np.where((f['parameter'][:, 1] > 0.5) & (f['parameter'][:, 1] < 1.5))[0]
            elif i == '-':
                selection = np.where(f['parameter'][:, 1] < 0.5)[0]
            f[i] = selection
        for p in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            catelogue = generate_binom_catelogue(maxcount=f['downsample_1.0'].shape[1] - 1, p=p)
            catelogue = coo_array(catelogue[:f['downsample_1.0'].shape[1], :f['downsample_1.0'].shape[1]])
            psim = f['downsample_1.0'] @ catelogue
            psim_coo = psim.tocoo()
            data_keep = filter_sparse_data(psim_coo)
            f['downsample_' + str(p)] = data_keep
        f.close()


if __name__=='__main__':
    #generate_library(shape=[60,60,60], path='one_gene_two_state_library_grid', ksyn_max=2.3)
    parser=argparse.ArgumentParser()
    parser.add_argument('--shape',help='shape of grid for generating library, default:[60 60 60]',type=int,nargs='*',metavar='N',default=[60,60,60])
    parser.add_argument('--sense',help='whether or not to compute sensitivity in the library,default is False',type=bool,default=False)
    parser.add_argument('--transition', help='whether or not to keep transition matrix in the library,default is False',
                        type=bool, default=False)
    parser.add_argument('--percentage', help='whether to use of percentage gene on in CME model,default is False',
                        type=bool, nargs='?', default=False)
    parser.add_argument('--path',help='library path',type=str,nargs='?',default='one_gene_two_state_library_grid')
    parser.add_argument('--mRNA',help='maximum MRNA value',type=int,default=298)
    args=parser.parse_args()
    ksyn_max=(np.log10(args.mRNA)+(np.log10(args.mRNA)**2-1.6)**0.5)/2
    ksyn_max=max(ksyn_max,2.3)
    generate_library(shape=args.shape,percentage=args.percentage,sense=args.sense,transition=args.transition,path=args.path,ksyn_max=ksyn_max)
    """
    f = shelve.open('one_gene_two_state_library_grid')
    keys = list(f.keys())
    f.close()
    if 'batch0' not in keys and 'downsample_1.0' not in keys:
        unbinding=np.linspace(-3,3,args.shape[0])
        binding=np.linspace(-3,3,args.shape[1])
        k_on=np.linspace(-0.3,2.3,args.shape[2])
        parameter=np.zeros((args.shape[0]*args.shape[1]*args.shape[2],5))
        parameter[:,0]=0
        parameter[:,-1]=1
        parameter[:,1]=np.repeat(k_on,repeats=args.shape[1]*args.shape[2])
        parameter[:,2] = np.tile(np.repeat(unbinding, repeats=args.shape[0]), reps=args.shape[2])
        parameter[:,3]=np.tile(binding,reps=args.shape[0]*args.shape[1])
        parameter[:,1:-1]=10**parameter[:,1:-1]
        batch = [0]
        max_state = [50, 70, 90, 110, 130, 150, 170, 190]
        for i in max_state:
            batch.append(np.argmax(parameter[batch[-1]:,1] > i) + batch[-1])
        batch.append(parameter.shape[0])
        max_state.append(210)
        start = time()
        batch_id = 0
        batch_name = 'batch0'
        save_path = 'one_gene_two_state_library_grid'
        error = []
        for i in tqdm(range(1, len(batch))):
            start2 = time()
            test = block_two_state_cme(parameter[batch[i - 1]:batch[i]], save_path, percentage=False, sense=False,
                                       keep_transition=False, batch=batch_id)
            # error.append(test.error)
            # test = block_four_state_cme(param_values[:70000], percentage=True, sense=True,keep_transition=False)
            batch_id = test.batch
            print('{} large system of size {} solved in {}s'.format(batch[i] - batch[i - 1], max_state[i - 1],
                                                                    time() - start2))
        print('total time:', time() - start)
    if 'downsample_1.0' not in keys:
        f = shelve.open('one_gene_two_state_library_grid')
        keys = list(f.keys())
        num_parameter = 0
        for i in keys:
            num_parameter += f[i].n
        parameter = np.empty((num_parameter, 5))
        distribution = sparse.zeros((1, 300))
        v0 = sparse.zeros((1, 300))
        va = sparse.zeros((1, 300))
        error = np.zeros(num_parameter)
        if 'S' in f[keys[0]].__dict__:
            S = np.zeros((5, num_parameter, 300))
        start = 0
        for i in keys:
            end = start + f[i].n
            temp = sparse.concatenate([f[i].v0, sparse.zeros((f[i].n, 300 - f[i].v0.shape[1]))], axis=1)
            v0 = sparse.concatenate([v0, temp], axis=0)
            temp = sparse.concatenate([f[i].va, sparse.zeros((f[i].n, 300 - f[i].va.shape[1]))], axis=1)
            va = sparse.concatenate([va, temp], axis=0)
            error[start:end] = f[i].error
            temp = sparse.concatenate([f[i].distribution, sparse.zeros((f[i].n, 300 - f[i].distribution.shape[1]))],
                                          axis=1)
            distribution = sparse.concatenate([distribution, temp], axis=0)
            parameter[start:end, 0] = f[i].k_off
            parameter[start:end, 1] = f[i].k_on
            parameter[start:end, 2] = f[i].f
            parameter[start:end, 3] = f[i].h
            parameter[start:end, 4] = f[i].kd
            try:
                S[5, start:end, 300] = f[i].S
            except:
                pass
            start = end
            del f[i]
        distribution=distribution[1:,:]
        distribution= distribution.to_scipy_sparse().tocoo()
        distribution = filter_sparse_data(distribution)
        f['parameter'] = parameter
        f['error'] = error
        f['downsample_1.0'] = distribution
        f['va'] = va[1:, :]
        f['v0'] = v0[1:, :]
        f['n'] = num_parameter
        try:
            f['S'] = S
        except:
            pass
        for i in ['+', '0', '-']:
            if i == '+':
                selection = np.where(f['parameter'][:, 1] > 1.5)[0]
            elif i == '0':
                selection = np.where((f['parameter'][:, 1] > 0.5) & (f['parameter'][:, 1] < 1.5))[0]
            elif i == '-':
                selection = np.where(f['parameter'][:, 1] < 0.5)[0]
            f[i] = selection
        for p in [0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            catelogue = generate_binom_catelogue(maxcount=f['downsample_1.0'].shape[1] - 1, p=p)
            catelogue = coo_array(catelogue[:f['downsample_1.0'].shape[1], :f['downsample_1.0'].shape[1]])
            psim = f['downsample_1.0'] @ catelogue
            psim_coo = psim.tocoo()
            data_keep = filter_sparse_data(psim_coo)
            f['downsample_'+str(p)]=data_keep
        f.close()
    """
    """
    percentage = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    kd = 1 / (1 / percentage - 1)  # 1/(k_on/(k_on+k_off))-1=k_off/k_on  = G/G*
    unbinding = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ksyn = np.array([3.5, 10, 20])
    param = np.zeros((ksyn.shape[0] * unbinding.shape[0] * percentage.shape[0], 5))
    param[:, 1] = np.repeat(ksyn, repeats=unbinding.shape[0] * percentage.shape[0])
    param[:, 2] = np.tile(unbinding, reps=ksyn.shape[0] * percentage.shape[0])
    param[:, 3] = np.tile(np.repeat(percentage, repeats=unbinding.shape[0]), reps=ksyn.shape[0])
    param[:, 4] = 1
    param[:, 3] = 1/(1/param[:,3]-1)*param[:,2]
    param=param[np.where((param[:,3]<10**3)&(param[:,3]>10**-3))[0],:]
    exp_histogram = block_two_state_cme(param,percentage=False).batch0.distribution.todense() + 10**-11
    exp_histogram = exp_histogram / exp_histogram.sum(axis=1)[:, None]
    max_cell_number = 50000
    repeat = 500
    capture_rate = False
    exp_data = {}
    exp_data['origin_histogram'] = exp_histogram
    exp_data_reform=[]
    if capture_rate:
        catelogue = generate_binom_catelogue(maxcount=exp_data['origin_histogram'].shape[1] - 1, p=capture_rate)
        exp_data['histogram']=exp_data['origin_histogram']@catelogue
    else:
        exp_data['histogram']=exp_data['origin_histogram']
    #exp_data['sample'] = np.zeros((exp_data['histogram'].shape[0], max_cell_number))
    for i in range(exp_data['histogram'].shape[0]):
        sample = np.random.choice(np.arange(exp_data['histogram'].shape[1]), size=max_cell_number,
                                  p=exp_data['histogram'][i, :])
        #exp_data['sample'][i, :] = sample
        exp_data_reform.append([i,param[i,:],exp_data['origin_histogram'][i,:],exp_data['histogram'][i,:],sample])
    #exp_data['sample'] = exp_data['sample'].astype(np.uint8)
    """
    """
    f=shelve.open('one_gene_two_state_library_grid')
    parameter=f['parameter']
    if capture_rate:
        psim_origin=f['downsample_1']
        psim_origin=psim_origin.todense()+10**-11
        psim_origin=psim_origin/psim_origin.sum(axis=1)[:,None]
        psim=f['downsample_'+str(capture_rate)]
    else:
        psim = f['downsample_1']
    psim=psim.tocsr()
    k_on=list(set(parameter[:,1]))
    k_on.sort()
    unbinding=list(set(parameter[:,2]))
    unbinding.sort()
    binding=list(set(parameter[:,3]))
    binding.sort()
    bounds = np.array([0, -1, -2, -5, -10, -100, -1000, -2000])[::-1]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cell_number=10000
    """
    """
    if capture_rate:
        save_path='nandor//PIC/'+str(capture_rate)+'_capture_downsample_'
    else:
        save_path='nandor//PIC/no_sampling_'
    #likelihood=np.zeros((exp_data['histogram'].shape[0],repeat,60,60,60),dtype='float16')
    arg_max_lib={}
    if capture_rate:
        key='downsampled_' + str(capture_rate)
    else:
        key='no downsampling'
    g = shelve.open('arg_max_lib_nandor')
    g[key]={}
    g.close()
    for cell_number in [700,350,100]:
        pool=Pool(6)
        with pool:
            result=list(tqdm(pool.imap(partial(likelihood_cal_and_plot,repeat=repeat,cell_number=cell_number,library_path='one_gene_two_state_library_grid',save_path=save_path,capture_rate=capture_rate),exp_data_reform),total=len(exp_data_reform)))
        pool.close()
        g=shelve.open('arg_max_lib_nandor',writeback=True)
        g[key][cell_number]=result
        g.close()
    """
    """
    for cell_number in [100,10000,350,700,1000,3500,7000]:
        arg_max_lib[cell_number]={}
        for i in tqdm(range(exp_data['histogram'].shape[0])):
            #data=np.repeat(exp_data['sample'][i:i+1,:],repeats=repeat,axis=0).astype('uint8')
            #data=list(data)
            data=exp_data['sample'][i,:]
            bin_max=exp_data['sample'].max()
            #pool = Pool(cpu_count() - 8)
            #with pool:
            #    result = list(pool.imap(partial(parallel_histogram, bins=bin_max + 1, density=True, random_sample=cell_number,seed=True),zip(data,np.arange(repeat))))
            #pool.close()
            result=[]
            for j in range(repeat):
                result.append(parallel_histogram([data,j],bins=bin_max+1,density=False,random_sample=cell_number,seed=True))
            result= np.array(result)
            psim_d=psim.copy()
            max_m = max(exp_data['origin_histogram'].shape[1], max(psim.indices) + 1)
            psim_d.resize((psim_d.shape[0], max_m))
            psim_d=psim_d.todense()+10**-11
            psim_d=psim_d/psim_d.sum(axis=1)[:,None]
            pexp=np.zeros((result.shape[0]+1,max_m))
            pexp[0,:exp_data['origin_histogram'].shape[1]]=exp_data['origin_histogram'][i,:]*cell_number
            pexp[1:,:result.shape[1]]=result
            del result
            #pexp = pexp + 10 ** -9
            #pexp = pexp / pexp.sum(axis=1)[:, None]
            temp = np.matmul(np.log(psim_d), pexp.T).T
            arg_max=np.argmax(temp[1:,:],axis=1)
            likelihood=np.reshape(temp,newshape=(repeat+1,60,60,60))
            pexp=pexp/pexp.sum(axis=1)[:,None]
            fig = plt.figure(1, figsize=(12, 6),dpi=300)
            plot_m_max=max(10,pexp.shape[1]-np.argmax(pexp[0,:][::-1]>10**-6))
            plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=2)
            plt.plot(pexp[:-1,:plot_m_max].T,alpha=10/repeat,color='blue',zorder=-2)
            plt.plot(pexp[-1, :plot_m_max].T, alpha=10/repeat, color='blue',label='sample_replicate')
            plt.plot(psim_d[arg_max[:-1], :plot_m_max].T, color='orange', alpha=10 / repeat, zorder=-1)
            plt.plot(psim_d[arg_max[-1], :plot_m_max].T, color='orange', alpha=10 / repeat, label='inference')
            if capture_rate:
                plt.plot(exp_data['histogram'][i,:plot_m_max],color='green',marker='o',markersize=3,label='downsample_groundtruth',zorder=10)
                plt.plot(exp_data['origin_histogram'][i,:plot_m_max],color='black',marker='v',markersize=3,label='undownsample_groundtruth',zorder=10)
            else:
                plt.plot(exp_data['histogram'][i,:plot_m_max],color='green',marker='o',markersize=5,label='groundtruth')
            plt.title('Distribution')
            if capture_rate:
                plt.plot(psim_origin[arg_max[:-1],:plot_m_max].T,color='red',alpha=10/repeat,zorder=1)
                plt.plot(psim_origin[arg_max[-1], :plot_m_max].T, color='red', alpha=10/repeat,label='undownsample inference')
            #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xticks(np.linspace(0,plot_m_max,10,dtype='int'))
            plt.xlabel('mRNA copy number')
            plt.ylabel('probability')
            leg = plt.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=2)
            burst_size=parameter[:,1]/parameter[:,2]
            burst_frequency=parameter[:,3]
            burst_lib=list(set(burst_size))
            frequency_lib=list(set(burst_frequency))
            frequency_lib.sort()
            burst_lib.sort()
            burst_dict={}
            frequency_dict={}
            burst_likelihood=[]
            for burst in burst_lib:
                burst_dict[burst]=set(np.where(burst_size==burst)[0])
            for frequency in frequency_lib:
                frequency_dict[frequency]=set(np.where(burst_frequency==frequency)[0])
            temp[0,:]=temp[0,:]-temp[0,:].max()
            for burst in burst_dict.keys():
                for frequency in frequency_dict.keys():
                    index=list(burst_dict[burst].intersection(frequency_dict[frequency]))
                    burst_likelihood.append([burst,frequency,np.nanmax(temp[0,index])])
            burst_likelihood=np.array(burst_likelihood)
            plt.scatter(np.log10(burst_likelihood[:, 0]), np.log10(burst_likelihood[:, 1]), c=burst_likelihood[:, 2],cmap='cool', norm=norm,zorder=-1)
            plt.colorbar(label='log_likelihood')
            burst_size = parameter[arg_max, 1] / parameter[arg_max, 2]
            burst_frequency = np.log10(parameter[arg_max, 3])
            plt.scatter(np.log10(burst_size),burst_frequency,alpha=20/repeat,color='darkorange',zorder=0)
            plt.scatter(np.log10(param[i,1]/param[i,2]),np.log10(param[i,3]),color='green',marker='o',s=20,zorder=1)
            plt.xlabel('burst size in log10')
            plt.xlim(-3.3,5.3)
            plt.ylim(-3,3)
            plt.ylabel('burst frequency k_on in log10')
            plt.title('Log Likelihood Surface of Ground Truth')
            plt.subplot2grid((3, 6), (2, 0), colspan=2, rowspan=1)
            temp_max = np.nanmax(likelihood[0:1,:], axis=(1,2))
            temp_max=temp_max-temp_max.max(axis=1)[:,None]
            CI_koff = (find_CI(np.log10(unbinding), temp_max[0, :]),temp_max[0,:])
            plt.fill_between(np.log10(unbinding), temp_max[0, :], -5, where=temp_max[0,:]>-1.92,color='violet',zorder=-1,alpha=0.5)
            #plt.plot(np.log10(unbinding), temp_max[1:,:].T,alpha=0.02,color='blue')
            plt.plot(np.log10(unbinding), temp_max[0, :], color='green',zorder=1)
            for j in range(repeat):
                plt.axvline(np.log10(parameter[arg_max[j],2]), color='darkorange',alpha=10/repeat,zorder=0)
            plt.axvline(np.log10(param[i, 2]), color='green')
            plt.xlabel('Koff in log10')
            plt.ylabel('log_likelihood')
            plt.ylim(-5,0)
            np.log10(parameter[arg_max,2])
            plt.subplot2grid((3, 6), (2, 2), colspan=2, rowspan=1)
            temp_max = np.nanmax(likelihood[0:1,:], axis=(1,3))
            temp_max = temp_max - temp_max.max(axis=1)[:, None]
            CI_kon = (find_CI(np.log10(binding), temp_max[0, :]),temp_max[0,:])
            plt.fill_between(np.log10(binding),temp_max[0,:],-5,where=temp_max[0,:]>-1.92,color='violet',zorder=-1,alpha=0.5)
            #plt.plot(np.log10(binding), temp_max[1:,:].T,alpha=0.02,color='blue')
            plt.plot(np.log10(binding), temp_max[0, :], color='green',zorder=1)
            for j in range(repeat):
                plt.axvline(np.log10(parameter[arg_max[j],3]), color='darkorange',alpha=10/repeat,zorder=0)
            plt.axvline(np.log10(param[i, 3]), color='green')
            plt.xlabel('Kon in log10')
            plt.title('Profile Likelihood')
            plt.ylim(-5, 0)
            plt.subplot2grid((3, 6), (2, 4), colspan=2, rowspan=1)
            temp_max = np.nanmax(likelihood[0:1,:], axis=(2,3))
            temp_max = temp_max - temp_max.max(axis=1)[:, None]
            CI_ksyn=(find_CI(np.log10(k_on),temp_max[0,:]),temp_max[0,:])
            plt.fill_between(np.log10(k_on), temp_max[0, :], -5, where=temp_max[0,:]>-1.92, color='violet',zorder=-1,label='95%CI',alpha=0.5)
            #plt.plot(np.log10(k_on), temp_max[1:-1,:].T,alpha=0.02,color='blue')
            #plt.plot(np.log10(k_on), temp_max[-1, :].T, alpha=0.02, color='blue',label='sample_replicate')
            plt.plot(np.log10(k_on), temp_max[0, :], color='green',zorder=1)
            for j in range(repeat):
                plt.axvline(np.log10(parameter[arg_max[j],1]), color='darkorange',alpha=10/repeat,zorder=0)
            plt.axvline(np.log10(param[i, 1]), color='green',zorder=1)
            plt.xlabel('Ksyn in log10')
            plt.ylim(-5, 0)
            plt.legend()
            fig.suptitle('ground truth:' + str(param[i, 1:-1]))
            fig.tight_layout()
            arg_max_lib[cell_number][i]=[CI_koff,CI_kon,CI_ksyn,arg_max]
            save_path2=save_path+str(cell_number)+'_cells/'
            if not os.path.isdir(save_path2):
                os.mkdir(save_path2)
            if capture_rate:
                plt.savefig(save_path2 + str(i) + 'png')
            else:
                plt.savefig(save_path2+str(i)+'png')
            #plt.show()
    print('done')
    """

