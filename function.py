from time import time
import psutil,shelve,os,subprocess,sys
try:
    from scipy.sparse import diags
except:
    subprocess.check_call([sys.executable,'-m','pip','install','scipy'])
    from scipy.sparse import diags
from scipy.stats import binom,chi2
from scipy.optimize import minimize
from scipy.sparse import coo_array
try:
    import sparse
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sparse'])
    import sparse
try:
    import h5py
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'h5py'])
    import h5py
try:
    import tensorflow as tf
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
    import tensorflow as tf
from copy import deepcopy
import numpy as np
from tqdm import tqdm
#from scipy.interpolate import interp1d
#from scipy.interpolate import RegularGridInterpolator
#import matplotlib.pyplot as plt
#from matplotlib import colors

class init:
    def __init__(self):

        pass


def generate_binom_catelogue(maxcount=20, p=0.1):
    if os.path.isfile('downsample.h5'):
        h5=h5py.File('downsample.h5','r')
    else:
        h5 = h5py.File('downsample.h5', 'a')
    if str(p) in h5.keys():
        catelogue = h5[str(p)][()]
        if catelogue.shape[0] < maxcount + 1:
            tmp = np.zeros((maxcount + 1, maxcount + 1))
            tmp[:catelogue.shape[0], :catelogue.shape[1]] = catelogue
            for i in range(catelogue.shape[0], maxcount + 1):
                for j in range(maxcount + 1):
                    tmp[i, j] = binom.pmf(j, i, p)
            h5.close()
            h5 = h5py.File('downsample.h5', 'a')
            del h5[str(p)]
            h5.create_dataset(str(p), data=tmp, compression='gzip')
            h5.close()
            return tmp
        else:
            return catelogue[:maxcount + 1, :maxcount + 1]
    h5.close()
    h5 = h5py.File('downsample.h5', 'a')
    catelogue = np.zeros((maxcount + 1, maxcount + 1))
    for i in range(maxcount + 1):
        for j in range(maxcount + 1):
            catelogue[i, j] = binom.pmf(j, i, p)
    h5.create_dataset(str(p), data=catelogue, compression='gzip')
    h5.close()
    return catelogue


def downsample_distribution(p, catelogue=[]):
    pnew = []
    for i in range(len(p)):
        pnew.append(np.sum(p.T * catelogue[:len(p), i]))
    return np.array(pnew) / np.sum(pnew)


def filter_sparse_data(data):
    indice_keep = np.where(data.data > 10**-11)[0]
    data_keep = data.data[indice_keep]
    col_keep = data.col[indice_keep]
    row_keep = data.row[indice_keep]
    return coo_array((data_keep, (row_keep, col_keep)))


class block_two_state_cme:
    def __init__(self, param, save_path=None, percentage=True, keep_transition=False, sense=False, device='/cpu:0',
                 batch=0, print_flag=True):
        self.percentage = percentage
        self.device = device
        self.keep_transition = keep_transition
        self.sense = sense
        self.print_flag=print_flag
        if any(param[:,0]==0):
            self.max_state = np.ceil(np.max(param[:, 1] / param[:, 4])).astype('int')
        else:
            try:
            #self.max_state = int(np.ceil(np.max(param[:, 0] * param[:, 1] / param[:, 4])))
                self.max_state = np.max(np.ceil(np.max(np.vstack((param[:, 0] * param[:, 1],param[:,0])),axis=0) / param[:, 4]))
            except:
                pass
        if self.max_state < 25:
            self.max_state = 25
        self.max_state = np.log10(self.max_state)
        self.max_state =np.ceil(10**(self.max_state+0.4/self.max_state)).astype('int')
        self.batch = batch
        if self.print_flag:
            print(self.max_state)

        start = 0
        end = 0
        while end < param.shape[0]:
            t1=time()
            memory = psutil.virtual_memory().available
            batch_size = int(memory * 0.45 / 16 / self.max_state / self.max_state)
            name = 'batch' + str(self.batch)
            end = start + batch_size
            setattr(self, name, init())
            current = getattr(self, name)
            if self.print_flag:
                print(name)
            current.n = param[start:end, 0].shape[0]
            current.k_off = param[start:end, 0]
            current.k_on = param[start:end, 1]
            current.f = param[start:end, 2]
            if not self.percentage:
                current.h = param[start:end, 3]
                current.p_on = current.h / (current.h + current.f)
            else:
                current.p_on = param[start:end, 3]
                current.h = current.f * current.p_on / (1 - current.p_on)
            current.kd = param[start:end, 4]
            self.solve(name)
            if not save_path is None:
                with shelve.open(save_path) as f:
                    f[name] = current
                f.close()
                if self.print_flag:
                    print(self.batch, param[start:end].shape[0], time() - t1)
                delattr(self, name)
            start = end
            self.batch += 1
    def solve(self, name):
        start=time()
        current = getattr(self, name)
        mask = np.ones(current.k_off.shape[0],dtype=bool)
        #mask[current.k_off*current.k_on<0.3] = False
        k_on=current.k_on
        mask[current.k_off==0]=False
        k_on[~mask]=current.k_on[~mask]
        k_on[mask]=current.k_on[mask]*current.k_off[mask]
        k_off = current.k_off
        k_off[~mask]=0
        identity = sparse.eye(self.max_state)
        with tf.device(self.device):
            self.dkd = sparse.COO.from_scipy_sparse(diags(np.arange(1, self.max_state), offsets=1) - diags(
                np.arange(self.max_state)))
            dA_dkd = self.dkd * current.kd[:, np.newaxis, np.newaxis]
            self.tridiag_l = sparse.COO.from_scipy_sparse(diags(np.ones(self.max_state - 1), offsets=-1) - diags(
                np.ones(self.max_state)))
            A = k_on[:, np.newaxis, np.newaxis] * self.tridiag_l - current.f[:, np.newaxis,
                                                                    np.newaxis] * identity + dA_dkd
            B = k_off[:, np.newaxis, np.newaxis] * self.tridiag_l - current.h[:, np.newaxis,
                                                                    np.newaxis] * identity + dA_dkd
            if self.sense or self.keep_transition:
                row1=sparse.concatenate([A,current.h[:,np.newaxis,np.newaxis]*identity],axis=2)
                row2=sparse.concatenate([current.f[:,np.newaxis,np.newaxis]*identity,B],axis=2)
                transition = sparse.concatenate([sparse.COO.from_numpy(np.ones((current.n, 1, int(self.max_state * 2)))), row1, row2], axis=1)
                if self.keep_transition:
                    current.transition = init()
                    current.transition.values = transition[:, 1:, ].data
                    current.transition.indices = transition[:, 1:, :].coords.T
                    current.transition.shape = transition[:, 1:, :].shape
                transition = tf.sparse.SparseTensor(values=transition.data,
                                                    indices=transition.coords.T, dense_shape=transition.shape)
                #start = time()
                #temp_sol=tf.experimental.numpy.abs(tf.linalg.svd(tf.sparse.to_dense(transition)[:,1:,:])[-1][:,:,-1])
                #temp_sol=temp_sol/tf.experimental.numpy.sum(temp_sol,axis=1)[:,None]
                #print(time() - start)

            A = tf.sparse.SparseTensor(values=A.data, indices=A.coords.T, dense_shape=A.shape)
            B = tf.sparse.SparseTensor(values=B.data, indices=B.coords.T, dense_shape=B.shape)
            I_hf = (current.h*current.f)[:,np.newaxis,np.newaxis] * identity
            I_hf = tf.sparse.SparseTensor(values=-I_hf.data, indices=I_hf.coords.T, dense_shape=I_hf.shape)
            testing=tf.sparse.add(tf.matmul(tf.sparse.to_dense(B), tf.sparse.to_dense(A)), I_hf)
            if self.print_flag:
                print('{} time for constructing matrix'.format(time() - start))

            #n_cores=cpu_count()
            #testing2=[]
            #batch_size=int(testing.shape[0]/n_cores)
            #for i in range(n_cores):
            #    testing2.append(np.array(testing[i*batch_size:(i+1)*batch_size,:,:]))
            #start = time()
            #temp=[]
            #for i in range(testing.shape[0]):
            #    temp.append(np.linalg.svd(testing[i,:,:])[-1][-1])
            #print(time()-start,'time looping 1 thread')
            #sleep(10)
            #pool=Pool(n_cores)
            #with pool:
            #    t1=list(pool.imap(loop_svd,testing2))
            #pool.close()
            #print(np.mean(t1),'time for solving svd in loop',time()-start-10,'total time looping')
            #del testing2
            start=time()
            current.va=tf.experimental.numpy.abs(tf.linalg.svd(testing)[-1][:,:,-1])
            if self.print_flag:
                print(time()-start,'time for solving svd in tensor')
            #current.va = tf.experimental.numpy.abs(tf.linalg.svd(tf.sparse.add(tf.matmul(tf.sparse.to_dense(B), tf.sparse.to_dense(A)),I_hf))[-1][:, :, -1])
            current.va = np.array(current.va/(tf.experimental.numpy.sum(current.va, axis=1)/current.p_on)[:, None])
            current.v0 = tf.squeeze(tf.matmul(tf.sparse.to_dense(A),(current.va/current.h[:,None])[:,:,None]))
            current.v0 = np.array(current.v0/(tf.experimental.numpy.sum(current.v0, axis=1)/(1-current.p_on))[:, None])
            current.error = current.va[:, -1]*current.k_on+current.v0[:, -1]*current.k_off
            current.va[current.va < 10 ** -12] = 0
            current.va = current.va / (np.sum(current.va, axis=1)/current.p_on)[:, None]
            current.v0[current.v0 < 10 ** -12] = 0
            current.v0 = current.v0 / (np.sum(current.v0, axis=1)/(1-current.p_on))[:, None]
            current.distribution = current.va+current.v0
            current.distribution[current.distribution<10**-11]=0
            current.v0 = sparse.COO.from_numpy(current.v0)
            current.va = sparse.COO.from_numpy(current.va)
            current.distribution = sparse.COO.from_numpy(current.distribution)
            #temp_sol=temp_sol[:,:int(temp_sol.shape[1]/2)]+temp_sol[:,int(temp_sol.shape[1]/2):]
            #self.error=tf.linalg.norm(current.distribution.todense()-temp_sol,axis=1)
            del A,B,I_hf,dA_dkd
            if self.sense:
                del row1,row2
                start = time()
                S = []
                for i, j in enumerate(['k_on', 'f', 'h', 'k_off', 'kd']):
                    if i == 2 and self.percentage:
                        temp = (current.f * np.log(1 - current.p_on))
                        b = tf.concat([np.zeros((current.n, 1)), (temp[:, np.newaxis] * current.v0).todense(), (-temp[:, np.newaxis] * current.v0).todense()], axis=1)[:, :, np.newaxis]
                    elif i == 2 and not self.percentage:
                        b = tf.concat([np.zeros((current.n, 1)), current.v0.todense(), -current.v0.todense()],axis=1)[:, :, np.newaxis]
                    # elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   -tf.einsum('BNi,Bi->BN', ((current.mask * current.k_off + (1 - current.mask)) * current.mask_a)[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.vab),
                    #                   -tf.einsum('BNi,Bi->BN',(current.mask * current.k_off + (1 - current.mask))[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.va0)], axis=1)[:, :, np.newaxis]
                    #elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   tf.einsum('BNi,Bi->BN',(k_off*mask+(1-mask))[:, np.newaxis, np.newaxis] * self.tridiag_l.todense(), current.va.todense()), np.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]
                    elif i==0:
                        b = tf.concat([np.zeros((current.n, 1)),tf.einsum('BNi,Bi->BN',k_off[:, np.newaxis, np.newaxis] * self.tridiag_l.todense(), current.va.todense()), np.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]

                    elif i == 1:
                        b = tf.concat([np.zeros((current.n, 1)), -current.va.todense(), current.va.todense()], axis=1)[:, :, np.newaxis]
                    elif i == 3:
                        b = tf.concat([np.zeros((current.n, 1)),
                                       tf.einsum('BNi,Bi->BN', current.k_on[:, np.newaxis, np.newaxis] * self.tridiag_l.todense(), current.va.todense()),
                                       tf.einsum('Ni,Bi->BN', self.tridiag_l.todense(), current.v0.todense())], axis=1)[:, :, np.newaxis]
                    elif i == 4:
                        b = tf.concat(
                            [np.zeros((current.n, 1)), tf.einsum('Ni,Bi->BN', self.dkd.todense(), current.va.todense()),
                             tf.einsum('Ni,Bi->BN', self.dkd.todense(), current.v0.todense())], axis=1)[:, :, np.newaxis]
                    S.append(tf.squeeze(tf.cast(tf.linalg.lstsq(tf.sparse.to_dense(transition), b, fast=False), tf.float32)))
                S = np.array(S, dtype=np.single)
                S=S.reshape(S.shape[0],S.shape[1],2,int(S.shape[2]/2))
                current.S = S
                if self.print_flag:
                    print('{} time for solving sensitivity'.format(time()-start))
    """
    def sensitivity(self, name):
        current = getattr(self, name)
        with tf.device(self.device):
            identity = np.eye(self.max_state)
            transition = tf.sparse.concat(1,[tf.sparse.from_dense(np.ones((current.n, 1, self.max_state * 2))),tf.sparse.SparseTensor(values=current.transition.values,indices=current.transition.indices,dense_shape=current.transition.shape)])
            S = []
            for i, j in enumerate(['k_on', 'f', 'h', 'k_off', 'kd']):
                if i == 2 and self.percentage:
                    temp = (current.f * np.log(1 - current.p_on))
                    b = np.concatenate([np.zeros((current.n, 1)), -temp[:, np.newaxis]*current.v2, temp[:, np.newaxis]*current.v2], axis=1)[:, :, np.newaxis]
                elif i == 2 and not self.percentage:
                    b = np.concatenate([np.zeros((current.n, 1)), -current.v2, current.v2], axis=1)[:, :, np.newaxis]
                elif i == 0:
                    temp = -(current.mask * current.k_off + (1 - current.mask))
                    values = (np.vstack(([np.array(self.tridiag_l.values)] * current.n)) * temp[:, None]).ravel()
                    indices = np.vstack(([self.tridiag_l.indices] * current.n))
                    indices = np.hstack((np.ones((indices.shape[0], 1), dtype=int), indices))
                    indices[:, 0] = indices[:, 0] * np.repeat(np.arange(current.n), self.tridiag_l.indices.shape[0])
                    temp = tf.sparse.SparseTensor(values=values, indices=indices,dense_shape=[current.n, self.max_state, self.max_state])
                    b = np.concatenate([np.zeros((current.n, 1)),tf.einsum('BNi,Bi->BN', tf.sparse.to_dense(temp), current.v1)], axis=1)[:, :, np.newaxis]
                    S.append(tf.squeeze(tf.linalg.lstsq(tf.sparse.to_dense(tf.sparse.slice(transition,[0,0,0],[current.n,self.max_state+1,self.max_state*2])), b,fast=False)))
                    continue
                elif i == 1:
                    b = np.concatenate([np.zeros((current.n, 1)),current.v1, -current.v1], axis=1)[:, :, np.newaxis]
                elif i == 3:
                    temp = current.mask * current.k_on
                    values = (np.vstack(([np.array(self.tridiag_l.values)] * current.n)) * temp[:, None]).ravel()
                    indices = np.vstack(([self.tridiag_l.indices] * current.n))
                    indices = np.hstack((np.ones((indices.shape[0], 1), dtype=int), indices))
                    indices[:, 0] = indices[:, 0] * np.repeat(np.arange(current.n), self.tridiag_l.indices.shape[0])
                    temp = tf.sparse.SparseTensor(values=values, indices=indices,
                                                  dense_shape=[current.n, self.max_state, self.max_state])

                    b = np.concatenate([np.zeros((current.n, 1)), -tf.einsum('BNi,Bi->BN', tf.sparse.to_dense(temp), current.v1), -tf.einsum('Ni,Bi->BN', tf.sparse.to_dense(self.tridiag_l), current.v2)], axis=1)[:, :, np.newaxis]
                elif i == 4:
                    b = np.concatenate([np.zeros((current.n, 1)), -tf.einsum('Ni,Bi->BN', tf.sparse.to_dense(tf.cast(self.dkd,tf.float64)), current.v1), -tf.einsum('Ni,Bi->BN', tf.sparse.to_dense(tf.cast(self.dkd,tf.float64)), current.v2)], axis=1)[:, :, np.newaxis]
                S.append(tf.squeeze(tf.linalg.lstsq(tf.sparse.to_dense(transition), b,fast=False)))
            S = np.array(S)
            current.S = np.array(tf.experimental.numpy.sum(tf.reshape(S, (S.shape[0], S.shape[1], 4, int(S.shape[-1] / 4))), axis=2))
    """


def eval_p(x0,p,index,value,cell_number=1,percentage=False,downsample=False):
    if index==0:
        parameter=np.array([[0,value,x0[0],x0[1],1],[0,value,x0[0],x0[1],1]])
    elif index==1:
        parameter = np.array([[0, x0[0], value, x0[1], 1], [0, x0[0], value, x0[1], 1]])
    else:
        parameter = np.array([[0, x0[0], x0[1], value, 1], [0, x0[0], x0[1], value, 1]])
    distribution=block_two_state_cme(parameter,percentage=percentage,print_flag=False).batch0.distribution.todense()[0,:]
    #plt.plot(distribution)
    #plt.show()
    if downsample!=1:
        catelogue=generate_binom_catelogue(distribution.shape[0]-1,downsample)
        distribution=distribution@catelogue
    max_m=max(p.shape[0],distribution.shape[0])
    distribution_d=np.zeros(max_m)
    distribution_d[:distribution.shape[0]]=distribution
    p=deepcopy(p)
    p.resize(max_m,refcheck=False)
    #np.clip(p,10**-11,1,out=p)
    np.clip(distribution_d,10**-11,1,out=distribution_d)
    p=p/p.sum()
    distribution_d=distribution_d/distribution_d.sum()
    return -cell_number*np.matmul(np.log(distribution_d),p.T)


def optimize_profile3(likelihood,shape,distribution,cell_number=1,percentage=False,probability=True,downsample=False,cutoff=1.92):
    #p=np.exp(likelihood)
    #p=p/p.sum()
    p=likelihood.reshape(shape)
    ksyn=10**np.linspace(-0.3,2.3,shape[0])
    koff=10**np.linspace(-3,3,shape[1])
    kon=10**np.linspace(-3,3,shape[2])
    parameter=[ksyn,koff,kon]
    profile=dict()
    for axis,name in zip(range(3),['ksyn','koff','kon']):
        if axis==0:
            l1=[0,1,2]
            l2=[0,1,2]
            temp=p
        elif axis==1:
            l1=[1,0,2]
            l2=[1,0,2]
            temp=np.moveaxis(p,[0,1,2],[1,0,2])
        else:
            l1=[2,0,1]
            l2=[1,2,0]
            temp=np.moveaxis(p,[0,1,2],[1,2,0])
        max_like=[]
        #detail=[]
        inference=[]
        bounds = ((parameter[l1[1]][0], parameter[l1[1]][-1]), (parameter[l1[2]][0], parameter[l1[2]][-1]))
        #for i in tqdm(range(temp.shape[0]),total=temp.shape[0]):
        min_LL=temp.min()
        for i in range(temp.shape[0]):
            j, k = np.unravel_index(np.argmin(temp[i, :, :]), shape=temp[i, :, :].shape)
            if probability:
                if temp[i,:,:].min()>(min_LL-np.log(0.9)*cell_number):
                    max_like.append(temp[i,:,:].min())
                    inference.append([parameter[l1[0]][i],parameter[l1[1]][j],parameter[l1[2]][k]])
                    continue
            else:
                if temp[i,:,:].min()>(min_LL+cutoff*cell_number/10):
                    max_like.append(temp[i,:,:].min())
                    inference.append([parameter[l1[0]][i],parameter[l1[1]][j],parameter[l1[2]][k]])
                    continue
            likelihood_min = np.inf

            #if 'res' in locals():
            #    initial=res.x
            #else:
            #    initial=[parameter[l1[1]][j],parameter[l1[2]][k]]
            initial = [parameter[l1[1]][j], parameter[l1[2]][k]]
            res=minimize(eval_p,x0=initial,args=(distribution,axis,parameter[l1[0]][i],cell_number,percentage,float(downsample)),bounds=bounds,method='L-BFGS-B')
            if res.fun<likelihood_min:
                likelihood_min=res.fun
                infer_min=res.x
            #detail.append([parameter[l1[0]][i],likelihood,infer_min])
            max_like.append(likelihood_min)
            max_param=np.concatenate(([parameter[l1[0]][i]],infer_min))
            inference.append(max_param)
            min_LL=min(min(max_like),min_LL)
        inference=np.array(inference)
        inference = inference[:,l2]
        max_like=np.array(max_like)
        #max_like=max_like-max_like.min()
        profile[name]={'max_like':max_like,'parameter':inference}
        del res
    return profile


def optimize_likelihood(initial,distribution,cell_number=1,percentage=False,probability=True,downsample=False,cutoff=1.92):
    def eval_p_3d(initial,distribution,percentage):
        param=np.hstack((np.array([0]),initial,np.array([1])))
        param=np.vstack((param,param))
        temp=block_two_state_cme(param,percentage=percentage).batch0.distribution
        max_m=max(temp.shape[1],distribution.shape[0])
        temp.resize((temp.shape[0],max_m))
        temp=np.clip(temp.todense()[0,:],a_min=10**-11,a_max=1)
        distribution_d=np.zeros(max_m)
        distribution_d[:distribution.shape[0]]=distribution
        return -np.matmul(np.log(temp),distribution_d.T)
    bounds=((10**-0.3,10**2.3),(10**-3,10**3),(10**-3,10**3))
    res=minimize(eval_p_3d,x0=initial,args=(distribution,percentage),bounds=bounds,method='L-BFGS-B')
    return res




def parallel_likelihood(pexp,psim_path,cell_number=100,repeat=20,max_cell_number=50000,shape=[60,60,60],percentage=False,downsample=1.0,probability=False,optimize=True,alpha=0.05,name=False,save_path=False):
    start_time=time()
    try:
        np.random.seed(pexp[1])
        name=pexp[2]
        pexp=pexp[0]
    except:
        np.random.seed(cell_number)
    try:
        cutoff=chi2.ppf(1-alpha,df=1)/2
        if repeat>0:
            sample=np.random.choice(np.arange(pexp.shape[0]),size=max_cell_number,p=pexp)
        psim=shelve.open(psim_path)['downsample_'+downsample]
        max_m = max(pexp.shape[0], psim.shape[1])
        psim.resize((psim.shape[0], max_m))
        psim=np.clip(psim.todense(),a_min=10**-11,a_max=1)
        sample_histogram=np.zeros((repeat+1,max_m))
        sample_histogram[0,:pexp.shape[0]]=pexp
        for i in range(1,repeat+1):
            sample_index=np.random.choice(np.arange(sample.shape[0]),replace=False,size=cell_number)
            sample_histogram[i,:]=np.histogram(sample[sample_index],bins=max_m,range=(0,max_m),density=True)[0]
        if downsample != '1.0':
            catelogue=generate_binom_catelogue(maxcount=sample_histogram.shape[1]+1,p=float(downsample))
            origin_sample_histogram=deepcopy(sample_histogram)
            sample_histogram=sample_histogram@catelogue
        likelihood=-cell_number*np.matmul(np.log(psim),sample_histogram.T).T
        profile_dict={'histogram':sample_histogram,'ksyn':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[0],3))},'koff':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[1],3))},'kon':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[2],3))}}
        if not optimize:
            ksyn = 10 ** np.linspace(-0.3, 2.3, shape[0])
            koff = 10 ** np.linspace(-3, 3, shape[1])
            kon = 10 ** np.linspace(-3, 3, shape[2])
            parameter = [ksyn, koff, kon]
        if repeat ==1:
            start=1
        else:
            start=0
        for i in range(start,repeat+1):
            if optimize:
                profile=optimize_profile3(likelihood[i,:],np.array(shape),distribution=sample_histogram[i,:],percentage=percentage,cell_number=cell_number,probability=probability,downsample=downsample,cutoff=cutoff)
                profile_dict['ksyn']['max_like'][i,:]=profile['ksyn']['max_like']
                profile_dict['ksyn']['parameter'][i, :,:] = profile['ksyn']['parameter']
                profile_dict['koff']['max_like'][i, :] = profile['koff']['max_like']
                profile_dict['koff']['parameter'][i, :,:] = profile['koff']['parameter']
                profile_dict['kon']['max_like'][i, :] = profile['kon']['max_like']
                profile_dict['kon']['parameter'][i, :,:] = profile['kon']['parameter']

            else:
                for axis,name in zip([0,1,2],['ksyn','koff','kon']):
                    if axis == 0:
                        l1 = [0, 1, 2]
                        l2 = [0, 1, 2]
                        temp = np.reshape(likelihood[i,:],shape=shape)
                    elif axis == 1:
                        l1 = [1, 0, 2]
                        l2 = [1, 0, 2]
                        temp = np.moveaxis(np.reshape(likelihood[i,:],shape=shape), [0, 1, 2], [1, 0, 2])
                    else:
                        l1 = [2, 0, 1]
                        l2 = [1, 2, 0]
                        temp = np.moveaxis(np.reshape(likelihood[i,:],shape=shape), [0, 1, 2], [1, 2, 0])
                    for index in tqdm(range(temp.shape[0]), total=temp.shape[0]):
                        j, k = np.unravel_index(np.argmax(temp[index, :, :]), shape=temp[index, :, :].shape)
                        profile_dict[name]['max_like'][i,index]=temp[index,j,k]
                        profile_dict[name]['parameter'][i,index,:]=np.concatenate(parameter[l1[0]][index],parameter[l1[1]][j],parameter[l1[2]][k])[l2]
        print(time()-start_time)
        if name:
            g=shelve.open(save_path,writeback=True)
            g[name]=profile_dict
            return
        return profile_dict
    except:
        return np.nan

"""
def plot(true_parameter,inference):
    #true parameter: array of three values
    #inference object: inference of experiment histogram: list of three dictionaries, [ksyn,koff,kon] respectively, each with two keys: the profile likelihood 2D arrat
    np.set_printoptions(precision=3)
    plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=2)
    plt.plot(pexp[:-1, :plot_m_max].T, alpha=10 / repeat, color='blue', zorder=-2)
    plt.plot(pexp[-1, :plot_m_max].T, alpha=10 / repeat, color='blue', label='sample_replicate')
    plt.plot(psim_d[arg_max[:-1], :plot_m_max].T, color='orange', alpha=10 / repeat, zorder=-1)
    plt.plot(psim_d[arg_max[-1], :plot_m_max].T, color='orange', alpha=10 / repeat, label='inference')
    if capture_rate:
        plt.plot(exp_data['histogram'][i, :plot_m_max], color='green', marker='o', markersize=3,
                 label='downsample_groundtruth', zorder=10)
        plt.plot(exp_data['origin_histogram'][i, :plot_m_max], color='black', marker='v', markersize=3,
                 label='undownsample_groundtruth', zorder=10)
    else:
        plt.plot(exp_data['histogram'][i, :plot_m_max], color='green', marker='o', markersize=5, label='groundtruth')
    plt.title('Distribution')
    if capture_rate:
        plt.plot(psim_origin[arg_max[:-1], :plot_m_max].T, color='red', alpha=10 / repeat, zorder=1)
        plt.plot(psim_origin[arg_max[-1], :plot_m_max].T, color='red', alpha=10 / repeat,
                 label='undownsample inference')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.linspace(0, plot_m_max, 10, dtype='int'))
    plt.xlabel('mRNA copy number')
    plt.ylabel('probability')
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.subplot2grid((3, 6), (0, 3), colspan=3, rowspan=2)
    burst_size = parameter[:, 1] / parameter[:, 2]
    burst_frequency = parameter[:, 3]
    burst_lib = list(set(burst_size))
    frequency_lib = list(set(burst_frequency))
    frequency_lib.sort()
    burst_lib.sort()
    burst_dict = {}
    frequency_dict = {}
    burst_likelihood = []
    for burst in burst_lib:
        burst_dict[burst] = set(np.where(burst_size == burst)[0])
    for frequency in frequency_lib:
        frequency_dict[frequency] = set(np.where(burst_frequency == frequency)[0])
    temp[0, :] = temp[0, :] - temp[0, :].max()
    for burst in burst_dict.keys():
        for frequency in frequency_dict.keys():
            index = list(burst_dict[burst].intersection(frequency_dict[frequency]))
            burst_likelihood.append([burst, frequency, np.nanmax(temp[0, index])])
    burst_likelihood = np.array(burst_likelihood)
    plt.scatter(np.log10(burst_likelihood[:, 0]), np.log10(burst_likelihood[:, 1]), c=burst_likelihood[:, 2],
                cmap='cool', norm=norm, zorder=-1)
    plt.colorbar(label='log_likelihood')
    burst_size = parameter[arg_max, 1] / parameter[arg_max, 2]
    burst_frequency = np.log10(parameter[arg_max, 3])
    plt.scatter(np.log10(burst_size), burst_frequency, alpha=20 / repeat, color='darkorange', zorder=0)
    plt.scatter(np.log10(param[i, 1] / param[i, 2]), np.log10(param[i, 3]), color='green', marker='o', s=20, zorder=1)
    plt.xlabel('burst size in log10')
    plt.xlim(-3.3, 5.3)
    plt.ylim(-3, 3)
    plt.ylabel('burst frequency k_on in log10')
    plt.title('Log Likelihood Surface of Ground Truth')
    plt.subplot2grid((3, 6), (2, 0), colspan=2, rowspan=1)
    temp_max = np.nanmax(likelihood[0:1, :], axis=(1, 2))
    temp_max = temp_max - temp_max.max(axis=1)[:, None]
    CI_koff = (find_CI(np.log10(unbinding), temp_max[0, :]), temp_max[0, :])
    plt.fill_between(np.log10(unbinding), temp_max[0, :], -5, where=temp_max[0, :] > -1.92, color='violet', zorder=-1,
                     alpha=0.5)
    # plt.plot(np.log10(unbinding), temp_max[1:,:].T,alpha=0.02,color='blue')
    plt.plot(np.log10(unbinding), temp_max[0, :], color='green', zorder=1)
    for j in range(repeat):
        plt.axvline(np.log10(parameter[arg_max[j], 2]), color='darkorange', alpha=10 / repeat, zorder=0)
    plt.axvline(np.log10(param[i, 2]), color='green')
    plt.xlabel('Koff in log10')
    plt.ylabel('log_likelihood')
    plt.ylim(-5, 0)
    np.log10(parameter[arg_max, 2])
    plt.subplot2grid((3, 6), (2, 2), colspan=2, rowspan=1)
    temp_max = np.nanmax(likelihood[0:1, :], axis=(1, 3))
    temp_max = temp_max - temp_max.max(axis=1)[:, None]
    CI_kon = (find_CI(np.log10(binding), temp_max[0, :]), temp_max[0, :])
    plt.fill_between(np.log10(binding), temp_max[0, :], -5, where=temp_max[0, :] > -1.92, color='violet', zorder=-1,
                     alpha=0.5)
    # plt.plot(np.log10(binding), temp_max[1:,:].T,alpha=0.02,color='blue')
    plt.plot(np.log10(binding), temp_max[0, :], color='green', zorder=1)
    for j in range(repeat):
        plt.axvline(np.log10(parameter[arg_max[j], 3]), color='darkorange', alpha=10 / repeat, zorder=0)
    plt.axvline(np.log10(param[i, 3]), color='green')
    plt.xlabel('Kon in log10')
    plt.title('Profile Likelihood')
    plt.ylim(-5, 0)
    plt.subplot2grid((3, 6), (2, 4), colspan=2, rowspan=1)
    temp_max = np.nanmax(likelihood[0:1, :], axis=(2, 3))
    temp_max = temp_max - temp_max.max(axis=1)[:, None]
    CI_ksyn = (find_CI(np.log10(k_on), temp_max[0, :]), temp_max[0, :])
    plt.fill_between(np.log10(k_on), temp_max[0, :], -5, where=temp_max[0, :] > -1.92, color='violet', zorder=-1,
                     label='95%CI', alpha=0.5)
    # plt.plot(np.log10(k_on), temp_max[1:-1,:].T,alpha=0.02,color='blue')
    # plt.plot(np.log10(k_on), temp_max[-1, :].T, alpha=0.02, color='blue',label='sample_replicate')
    plt.plot(np.log10(k_on), temp_max[0, :], color='green', zorder=1)
    for j in range(repeat):
        plt.axvline(np.log10(parameter[arg_max[j], 1]), color='darkorange', alpha=10 / repeat, zorder=0)
    plt.axvline(np.log10(param[i, 1]), color='green', zorder=1)
    plt.xlabel('Ksyn in log10')
    plt.ylim(-5, 0)
    plt.legend()
    fig.suptitle('ground truth:' + str(param[i, 1:-1]))
    fig.tight_layout()
    arg_max_lib[cell_number][i] = [CI_koff, CI_kon, CI_ksyn, arg_max]
    save_path2 = save_path + str(cell_number) + '_cells/'
    if not os.path.isdir(save_path2):
        os.mkdir(save_path2)
    if capture_rate:
        plt.savefig(save_path2 + str(i) + 'png')
    else:
        plt.savefig(save_path2 + str(i) + 'png')
    # plt.show()
"""

if __name__=='__main__':
    from multiprocessing import Pool, cpu_count
    from functools import partial
    distribution=block_two_state_cme(np.array([[0,3.5,23,10,1],[0,3.44578,23.598,9.249,1],[0,3.44578,23,10,1],[0,3.44578,23.598,9.249,1]]),percentage=False).batch0.distribution.todense()[0,:]
    psim = shelve.open('one_gene_two_state_library_grid')['downsample_1']
    profile=parallel_likelihood((distribution,1),psim=psim,cell_number=1000,repeat=0,max_cell_number=50000,shape=[60,60,60],percentage=False,optimize=True,probability=False)

    percentage = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    kd = 1 / (1 / percentage - 1)  # 1/(k_on/(k_on+k_off))-1=k_off/k_on  = G/G*
    unbinding = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ksyn = np.array([3.5, 10, 20])
    parameter = np.zeros((ksyn.shape[0] * unbinding.shape[0] * percentage.shape[0], 5))
    parameter[:, 1] = np.repeat(ksyn, repeats=unbinding.shape[0] * percentage.shape[0])
    parameter[:, 2] = np.tile(unbinding, reps=ksyn.shape[0] * percentage.shape[0])
    parameter[:, 3] = np.tile(np.repeat(percentage, repeats=unbinding.shape[0]), reps=ksyn.shape[0])
    parameter[:, 4] = 1
    parameter=parameter[(parameter[:,2]<10**3) & (parameter[:,2]>10**-3)]
    exp_histogram = np.clip(block_two_state_cme(parameter).batch0.distribution.todense(), a_min=10 ** -11, a_max=1)
    exp_histogram = exp_histogram / exp_histogram.sum(axis=1)[:, None]
    cell_number = np.array([100, 350,700, 1000,3500, 7000])[::-1]
    max_cell_number = 50000
    repeat = 300
    repeat=30
    capture_rate = [1, 0.3, 0.15]
    exp_data = {}
   # exp_data['histogram'] = exp_histogram[[1,3,8,10,13,15,22,25,27,45,50,52,57,63,69,75,81,87,92,103,110,117,128,151],:]
    exp_data['histogram']=exp_histogram
    f=shelve.open('one_gene_two_state_library_grid')
    param=f['parameter']
    f.close()
    test=np.tile(exp_data['histogram'][27,:],repeat).reshape((repeat,exp_data['histogram'].shape[1]))
    for capture in capture_rate:
        psim=shelve.open('one_gene_two_state_library_grid')['downsample_'+str(capture)]
        for cell in cell_number:
            #parallel_likelihood((test[0,:],1),psim=psim,cell_number=10000,repeat=1,max_cell_number=max_cell_number,shape=[60,60,60],percentage=False,optimize=True,probability=False)
            pool=Pool(cpu_count())
            with pool:
                #result=list(tqdm(pool.imap(partial(parallel_likelihood,psim=psim,cell_number=cell,repeat=repeat,max_cell_number=max_cell_number,shape=[60,60,60],percentage=False,optimize=True,probability=False),exp_data['histogram']),total=exp_data['histogram'].shape[0]))
                result = list(tqdm(pool.imap(partial(parallel_likelihood, psim=psim, cell_number=1000, repeat=1,
                                                     max_cell_number=max_cell_number, shape=[60, 60, 60],
                                                     percentage=False, optimize=True, probability=False,downsample=str(capture)),
                                             zip(test,np.arange(30))), total=test.shape[0]))
            h=shelve.open('synthetic_data_inference',writeback=True)
            h['downsample_' + str(capture)]=dict()
            h['downsample_'+str(capture)][str(cell)+'cells']=result
            h.close()
            pool.close()
