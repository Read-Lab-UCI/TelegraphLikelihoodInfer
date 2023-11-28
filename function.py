from time import time,sleep
import psutil,shelve,os,subprocess,sys
try:
    from scipy.sparse import diags
except:
    subprocess.check_call([sys.executable,'-m','pip','install','scipy'])
    from scipy.sparse import diags
from scipy.stats import binom,chi2
from scipy.optimize import minimize
from scipy.sparse import coo_array,csr_array,lil_array
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow==2.10.0'])
    import tensorflow as tf
from copy import deepcopy
import numpy as np
from tqdm import tqdm
#from scipy.interpolate import interp1d
#from scipy.interpolate import RegularGridInterpolator
#import matplotlib.pyplot as plt
#from matplotlib import colors
from pytorch_test import block_two_state_cme

class init:
    def __init__(self):

        pass

def compute_cme_one_gene_two_state(parameters, parameter_diction={}, shape=(), percentage=False, testing=False):
    if type(parameters) == int:
        index = np.unravel_index(parameters, shape=shape)
        parameters = []
        for i in range(len(index)):
            parameters.append(parameter_diction[i][0][index[i]])
    multiplier = parameters[1]
    regulator_unbind = parameters[2]
    transcription_off = parameters[0]
    mdegrade_rate = parameters[4]

    if percentage:
        p_on = parameters[3]
        regulator_bind = regulator_unbind * p_on / (1 - p_on)
    else:
        regulator_bind = parameters[3]
        p_on = regulator_bind / (regulator_bind + regulator_unbind)
    p_off = 1 - p_on
    #print(regulator_bind)
    if transcription_off < 1:
        transcription_on = multiplier
    else:
        transcription_on = multiplier * transcription_off
    tmp = max(transcription_on, transcription_off)
    #if tmp > 300 or tmp < 1:
    #    return csr_array(np.array([-np.inf]))
    if multiplier > 1 or transcription_off==0:
        mrna_max = transcription_on / mdegrade_rate
    else:
        mrna_max = transcription_off / mdegrade_rate
    mrna_max = int(mrna_max)
    if mrna_max < 20:
        mrna_max = 48
    else:
        mrna_max = mrna_max * 2

    # print('mrna max:',mrna_max)
    index = np.linspace(0, mrna_max, mrna_max + 1)
    one = np.ones(mrna_max + 1)
    one[-1] = 0
    transition = lil_array(((mrna_max + 1) * 2, (mrna_max + 1) * 2))
    transition_on = diags([-regulator_unbind * np.ones(
        mrna_max + 1) - transcription_on * one - mdegrade_rate * np.linspace(0, mrna_max, num=mrna_max + 1),
                           mdegrade_rate * np.linspace(1, mrna_max, num=mrna_max),
                           transcription_on *one], (0, 1, -1)).tolil()
    transition_off = diags([-regulator_bind * np.ones(
        mrna_max + 1) - transcription_off * one - mdegrade_rate * np.linspace(0, mrna_max, num=mrna_max + 1),
                            mdegrade_rate * np.linspace(1, mrna_max, num=mrna_max),
                            transcription_off*one], (0, 1, -1)).tolil()
    transition[:mrna_max + 1, :mrna_max + 1] = transition_on
    transition[mrna_max + 1:, mrna_max + 1:] = transition_off
    transition[:mrna_max + 1, mrna_max + 1:] = diags(regulator_bind * np.ones(mrna_max + 1)).tolil()
    transition[mrna_max + 1:, :mrna_max + 1] = diags(regulator_unbind * np.ones(mrna_max + 1)).tolil()
    try:
        v = np.linalg.svd(transition.toarray())[-1][-1]
        #sigma,v=eigs(transition,k=1,sigma=0,OPpart='r')
        v=np.abs(np.real(v))
        #test=block_two_state_cme(np.array([parameters,parameters]),keep_transition=True)
        v = v / np.sum(v)
        #v_test=np.linalg.eig(transition.toarray())[-1]
        if testing:
            return v.T
        else:
            v = np.sum(v.reshape(2, int(v.ravel().shape[0] / 2)), axis=0)
            v[v < 10**-12] = 0
            v = v/np.sum(v)
            return csr_array(v.ravel())
    except:
        print('null space error', parameters)
        return csr_array(np.array([-np.inf]))

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

"""
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
        if keep_transition:
            factor=2
        else:
            factor=1

        start = 0
        end = 0
        while end < param.shape[0]:
            t1=time()
            memory = psutil.virtual_memory().available
            batch_size = int(memory * 0.45 / 16 / self.max_state / self.max_state/factor)
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
            one=np.ones(self.max_state)
            one[-1]=0
            self.dkd = sparse.COO.from_scipy_sparse(diags(np.arange(1, self.max_state), offsets=1) - diags(
                np.arange(self.max_state)))
            dA_dkd = self.dkd * current.kd[:, np.newaxis, np.newaxis]
            self.tridiag_l = sparse.COO.from_scipy_sparse(diags(np.ones(self.max_state - 1), offsets=-1) - diags(
                one))
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


def eval_p(x0,p,index,value,cell_number=100000,percentage=False,downsample=False):
    if index==0:
        parameter=np.array([[0,value,x0[0],x0[1],1],[0,value,x0[0],x0[1],1]])
    elif index==1:
        parameter = np.array([[0, x0[0], value, x0[1], 1], [0, x0[0], value, x0[1], 1]])
    else:
        parameter = np.array([[0, x0[0], x0[1], value, 1], [0, x0[0], x0[1], value, 1]])
    #distribution=block_two_state_cme(parameter,percentage=percentage,print_flag=False).batch0.distribution.todense()[0,:]
    distribution=compute_cme_one_gene_two_state(parameter[0,:]).todense().squeeze()
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


def optimize_profile3(likelihood,shape,distribution,cell_number=100000,percentage=False,probability=True,downsample=False,cutoff=1.92):
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
            start=time()
            j, k = np.unravel_index(np.argmin(temp[i, :, :]), shape=temp[i, :, :].shape)
            #if probability:
            #    if temp[i,:,:].min()>(min_LL-np.log(0.9)*cell_number):
            #        max_like.append(temp[i,:,:].min())
            #        inference.append([parameter[l1[0]][i],parameter[l1[1]][j],parameter[l1[2]][k]])
            #        continue
            #else:
            #    if temp[i,:,:].min()>(min_LL+cutoff*cell_number/2):
            #        max_like.append(temp[i,:,:].min())
            #        inference.append([parameter[l1[0]][i],parameter[l1[1]][j],parameter[l1[2]][k]])
            #        continue
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
            print(i,time()-start)
        inference=np.array(inference)
        inference = inference[:,l2]
        max_like=np.array(max_like)/cell_number
        #max_like=max_like-max_like.min()
        profile[name]={'max_like':max_like,'parameter':inference}
        del res
    return profile


def optimize_likelihood(initial,distribution,cell_number=100000,percentage=False,probability=True,downsample=False,cutoff=1.92):
    def eval_p_3d(initial,distribution,percentage):
        param=np.hstack((np.array([0]),initial,np.array([1])))
        param=np.vstack((param,param))
        temp=block_two_state_cme(param,percentage=percentage).batch0.distribution
        max_m=max(temp.shape[1],distribution.shape[0])
        temp.resize((temp.shape[0],max_m))
        temp=np.clip(temp.todense()[0,:],a_min=10**-11,a_max=1)
        distribution_d=np.zeros(max_m)
        distribution_d[:distribution.shape[0]]=distribution
        return -np.matmul(np.log(temp),distribution_d.T)*cell_number
    bounds=((10**-0.3,10**2.3),(10**-3,10**3),(10**-3,10**3))
    res=minimize(eval_p_3d,x0=initial,args=(distribution,percentage),bounds=bounds,method='L-BFGS-B')
    return res

def parallel_likelihood(pexp_list,psim_path,repeat=0,max_cell_number=50000,shape=[60,60,60],percentage=False,downsample='1.0',probability=False,optimize=True,alpha=0.05,save_path=False,self_infer=False,coarse_grain=5):
    start_time=time()
    np.random.seed(pexp_list[1])
    save_name=str(pexp_list[2])
    pexp=pexp_list[0].ravel()
    cell=pexp_list[1]
    try:
        cutoff = chi2.ppf(1-alpha, df=1)/2
        if repeat>0:
            sample=np.random.choice(np.arange(pexp.shape[0]),size=max_cell_number,p=pexp)
        g=shelve.open(psim_path,'r')
        psim=g['downsample_'+str(downsample)]
        parameter=g['parameter']
        g.close()
        ksyn=parameter[:,1].reshape(60,60,60)[:,0,0]
        koff=parameter[:,2].reshape(60,60,60)[0,:,0]
        kon =parameter[:,3].reshape(60,60,60)[0,0,:]
        parameter=[ksyn,koff,kon]
        print('successful loaded library parallel')
        max_m = max(pexp.shape[0], psim.shape[1])
        psim.resize((psim.shape[0], max_m))
        psim=np.clip(psim.todense(),a_min=10**-11,a_max=1)
        sample_histogram=np.zeros((repeat+1,max_m))
        sample_histogram[0,:pexp.shape[0]]=pexp
        for i in range(1,repeat+1):
            sample_index=np.random.choice(np.arange(sample.shape[0]),replace=False,size=cell)
            sample_histogram[i,:]=np.histogram(sample[sample_index],bins=max_m,range=(0,max_m),density=True)[0]
        if downsample != '1.0':
            catelogue=generate_binom_catelogue(maxcount=sample_histogram.shape[1]-1,p=float(downsample))
            #origin_sample_histogram=deepcopy(sample_histogram)
            sample_histogram=sample_histogram@catelogue
        likelihood=-cell*np.matmul(np.log(psim),sample_histogram.T).T
        if self_infer:
            n_cell_number=3
        else:
            n_cell_number=1
        profile_dict={'cell_number':cell,'histogram':sample_histogram,'MLE':np.zeros((repeat+2,3)),'CI':np.zeros((repeat+1,n_cell_number,3,5)),'ksyn':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[0],3))},'koff':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[1],3))},'kon':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[2],3))}}
        key_dict = {0: 'ksyn', 1: 'koff', 2: 'kon'}
        start=time()
        for i in range(repeat+1):
            tmp=likelihood[i,:].reshape(shape)
            for j in key_dict.keys():
                if j==1:
                    tmp=np.moveaxis(tmp,[0,1,2],[1,0,2])
                    key=[0,2]
                elif j==2:
                    tmp=np.moveaxis(tmp,[0,1,2],[1,2,0])
                    key=[0,1]
                else:
                    key=[1,2]
                for k in range(tmp.shape[0]):
                    min_index=np.unravel_index(tmp[k,:,:].argmin(),shape=(shape[key[0]],shape[key[1]]))
                    profile_dict[key_dict[j]]['parameter'][i,k,j]=parameter[j][k]
                    profile_dict[key_dict[j]]['parameter'][i,k, key[0]] = parameter[key[0]][min_index[0]]
                    profile_dict[key_dict[j]]['parameter'][i,k, key[1]] = parameter[key[1]][min_index[1]]
                    profile_dict[key_dict[j]]['max_like'][i,k]=tmp[k,min_index[0],min_index[1]]
        print('coarse grain profile in {}s'.format(time()-start))
        #profile_dict['ksyn']['max_like']=likelihood.reshape((repeat+1,shape[0],shape[1],shape[2])).min(axis=(2,3))/cell
        #profile_dict['koff']['max_like'] = likelihood.reshape((repeat + 1, shape[0], shape[1], shape[2])).min(axis=(1, 3))/cell
        #profile_dict['kon']['max_like'] = likelihood.reshape((repeat + 1, shape[0], shape[1], shape[2])).min(axis=(1, 2))/cell
        start=time()
        profile_dict['MLE']=find_MLE(profile_dict)
        if self_infer:
            profile_dict['CI'][:, 0, :, :] = find_CI(profile_dict, cutoff+coarse_grain+3, cell=1000)
            profile_dict['CI'][:, 1, :, :] = find_CI(profile_dict, cutoff+coarse_grain+3, cell=10000)
            profile_dict['CI'][:, 2, :, :] = find_CI(profile_dict, cutoff+coarse_grain+3, cell=100000)
        else:
            profile_dict['CI'][:, 0, :, :] = find_CI(profile_dict, cutoff+coarse_grain+3, cell=profile_dict['cell_number'])
        print('coarse grain CI in {}s'.format(time() - start))
        for i in tqdm(range(repeat + 1)):
            minimum_like=likelihood[i,:].min()
            tmp = likelihood[i, :] - minimum_like
            index_to_search = np.nonzero(tmp < cutoff + coarse_grain)
            index_to_search = np.unravel_index(index_to_search, shape=shape)
            for name in key_dict.keys():
                #plt.plot(profile_dict[key_dict[name]]['max_like'][i,:]-minimum_like,label='coarse grain',marker='o')
                for cell_index in range(n_cell_number):
                    if np.log10(profile_dict['CI'][i,cell_index,name,3])>4:
                        continue
                    index=set(index_to_search[name].ravel())
                    index_larger=set(np.where(profile_dict[key_dict[name]]['max_like'][i,:]-minimum_like>cutoff+3)[0])
                    index=index.intersection(index_larger)
                    for j in index:
                        if profile_dict[key_dict[name]]['max_like'][i,j]-minimum_like<cutoff+3:
                            continue
                        bounds = [[], []]
                        grid_index=np.nonzero(index_to_search[name].flatten()==j)
                        other_keys=list(key_dict.keys())
                        other_keys.remove(name)
                        #other_keys=np.array(other_keys)
                        bounds[0]=(parameter[other_keys[0]][index_to_search[other_keys[0]].ravel()[grid_index].min()],parameter[other_keys[0]][index_to_search[other_keys[0]].ravel()[grid_index].max()])
                        bounds[1]=(parameter[other_keys[1]][index_to_search[other_keys[1]].ravel()[grid_index].min()],parameter[other_keys[1]][index_to_search[other_keys[1]].ravel()[grid_index].max()])
                        tmp=tmp.reshape(60,60,60)
                        if name==0:
                            min_index=np.unravel_index(tmp[j,:,:].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                        elif name==1:
                            min_index=np.unravel_index(tmp[:,j,:].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                        else:
                            min_index=np.unravel_index(tmp[:,:,j].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                        initial=[parameter[other_keys[0]][min_index[0]],parameter[other_keys[1]][min_index[1]]]
                        res = minimize(eval_p, x0=initial, args=(sample_histogram[i,:], name, parameter[name][j], cell, percentage, float(downsample)), bounds=bounds,method='L-BFGS-B')
                        profile_dict[key_dict[name]]['max_like'][i,j]=res.fun
                        profile_dict[key_dict[name]]['parameter'][i,j,other_keys]=res.x
                #plt.plot(profile_dict[key_dict[name]]['max_like'][i, :] - profile_dict[key_dict[name]]['max_like'][i, :].min(),label='optimized')
                #plt.ylim(0,2.2)
                #plt.legend()
                #plt.show()
        profile_dict['MLE']=find_MLE(profile_dict)
        if self_infer:
            profile_dict['CI'][:, 0, :, :] = find_CI(profile_dict, cutoff, cell=1000)
            profile_dict['CI'][:, 1, :, :] = find_CI(profile_dict, cutoff, cell=10000)
            profile_dict['CI'][:, 2, :, :] = find_CI(profile_dict, cutoff, cell=100000)
        else:
            profile_dict['CI'][:, 0, :, :] = find_CI(profile_dict, cutoff, cell=profile_dict['cell_number'])

        print(time()-start_time)
        if save_path:
            while True:
                try:
                    g=shelve.open(save_path+'_'+save_name,writeback=True)
                    g[save_name]=profile_dict
                    g.close()
                    print('success saved output')
                    break
                except:
                    sleep(10)
            return
        return profile_dict
    except:
        print('error in between')
        return np.nan

def find_MLE(profile):
    param=np.zeros((profile['histogram'].shape[0]+1,3))
    for j in range(param.shape[0]-1):
        param_temp=[]
        mle = []
        for i in ['ksyn','koff','kon']:
            index=np.argmin(profile[i]['max_like'][j])
            param_temp.append(profile[i]['parameter'][j][index,:])
            mle.append(profile[i]['max_like'][j][index])
        index=np.argmin(mle)
        param[j,:]=param_temp[index]
    return param

def find_cell_CI(profile_dict, cutoff=1.92,error_width=0.3):
    cell_array=[]
    for i in range(profile_dict['histogram'].shape[0]):
        cell_temp = []
        for index, j in enumerate(['ksyn', 'koff', 'kon']):
            tmp = profile_dict[j]['max_like'][i,:].ravel()
            tmp = tmp - tmp.min()
            tmp_inter = interp1d(profile_dict[j]['parameter'][i, :, index], tmp, kind='linear')
            min_index = tmp.argmin()
            if min_index == 0:
                lower_bound = 10 ** -0.3
                upper_bound = lower_bound * (1+error_width)
                cell = int(np.ceil(cutoff / tmp_inter(upper_bound)))
            elif min_index == tmp.shape[0] - 1:
                upper_bound = 10 ** 2.3
                lower_bound = 10 ** 2.3 *(1-error_width)
                cell = int(np.ceil(1.92 / tmp_inter(lower_bound)))
            else:
                max_para = profile_dict[j]['parameter'][i, min_index, index]
                try:
                    lower_bound = tmp_inter(max_para * (1-error_width))
                except:
                    lower_bound =tmp_inter(profile_dict[j]['parameter'][i, 0, index])
                try:
                    upper_bound = tmp_inter(max_para * (1+error_width))
                except:
                    upper_bound = tmp_inter(profile_dict[j]['parameter'][i, -1, index])
                cell = int(np.ceil(1.92 / min(lower_bound,upper_bound)))
            cell_temp.append(cell)
        cell_array.append(cell_temp)
    profile_dict['cell_number with {} uncertainty'.format(error_width)]=cell_array
    return profile_dict

def find_CI(profile_dict,cutoff=1.92,cell=0,normalize=True,infer_factor=3):
    width_array=[]
    for i in range(profile_dict['histogram'].shape[0]):
        temp_width=[]
        for index,j in enumerate(['ksyn','koff','kon']):
            tmp=profile_dict[j]['max_like'][i,:].ravel()
            tmp=tmp-tmp.min()
            tmp=tmp*cell
            min_index=tmp.argmin()
            #if min_index==0:
            #    upper_int = interp1d(tmp[min_index:], profile_dict[j]['parameter'][i, min_index:, index])
            #    lower_bound=np.log10( profile_dict[j]['parameter'][i, min_index, index])
            #elif min_index==tmp.shape[0]-1:
            #    lower_int=interp1d(tmp[:min_index+1],profile_dict[j]['parameter'][i,:min_index+1,index])
            #    upper_bound=np.log10( profile_dict[j]['parameter'][i, min_index, index])
            #else:
            #    lower_int=interp1d(tmp[:min_index+1],profile_dict[j]['parameter'][i,:min_index+1,index])
            #    upper_int=interp1d(tmp[min_index:],profile_dict[j]['parameter'][i,min_index:,index])
            if min_index==0:
                lower_bound=profile_dict[j]['parameter'][i,0,index]
            else:
                lower_int=interp1d(profile_dict[j]['parameter'][i,:min_index+1,index],tmp[:min_index+1])
                param=np.linspace(profile_dict[j]['parameter'][i,0,index],profile_dict[j]['parameter'][i,min_index,index],profile_dict[j]['parameter'][i,:min_index,index].shape[0]*infer_factor+1)
                lower_profile=lower_int(param)
                lower_bound=np.where(lower_profile<cutoff)[0]
                if lower_bound.shape[0]==1:
                    lower_bound=(profile_dict[j]['parameter'][i,min_index,index]-profile_dict[j]['parameter'][i,min_index-1,index])/(profile_dict[j]['max_like'][i,min_index]-profile_dict[j]['max_like'][i,min_index-1])*cutoff+profile_dict[j]['parameter'][i,min_index-1,index]
                else:
                    lower_bound=param[lower_bound[0]]
            if min_index==profile_dict[j]['parameter'][i,:,index].shape[0]-1:
                upper_bound=profile_dict[j]['parameter'][i,-1,index]
            else:
                upper_int=interp1d(profile_dict[j]['parameter'][i,min_index:,index],tmp[min_index:])
                param=np.linspace(profile_dict[j]['parameter'][i,min_index,index],profile_dict[j]['parameter'][i,-1,index],profile_dict[j]['parameter'][i,min_index:,index].shape[0]*infer_factor+1)
                upper_profile=upper_int(param)
                upper_bound=np.where(upper_profile<cutoff)[0]
                if upper_bound.shape[0]==1:
                    upper_bound=(profile_dict[j]['parameter'][i,min_index+1,index]-profile_dict[j]['parameter'][i,min_index,index])/(profile_dict[j]['max_like'][i,min_index+1]-profile_dict[j]['max_like'][i,min_index])*cutoff+profile_dict[j]['parameter'][i,min_index,index]
                else:
                    upper_bound=param[upper_bound[-1]]
            #try:
            #    lower_int = interp1d(tmp[:min_index + 1], profile_dict[j]['parameter'][i, :min_index + 1, index])
            #    lower_bound=lower_int(cutoff)
            #except:
            #    lower_bound=profile_dict[j]['parameter'][i, 0, index]
            #try:
            #    upper_int = interp1d(tmp[min_index:], profile_dict[j]['parameter'][i, min_index:, index])
            #    upper_bound=upper_int(cutoff)
            #except:
            #    upper_bound = profile_dict[j]['parameter'][i, -1, index]
            if upper_bound-lower_bound<0.001:
                pass
            width=max(upper_bound-lower_bound,0.001)
            temp_width.append([lower_bound,upper_bound,width,width/profile_dict['MLE'][i,index],width/(profile_dict[j]['parameter'][i,-1,index]-profile_dict[j]['parameter'][i,0,index])])
        width_array.append(temp_width)
    width_array=np.array(width_array)
    return width_array



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
    param=np.array([[0,3.5,23,10,1],[0,3.5,0.23,10,1],[0,3.5,0.23,0.1,1],[0,3.5,0.23,0.01,1],[0,3.5,23,100,1],[0,3.5,2.3,100,1],[0,3.5,2.3,10,1],[0,3.5,2.3,1,1],[0,10,23,10,1],[0,10,0.23,10,1],[0,10,0.23,0.1,1],[0,10,0.23,0.01,1],[0,10,23,100,1],[0,10,2.3,100,1],[0,10,2.3,10,1],[0,10,2.3,1,1]])
    distribution=block_two_state_cme(param,percentage=False).batch0.distribution.todense()
    for i in range(16):#distribution.shape[0]):
        parallel_likelihood([distribution[i,:],100000,'test'+str(i)],psim_path='one_gene_two_state_library_grid',repeat=0,shape=[60,60,60],save_path='test')

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
