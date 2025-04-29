from time import time,sleep
import traceback
import psutil,shelve,os,subprocess,sys
try:
    from scipy.sparse import diags,vstack
except:
    subprocess.check_call([sys.executable,'-m','pip','install','scipy'])
    from scipy.sparse import diags
from scipy.stats import binom,chi2
from scipy.optimize import minimize
from scipy.sparse import coo_array,csr_array,lil_array
import torch
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

from copy import deepcopy
import numpy as np
from tqdm import tqdm

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

    if transcription_off < 1:
        transcription_on = multiplier
    else:
        transcription_on = multiplier * transcription_off
    tmp = max(transcription_on, transcription_off)

    if multiplier > 1 or transcription_off==0:
        mrna_max = transcription_on / mdegrade_rate
    else:
        mrna_max = transcription_off / mdegrade_rate
    mrna_max = max(mrna_max,25)
    mrna_max=int(mrna_max+5*mrna_max**0.5)


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


class block_two_state_cme:
    def __init__(self, param, save_path=None, percentage=False, keep_transition=False, sense=False, device='cpu',
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
        self.max_state=int(self.max_state+4*self.max_state**0.5)
        if self.max_state < 25:
            self.max_state = 25
        #self.max_state = np.log10(self.max_state)
        #self.max_state =np.ceil(10**(self.max_state+0.4/self.max_state)).astype('int')
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
            batch_size = int(memory * 0.45 / 32 / self.max_state / self.max_state/factor)
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
        torch.device(self.device)
        try:
            self.dkd = sparse.COO.from_scipy_sparse(diags(np.arange(1, self.max_state), offsets=1) - diags(
                np.arange(self.max_state)))
            dA_dkd = self.dkd * current.kd[:, np.newaxis, np.newaxis]
            one=np.ones(self.max_state)
            one[-1]=0
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
                    current.transition.indices = transition[:, 1:, :].coords
                    current.transition.shape = transition[:, 1:, :].shape
                transition=torch.sparse_coo_tensor(transition.coords, transition.data, (transition.shape))
            A = torch.sparse_coo_tensor( A.coords,A.data, A.shape)
            B = torch.sparse_coo_tensor(B.coords, B.data, B.shape)
            I_hf = (current.h*current.f)[:,np.newaxis,np.newaxis] * identity
            I_hf = torch.sparse_coo_tensor(I_hf.coords,-I_hf.data, I_hf.shape)
            testing=torch.add(torch.bmm(B, A.to_dense()), I_hf)
            if self.print_flag:
                print('{} time for constructing matrix'.format(time() - start))

            start=time()
            current.va=torch.abs(torch.linalg.svd(testing)[-1][:,-1,:])
            if self.print_flag:
                print(time()-start,'time for solving svd in tensor')
            current.va = np.array(current.va/(torch.sum(current.va, axis=1)/current.p_on)[:, None])
            current.v0 = torch.squeeze(torch.bmm(A,torch.from_numpy((current.va/current.h[:,None])[:,:,None])))
            current.v0 = np.array(current.v0/(torch.sum(current.v0, axis=1)/(1-current.p_on))[:, None])
            current.error = current.va[:, -1]*current.k_on+current.v0[:, -1]*current.k_off
            current.va[current.va < 10 ** -12] = 0
            current.va = current.va / (np.sum(current.va, axis=1)/current.p_on)[:, None]
            current.v0[current.v0 < 10 ** -12] = 0
            current.v0 = current.v0 / (np.sum(current.v0, axis=1)/(1-current.p_on))[:, None]
            current.distribution = current.va+current.v0
            current.distribution[current.distribution<10**-12]=0
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
                for i, j in enumerate(['k_off','k_on', 'f', 'h', 'kd']):
                    if i == 3 and self.percentage:
                        temp = (current.f /(1-current.p_on)**2)
                        b = torch.concat([torch.zeros((current.n, 1)), torch.from_numpy((temp[:, np.newaxis] * current.v0).todense()), torch.from_numpy((-temp[:, np.newaxis] * current.v0).todense())], axis=1)[:, :, np.newaxis]
                    elif i == 3 and not self.percentage:
                        b =torch.tensor(current.h)[:,None,None]*torch.concat([torch.zeros((current.n, 1)), torch.from_numpy(current.v0.todense()), torch.from_numpy(-current.v0.todense())],axis=1)[:, :, np.newaxis]
                    # elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   -tf.einsum('BNi,Bi->BN', ((current.mask * current.k_off + (1 - current.mask)) * current.mask_a)[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.vab),
                    #                   -tf.einsum('BNi,Bi->BN',(current.mask * current.k_off + (1 - current.mask))[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.va0)], axis=1)[:, :, np.newaxis]
                    #elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   tf.einsum('BNi,Bi->BN',(k_off*mask+(1-mask))[:, np.newaxis, np.newaxis] * self.tridiag_l.todense(), current.va.todense()), np.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]
                    elif i==1:
                        temp=np.zeros((current.n))
                        temp[mask]=current.k_off[mask]
                        temp[~mask]=1
                        b = torch.concat([torch.zeros((current.n, 1)),torch.einsum('BNi,Bi->BN',torch.from_numpy(temp[:, np.newaxis, np.newaxis]*self.tridiag_l.todense()*current.k_on[:,None,None]), torch.from_numpy(current.va.todense())), torch.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]
                    elif i == 0:
                        temp=np.zeros((current.n))
                        temp[mask]=current.k_on[mask]
                        temp[~mask]=0
                        b = torch.concat([torch.zeros((current.n, 1)),
                                       torch.einsum('BNi,Bi->BN', torch.from_numpy(temp[:, np.newaxis, np.newaxis] * self.tridiag_l.todense()), torch.from_numpy(current.va.todense())),
                                       torch.einsum('Ni,Bi->BN', torch.from_numpy(self.tridiag_l.todense()), torch.from_numpy(current.v0.todense()))], axis=1)[:, :, np.newaxis]
                    elif i == 2:
                        b =torch.tensor(current.f)[:,None,None]*torch.concat([torch.zeros((current.n, 1)), torch.from_numpy(-current.va.todense()), torch.from_numpy(current.va.todense())], axis=1)[:, :, np.newaxis]
                    elif i == 4:
                        b = torch.concat(
                            [torch.zeros((current.n, 1)), torch.einsum('Ni,Bi->BN', torch.from_numpy(self.dkd.todense()), torch.from_numpy(current.va.todense())),
                             torch.einsum('Ni,Bi->BN', torch.from_numpy(self.dkd.todense()), torch.from_numpy(current.v0.todense()))], axis=1)[:, :, np.newaxis]
                    S.append(torch.squeeze(torch.linalg.lstsq(transition.to_dense(), b)[0].type(torch.float32)).numpy())
                S = np.array(S, dtype=np.single)
                S=S.reshape(S.shape[0],S.shape[1],2,int(S.shape[2]/2)).sum(axis=2)
                current.S = S
                if self.print_flag:
                    print('{} time for solving sensitivity'.format(time()-start))
        except Exception:
            print(traceback.print_exc())

def eval_p(x0,p,index,value,min_like,cell_number=100000,percentage=False,downsample=False):
    if index==0:
        parameter=np.array([[0,value,x0[0],x0[1],1],[0,value,x0[0],x0[1],1]])
    elif index==1:
        parameter = np.array([[0, x0[0], value, x0[1], 1], [0, x0[0], value, x0[1], 1]])
    else:
        parameter = np.array([[0, x0[0], x0[1], value, 1], [0, x0[0], x0[1], value, 1]])
    #distribution=block_two_state_cme(parameter,percentage=percentage,print_flag=False).batch0.distribution.todense()[0,:]
    distribution=compute_cme_one_gene_two_state(parameter[0,:]).todense().squeeze()
    if downsample!=1:
        catelogue=generate_binom_catelogue(distribution.shape[0]-1,downsample)
        distribution=distribution@catelogue
        print('downsampled')
    max_m=max(p.shape[0],distribution.shape[0])
    distribution_d=np.zeros(max_m)
    distribution_d[:distribution.shape[0]]=distribution
    p=deepcopy(p)
    p.resize(max_m,refcheck=False)
    np.clip(distribution_d,10**-11,1,out=distribution_d)
    p=p/p.sum()
    distribution_d=distribution_d/distribution_d.sum()
    return -cell_number*np.matmul(np.log(distribution_d),p.T)-min_like


class Opobj:
    def __init__(self,distribution,index,value,downsample,cell_number,percentage,min_like):
        self.p=distribution
        self.downsample=downsample
        self.cell_number=cell_number
        self.percentage=percentage
        self.index=index
        self.value=value
        self.min_like=min_like
        self.obj=[]
        self.minimum=np.inf

    def eval_p1(self,x0):
        if self.index == 0:
            self.parameter = np.array([0, self.value, x0[0], x0[1], 1])
        elif self.index == 1:
            self.parameter = np.array([0, x0[0], self.value, x0[1], 1])
        else:
            self.parameter = np.array([0, x0[0], x0[1], self.value, 1])
        # distribution=block_two_state_cme(parameter,percentage=percentage,print_flag=False).batch0.distribution.todense()[0,:]
        distribution = compute_cme_one_gene_two_state(self.parameter).todense().squeeze()

        if self.downsample != 1:
            catelogue = generate_binom_catelogue(distribution.shape[0] - 1, self.downsample)
            distribution = distribution @ catelogue
        max_m = max(self.p.shape[0], distribution.shape[0])
        distribution_d = np.zeros(max_m)
        distribution_d[:distribution.shape[0]] = distribution
        p = deepcopy(self.p)
        p.resize(max_m, refcheck=False)
        np.clip(distribution_d, 10 ** -11, 1, out=distribution_d)
        p = p / p.sum()
        distribution_d = distribution_d / distribution_d.sum()
        likelihood=-self.cell_number * np.matmul(np.log(distribution_d), p.T)
        return likelihood

    def callback(self,xk,f):
        self.obj.append(f)
        if len(self.obj)==1:
            flag=False
            pass
        elif np.abs(self.obj[-1]-self.minimum)<0.01 or np.abs(self.obj[-1]-self.minimum)/np.maximum(self.obj[-1],self.minimum)<0.0001:
            flag = True
        else:
            flag= False
        self.minimum = np.minimum(self.minimum, self.obj[-1])
        return flag



def optimize_profile3(likelihood,shape,distribution,cell_number=100000,percentage=False,downsample=False,ksyn_max=2.3):
    p=likelihood.reshape(shape)
    ksyn=10**np.linspace(-0.3,ksyn_max,shape[0])
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
        inference=[]
        bounds = ((parameter[l1[1]][0], parameter[l1[1]][-1]), (parameter[l1[2]][0], parameter[l1[2]][-1]))
        min_LL=temp.min()
        for i in range(temp.shape[0]):
            start=time()
            j, k = np.unravel_index(np.argmin(temp[i, :, :]), shape=temp[i, :, :].shape)
            likelihood_min = np.inf
            initial = [parameter[l1[1]][j], parameter[l1[2]][k]]
            res=minimize(eval_p,x0=initial,args=(distribution,axis,parameter[l1[0]][i],cell_number,percentage,float(downsample)),bounds=bounds,method='L-BFGS-B')
            if res.fun<likelihood_min:
                likelihood_min=res.fun
                infer_min=res.x
            max_like.append(likelihood_min)
            max_param=np.concatenate(([parameter[l1[0]][i]],infer_min))
            inference.append(max_param)
            min_LL=min(min(max_like),min_LL)
            print(i,time()-start)
        inference=np.array(inference)
        inference = inference[:,l2]
        max_like=np.array(max_like)/cell_number
        profile[name]={'max_like':max_like,'parameter':inference}
        del res
    return profile


def optimize_likelihood(initial,distribution,downsample=False,cell_number=100000,percentage=False,probability=True,cutoff=1.92):
    def eval_p_3d(initial,distribution,downsample,cell_number,percentage=False):
        param=np.hstack((np.array([0]),initial,np.array([1])))
        param=np.vstack((param,param))
        temp=compute_cme_one_gene_two_state(param[0,:],percentage=percentage)
        max_m=max(temp.shape[1],distribution.shape[0])
        temp.resize((temp.shape[0],max_m))
        temp=np.clip(temp.todense()[0,:],a_min=10**-11,a_max=1)
        distribution_d=np.zeros(max_m)
        distribution_d[:distribution.shape[0]]=distribution
        if downsample:
            category=generate_binom_catelogue(maxcount=temp.shape[0]-1,p=float(downsample))
            temp=temp@category
        return -np.matmul(np.log(temp),distribution_d.T)*cell_number
    bounds=((10**-0.3,10**2.3),(10**-3,10**3),(10**-3,10**3))
    res=minimize(eval_p_3d,x0=initial,args=(distribution,downsample,cell_number,percentage),bounds=bounds,method='L-BFGS-B')
    return res

def parallel_likelihood(pexp_list,psim_path,extra_lib=None,repeat=0,max_cell_number=50000,shape=[60,60,60],percentage=False,downsample='1.0',probability=False,optimize=True,alpha=0.05,save_path=False,self_infer=False,coarse_grain=2):
    start_time=time()
    np.random.seed(pexp_list[1])
    save_name=str(pexp_list[2])
    pexp=pexp_list[0].ravel()
    cell=pexp_list[1]
    print(save_name)
    try:
        cutoff = chi2.ppf(1-alpha, df=1)/2
        if repeat>0:
            sample=np.random.choice(np.arange(pexp.shape[0]),size=max_cell_number,p=pexp)
        g=shelve.open(psim_path,'r')
        psim=g['downsample_'+str(downsample)]
        parameter=g['parameter']
        g.close()
        if extra_lib != None:
            h=shelve.open(extra_lib,'r')
            extra=h['downsample_' + str(downsample)]
            parameter=np.vstack((parameter,h['parameter']))
            psim.resize((psim.shape[0],extra.shape[1]))
            psim=vstack([psim,extra])
            shape[0]=int(psim.shape[0]/shape[1]/shape[2])
        ksyn=parameter[:,1].reshape(shape)[:,0,0]
        koff=parameter[:,2].reshape(shape)[0,:,0]
        kon =parameter[:,3].reshape(shape)[0,0,:]
        parameter=[ksyn,koff,kon]
        max_m = max(pexp.shape[0], psim.shape[1])
        psim.resize((psim.shape[0], max_m))
        psim=np.clip(psim.todense(),a_min=10**-11,a_max=1)
        sample_histogram=np.zeros((repeat+1,max_m))
        sample_histogram[0,:pexp.shape[0]]=pexp
        for i in range(1,repeat+1):
            sample_index=np.random.choice(np.arange(sample.shape[0]),replace=False,size=cell)
            sample_histogram[i,:]=np.histogram(sample[sample_index],bins=max_m,range=(0,max_m),density=True)[0]
        likelihood=np.array(-np.matmul(np.log(psim),sample_histogram.T).T)
        if self_infer:
            n_cell_number=4
        else:
            n_cell_number=1
        profile_dict={'cell_number':cell,'histogram':sample_histogram,'MLE':np.zeros((repeat+2,3)),'CI':np.zeros((repeat+1,n_cell_number,3,4)),'ksyn':{'max_like':np.zeros((repeat+1,shape[0])),'parameter':np.zeros((repeat+1,shape[0],3))},'koff':{'max_like':np.zeros((repeat+1,shape[1])),'parameter':np.zeros((repeat+1,shape[1],3))},'kon':{'max_like':np.zeros((repeat+1,shape[2])),'parameter':np.zeros((repeat+1,shape[2],3))}}
        key_dict = {0: 'ksyn', 1: 'koff', 2: 'kon'}
        for i in range(repeat+1):
            for j in key_dict.keys():
                tmp = likelihood[i, :].reshape(shape[0], shape[1], shape[2])
                tmp_min=tmp.min()
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
        start = time()
        find_MLE_CI(profile_dict,shape=shape,cell=cell,coarse_grain=0,self_infer=self_infer)
        if optimize:
            for i in tqdm(range(repeat + 1)):
                minimum_like=-cell*np.sum(profile_dict['histogram'][i,:]*np.log(np.clip(profile_dict['histogram'][i,:],a_min=10**-11,a_max=1)))
                tmp=likelihood[i,:]
                tmp=tmp/cell
                tmp_min=tmp.min()
                tmp=tmp-tmp_min
                arg_min=tmp.argmin()
                arg_min=np.unravel_index(arg_min,shape=shape)
                index_to_search = np.nonzero(tmp < (cutoff/cell + coarse_grain/10))
                index_to_search = np.unravel_index(index_to_search, shape=shape)
                for name in key_dict.keys():
                    for cell_index in range(n_cell_number):
                        coarse_threshold=np.log10(10*9) if name == 0 else np.log10(100*99)
                        fine_threshold = np.log10(3*2) if name == 0 else np.log10(6*5)
                        if profile_dict['CI'][i,cell_index,name,3]>coarse_threshold or profile_dict['CI'][i,cell_index,name,3]<fine_threshold:
                            continue
                        index=set(index_to_search[name].ravel())
                        index=list(index)
                        index.sort()
                        index_arg_min=np.where(index==arg_min[name])[0]
                        for index_part in [index[:index_arg_min[0]][::-1],index[index_arg_min[0]:]]:
                            fluctuation=cell/20
                            for j_index,j in enumerate(index_part):
                                if profile_dict[key_dict[name]]['max_like'][i,j]*cell<cutoff+tmp_min*cell:
                                    continue
                                sub_index=np.where(index_to_search[name].ravel()==j)[0]
                                other_keys = list(key_dict.keys())
                                other_keys.remove(name)
                                bounds = [[], []]
                                bounds[0]=[parameter[other_keys[0]][index_to_search[other_keys[0]].ravel()[sub_index].min()],parameter[other_keys[0]][index_to_search[other_keys[0]].ravel()[sub_index].max()]]
                                bounds[1] = [parameter[other_keys[1]][index_to_search[other_keys[1]].ravel()[sub_index].min()],
                                             parameter[other_keys[1]][index_to_search[other_keys[1]].ravel()[sub_index].max()]]
                                tmp=tmp.reshape(shape)
                                if name==0:
                                    min_index=np.unravel_index(tmp[j,:,:].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                                elif name==1:
                                    min_index=np.unravel_index(tmp[:,j,:].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                                else:
                                    min_index=np.unravel_index(tmp[:,:,j].argmin(),shape=(shape[other_keys[0]],shape[other_keys[1]]))
                                initial=[parameter[other_keys[0]][min_index[0]],parameter[other_keys[1]][min_index[1]]]
                                func=Opobj(sample_histogram[i,:],name,parameter[name][j],float(downsample),cell,percentage,minimum_like)
                                res=minimize(func.eval_p1,x0=initial,method='L-BFGS-B',bounds=bounds,callback=func.callback)
                                #res = minimize(eval_p, x0=initial, args=(sample_histogram[i,:], name, parameter[name][j], minimum_like, cell, percentage, float(downsample)), bounds=bounds,method='L-BFGS-B')
                                profile_dict[key_dict[name]]['max_like'][i,j]=(res.fun)/cell
                                profile_dict[key_dict[name]]['parameter'][i,j,other_keys]=res.x
                                tmp_min=min(tmp_min,res.fun/cell)
                                if j_index==1:
                                    fluctuation = max(np.abs(res.fun - profile_dict[key_dict[name]]['max_like'][i, index_part[0]]*cell)*10,cell/10)
                                    continue
                                elif j_index==0:
                                    continue
                                elif res.fun-tmp_min*cell>cutoff+fluctuation:
                                    break

            find_MLE_CI(profile_dict, shape=shape, cell=cell,self_infer=self_infer)
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

def find_MLE_CI(profile_dict,shape,cutoff=1.92,cell=0,self_infer=False,coarse_grain=0):
    MLE_data=np.zeros((profile_dict['histogram'].shape[0],np.sum(shape),4))
    MLE_data[:,:shape[0],0]=profile_dict['ksyn']['max_like']
    MLE_data[:, :shape[0], 1:] = profile_dict['ksyn']['parameter']
    MLE_data[:,shape[0]:shape[0]+shape[1],0]=profile_dict['koff']['max_like']
    MLE_data[:,shape[0]:shape[0]+shape[1], 1:] = profile_dict['koff']['parameter']
    MLE_data[:,shape[0]+shape[1]:,0]=profile_dict['kon']['max_like']
    MLE_data[:, shape[0]+shape[1]:, 1:] = profile_dict['kon']['parameter']
    profile_dict['MLE']=MLE_data[np.arange(MLE_data.shape[0]),MLE_data[:,:,0].argmin(axis=1),1:]
    if not self_infer:
        total_cell=[cell]
    else:
        total_cell=[100,1000,10000,100000]
    profile_dict['profile']={'ksyn':[],'koff':[],'kon':[]}
    profile_dict['CI']=np.zeros((MLE_data.shape[0],len(total_cell),3,4))
    for i in range(MLE_data.shape[0]):
        temp=MLE_data[i,:,:]
        temp[:,0]=temp[:,0]-temp[:,0].min()
        for name,index in zip(['ksyn','koff','kon'],range(1,4)):
            tmp = temp[temp[ :, index].argsort()][:, [0, index]]
            pair = [tmp[0, :]]
            for j in range(1, tmp.shape[0]):
                if tmp[j, 1] == pair[-1][1]:
                    pair[-1][0] = min(tmp[j, 0], pair[-1][0])
                else:
                    pair.append(tmp[j, :])
            profile_dict['profile'][name]=np.array(pair)
        for j,cell in enumerate(total_cell):
            for k,name in zip(range(1,MLE_data.shape[2]),['ksyn','koff','kon']):
                tmp=profile_dict['profile'][name]+0
                tmp[:,0]=tmp[:,0]*cell
                cut=cutoff+coarse_grain*cell/10
                index=np.where(tmp[:,0]<cut)[0]
                if len(index)==1:
                    if index[0]==0:
                        lower_bound=tmp[0,1]
                        upper_bound=tmp[0,1]+(tmp[1,1]-tmp[0,1])/(tmp[1,0]-tmp[0,0])*cut
                    elif index[0]==tmp.shape[0]-1:
                        upper_bound=tmp[-1,1]
                        lower_bound=tmp[-1,1]-(tmp[-1,1]-tmp[-2,1])/(tmp[-2,0]-tmp[-1,0])*cut
                    else:
                        lower_bound=tmp[index[0],1]-(tmp[index[0],1]-tmp[index[0]-1,1])/(tmp[index[0]-1,0]-tmp[index[0],0])*cut
                        upper_bound=tmp[index[-1],1]+(tmp[index[-1]+1,1]-tmp[index[-1],1])/(tmp[index[-1]+1,0]-tmp[index[-1],0])*cut
                else:
                    if index[0]==0:
                        lower_bound=tmp[0,1]
                    else:
                        lower_bound=tmp[index[0],1]-(tmp[index[0],1]-tmp[index[0]-1,1])/(tmp[index[0]-1,0]-tmp[index[0],0])*(cut-tmp[index[0],0])
                    if index[-1]==tmp.shape[0]-1:
                        upper_bound=tmp[-1,1]
                    else:
                        upper_bound=tmp[index[-1],1]+(tmp[index[-1]+1,1]-tmp[index[-1],1])/(tmp[index[-1]+1,0]-tmp[index[-1],0])*(cut-tmp[index[-1],0])
                width=np.log10(upper_bound/lower_bound)
                metric=width+np.log10(max(upper_bound-lower_bound,0.001)/profile_dict['MLE'][i,k-1])
                profile_dict['CI'][i,j,k-1,:]=[lower_bound,upper_bound,width,metric]

    return


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
