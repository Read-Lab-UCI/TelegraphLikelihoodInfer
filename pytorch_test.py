import torch,psutil,shelve
import numpy as np
import sparse
from scipy.sparse import diags
from time import time
import matplotlib.pyplot as plt
#from function import *

class init:
    def __init__(self):

        pass

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
                for i, j in enumerate(['k_on', 'f', 'h', 'k_off', 'kd']):
                    if i == 2 and self.percentage:
                        temp = (current.f * np.log(1 - current.p_on))
                        b = torch.concat([torch.zeros((current.n, 1)), torch.from_numpy((temp[:, np.newaxis] * current.v0).todense()), torch.from_numpy((-temp[:, np.newaxis] * current.v0).todense())], axis=1)[:, :, np.newaxis]
                    elif i == 2 and not self.percentage:
                        b = torch.concat([torch.zeros((current.n, 1)), torch.from_numpy(current.v0.todense()), torch.from_numpy(-current.v0.todense())],axis=1)[:, :, np.newaxis]
                    # elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   -tf.einsum('BNi,Bi->BN', ((current.mask * current.k_off + (1 - current.mask)) * current.mask_a)[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.vab),
                    #                   -tf.einsum('BNi,Bi->BN',(current.mask * current.k_off + (1 - current.mask))[:, np.newaxis,np.newaxis] * self.tridiag_l.todense(), current.va0)], axis=1)[:, :, np.newaxis]
                    #elif i == 0:
                    #    b = tf.concat([np.zeros((current.n, 1)),
                    #                   tf.einsum('BNi,Bi->BN',(k_off*mask+(1-mask))[:, np.newaxis, np.newaxis] * self.tridiag_l.todense(), current.va.todense()), np.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]
                    elif i==0:
                        b = torch.concat([torch.zeros((current.n, 1)),torch.einsum('BNi,Bi->BN',torch.from_numpy(k_off[:, np.newaxis, np.newaxis]*self.tridiag_l.todense()), torch.from_numpy(current.va.todense())), torch.zeros((current.n,self.max_state))], axis=1)[:, :,np.newaxis]

                    elif i == 1:
                        b = torch.concat([torch.zeros((current.n, 1)), torch.from_numpy(-current.va.todense()), torch.from_numpy(current.va.todense())], axis=1)[:, :, np.newaxis]
                    elif i == 3:
                        b = torch.concat([torch.zeros((current.n, 1)),
                                       torch.einsum('BNi,Bi->BN', torch.from_numpy(current.k_on[:, np.newaxis, np.newaxis] * self.tridiag_l.todense()), torch.from_numpy(current.va.todense())),
                                       torch.einsum('Ni,Bi->BN', torch.from_numpy(self.tridiag_l.todense()), torch.from_numpy(current.v0.todense()))], axis=1)[:, :, np.newaxis]
                    elif i == 4:
                        b = torch.concat(
                            [torch.zeros((current.n, 1)), torch.einsum('Ni,Bi->BN', torch.from_numpy(self.dkd.todense()), torch.from_numpy(current.va.todense())),
                             torch.einsum('Ni,Bi->BN', torch.from_numpy(self.dkd.todense()), torch.from_numpy(current.v0.todense()))], axis=1)[:, :, np.newaxis]
                    S.append(torch.squeeze(torch.linalg.lstsq(transition.to_dense(), b)[0].type(torch.float32)).numpy())
                S = np.array(S, dtype=np.single)
                S=S.reshape(S.shape[0],S.shape[1],2,int(S.shape[2]/2))
                current.S = S
                if self.print_flag:
                    print('{} time for solving sensitivity'.format(time()-start))
        except:
            pass
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



if __name__=='__main__':
    param=np.array([[0,3.6,8,0.6,1],[0,20,0.08,0.1,1]])
    test=block_two_state_cme(param)
    plt.plot(test.batch0.distribution.todense().T)
    plt.show()
    g=shelve.open('nandor_data/dko.norm_histogram')
    keys=['RP11-206L10.3','RP11-206L10.2','NRL','PCK2','FITM1','PSME1','EMC9']
    for i in range(12640,12650):
        if g['histogram'][i][-1] in keys:
            print(i)
            start=time()
            parallel_likelihood(g['histogram'][i],psim_path='nandor_data/one_gene_two_state_library_grid')
            print(g['histogram'][i][-1],time()-start)
    print('run')