import shelve,os
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool,cpu_count
import scipy

def get_m2_m3(psim):
    index=np.argmin(psim.cumsum()<0.999)
    if index==0:
        return 0,0
    index=index+1
    d=interp1d(np.linspace(0,1,index),psim[:index])
    x=np.linspace(0,1,21)
    y=d(x)
    mean=(x*y).sum()
    m2=((x-mean)**2*y).sum()
    m3=((x-mean)**3*y).sum()
    return m2,m3

def get_gaussian_mixture(psim):
    def likelihood(x,start,psim):
        total=psim.sum()
        pexp=scipy.stats.norm.pdf(x=np.arange(start,psim.shape[0]+start),loc=x[0],scale=x[1])
        if pexp.sum()!=0:
            pexp=pexp/pexp.sum()*total
        error=np.linalg.norm(pexp-psim)*100
        return error
    def likelihood_p(x,start,psim):
        total=psim.sum()
        pexp=scipy.stats.poisson.pmf(k=np.arange(start,psim.shape[0]+start),mu=x[0])
        pexp=pexp/pexp.sum()*total
        error=np.linalg.norm(pexp-psim)*100
        return error
    psim=psim/psim.sum()
    if ((psim[1:] - psim[:-1])[::-1] > 0).any():
        lamda2 = psim.shape[0] - np.argmax((psim[1:] - psim[:-1])[::-1] > 0)-1
        bounds=[[0,lamda2],[0.4,max(20*lamda2**0.5,5)]]
        x0=[lamda2,lamda2**0.5]
    else:
        lamda2=0
        bounds=[[0,2],[0.4,20*2]]
        x0=[lamda2,1]
    position=lamda2 if lamda2==0 else lamda2+3
    result=scipy.optimize.minimize(likelihood,x0=x0,args=(position,psim[position:]),bounds=bounds)
    p1=scipy.stats.norm.pdf(x=np.arange(psim.shape[0]),loc=result.x[0],scale=result.x[1])
    p1=p1/p1.sum()
    p1_r=p1[:position].sum()/p1[position:].sum()
    p1[position:]=p1[position:]/p1[position:].sum()*psim[position:].sum()
    if lamda2>0:
        p1[:position]=p1[:position]/p1[:position].sum()*p1_r*p1[position:].sum()
    remain=np.clip((psim-p1)*np.sign(psim-p1),a_min=10**-11,a_max=1)
    if p1.sum()!=0:
    #lamda1 = np.argmax(psim[:-1]-psim[1:]>0)
        lamda1=np.argmax(remain)
        result1=scipy.optimize.minimize(likelihood,x0=[lamda1,0.1],args=(0,remain),bounds=bounds)
        #result2 = scipy.optimize.minimize(likelihood_p, x0=[lamda1], args=(position, psim[position:]), bounds=[bounds[1]])
        p2 = scipy.stats.norm.pdf(np.arange(psim.shape[0]), loc=result1.x[0], scale=result1.x[1] ** 0.5 + 0.001)
        p2 = p2 / p2.sum() * remain.sum()*np.sign(1-p1.sum())
        #p2_=scipy.stats.poisson.pmf(np.arange(psim.shape[0]),mu=result2.x[0])*remain.sum()
        return [result1.x[0],result.x[0],result1.x[1],result.x[1],1-p1.sum(),p1.sum(),np.abs(result1.x[0]-result.x[0])/(2*(result.x[1]**2+result1.x[1]**2))**0.5,np.abs(result1.x[0]-result.x[0])/(2*(result.x[1]+result1.x[1])),((p1.argmax()+1)/(np.abs(p2).argmax()+1))**0.5*(p1.max()+np.abs(p2).max()),np.linalg.norm(psim-p1-p2)]
    else:
        return [0,result.x[0], 0, result.x[1],0,1,np.abs(result.x[0]-0)/(2*(result.x[1]**2+0))**0.5,np.abs(result.x[0]-0)/(2*(result.x[1]+0)),((p1.argmax()+1)/(np.abs(remain).argmax()+1))**0.5*(p1.max()+np.abs(remain).max()),np.linalg.norm(psim-p1)]

def get_distribution_mixture(psim):
    psim=psim/psim.sum()
    lamda2=psim.shape[0]-np.argmax((psim[1:]-psim[:-1])[::-1]>0)
    lamda1=0
    weight1=1-psim[0]
    p=psim.copy()
    p[0]=0
    p=p/weight1




if __name__=='__main__':
    parallel=True
    param_index=[50,14,4]
    choose=np.ravel_multi_index(param_index,dims=(60,60,60))
    choose=204282
    psim=shelve.open('../library_300','r')['downsample_0.3'].todense()
    x=get_gaussian_mixture(psim[choose,:])
    p1_1 = scipy.stats.norm.pdf(np.arange(psim.shape[1]), loc=x[0], scale=x[2])
    p1_1 = p1_1 / p1_1.sum() * x[4]
    p1_2 = scipy.stats.norm.pdf(np.arange(psim.shape[1]), loc=x[1], scale=x[3])
    p1_2 = p1_2 / p1_2.sum() * x[5]
    plt.plot(p1_1 + p1_2)
    plt.plot(psim[choose, :])
    plt.title(choose)
    plt.show()
    #dist=scipy.stats.poisson.pmf(np.arange(50),mu=0.6)*0.996+scipy.stats.poisson.pmf(np.arange(50),mu=6)*0.004
    #x=get_poisson_mixture_fit_2(dist)
    #index=[149250,    169213,    133221,    133342,    133403,    136822,    136943,    137000,    140423,    140544,    144203,    151218,    162196,    162201,    162266,    163717,    164017,    165867,    167317,    167679,    169213,    111622,    111804,    115283,    115344,    118823,    118884,    122424,    129686,    133226,    136887,    140427,    140488,    140548,    144028,    144088,    144149,    162148,    162270,    26904,    30565,    44461,    55354,    62615,    99950,    103489,    103550,    107089,    107150,    162630]
    #test_fit=np.zeros((len(index),4))
    #for j,i in enumerate(index):
    #    test_fit[j,:]=get_poisson_mixture_fit_2(psim[i,:])
    #x=get_poisson_mixture_fit(psim[158600,:])
    moments=np.zeros((216000,4))
    moments[:, 0] = np.sum(np.arange(psim.shape[1])[None,:] * psim, axis=1)
    moments[:, 1] = np.sum(((np.arange(psim.shape[1])[:, None] - moments[:,0]) ** 2).T * psim, axis=1)
    moments[:, 2] = np.sum(((np.arange(psim.shape[1])[:, None] - moments[:,0]) ** 3).T * psim, axis=1)
    moments[:, 3] = np.sum(((np.arange(psim.shape[1])[:, None] - moments[:,0]) ** 4).T * psim, axis=1)
    g = shelve.open('../self_infer/downsample_0.3/library_300_infer', 'r')
    data = np.zeros((4, 216000, 3))
    for i in tqdm(range(216000)):
        # data[:,i,:]=find_MLE_CI2(g[str(i)],self_infer=True).squeeze()[:,:,2]
        data[:, i, :] = g[str(i)]['CI'].squeeze()[:, :, 2]
    g.close()
    metric=data
    metric=(metric-metric.min(axis=(0,1))[np.newaxis,np.newaxis,:])/(metric.max(axis=(0,1))-metric.min(axis=(0,1)))[np.newaxis,np.newaxis,:]
    index_id = np.where((metric[0, :, :].max(axis=1) < 0.333) & (moments[:, 1] < 50) & (np.abs(moments[:, 2]) < 15))[0]
    index_un = np.where((metric[0, :, :].max(axis=1) > 0.7) & (moments[:, 1] < 50) & (np.abs(moments[:, 2]) < 15))[0]
    mix=[]
    pool=Pool(cpu_count())
    if os.path.exists('gaussian_fit.mat'):
        mix=scipy.io.loadmat('gaussian_fit.mat')
    else:
        if parallel:
            with pool:
                mix=list(tqdm(pool.imap(get_gaussian_mixture,psim),total=216000))
        else:
            for i in tqdm(range(216000)):
                mix.append(get_gaussian_mixture(psim[i,:]))
        mix=np.array(mix).reshape(216000,len(mix[0]))
        scipy.io.savemat('gaussian_fit.mat',{'gaussian':mix[:,:6]})
    print('done')