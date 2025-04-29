import scipy
from function2 import *
import shelve
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os

def get_CI(downsample):
    g=shelve.open('../self_infer/downsample_'+downsample+'/library_300_infer','r')
    CI=np.zeros((216000,19,4,5))
    for j in tqdm(range(216000)):
        for k,cell in enumerate([10**2,3*10**2,7*10**2,10**3,3*10**3,7*10**3,10**4,3*10**4,7*10**4,10**5,3*10**5,7*10**5,10**6,3*10**6,7*10**6,10**7,3*10**7,7*10**7,10**8]):
            CI[j,k]=find_MLE_CI(g[str(j)],[60,60,60],cell=cell).squeeze()
    return CI

if __name__=='__main__':

    fig,ax=plt.subplot_mosaic([['a','a','a','a'],['z','z','z','z'],['b','c','d','e'],['f','g','h','i']],figsize=(12,12),height_ratios=[2,0.15,1,1],dpi=400)
    fig.subplots_adjust(hspace=0.05)
    if os.path.exists('../CI_all.mat'):
        CI=scipy.io.loadmat('../CI_all.mat')['CI']
    else:
        CI=np.zeros((4,216000,19,4,5))
        pool=Pool(4)
        with pool:
            temp=list(pool.imap(get_CI,['0.3','0.5','0.7','1.0']))
        for i in range(4):
            CI[i]=temp[i]
        scipy.io.savemat('../CI_all.mat', {'CI': CI})
    Iden=(CI[:,:,:,:-1,2]/(np.array([2.6,6,6])[np.newaxis,np.newaxis,np.newaxis,:])).max(axis=3)
    Iden=np.clip(Iden,a_min=0,a_max=1)
    percentage=np.zeros((4,19))
    for i in range(4):
        for j in range(19):
            percentage[i,j]=np.where(Iden[i,:,j]<1/3)[0].shape[0]/216000
    ax['a'].plot(np.log10([10**2,3*10**2,7*10**2,10**3,3*10**3,7*10**3,10**4,3*10**4,7*10**4,10**5,3*10**5,7*10**5,10**6,3*10**6,7*10**6,10**7,3*10**7,7*10**7,10**8]),percentage.T,marker='o')
    ax['a'].legend(['0.3 capture rate','0.5 capture rate','0.7 capture rate','1.0 capture rate'])
    ax['a'].set_xlabel('number of cells in log10',fontsize=18)
    ax['a'].set_ylabel('fraction of parameter sets identifiable',fontsize=18)
    ax['a'].tick_params(labelsize=15)
    ax['a'].text(0, 1.05, 'A', transform=ax['a'].transAxes, size=20, weight='bold')
    ax['z'].axis("off")
    psim=compute_cme_one_gene_two_state(np.array([0,3.5,0.1,0.23,1]),testing=True)
    try:
        psim=psim.reshape((2,int(psim.shape[0]/2)))
    except:
        pass
    psim_c=np.zeros((2,300))
    psim_c[:,:psim.shape[1]]=psim
    psim=np.clip(psim_c,a_min=10**-11,a_max=1)
    last_index=np.argmin(np.cumsum(psim.sum(axis=0))<0.999)
    ax['e'].plot(psim[0,:last_index],label='active',color='orange')
    ax['e'].plot(psim[1,:last_index],label='inactive',color='red')
    ax['e'].plot(psim[:,:last_index].sum(axis=0),label='overall',color='black')
    ax['e'].set_xticks([])
    ax['e'].set_yticklabels(np.arange(0, 0.2, 5), fontsize=18)
    ax['e'].set_ylabel('Probability',fontsize=18)
    ax['e'].legend(fontsize=13)
    ax['e'].text(-0.2, 1, 'E', transform=ax['e'].transAxes, size=20, weight='bold')
    catelogue=generate_binom_catelogue(maxcount=psim.sum(axis=0).shape[0]-1,p=0.3)
    psim_d=psim@catelogue
    result=parallel_likelihood([psim.sum(axis=0),100000,'test'],psim_path='../library_300',self_infer=True,downsample='1.0')
    result_d=parallel_likelihood([psim_d[:,:119].sum(axis=0),100000,'test'],psim_path='../library_300',self_infer=True,downsample='0.3')
    ax['b'].plot(np.log10(result['ksyn']['parameter'].squeeze()[:,0]),(result['ksyn']['max_like'].squeeze()-result['ksyn']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['b'].plot(np.log10(result['ksyn']['parameter'].squeeze()[:,0]),(result['ksyn']['max_like'].squeeze()-result['ksyn']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['b'].plot(np.log10(result['ksyn']['parameter'].squeeze()[:,0]),(result['ksyn']['max_like'].squeeze()-result['ksyn']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['b'].set_ylim(-0.2,2.2)
    ax['b'].axvline(np.log10(3.5), ymin=0, ymax=2.2, color='black')
    ax['b'].scatter([np.log10(result['MLE'].squeeze()[0])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['b'].legend(fontsize=13)
    ax['b'].set_ylabel('PL',fontsize=18)
    ax['b'].set_xticks([])
    ax['b'].set_yticks([0,1,1.92])
    ax['b'].set_yticklabels([0,1,1.92],fontsize=18)
    ax['b'].text(-0.2, 1, 'B', transform=ax['b'].transAxes, size=20, weight='bold')

    ax['c'].plot(np.log10(result['koff']['parameter'].squeeze()[:,1]),(result['koff']['max_like'].squeeze()-result['koff']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['c'].plot(np.log10(result['koff']['parameter'].squeeze()[:,1]),(result['koff']['max_like'].squeeze()-result['koff']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['c'].plot(np.log10(result['koff']['parameter'].squeeze()[:,1]),(result['koff']['max_like'].squeeze()-result['koff']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['c'].set_ylim(-0.2,2.2)
    ax['c'].axvline(np.log10(0.1), ymin=0, ymax=2.2, color='black')
    ax['c'].scatter([np.log10(result['MLE'].squeeze()[1])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['c'].set_xticks([])
    ax['c'].set_yticks([])
    ax['c'].text(-0.2, 1, 'C', transform=ax['c'].transAxes, size=20, weight='bold')

    ax['d'].plot(np.log10(result['kon']['parameter'].squeeze()[:,2]),(result['kon']['max_like'].squeeze()-result['kon']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['d'].plot(np.log10(result['kon']['parameter'].squeeze()[:,2]),(result['kon']['max_like'].squeeze()-result['kon']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['d'].plot(np.log10(result['kon']['parameter'].squeeze()[:,2]),(result['kon']['max_like'].squeeze()-result['kon']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['d'].set_ylim(-0.2,2.2)
    ax['d'].axvline(np.log10(0.23), ymin=0, ymax=2.2, color='black')
    ax['d'].scatter([np.log10(result['MLE'].squeeze()[2])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['d'].set_xticks([])
    ax['d'].set_yticks([])
    ax['d'].text(-0.2, 1, 'D', transform=ax['d'].transAxes, size=20, weight='bold')

    ax['i'].plot(psim_d[0,:last_index],label='active',color='orange')
    ax['i'].plot(psim_d[1,:last_index],label='inactive',color='red')
    ax['i'].plot(psim_d[:,:last_index].sum(axis=0),label='overall',color='black')
    ax['i'].set_xticks(np.arange(0,9,2))
    ax['i'].set_xticklabels(np.arange(0,9,2),fontsize=18)
    ax['i'].set_yticklabels(np.arange(0, 0.4, 5), fontsize=18)
    ax['i'].set_xlabel('mRNA copy',fontsize=18)
    ax['i'].set_ylabel('Probability',fontsize=18)
    ax['i'].legend(fontsize=13)
    ax['i'].text(-0.2, 1, 'I', transform=ax['i'].transAxes, size=20, weight='bold')

    ax['f'].plot(np.log10(result_d['ksyn']['parameter'].squeeze()[:,0]),(result_d['ksyn']['max_like'].squeeze()-result_d['ksyn']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['f'].plot(np.log10(result_d['ksyn']['parameter'].squeeze()[:,0]),(result_d['ksyn']['max_like'].squeeze()-result_d['ksyn']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['f'].plot(np.log10(result_d['ksyn']['parameter'].squeeze()[:,0]),(result_d['ksyn']['max_like'].squeeze()-result_d['ksyn']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['f'].set_ylim(-0.2,2.2)
    ax['f'].scatter([np.log10(result_d['MLE'].squeeze()[0])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['f'].axvline(np.log10(3.5), ymin=0, ymax=2.2, color='black')
    ax['f'].legend(fontsize=13)
    ax['f'].set_xlabel('ksyn in log10', fontsize=18)
    ax['f'].set_xticklabels([0,1,2],fontsize=18)
    ax['f'].set_ylabel('PL', fontsize=18)
    ax['f'].set_yticks([0,1,1.92])
    ax['f'].set_yticklabels([0,1,1.92],fontsize=18)
    ax['f'].text(-0.2, 1, 'F', transform=ax['f'].transAxes, size=20, weight='bold')

    ax['g'].plot(np.log10(result_d['koff']['parameter'].squeeze()[:,1]),(result_d['koff']['max_like'].squeeze()-result_d['koff']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['g'].plot(np.log10(result_d['koff']['parameter'].squeeze()[:,1]),(result_d['koff']['max_like'].squeeze()-result_d['koff']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['g'].plot(np.log10(result_d['koff']['parameter'].squeeze()[:,1]),(result_d['koff']['max_like'].squeeze()-result_d['koff']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['g'].set_ylim(-0.2,2.2)
    ax['g'].axvline(np.log10(0.1), ymin=0, ymax=2.2, color='black')
    ax['g'].scatter([np.log10(result_d['MLE'].squeeze()[1])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['g'].set_xlabel('koff in log10', fontsize=18)
    ax['g'].set_yticks([])
    ax['g'].set_xticks([-2,0,2])
    ax['g'].set_xticklabels([-2, 0, 2],fontsize=18)
    ax['g'].text(-0.2, 1, 'G', transform=ax['g'].transAxes, size=20, weight='bold')

    ax['h'].plot(np.log10(result_d['kon']['parameter'].squeeze()[:,2]),(result_d['kon']['max_like'].squeeze()-result_d['kon']['max_like'].squeeze().min())*1000,label='1Kcells',color='lightblue')
    ax['h'].plot(np.log10(result_d['kon']['parameter'].squeeze()[:,2]),(result_d['kon']['max_like'].squeeze()-result_d['kon']['max_like'].squeeze().min())*10000,label='10Kcells',color='blue')
    ax['h'].plot(np.log10(result_d['kon']['parameter'].squeeze()[:,2]),(result_d['kon']['max_like'].squeeze()-result_d['kon']['max_like'].squeeze().min())*100000,label='100Kcells',color='blueviolet')
    ax['h'].set_ylim(-0.2,2.2)
    ax['h'].axvline(np.log10(0.23), ymin=0, ymax=2.2, color='black')
    ax['h'].scatter([np.log10(result_d['MLE'].squeeze()[2])]*5,np.linspace(0,2.2,5),marker='*',color='saddlebrown')
    ax['h'].set_xlabel('kon in log10', fontsize=18)
    ax['h'].set_yticks([])
    ax['h'].set_xticks([-2,0,2])
    ax['h'].set_xticks([-2,0,2])
    ax['h'].set_xticklabels([-2, 0, 2],fontsize=18)
    ax['h'].text(-0.2, 1, 'H', transform=ax['h'].transAxes, size=20, weight='bold')
    fig.text(0.38, 0.52, s='ksyn: 3.5, koff: 0.1, kon: 0.23', fontsize=20)
    fig.tight_layout()
    fig.savefig('fig4')
    fig.savefig('fig4.eps',format='eps')
    fig.savefig('fig4.svg',format='svg')



    print('done')