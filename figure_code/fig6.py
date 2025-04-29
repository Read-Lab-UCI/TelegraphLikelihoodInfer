import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import shelve,scipy
import numpy as np
from function3 import compute_cme_one_gene_two_state,parallel_likelihood
if __name__=='__main__':
    #fig,ax=plt.subplots(1,3,dpi=400,figsize=(15,5))
    """
    param=np.array([0,10,0.1,0.05,1])
    param2=np.array([0,1,0.01,0.05,1])
    p=compute_cme_one_gene_two_state(param)
    p2=compute_cme_one_gene_two_state(param2)
    plt.plot(p.todense().ravel()[:20])
    plt.plot(p2.todense().ravel()[:20])
    plt.ylim(0,0.3)
    plt.show()
    data=scipy.io.loadmat('../CI_all.mat')
    g=shelve.open('../library_300','r')
    burst_freq=g['parameter'][:,-2]
    burst_size=g['parameter'][:,1]/g['parameter'][:,2]
    test=parallel_likelihood([p.todense(),1000,'test'],psim_path='../library_300',optimize=False)
    test2 = parallel_likelihood([p2.todense(), 1000, 'test2'], psim_path='../library_300',full_optimize=True)
    plt.plot(np.log10(test['ksyn']['parameter'].squeeze()[:, 0]),
             (test['ksyn']['max_like'].squeeze() - test['ksyn']['max_like'].min()) * test['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    plt.plot(np.log10(test['koff']['parameter'].squeeze()[:, 1]),
             (test['koff']['max_like'].squeeze() - test['koff']['max_like'].min()) * test['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    plt.plot(np.log10(test['kon']['parameter'].squeeze()[:, -1]),
             (test['kon']['max_like'].squeeze() - test['kon']['max_like'].min()) * test['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    plt.plot(np.log10(test2['ksyn']['parameter'].squeeze()[:, 0]),
             (test2['ksyn']['max_like'].squeeze() - test2['ksyn']['max_like'].min()) * test2['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    plt.plot(np.log10(test2['koff']['parameter'].squeeze()[:, 1]),
             (test2['koff']['max_like'].squeeze() - test2['koff']['max_like'].min()) * test2['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    plt.plot(np.log10(test2['kon']['parameter'].squeeze()[:, -1]),
             (test2['kon']['max_like'].squeeze() - test2['kon']['max_like'].min()) * test2['cell_number'])
    plt.ylim(0, 2.2)
    plt.show()
    """
    import matplotlib
    matplotlib.rc('font', size=15)
    norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=2)
    #CI = scipy.io.loadmat('../CI_all.mat')['CI']
    CI=np.load('../self_infer/downsample_1.0/new_version/CI.npy')
    parameter = shelve.open('../library_300', 'r')['parameter']
    index=np.where(parameter[:,2]>parameter[:,3]/4)
    burst_size = np.log10(parameter[:, 1] / parameter[:, 2])[index]
    burst_frequency = np.log10(parameter[:,2]*parameter[:, 3]/(parameter[:,2]+parameter[:,3]))[index]
    fig,ax=plt.subplot_mosaic([['a','b','d']],width_ratios=[1,1.2,1],figsize=(16,5),dpi=400)
    tmp=(CI[index,1,:,2].squeeze()/np.array([0.7,2,2])[None,:]).max(axis=1)
    sort_index=tmp.argsort()
    surface=ax['a'].scatter(burst_size[sort_index],burst_frequency[sort_index],c=tmp[sort_index],s=10,norm=norm,cmap='RdBu')
    ax['a'].set_title('increasing order',fontsize=18)
    ax['b'].scatter(burst_size[sort_index[::-1]],burst_frequency[sort_index[::-1]],c=tmp[sort_index[::-1]],s=10,norm=norm,cmap='RdBu')
    ax['b'].set_title('decreasing order',fontsize=18)
    ax['a'].scatter(np.log10(2.667/0.52),np.log10(0.52*0.12/0.64),marker='o',color='orange',s=60)
    ax['b'].scatter(np.log10(20/3.9), np.log10(3.9*0.1/4), marker='v',color='blue',s=60)
    ax['a'].set_yticks([-3,0])
    ax['a'].set_yticklabels([-3,0],fontsize=15)
    ax['b'].set_yticks([-3,0])
    ax['b'].set_yticklabels([-3,0],fontsize=15)
    ax['a'].set_xticks([-2, 0, 2,4])
    ax['b'].set_xticks([-2, 0, 2, 4])
    ax['a'].set_xticklabels([-2,0,2,4],fontsize=15)
    ax['b'].set_xticklabels([-2, 0, 2, 4], fontsize=15)
    ax['a'].set_ylabel('burst frequency in log10',fontsize=18)
    dist2 = compute_cme_one_gene_two_state(np.array([0, 2.667, 0.52, 0.12, 1])).todense().ravel()
    test2 = parallel_likelihood((dist2, 1000, 'test'), '../library_300', self_infer=False)
    """
    index = np.arange(dist2.shape[0])[::4]
    ax['d'].set_ylim(0,0.003)
    ax['d'].set_xticks(np.linspace(0,60,4))
    ax['d'].set_xticklabels(np.linspace(0, 60, 4).astype('int'),fontsize=15)
    ax['d'].set_yticks(np.linspace(0,0.003,4))
    #ax['d'].set_yticklabels(np.linspace(0, 0.003, 4), fontsize=15)
    ax['d'].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax['d'].plot(index,dist2[::4], label='un-identifiable',marker='o',color='blue')
    """
    dist1 = compute_cme_one_gene_two_state(np.array([0, 20, 3.9, 0.1, 1])).todense().ravel()
    test1 = parallel_likelihood((dist1, 1000, 'test'), '../library_300', self_infer=False)
    #index = np.arange(dist1.shape[0])[::4]
    #ax['d'].plot(index, dist1[::4], label='identifiable', marker='v', color='orange')
    ax['d'].plot(dist1[:10],label='identifiable',marker='v',color='blue')
    ax['d'].plot(dist2[:10], label='un-identifiable', marker='o', color='orange')
    ax['d'].set_ylim(0,0.6)
    ax['d'].set_xlim(0,10)
    ax['d'].legend(fontsize=18)
    ax['d'].set_xlabel('mRNA copy number',fontsize=18)
    ax['d'].set_ylabel('Probability',fontsize=18)
    ax['a'].set_xlabel('burst size in log10', fontsize=18)
    ax['b'].set_xlabel('burst size in log10',fontsize=18)
    fig.text(0.01, 0.95, 'A', ha='center', fontsize=20,weight='bold')
    fig.text(0.33, 0.95, 'B', ha='center', fontsize=20,weight='bold')
    fig.text(0.68, 0.95, 'C', ha='center', fontsize=20,weight='bold')
    cbar=fig.colorbar(surface,ax=ax['b'],orientation='vertical')
    #cbar.set_label(label=r'$Log_{threshold}(Upper/Lower)$',size=18)
    cbar.set_label(label='Alternative Precision Metric', size=18)
    cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.show()
    fig.savefig('fig6')
    """
    fig2, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']])
    sort_index = CI[-1, :, 1, 0, 3].argsort()[::-1]
    surface = ax['a'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 0, 3])
    ax['a'].set_title('ksyn')
    ax['a'].set_xticks([])
    fig2.colorbar(surface, ax=ax['a'])
    sort_index = CI[-1, :, 1, 1, 3].argsort()[::-1]
    surface = ax['b'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 1, 3])
    ax['b'].set_title('koff')
    ax['b'].set_xticks([])
    ax['b'].set_yticks([])
    fig2.colorbar(surface, ax=ax['b'])
    sort_index = CI[-1, :, 1, 2, 3].argsort()[::--1]
    surface = ax['c'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 2, 3])
    ax['c'].set_title('burst frequency(kon)')
    fig2.colorbar(surface, ax=ax['c'])
    sort_index = CI[-1, :, 1, 3, 3].argsort()[::-1]
    surface = ax['d'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 3, 3])
    ax['d'].set_title('burst size(kysn/koff)')
    fig2.colorbar(surface, ax=ax['d'])
    fig2.text(x=0.33, y=0.01, s='burst size in log10', fontsize=18)
    fig2.text(x=0.02, y=0.22, s='burst frequency in log10', rotation='vertical', fontsize=18)
    ax['a'].text(-0.15, 1, s='a', transform=ax['a'].transAxes, size=20, weight='bold')
    ax['b'].text(-0.15, 1, s='b', transform=ax['b'].transAxes, size=20, weight='bold')
    ax['c'].text(-0.15, 1, s='c', transform=ax['c'].transAxes, size=20, weight='bold')
    ax['d'].text(-0.15, 1, s='d', transform=ax['d'].transAxes, size=20, weight='bold')
    fig2.show()
    fig2.savefig('frequency_vs_size_uncertainty')
    fig2.savefig('frequency_vs_size_uncertainty.eps', format='eps')
    fig2.savefig('frequency_vs_size_uncertainty.svg', format='svg')

    fig3, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']])
    sort_index = CI[-1, :, 3, 0, 2].argsort()[::-1]
    surface = ax['a'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 0, 2])
    ax['a'].set_title('ksyn')
    ax['a'].set_xticks([])
    fig3.colorbar(surface, ax=ax['a'])
    sort_index = CI[-1, :, 3, 1, 2].argsort()[::-1]
    surface = ax['b'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 1, 2])
    ax['b'].set_title('koff')
    ax['b'].set_xticks([])
    ax['b'].set_yticks([])
    fig3.colorbar(surface, ax=ax['b'])
    sort_index = CI[-1, :, 3, 2, 2].argsort()[::-1]
    surface = ax['c'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 2, 2])
    ax['c'].set_title('burst frequency(kon)')
    fig3.colorbar(surface, ax=ax['c'])
    sort_index = CI[-1, :, 3, 3, 2].argsort()[::-1]
    surface = ax['d'].scatter(burst_size[sort_index], burst_frequency[sort_index], c=CI[-1, sort_index, 1, 3, 2])
    ax['d'].set_title('burst size(kysn/koff)')
    fig3.colorbar(surface, ax=ax['d'])
    fig3.text(x=0.33, y=0.01, s='burst size in log10', fontsize=18)
    fig3.text(x=0.02, y=0.22, s='burst frequency in log10', rotation='vertical', fontsize=18)
    ax['a'].text(-0.15, 1, s='a', transform=ax['a'].transAxes, size=20, weight='bold')
    ax['b'].text(-0.15, 1, s='b', transform=ax['b'].transAxes, size=20, weight='bold')
    ax['c'].text(-0.15, 1, s='c', transform=ax['c'].transAxes, size=20, weight='bold')
    ax['d'].text(-0.15, 1, s='d', transform=ax['d'].transAxes, size=20, weight='bold')
    fig3.show()
    fig3.savefig('frequency_vs_size_width')
    fig3.savefig('frequency_vs_size_width.eps', format='eps')
    fig3.savefig('frequency_vs_size_width.svg', format='svg')
    """
    plt.show()
