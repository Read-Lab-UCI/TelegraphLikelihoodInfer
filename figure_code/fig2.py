import numpy as np
import matplotlib.pyplot as plt
import shelve
from function2 import compute_cme_one_gene_two_state, parallel_likelihood, find_MLE_CI, Opobj
from multiprocessing import Pool,cpu_count
from functools import partial
from tqdm import tqdm


if __name__=='__main__':
    fig,ax=plt.subplot_mosaic([['dist1','dist1','dist1','2dim1','2dim1','2dim1'],
                        ['dist1','dist1','dist1','2dim1','2dim1','2dim1'],
                        ['ksyn1','ksyn1','koff1','koff1','kon1','kon1'],
                        ['dist2', 'dist2', 'dist2', '2dim2', '2dim2', '2dim2'],
                        ['dist2', 'dist2', 'dist2', '2dim2', '2dim2', '2dim2'],
                        ['ksyn2', 'ksyn2', 'koff2', 'koff2', 'kon2', 'kon2'],#],figsize=(10,10),dpi=400)
                        ['z', 'z', 'z', 'z', 'z', 'z']],figsize=(10,10),dpi=300,height_ratios=[1,1,1,1,1,1,0.1])
    p = shelve.open('../library_300', 'r')['downsample_1.0'].todense()
    parameter = shelve.open('../library_300', 'r')['parameter']
    param1=np.array([0,10,0.1,0.05,1])
    param2=np.array([0,1,0.01,0.05,1])
    repeat=200
    burst = np.zeros((216000, 3))
    burst[:, 1] = parameter[:, 1] / parameter[:, 2]
    burst[:, 2] = parameter[:,2]*parameter[:, 3]/(parameter[:,3]+parameter[:,2])
    g=shelve.open('figure2_data')
    keys=list(g.keys())
    g.close()
    all_result = []
    chosen=np.where(parameter[:,2]/parameter[:,3]>100)[0]
    for i,label,position in zip(['1','2'],[['A','B','C','',''],['D','E','F','','']],[0.64,0.17]):
        psim = np.abs(compute_cme_one_gene_two_state(vars()['param' + i]).todense().ravel())
        psim = psim / psim.sum()
        psim_c = np.zeros(p.shape[1])
        psim_c[:psim.shape[0]] = psim
        psim_c = np.clip(psim_c, a_min=10 ** -11, a_max=1)
        likelihood = -200 * np.matmul(np.log(np.clip(p, a_min=10 ** -11, a_max=1)), psim_c.T)
        if str(vars()['param'+i]) not in keys:
            sample_data=np.random.choice(np.arange(psim_c.shape[0]),size=10000,p=psim_c)
            sample_hist=np.zeros((repeat,psim_c.shape[0]))
            for j in range(0,repeat):
                sample_hist[j,:]=np.histogram(sample_data[np.random.choice(np.arange(sample_data.shape[0]),200)],bins=np.arange(psim_c.shape[0]+1))[0]
            sample_hist=sample_hist/200
            result_0 = parallel_likelihood([psim_c, 200, 'test'], psim_path='../library_300', optimize=True,
                                           full_optimize=True)
            pool=Pool(cpu_count()-2)
            with pool:
                result=list(pool.imap(partial(parallel_likelihood, psim_path='../library_300',optimize=True), zip(sample_hist,[200]*(repeat),['test']*(repeat))))
            result.insert(0,result_0)
            g = shelve.open('figure2_data', writeback=True)
            g[str(vars()['param' + i])] = result
            g.close()
        else:
            g = shelve.open('figure2_data', 'r')
            result=g[str(vars()['param'+i])]
            g.close()
        #result = parallel_likelihood(pexp_list=[psim, 200, 'test1'], psim_path='../library_300', repeat=200, optimize=True)
        burst[:,0]=likelihood-likelihood.min()

        #heatmap=ax['2dim' + i].tricontourf(np.log10(burst[:, 1]), np.log10(burst[:, 2]), burst[:, 0], levels=60)
        sort_index=burst[:,0].argsort()[::-1]
        heatmap =ax['2dim' + i].scatter(np.log10(burst[sort_index, 1]), np.log10(burst[sort_index, 2]), c=burst[sort_index, 0],norm='log')
        ax['2dim' + i].text(-0.1, 1, label[1], transform=ax['2dim' + i].transAxes, size=20, weight='bold')
        last_index=np.argmin(np.cumsum(psim_c)<0.999)
        last_index=(last_index//5+1)*5
        if i=='1':
            ticklabels=np.linspace(0,last_index,5).astype('int')
        else:
            ticklabels=np.arange(last_index)
        ax['dist' + i].set_xticks(ticklabels)
        ax['dist' + i].set_xticklabels(ticklabels, fontsize=15)
        for j in range(1,repeat+1):
            if j != repeat+1:
                ax['dist'+i].plot(result[j]['histogram'][0,:last_index].T,alpha=0.05,color='red',linewidth=1)
            ax['2dim' + i].scatter(np.log10(result[j]['MLE'][:, -1]), np.log10(result[j]['MLE'][:, -2]*result[j]['MLE'][:,1]/(result[j]['MLE'][:,-2]+result[j]['MLE'][:,1])), alpha=0.05,color='red')
            ax['ksyn' + i].vlines(np.log10(result[j]['MLE'][:, 0]), ymin=-0.2, ymax=2.2, alpha=0.05, colors='red',linewidth=1)
            ax['koff' + i].vlines(np.log10(result[j]['MLE'][:, 1]), ymin=-0.2, ymax=2.2, alpha=0.05, colors='red',linewidth=1)
            ax['kon'  + i].vlines(np.log10(result[j]['MLE'][:, 2]), ymin=-0.2, ymax=2.2, alpha=0.05,colors='red',linewidth=1)
        ax['ksyn' + i].vlines(np.log10(result[j]['MLE'][:, 0]), ymin=0, ymax=2.2, alpha=0.05, colors='red',linewidth=1,label='replicate MLE')
        ax['dist'+i].plot(result[-1]['histogram'][0, :last_index].T, alpha=0.05, color='red', linewidth=1,label='sample_replicate')
        ax['dist'+i].plot(result[0]['histogram'][0,:last_index].ravel(),alpha=1,color='black',label='ground truth',marker='v',markersize=8)
        ax['dist'+i].set_xlabel('mRNA copy number',fontsize=15)
        ax['dist'+i].set_ylabel('probability',fontsize=15)
        ax['dist' + i].text(-0.2, 1, label[0], transform=ax['dist' + i].transAxes, size=20, weight='bold')
        ticklabels=np.round(np.linspace(0,np.round(np.clip(result[0]['histogram'].max()*2,a_min=10**-11,a_max=0.6),decimals=1),5),decimals=2)
        ax['dist'+i].set_yticks(ticklabels)
        ax['dist'+i].set_yticklabels(ticklabels,fontsize=15)
        ax['dist' + i].set_ylim(0, ticklabels[-1])
        leg=ax['dist'+i].legend(fontsize=15)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        ax['2dim'+i].scatter(np.log10(vars()['param'+i][1]/vars()['param'+i][2]),np.log10(vars()['param'+i][3]*vars()['param'+i][2]/(vars()['param'+i][3]+vars()['param'+i][2])),color='black',marker='v',s=70)
        ax['2dim'+i].scatter(np.log10(result[0]['MLE'][0,-1]),np.log10(result[0]['MLE'][0,-2]*result[0]['MLE'][0,1]/(result[0]['MLE'][0,-2]+result[0]['MLE'][0,1])),color='saddlebrown',marker='*',s=70)
        ax['2dim'+i].set_xlabel('log10(burst size)',fontsize=15)
        ax['2dim'+i].set_ylabel('log10(burst frequency)',fontsize=15)
        ax['2dim' + i].set_xticks([-2,0,2,4])
        ax['2dim' + i].set_xticklabels([-2,0,2,4], fontsize=15)
        ax['2dim' + i].set_yticks([-3,0,3])
        ax['2dim' + i].set_yticklabels([-3,0,3], fontsize=15)
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_title('PL',fontsize=15)

        ax['ksyn'+i].axvline(np.log10(vars()['param'+i][1]),ymin=-0.2,ymax=2.2,color='black',label='ground truth')
        ax['ksyn'+i].plot(np.log10(result[0]['ksyn']['parameter'][0,:,0]),(result[0]['ksyn']['max_like'][0,:]-result[0]['ksyn']['max_like'][0,:].min())*200,color='blue',label='PL')
        ax['ksyn'+i].scatter([np.log10(result[0]['MLE'][0, 0])]*4, np.linspace(0,2.2,4), marker='*', color='saddlebrown',zorder=10,label='MLE')
        ax['ksyn'+i].set_ylim(-0.2,2.2)
        ax['ksyn'+i].axhline(y=1.92, color='green', linestyle='-')
        ax['ksyn'+i].set_xlim(-0.3,2.3)
        ax['ksyn'+i].set_ylabel('PL',fontsize=15)
        ax['ksyn' + i].set_yticks([0, 1.92])
        ax['ksyn' + i].set_yticklabels([0, 1.92], fontsize=15)
        fig.text(0.015,position,label[2],fontsize=20,weight='bold')
        #leg=ax['ksyn'+i].legend(fontsize=15,bbox_to_anchor=(0.5, 0.05),ncol=4,fancybox=False,frameon=False)
        #for lh in leg.legendHandles:
        #    lh.set_alpha(1)

        ax['koff'+i].plot(np.log10(result[0]['koff']['parameter'][0,:,1]),(result[0]['koff']['max_like'][0,:]-result[0]['koff']['max_like'][0,:].min())*200,color='blue')
        ax['koff'+i].axvline(np.log10(vars()['param'+i][2]),ymin=-0.2,ymax=2.2,color='black',label='ground truth')
        ax['koff'+i].scatter([np.log10(result[0]['MLE'][0, 1])]*4, np.linspace(0,2.2,4), marker='*', color='saddlebrown',zorder=10)
        ax['koff'+i].set_ylim(-0.2,2.2)
        ax['koff' + i].axhline(y=1.92, color='green', linestyle='-')
        ax['koff'+i].set_xlim(-3,3)
        ax['koff'+i].set_yticks([])
        #ax['koff' + i].set_yticks([0, 2])
        #ax['koff' + i].set_yticklabels([0, 2], fontsize=18)
        fig.text(0.35, position, label[3], fontsize=20, weight='bold')

        ax['kon'+i].axvline(np.log10(vars()['param'+i][3]),ymin=-0.2,ymax=2.2,color='black',label='ground truth')
        ax['kon'+i].plot(np.log10(result[0]['kon']['parameter'][0,:,2]),(result[0]['kon']['max_like'][0,:]-result[0]['kon']['max_like'][0,:].min())*200,color='blue')
        ax['kon'+i].scatter([np.log10(result[0]['MLE'][0,2])]*4,np.linspace(0,2.2,4),marker='*',color='saddlebrown',zorder=10)
        ax['kon'+i].set_ylim(-0.2,2.2)
        ax['kon' + i].axhline(y=1.92, color='green', linestyle='-')
        ax['kon'+i].set_xlim(-3,3)
        ax['kon'+i].set_yticks([])
        ax['koff' + i].set_xticks([-2, 0, 2])
        ax['koff' + i].set_xticklabels([-2, 0, 2], fontsize=15)
        ax['kon' + i].set_xticks([-2, 0, 2])
        ax['kon' + i].set_xticklabels([-2, 0, 2], fontsize=15)
        ax['ksyn' + i].set_xticks([0, 1, 2])
        ax['ksyn' + i].set_xticklabels([0, 1, 2], fontsize=15)
        if i=='2':
            ax['ksyn' + i].set_xlabel('ksyn in log10', fontsize=18)
            ax['kon' + i].set_xlabel('kon in log10', fontsize=18)
            ax['koff' + i].set_xlabel('koff in log10', fontsize=18)

        fig.text(0.68, position, label[4], fontsize=20, weight='bold')
    handles,labels=ax['ksyn'+i].get_legend_handles_labels()
    leg=fig.legend(handles,labels, bbox_to_anchor=(0.5, 0.02),loc='center',ncol=4,fontsize=18,frameon=False)
    leg.legendHandles[0].set_alpha(1)
    ax['z'].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(hspace=1, wspace=0.8)
    fig.show()
    fig.savefig('fig2')
    fig.savefig('fig2.svg',format='svg')
    fig.savefig('fig2.eps',format='eps')
    print('done')