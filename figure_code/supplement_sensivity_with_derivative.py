import torch

from function3 import block_two_state_cme
import numpy as np
from function3 import compute_cme_one_gene_two_state
import matplotlib.pyplot as plt
import shelve
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

if __name__=='__main__':

    #device='cuda'
    #param=np.array([[0,5,0.01,0.046,1],[0,15,0.01,0.046,1],[0,45,0.01,0.046,1],[0,65,0.01,0.046,1],[0,100,0.01,0.046,1],[0,200,0.01,0.046,1]])
    #temp=block_two_state_cme(param, sense=True,keep_transition=True,device=device)
    """
    for index,i in enumerate(param):
        tmp=compute_cme_one_gene_two_state(i)
        plt.plot(tmp.todense().ravel())
        plt.plot(temp.batch0.distribution.todense()[index])
        plt.show()
    """
    b=shelve.open('../library_with_sense_log','r')
    singular_value=np.zeros(60*60*60)
    start=0
    for i in range(len(b.keys())):
        tmp=torch.svd(torch.swapaxes(torch.tensor(b['batch'+str(i)].S),0,1)[:,:4,:],compute_uv=False)
        tmp=tmp.S.min(axis=1).values
        singular_value[start:start+tmp.shape[0]]=tmp
        start+=tmp.shape[0]
    singular_value=singular_value.reshape(60,60,60)
    g = shelve.open('../library_300_with_sense_2', 'r')
    S_single_value = np.min(g['S_single_value'], axis=1).reshape(60, 60, 60)
    ksyn = np.linspace(-0.3, 2.3, 60)
    koff = np.linspace(-3, 3, 60)
    kon = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(kon, koff)
    fig = plt.figure(figsize=(6, 6), dpi=400)
    gs0 = GridSpec(2, 2)
    ax1 = fig.add_subplot(gs0[0, 0], projection='3d')
    ax1.contourf(X, Y, singular_value[15, :, :], zdir='z', offset=ksyn[15], levels=60, cmap='viridis_r',
                 vmin=singular_value.min(), vmax=singular_value.max())
    ax1.contourf(X, Y, singular_value[30, :, :], zdir='z', offset=ksyn[30], levels=60, cmap='viridis_r',
                 vmin=singular_value.min(), vmax=singular_value.max())
    surface = ax1.contourf(X, Y, singular_value[45, :, :], zdir='z', offset=ksyn[45], levels=60, cmap='viridis_r',
                           vmin=singular_value.min(), vmax=singular_value.max())
    ax1.set_zlim(ksyn[15], ksyn[45])
    ax1.set_zticks([0.3, 1.0, 1.5])
    ax1.view_init(azim=250)
    ax1.text2D(-0.1, 1.1, 'A', transform=ax1.transAxes, size=12, weight='bold')
    ticks = np.linspace(0, 0.05, 6)
    cbar = plt.colorbar(surface, ax=ax1, extend='both', ticks=ticks)  # format=tkr.FormatStrFormatter('%.2f'))
    cbar.formatter.set_powerlimits((0, 0))

    log_singular=np.clip(np.log10(singular_value),a_min=-9, a_max=-1)
    ax1 = fig.add_subplot(gs0[1, 0], projection='3d')
    ax1.contourf(X, Y, log_singular[15, :, :], zdir='z', offset=ksyn[15], levels=60, cmap='viridis_r',
                 vmin=log_singular.min(), vmax=log_singular.max())
    ax1.contourf(X, Y, log_singular[30, :, :], zdir='z', offset=ksyn[30], levels=60, cmap='viridis_r',
                 vmin=log_singular.min(), vmax=log_singular.max())
    surface = ax1.contourf(X, Y, log_singular[45, :, :], zdir='z', offset=ksyn[45], levels=60, cmap='viridis_r',
                           vmin=log_singular.min(), vmax=log_singular.max())
    ax1.set_zlim(ksyn[15], ksyn[45])
    ax1.set_zticks([0.3, 1.0, 1.5])
    ax1.view_init(azim=250)
    ax1.text2D(-0.1, 1.1, 'C', transform=ax1.transAxes, size=12, weight='bold')
    ticks = np.linspace(-9, -1, 9)
    cbar = plt.colorbar(surface, ax=ax1, extend='both', ticks=ticks)  # format=tkr.FormatStrFormatter('%.2f'))


    ax2 = fig.add_subplot(gs0[0, 1], projection='3d')
    surface=ax2.contourf(X, Y, S_single_value[15, :, :], zdir='z', offset=ksyn[15], levels=60, cmap='viridis_r',
                 vmin=S_single_value.min(), vmax=S_single_value.max())
    ax2.contourf(X, Y, S_single_value[30, :, :], zdir='z', offset=ksyn[30], levels=60, cmap='viridis_r',
                 vmin=S_single_value.min(), vmax=S_single_value.max())
    ax2.contourf(X, Y, S_single_value[45, :, :], zdir='z', offset=ksyn[45], levels=60, cmap='viridis_r',
                           vmin=S_single_value.min(), vmax=S_single_value.max())
    ax2.set_zlim(ksyn[15], ksyn[45])
    ax2.set_zticks([0.3, 1.0, 1.5])
    ax2.view_init(azim=250)
    ax2.text2D(-0.1, 1.1, 'B', transform=ax2.transAxes, size=12, weight='bold')
    ticks = np.linspace(0, 0.08, 9)
    cbar = plt.colorbar(surface, ax=ax2, extend='both', ticks=ticks)  # format=tkr.FormatStrFormatter('%.2f'))
    cbar.formatter.set_powerlimits((0, 0))

    log_S=np.clip(np.log10(S_single_value),a_min=-9,a_max=-1)
    ax2 = fig.add_subplot(gs0[1, 1], projection='3d')
    surface = ax2.contourf(X, Y, log_S[15], zdir='z', offset=ksyn[15], levels=60, cmap='viridis_r',
                           vmin=log_S.min(), vmax=log_S.max())
    ax2.contourf(X, Y, log_S[30], zdir='z', offset=ksyn[30], levels=60, cmap='viridis_r',vmin=log_S.min(), vmax=log_S.max())
    ax2.contourf(X, Y, log_S[45], zdir='z', offset=ksyn[45], levels=60, cmap='viridis_r',vmin=log_S.min(), vmax=log_S.max())
    ax2.set_zlim(ksyn[15], ksyn[45])
    ax2.set_zticks([0.3, 1.0, 1.5])
    ax2.view_init(azim=250)
    ax2.text2D(-0.1, 1.1, 'D', transform=ax2.transAxes, size=12, weight='bold')
    ticks = np.linspace(-9, -1, 9)
    cbar = plt.colorbar(surface, ax=ax2, extend='both', ticks=ticks)  # format=tkr.FormatStrFormatter('%.2f'))
    fig.text(0.15,0.95, 'with respect to ln('+r'$\theta$'+')',fontsize=12)
    fig.text(0.6, 0.95, 'with respect to '+ r'$\theta$',fontsize=12)
    fig.text(0.05, 0.65, 'in actual value',rotation=90,fontsize=12)
    fig.text(0.05, 0.3, 'in log10', rotation=90,fontsize=12)
    fig.savefig('comparison with different derivative')
    fig.savefig('comparison with different derivative.svg',format='svg')
    fig.savefig('comparison with different derivative.eps', format='eps')
    """
    b=shelve.open('library_300_with_sense','r')
    S=b['S']
    singular_values=np.zeros((216000,4))
    for i in tqdm(range(216000)):
        singular_values[i]=np.linalg.svd(S[1:,i,:])[1]
    print('done')
    """
