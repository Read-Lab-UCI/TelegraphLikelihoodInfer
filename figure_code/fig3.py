import shelve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy
from function2 import *
from sklearn.preprocessing import normalize

data=scipy.io.loadmat('self_infer_CI_lib_downsample_1.0.mat')
data2=scipy.io.loadmat('self_infer_CI_lib_downsample_0.3.mat')
catelogue=generate_binom_catelogue(data['distribution'].shape[1]-1,0.3)
ksyn=np.linspace(-0.3,2.3,60)
koff=np.linspace(-3,3,60)
kon=np.linspace(-3,3,60)
X,Y=np.meshgrid(kon,koff)
metric_index=0
save_name=['APM','standard uncertainty','Log_base_threshold_Upper_Lower_standard_error'][metric_index]
metric_name=['Alternative Precision Metric','Precision Metric',r'$Log_{threshold}(Upper/Lower)$'+' standard error'][metric_index]
metric=np.zeros((data['CI'].shape[0],data['CI'].shape[1],4,3))
metric[:,:,:3,0]=np.log10(data['CI'][:,:,:,1]/data['CI'][:,:,:,0])/np.array([0.7,2,2])[None,None,:]
metric[:,:,:3,1]=np.log10((data['CI'][:,:,:,1]-data['CI'][:,:,:,0])/data['MLE'][:,np.newaxis,:])
metric[:,:,:3,2]=metric[:,:,:3,0]+metric[:,:,:3,1]
metric[:,:,3,:]=metric[:,:,:3,:].max(axis=2)
metric[:,:,:3,0]=np.clip(metric[:,:,:3,0],a_min=0,a_max=4)
metric[:,:,:3,1]=np.clip(metric[:,:,:3,1],a_min=-6,a_max=6)


metric2=np.zeros((data2['CI'].shape[0],data2['CI'].shape[1],4,3))
metric2[:,:,:3,0]=np.log10(data2['CI'][:,:,:,1]/data2['CI'][:,:,:,0])/np.array([0.7,2,2])[None,None,:]
metric2[:,:,:3,1]=np.log10((data2['CI'][:,:,:,1]-data2['CI'][:,:,:,0])/data2['MLE'][:,np.newaxis,:])
metric2[:,:,:3,2]=metric2[:,:,:3,0]+metric2[:,:,:3,1]
metric2[:,:,3,:]=metric2[:,:,:3,:].max(axis=2)
metric2[:,:,:3,0]=np.clip(metric2[:,:,:3,0],a_min=0,a_max=4)
metric2[:,:,:3,1]=np.clip(metric2[:,:,:3,1],a_min=-6,a_max=6)
#metric=(metric-metric.min(axis=(0,1)))/(metric.max(axis=(0,1))-metric.min(axis=(0,1)))
#metric_min=np.array([-5.3,-4.25,-4.67,0])
#metric_max=np.array([5.2,12,12,0])
#metric_min=metric.min(axis=(0,1))
#metric_max=metric.max(axis=(0,1))
#metric=(metric-metric_min[None,None,:])/(metric_max[None,None,:]-metric_min[None,None,:])
metric=metric[:,:,:,metric_index]
metric2=metric2[:,:,:,metric_index]
cell=2
min_c=np.round(metric[:,:,3].min()).astype('int')
max_c=np.round(metric[:,:,3].max()).astype('int')

metric=metric.reshape(60,60,60,4,4)
metric2=metric2.reshape(60,60,60,4,4)
fig=plt.figure(figsize=(12,12),dpi=400)
gs0 = GridSpec(5,4, height_ratios=[1,2.2,2.2,0.2,1])
#fig2,axes=plt.subplots(2,4,height_ratios=[2,1])
#axes[0,0]=plt.subplot(2,3,3, projection='3d')
levels = np.linspace(min_c, max_c, 60)
color=np.linspace(0,0.85,7)
dummy_scale=np.linspace(min_c,max_c,60)
dummy_scale=np.tile(dummy_scale,reps=(60,1))
colormap='viridis'
scatter_color='bone'
for i,j in zip(range(4),['ksyn','koff','kon','max of three']):
    ax1 = fig.add_subplot(gs0[1,i],projection='3d')
    metric[[5,17,29],-1,-1,cell,i]=min_c
    metric[[5,17,29],-1,-2,cell,i]=max_c
    surface=ax1.contourf(X,Y,metric[5,:,:,cell,i],zdir='z', offset=ksyn[5],cmap=colormap,levels=levels)
    ax1.contourf(X,Y,metric[17,:,:,cell,i],zdir='z', offset=ksyn[17],cmap=colormap,levels=levels)
    ax1.contourf(X,Y,metric[29,:,:,cell,i],zdir='z', offset=ksyn[29],cmap=colormap,levels=levels)
    ax1.set_zlim(ksyn[5],ksyn[29])
    ax1.view_init(azim=250)
    #cbar = fig.colorbar(surface, ax=ax1, extend='both')
    #cbar.ax.set_ylim(0, 1)
    if i==0:
        ax1.set_ylabel('log10(koff)',fontsize=13)
        ax1.set_xlabel('log10(kon)',fontsize=13)
        ax1.set_zlabel('log10(ksyn)',fontsize=13)
        ax1.set_zticks([ksyn[5],ksyn[17],ksyn[29]])
        ax1.zaxis.set_major_formatter('{x:3<2.2f}')
    else:
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax1.zaxis.set_ticklabels([])
    #ax1.set_title('Metric of '+j)
    if i==0:
        ax1.scatter([kon[10]] * 7,[koff[10]] * 7, ksyn[np.arange(5, 30, 4)], c=color[::-1], cmap=scatter_color,alpha=1)
    elif i==1:
        ax1.scatter([kon[10]] * 7, koff[np.arange(10, 50, 6)],  [ksyn[29]] * 7, c=color, cmap=scatter_color,alpha=1)
    elif i==2:
        ax1.scatter(kon[np.arange(10, 50, 6)], [koff[10]] * 7, [ksyn[29]] * 7, c=color, cmap=scatter_color,alpha=1)
    if i==3:
        index=np.empty(shape=(3,0))
        #ax1.scatter([koff[10]]*7,[kon[10]]*7,ksyn[np.arange(5,30,4)],c=color[::-1],cmap='bone')
        index=np.hstack((index,np.array([range(5,30,4)[::-1],[10]*7,[10]*7])))
        #ax1.scatter(koff[np.arange(10, 50, 6)], [kon[10]] * 7, [ksyn[29]] * 7, c=color,cmap='bone')
        index = np.hstack((index,np.array([[29] * 7, range(10, 50, 6), [10] * 7])))
        #ax1.scatter([koff[10]] * 7, kon[np.arange(10, 50, 6)], [ksyn[29]] * 7, c=color,cmap='bone')
        index = np.hstack((index,np.array([[29] * 7, [10] * 7, range(10, 50, 6)])))
        ax1.scatter(kon[np.arange(10, 50, 6)],koff[np.arange(10, 50, 6)], [ksyn[29]] * 7, c=color,cmap=scatter_color,alpha=1)
        index = np.hstack((index,np.array([[29] * 7, range(10, 50, 6), range(10, 50, 6)])))
        index=np.array(index,int)
        #c=ax1.pcolor(metric[29,:,:,cell,i],vmin=min_c,vmax=max_c)
        #fig.colorbar(c,ax=ax1)
#ax1 = fig.add_subplot(gs0[1,0],projection='3d')
surface=ax1.contourf(X,Y,dummy_scale,zdir='z', offset=ksyn[59],cmap=colormap)
surface.set_clim(min_c, max_c)
ax1.set_xticks(ticks=[])
ax1.set_yticks(ticks=[])
ax1.set_zticks(ticks=[])
index=np.ravel_multi_index(index,dims=(60,60,60))
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.bone(color))

for i,j in zip(range(4),['varying ksyn','varying koff','varying kon','along diagonal']):
    ax2 = fig.add_subplot(gs0[0,i])
    try:
        psim=data['distribution'].todense()[index[i*7:(i+1)*7],:]
    except:
        psim = data['distribution'][index[i * 7:(i + 1) * 7], :]
    cumsum=np.cumsum(psim,axis=1)
    last_index=np.argmax(cumsum>0.999,axis=0).max()
    ax2.plot(psim[:,:20].T)
    ax2.set_title(j,fontsize=16)
    ax2.set_xlabel('mRNA copy',fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(0,0.4)
    if i==0:
        ax2.set_ylabel('Probability',fontsize=13)

for i,j in zip(range(4),['varying ksyn','varying koff','varying kon','along diagonal']):
    ax2 = fig.add_subplot(gs0[4,i])
    try:
        psim=data2['distribution'].todense()[index[i*7:(i+1)*7],:]

    except:
        psim = data2['distribution'][index[i * 7:(i + 1) * 7], :]
    cumsum=np.cumsum(psim,axis=1)
    last_index=np.argmax(cumsum>0.999,axis=0).max()
    ax2.plot(psim[:,:20].T)
    ax2.set_title(j,fontsize=16)
    ax2.set_xlabel('mRNA copy',fontsize=13)
    ax2.tick_params(labelsize=12)
    ax2.set_ylim(0,0.4)
    if i==0:
        ax2.set_ylabel('Probability',fontsize=13)

for i,j in zip(range(4),['ksyn','koff','kon','max of three']):
    ax2 = fig.add_subplot(gs0[2,i],projection='3d')
    metric2[[5,17,29],-1,-1,cell,i]=min_c
    metric2[[5,17,29],-1,-2,cell,i]=max_c
    surface=ax2.contourf(X,Y,metric2[5,:,:,cell,i],zdir='z', offset=ksyn[5],cmap=colormap,levels=levels)
    ax2.contourf(X,Y,metric2[17,:,:,cell,i],zdir='z', offset=ksyn[17],cmap=colormap,levels=levels)
    ax2.contourf(X,Y,metric2[29,:,:,cell,i],zdir='z', offset=ksyn[29],cmap=colormap,levels=levels)
    ax2.set_zlim(ksyn[5],ksyn[29])
    #   ax2.tick_params(labelsize=12)
    ax2.view_init(azim=250)
    #cbar = fig.colorbar(surface, ax=ax1, extend='both')
    #cbar.ax.set_ylim(0, 1)
    if i==0:
        ax2.set_ylabel('log10(koff)',fontsize=13)
        ax2.set_xlabel('log10(kon)',fontsize=13)
        ax2.set_zlabel('log10(ksyn)',fontsize=13)
        ax2.set_zticks([ksyn[5],ksyn[17],ksyn[29]])
        ax2.zaxis.set_major_formatter('{x:3<2.2f}')
    else:
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax2.zaxis.set_ticklabels([])
    #ax1.set_title('Metric of '+j)
    if i==0:
        ax2.scatter([kon[10]] * 7, [koff[10]] * 7, ksyn[np.arange(5, 30, 4)], c=color[::-1], cmap=scatter_color,alpha=1)
    elif i==1:
        ax2.scatter([kon[10]] * 7, koff[np.arange(10, 50, 6)], [ksyn[29]] * 7, c=color, cmap=scatter_color,alpha=1)
    elif i==2:
        ax2.scatter(kon[np.arange(10, 50, 6)], [koff[10]] * 7, [ksyn[29]] * 7, c=color, cmap=scatter_color,alpha=1)
    if i==3:
        index=np.empty(shape=(3,0))
        #ax1.scatter([koff[10]]*7,[kon[10]]*7,ksyn[np.arange(5,30,4)],c=color[::-1],cmap='bone')
        index=np.hstack((index,np.array([range(5,30,4)[::-1],[10]*7,[10]*7])))
        #ax1.scatter(koff[np.arange(10, 50, 6)], [kon[10]] * 7, [ksyn[29]] * 7, c=color,cmap='bone')
        index = np.hstack((index,np.array([[29] * 7, range(10, 50, 6), [10] * 7])))
        #ax1.scatter([koff[10]] * 7, kon[np.arange(10, 50, 6)], [ksyn[29]] * 7, c=color,cmap='bone')
        index = np.hstack((index,np.array([[29] * 7, [10] * 7, range(10, 50, 6)])))
        ax2.scatter(kon[np.arange(10, 50, 6)], koff[np.arange(10, 50, 6)], [ksyn[29]] * 7, c=color,cmap=scatter_color,alpha=1)
        index = np.hstack((index,np.array([[29] * 7, range(10, 50, 6), range(10, 50, 6)])))
        index=np.array(index,int)
        #c=ax1.pcolor(metric[29,:,:,cell,i],vmin=min_c,vmax=max_c)
        #fig.colorbar(c,ax=ax1)
surface=ax2.contourf(X,Y,dummy_scale,zdir='z', offset=ksyn[59],cmap=colormap)
surface.set_clim(min_c, max_c)
ax2.set_xticks(ticks=[])
ax2.set_yticks(ticks=[])
ax2.set_zticks(ticks=[])
cax = plt.axes((0.9, 0.32, 0.02, 0.3))
cbar=fig.colorbar(surface,cax=cax,extend='both')
cbar.ax.set_ylim(min_c,max_c)
cbar.ax.tick_params(labelsize=14)
index=np.ravel_multi_index(index,dims=(60,60,60))
fig.text(0.05, 0.9, 'A', ha='center', fontsize=20,weight='bold')
fig.text(0.05, 0.71, 'B', ha='center', fontsize=20,weight='bold')
fig.text(0.05, 0.45, 'C', ha='center', fontsize=20,weight='bold')
fig.text(0.05, 0.25, 'D', ha='center', fontsize=20,weight='bold')
fig.text(0.97, 0.32, metric_name, ha='center',rotation=90, fontsize=20)
fig.text(0.18, 0.49, 'ksyn', ha='center', fontsize=18)
fig.text(0.4, 0.49, 'koff', ha='center', fontsize=18)
fig.text(0.6, 0.49, 'kon', ha='center', fontsize=18)
fig.text(0.82, 0.49, 'max of three', ha='center', fontsize=18)
"""
h=shelve.open('../library_300_with_sense','r')
S_single_value=h['S_single_value']
for i in range(4):
    ax1 = fig.add_subplot(gs0[1,i],projection='3d')
    if i==3:
        temp=np.min(S_single_value[:,1:4],axis=1)
    else:
        temp=S_single_value[:,i+1]
    hist=np.histogram(temp,bins=20)
    most=np.cumsum(hist[0]/temp.shape[0])
    index=np.argmin(most<0.95)
    levels = np.linspace(temp.min(), hist[1][index+1], 60)
    ax1.contourf(X, Y, np.reshape(temp, (60, 60, 60))[5, :, :], offset=ksyn[5], alpha=0.7, levels=levels)
    ax1.contourf(X, Y, np.reshape(temp, (60, 60, 60))[17, :, :], offset=ksyn[17], alpha=0.7, levels=levels)
    ax1.contourf(X, Y, np.reshape(temp, (60, 60, 60))[29, :, :], offset=ksyn[29], alpha=0.7, levels=levels)
    ax1.set_zlim(ksyn[5], ksyn[29])
    ax1.view_init(azim=250)
"""
#fig.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.95,wspace=0.4,hspace=0)
fig.tight_layout()
#fig.suptitle(metric_name+' 10K cells',fontsize=20)
fig.savefig(save_name+'_10K_cell')
#fig.savefig(save_name+'_10K_cell.svg')
fig.show()
"""
fig.suptitle('Precision Metric with ratio VS Sensitivity Singular Value',fontsize=20)
fig.savefig('Precision Metric with ratio VS Sensitivity Singular Value')
"""
print('done')




