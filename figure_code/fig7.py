import matplotlib.pyplot as plt
import shelve
import numpy as np
import plotly.express as px
from function2 import find_MLE_CI

if __name__=='__main__':
    fig = plt.figure(figsize=(6.5,6.5*0.6),dpi=300)
    max_colorbar=np.array([1,1,1])
    for plot_index,index_name,path,name in zip([1,2],['a','b'],['../larsson_data/SS3_cast_UMIs_concat_infer','../nandor_data/wt.norm_infer'],['CAST c57','HUES64 WT']):
        g=shelve.open(path,writeback=True)
        ax=fig.add_subplot(1,2,plot_index,projection='3d')
        mle=np.zeros((len(g.keys()),3))
        CI=np.zeros((len(g.keys()),3))
        for index,j in enumerate(g.keys()):
            mle[index]=g[j]['MLE'].squeeze()[:3]
            #CI[index]=find_MLE_CI(g[j],cell=g[j]['cell_number']).squeeze()[:-1,2]
            CI[index]=np.log10(g[j]['CI'].squeeze()[:3,1]/g[j]['CI'].squeeze()[:3,0])
        g.close()
        CI_norm=CI/(np.array([0.7,2,2])[np.newaxis,:])
        CI_norm=np.clip(CI_norm,a_min=0,a_max=4)
        surface=ax.scatter(np.log10(mle[:,1]),np.log10(mle[:,2]),np.log10(mle[:,0]),c=CI_norm.max(axis=1))
        ax.view_init(azim=250)
        ax.set_xlabel('koff in log10',fontsize=10)
        ax.set_ylabel('kon in log10',fontsize=10)
        ax.set_zlabel('ksyn in log10',fontsize=10)
        ax.set_xticks([-3,0,3])
        ax.set_yticks([-3, 0, 3])
        ax.set_zticks([0, 1.5, 3])
        ax.set_xticklabels([-3,0,3],fontsize=10)
        ax.set_yticklabels([-3, 0, 3], fontsize=10)
        ax.set_zticklabels([0, 1.5, 3], fontsize=10)
        ax.set_title(name,fontsize=18)
        ax.text2D(-0.1,1.1,index_name,fontsize=15)
        sec=np.where(CI_norm.max(axis=1)<0.4)
        fig2 = px.scatter_3d(x=np.log10(mle[sec, 1]).ravel(),y=np.log10(mle[sec, 2]).ravel(),z=np.log10(mle[sec, 0]).ravel(),color=CI_norm.max(axis=1)[sec],title='plotly',opacity=0.8,color_continuous_scale='viridis',range_color=(0, 1))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.1,0.06, 0.8, 0.04])
    cbar_ax.axes.tick_params(labelsize=10)
    fig.colorbar(surface, cax=cbar_ax,orientation='horizontal',aspect=40)
    fig.savefig('fig7')
    fig.savefig('fig7.svg')
    print('done')

