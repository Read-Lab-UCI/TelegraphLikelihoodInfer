import matplotlib.pyplot as plt
from function import *
from tqdm import tqdm
import shelve,os
from multiprocessing import Pool,cpu_count


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

def find_CI_2(profile_dict,cutoff=1.92,cell=0):
    width_array=[]
    for i in range(profile_dict['histogram'].shape[0]):
        temp_width=[]
        for index,j in enumerate(['ksyn','koff','kon']):
            tmp=profile_dict[j]['max_like'][i,:].ravel()
            tmp=tmp-tmp.min()
            if cell:
                tmp=tmp*cell
                #plt.plot(tmp)
                #plt.ylim(0,2.2)
                #plt.show()
            else:
                tmp=tmp*profile_dict['cell_number']
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
            try:
                lower_int = interp1d(tmp[:min_index + 1], profile_dict[j]['parameter'][i, :min_index + 1, index])
                lower_bound=lower_int(cutoff)
            except:
                lower_bound=profile_dict[j]['parameter'][i, 0, index]
            try:
                upper_int = interp1d(tmp[min_index:], profile_dict[j]['parameter'][i, min_index:, index])
                upper_bound=upper_int(cutoff)
            except:
                upper_bound = profile_dict[j]['parameter'][i, -1, index]
            width=upper_bound-lower_bound
            temp_width.append([lower_bound,upper_bound,width])
        width_array.append(temp_width)
    return np.array(width_array)


def find_CI(profile_dict,cutoff=1.92,cell=0,infer_factor=3):
    width_array=[]
    for i in range(profile_dict['histogram'].shape[0]):
        temp_width=[]
        for index,j in enumerate(['ksyn','koff','kon']):
            tmp=profile_dict[j]['max_like'][i,:].ravel()
            tmp=tmp-tmp.min()
            if cell:
                tmp=tmp*cell
            else:
                tmp=tmp*profile_dict['cell_number']
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
                lower_int=interp1d(profile_dict[j]['parameter'][i,:min_index+1,index],tmp[:min_index+1],fill_value=(tmp[0],tmp[min_index]),bounds_error=False)
                param=10**np.linspace(np.log10(profile_dict[j]['parameter'][i,0,index]),np.log10(profile_dict[j]['parameter'][i,min_index,index]),profile_dict[j]['parameter'][i,:min_index,index].shape[0]*infer_factor+1)
                lower_profile=lower_int(param)
                lower_bound=param[np.where(lower_profile<cutoff)[0][0]]
            if min_index==len(profile_dict[j]['parameter'][i,:,index])-1:
                upper_bound = profile_dict[j]['parameter'][i, -1, index]
            else:
                upper_int=interp1d(profile_dict[j]['parameter'][i,min_index:,index],tmp[min_index:],fill_value=(tmp[min_index],tmp[-1]),bounds_error=False)
                param=10**np.linspace(np.log10(profile_dict[j]['parameter'][i,min_index,index]),np.log10(profile_dict[j]['parameter'][i,-1,index]),profile_dict[j]['parameter'][i,min_index:,index].shape[0]*infer_factor+1)
                upper_profile=upper_int(param)
                upper_bound=param[np.where(upper_profile<cutoff)[0][-1]]

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
            width=np.clip(upper_bound-lower_bound,a_min=10**-4,a_max=upper_bound)
            temp_width.append([lower_bound,upper_bound,width,width/profile_dict['MLE'][i,index],width/(profile_dict[j]['parameter'][i,-1,index]-profile_dict[j]['parameter'][i,0,index])])
        width_array.append(temp_width)
    width_array=np.array(width_array)
    return width_array

def get_ci(shelve_name,cutoff=1.92,normalize=True):
    shelve_path='self_infer/library_300_infer_batch_'+str(shelve_name)
    try:
        g=shelve.open(shelve_path,writeback=True)
    except:
        return [False,0]
    MLE=np.zeros((len(list(g.keys())),3))
    result = np.zeros((len(list(g.keys())), 3,g[list(g.keys())[0]]['histogram'].shape[0], 3, 5))
    if 'CI with 1000 cells' not in g[list(g.keys())[0]].keys():
        for iteration,i in enumerate(list(g.keys())):
            MLE[iteration,:]=g[str(i)]['MLE'][0,:]
            result[iteration,0,:,:,:]=g[str(i)]['CI with 1000 cells']
            result[iteration,1,:,:,:]=g[str(i)]['CI with 10000 cells']
            result[iteration, 2, :,:, :] = g[str(i)]['CI with 100000 cells']
            #if normalize:
            #    result[iteration, 0,:, :, 3] = result[iteration, 0,:, :, 2] / g[i]['MLE']
            #    result[iteration, 1,:, :, 3] = result[iteration, 0,:, :, 2] / g[i]['MLE']
            #    result[iteration, 1,:, :, 3] = result[iteration, 0,:, :, 2] / g[i]['MLE']
            #else:
            #    result[iteration, 0,:, :, 3] = result[iteration,0,:,:,2] / [g[i]['ksyn']['parameter'][0][-1, 0] - g[i]['ksyn']['parameter'][0][0, 0],g[i]['koff']['parameter'][0][-1, 1] - g[i]['koff']['parameter'][0][0, 1],g[i]['kon']['parameter'][0][-1, 2] - g[i]['kon']['parameter'][0][0, 2]]
            #    result[iteration, 1,:, :, 3] = result[iteration, 1, :, 2] / [
            #        g[i]['ksyn']['parameter'][0][-1, 0] - g[i]['ksyn']['parameter'][0][0, 0],
            #        g[i]['koff']['parameter'][0][-1, 1] - g[i]['koff']['parameter'][0][0, 1],
            #        g[i]['kon']['parameter'][0][-1, 2] - g[i]['kon']['parameter'][0][0, 2]]
            #    result[iteration, 2,:, :, 3] = result[iteration, 2,:, :, 2] / [
            #        g[i]['ksyn']['parameter'][0][-1, 0] - g[i]['ksyn']['parameter'][0][0, 0],
            #        g[i]['koff']['parameter'][0][-1, 1] - g[i]['koff']['parameter'][0][0, 1],
            #        g[i]['kon']['parameter'][0][-1, 2] - g[i]['kon']['parameter'][0][0, 2]]

        return [True, MLE,result]
    for iteration,i in enumerate(list(g.keys())):
        g[i]['MLE'] = find_MLE(g[i])
        MLE[iteration,:]=g[i]['MLE'].squeeze()[0]
        for index,cell_number in enumerate([1000,10000,100000]):
            width_array=np.array(find_CI(g[i],1.92,cell_number))
            g[i]['CI with {} cells'.format(cell_number)]=width_array
            result[iteration,index,:,:,:]=width_array
    return [True,MLE,result]

if __name__=='__main__':
    parallel=False
    normalize=True
    test=False
    if test:
        flag,test_ci=get_ci('test_100000',normalize=normalize)
        param = np.array([[0, 3.5, 23, 10, 1], [0, 3.5, 0.23, 10, 1], [0, 3.5, 0.23, 0.1, 1], [0, 3.5, 0.23, 0.01, 1],
                         [0, 3.5, 23, 100, 1], [0, 3.5, 2.3, 100, 1], [0, 3.5, 2.3, 10, 1], [0, 3.5, 2.3, 1, 1],
                        [0, 10, 23, 10, 1], [0, 10, 0.23, 10, 1], [0, 10, 0.23, 0.1, 1], [0, 10, 0.23, 0.01, 1],
                         [0, 10, 23, 100, 1], [0, 10, 2.3, 100, 1], [0, 10, 2.3, 10, 1], [0, 10, 2.3, 1, 1]])
        histogram=block_two_state_cme(param,percentage=False).batch0
        g = shelve.open('test_100000')
        for index,i in enumerate(list(g.keys())):
            fig,ax=plt.subplots(1,4,figsize=(20,5))
            ax[0].plot(np.log10(g[i]['ksyn']['parameter'][0][:,0]),(g[i]['ksyn']['max_like'][0]-g[i]['ksyn']['max_like'][0].min())*1000,label='1K cells',c='b')
            ax[0].plot(np.log10(g[i]['ksyn']['parameter'][0][:, 0]),(g[i]['ksyn']['max_like'][0] - g[i]['ksyn']['max_like'][0].min()) * 10000,label='10K cells',c='orange')
            ax[0].plot(np.log10(g[i]['ksyn']['parameter'][0][:, 0]),(g[i]['ksyn']['max_like'][0] - g[i]['ksyn']['max_like'][0].min()) * 100000,label='100K cells',c='green')
            ax[0].axvline(np.log10(param[index,1]),c='r',label='ground truth')
            ax[0].scatter([np.log10(g[i]['MLE'][0])]*10,np.linspace(0,2.2,10),c='black',marker='*',label='inferred',s=60,zorder=10)
            ax[0].legend()
            ax[0].set_ylim(0,2.2)
            ax[0].set_ylabel('negative log likelihood')
            ax[0].set_xlabel('ksyn')
            ax[1].plot(np.log10(g[i]['koff']['parameter'][0][:, 1]),(g[i]['koff']['max_like'][0] - g[i]['koff']['max_like'][0].min()) * 1000,c='b')
            ax[1].plot(np.log10(g[i]['koff']['parameter'][0][:, 1]),(g[i]['koff']['max_like'][0] - g[i]['koff']['max_like'][0].min()) * 10000,c='orange')
            ax[1].plot(np.log10(g[i]['koff']['parameter'][0][:, 1]),(g[i]['koff']['max_like'][0] - g[i]['koff']['max_like'][0].min()) * 100000,c='green')
            ax[1].axvline(np.log10(param[index, 2]), c='r', label='ground truth')
            ax[1].scatter([np.log10(g[i]['MLE'][1])]*10,np.linspace(0,2.2,10),c='black',marker='*',label='inferred',s=60,zorder=10)
            ax[1].set_ylim(0, 2.2)
            ax[1].set_xlabel('koff')
            ax[2].plot(np.log10(g[i]['kon']['parameter'][0][:, 2]),(g[i]['kon']['max_like'][0] - g[i]['kon']['max_like'][0].min()) * 1000,c='b')
            ax[2].plot(np.log10(g[i]['kon']['parameter'][0][:, 2]),(g[i]['kon']['max_like'][0] - g[i]['kon']['max_like'][0].min()) * 10000,c='orange')
            ax[2].plot(np.log10(g[i]['kon']['parameter'][0][:, 2]),(g[i]['kon']['max_like'][0] - g[i]['kon']['max_like'][0].min()) * 100000,c='green')
            ax[2].axvline(np.log10(param[index, 3]), c='r', label='ground truth')
            ax[2].scatter([np.log10(g[i]['MLE'][2])]*10, np.linspace(0,2.2,10), c='black', marker='*', label='inferred',s=60,zorder=10)
            ax[2].set_ylim(0, 2.2)
            ax[2].set_xlabel('kon')
            ax[3].plot(histogram.v0.todense()[index,:20],label='gene inactive')
            ax[3].plot(histogram.va.todense()[index,:20],label='gene active')
            ax[3].plot(histogram.distribution.todense()[index, :20],label='overall')
            ax[3].legend()
            ax[3].set_xlabel('mRNA copy number')
            ax[3].set_ylabel('probability')
            fig.suptitle('ksyn:{}, koff:{}, kon:{}'.format(param[index,1],param[index,2],param[index,3]))
            fig.savefig('Ncell/'+i + '.png')
        exit()
    index=np.arange(0,216000,30)
    param = shelve.open('library_300')['parameter']
    save_dict={'parameter':param[:,1:4]}
    MLE_save=np.zeros((216000,3))
    CI_MLE=np.zeros((216000,3,3))
    CI_range=np.zeros((216000,3,3))
    CI_all=np.zeros((216000,3,3,5))
    if parallel:
        pool=Pool(cpu_count()-2)
        with pool:
            result=list(pool.imap(get_ci,index,normalize=normalize))
    else:
        result=[]
        matrix=[]
        fig1, ax1 = plt.subplots(1,3,figsize=(15,5))
        fig2, ax2 = plt.subplots(1,3,figsize=(15,5))
        fig3, ax3 = plt.subplots(1,3,figsize=(15,5))
        fig4, ax4 = plt.subplots(1,3,figsize=(15,5))
        size=10
        alpha=0.1
        for i in tqdm(index):
            flag,MLE,temp_result=get_ci(i,normalize=normalize)
            temp_result=temp_result.squeeze()
            location=temp_result[:,:,:,3].argmax(axis=2)
            CI_MLE[i:i + temp_result.shape[0], 0, :] = temp_result[:, 0,location[:,0],:][:,0,[0,1,3]]
            CI_MLE[i:i + temp_result.shape[0], 1, :] = temp_result[:, 1, location[:, 1], :][:, 0, [0, 1, 3]]
            CI_MLE[i:i + temp_result.shape[0], 2, :] = temp_result[:, 2, location[:, 2], :][:, 0, [0, 1, 3]]
            location_n=temp_result[:,:,:,4].argmax(axis=2)
            CI_range[i:i + temp_result.shape[0], 0, :] = temp_result[:, 0,location_n[:,0],:][:,0,[0,1,4]]
            CI_range[i:i + temp_result.shape[0], 1, :] = temp_result[:, 1, location_n[:, 1], :][:, 0, [0, 1, 4]]
            CI_range[i:i + temp_result.shape[0], 2, :] = temp_result[:, 2, location_n[:, 2], :][:, 0, [0, 1, 4]]
            MLE_save[i:i+30,:]=MLE
            CI_all[i:i+temp_result.shape[0],:,:,:]

            if flag:
                n_param = temp_result.shape[0]
                result.append(temp_result)
                tmp_param=param[i:i+n_param,:]

                try:
                    matrix1 = np.log10(np.max(tmp_param[:, 2:4], axis=1) / tmp_param[:, 1])
                    matrix2 = np.log10(np.max(tmp_param[:, 2:4], axis=1) / tmp_param[:, 1] * tmp_param[:, 2:4].sum(axis=1) / tmp_param[:, 2])
                    matrix3 = np.log10(np.max(tmp_param[:, 2:4], axis=1) / tmp_param[:, 1] * tmp_param[:, 2:4].sum(axis=1) / tmp_param[:, 3])
                    matrix4 = np.log10(np.max(tmp_param[:, 2:4], axis=1) / tmp_param[:, 1] * tmp_param[:, 2:4].sum(axis=1))
                    #ax1[2].scatter(matrix1,result[-1][:,0,0,2],c='b',alpha=0.01,s=size)
                    #ax1[3].scatter(matrix1, result[-1][:, 0, 1, 2],c='b',alpha=0.01,s=size)
                    #ax1[4].scatter(matrix1, result[-1][:, 0, 2, 2],c='b',alpha=0.01,s=size)
                    if normalize:
                        max_unc=np.log10(temp_result[:,:,:,3].max(axis=2))
                        min_unc=np.log10(temp_result[:,:,:,3].min(axis=2))
                    else:
                        max_unc = temp_result[:, :, :, 4].max(axis=2)
                        min_unc = temp_result[:, :, :, 4].min(axis=2)
                    ax1[0].scatter(matrix1, max_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax1[1].scatter(matrix1, min_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax1[2].scatter(matrix1, result[-1][:, 1, 0, 2], c='y',alpha=0.01,s=size)
                    #ax1[3].scatter(matrix1, result[-1][:, 1, 1, 2], c='y',alpha=0.01,s=size)
                    #ax1[4].scatter(matrix1, result[-1][:, 1, 2, 2], c='y',alpha=0.01,s=size)
                    ax1[1].scatter(matrix1, max_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax1[1].scatter(matrix1, min_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax1[2].scatter(matrix1, result[-1][:, 2, 0, 2], c='r',alpha=0.01,s=size)
                    #ax1[3].scatter(matrix1, result[-1][:, 2, 1, 2], c='r',alpha=0.01,s=size)
                    #ax1[4].scatter(matrix1, result[-1][:, 2, 2, 2], c='r',alpha=0.01,s=size)
                    ax1[2].scatter(matrix1, max_unc[:,2], c='r', alpha=alpha,s=size)
                    #[1].scatter(matrix1, min_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax2[2].scatter(matrix2, result[-1][:, 0, 0, 2], c='b',alpha=0.01,s=size)
                    #ax2[3].scatter(matrix2, result[-1][:, 0, 1, 2], c='b',alpha=0.01,s=size)
                    #ax2[4].scatter(matrix2, result[-1][:, 0, 2, 2], c='b',alpha=0.01,s=size)
                    ax2[0].scatter(matrix2, max_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax2[1].scatter(matrix2, min_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax2[2].scatter(matrix2, result[-1][:, 1, 0, 2], c='y',alpha=0.01,s=size)
                    #ax2[3].scatter(matrix2, result[-1][:, 1, 1, 2], c='y',alpha=0.01,s=size)
                    #ax2[4].scatter(matrix2, result[-1][:, 1, 2, 2], c='y',alpha=0.01,s=size)
                    ax2[1].scatter(matrix2, max_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax2[1].scatter(matrix2, min_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax2[2].scatter(matrix2, result[-1][:, 2, 0, 2], c='r',alpha=0.01,s=size)
                    #ax2[3].scatter(matrix2, result[-1][:, 2, 1, 2], c='r',alpha=0.01,s=size)
                    #ax2[4].scatter(matrix2, result[-1][:, 2, 2, 2], c='r',alpha=0.01,s=size)
                    ax2[2].scatter(matrix2, max_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax2[1].scatter(matrix2, min_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax3[2].scatter(matrix3, result[-1][:, 0, 0, 2], c='b',alpha=0.01,s=size)
                    #ax3[3].scatter(matrix3, result[-1][:, 0, 1, 2], c='b',alpha=0.01,s=size)
                    #ax3[4].scatter(matrix3, result[-1][:, 0, 2, 2], c='b',alpha=0.01,s=size)
                    ax3[0].scatter(matrix3, max_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax3[1].scatter(matrix3, min_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax3[2].scatter(matrix3, result[-1][:, 1, 0, 2], c='y',alpha=0.01,s=size)
                    #ax3[3].scatter(matrix3, result[-1][:, 1, 1, 2], c='y',alpha=0.01,s=size)
                    #ax3[4].scatter(matrix3, result[-1][:, 1, 2, 2], c='y',alpha=0.01,s=size)
                    ax3[1].scatter(matrix3, max_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax3[1].scatter(matrix3, min_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax3[2].scatter(matrix3, result[-1][:, 2, 0, 2], c='r', alpha=0.01,s=size)
                    #ax3[3].scatter(matrix3, result[-1][:, 2, 1, 2], c='r',alpha=0.01,s=size)
                    #ax3[4].scatter(matrix3, result[-1][:, 2, 2, 2], c='r',alpha=0.01,s=size)
                    ax3[2].scatter(matrix3, max_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax3[1].scatter(matrix3, min_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax4[2].scatter(matrix4, result[-1][:, 0, 0, 2], c='b', alpha=0.01,s=size)
                    #ax4[3].scatter(matrix4, result[-1][:, 0, 1, 2], c='b',alpha=0.01,s=size)
                    #ax4[4].scatter(matrix4, result[-1][:, 0, 2, 2], c='b',alpha=0.01,s=size)
                    ax4[0].scatter(matrix4, max_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax4[1].scatter(matrix4, min_unc[:,0], c='b', alpha=alpha,s=size)
                    #ax4[2].scatter(matrix4, result[-1][:, 1, 0, 2], c='y', alpha=0.01,s=size)
                    #ax4[3].scatter(matrix4, result[-1][:, 1, 1, 2], c='y',alpha=0.01,s=size)
                    #ax4[4].scatter(matrix4, result[-1][:, 1, 2, 2], c='y',alpha=0.01,s=size)
                    ax4[1].scatter(matrix4, max_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax4[1].scatter(matrix4, min_unc[:,1], c='y', alpha=alpha,s=size)
                    #ax4[2].scatter(matrix4, result[-1][:, 2, 0, 2], c='r',alpha=0.01,s=size)
                    #ax4[3].scatter(matrix4, result[-1][:, 2, 1, 2], c='r',alpha=0.01,s=size)
                    #ax4[4].scatter(matrix4, result[-1][:, 2, 2, 2], c='r',alpha=0.01,s=size)
                    ax4[2].scatter(matrix4, max_unc[:,2], c='r', alpha=alpha,s=size)
                    #ax4[1].scatter(matrix4, min_unc[:,2], c='r', alpha=alpha,s=size)
                except:
                    print(i)
        if normalize:
            s='maximum confidence interval normalized by MLE in log10'
        else:
            s='maximum confidence interval normalized by parameter range'
        ax1[0].set_title('1000 cells')
        ax1[1].set_title('10000 cells')
        ax1[2].set_title('100000 cells')
        #ax1[2].set_title('ksyn')
        #ax1[3].set_title('koff')
        #ax1[4].set_title('kon')
        ax1[0].set_ylabel(s)
        #ax1[2].set_ylabel('confidence interval/MLE in log10')
        ax1[0].legend()
        fig1.text(0.5, 0.04, 'log10(max(kon,koff)/ksyn)', ha='center')
        ax2[0].set_title('1000 cells')
        ax2[1].set_title('10000 cells')
        ax2[2].set_title('100000 cells')
        #ax2[2].set_title('ksyn')
        #ax2[3].set_title('koff')
        #ax2[4].set_title('kon')
        ax2[0].set_ylabel(s)
        #ax2[2].set_ylabel('confidence interval/MLE in log10')
        ax2[0].legend()
        fig2.text(0.5, 0.04, 'log10(max(kon,koff)/ksyn/Poff)', ha='center')
        ax3[0].set_title('1000 cells')
        ax3[1].set_title('10000 cells')
        ax3[2].set_title('100000 cells')
        #ax3[2].set_title('ksyn')
        #ax3[3].set_title('koff')
        #ax3[4].set_title('kon')
        ax3[0].set_ylabel(s)
        #ax3[2].set_ylabel('confidence interval/MLE in log10')
        ax3[0].legend()
        fig3.text(0.5, 0.04, 'log10(max(kon,koff)/ksyn/Pon)', ha='center')
        ax4[0].set_title('1000 cells')
        ax4[1].set_title('10000 cells')
        ax4[2].set_title('100000 cells')
        #ax4[2].set_title('ksyn')
        #ax4[3].set_title('koff')
        #ax4[4].set_title('kon')
        ax4[0].set_ylabel(s)
        #ax4[2].set_ylabel('confidence interval/MLE in log10')
        ax4[0].legend()
        fig4.text(0.5, 0.04, 'log10(max(kon,koff)/ksyn/(kon+koff)', ha='center')
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
