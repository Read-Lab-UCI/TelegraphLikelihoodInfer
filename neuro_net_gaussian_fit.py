import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy,tqdm,scipy,shelve,os
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.interpolate import interp1d
from joblib import dump, load

def get_m2_m3(psim):
    index=np.argmin(psim.cumsum()<0.999)
    index=index+1
    d=interp1d(np.linspace(0,1,index),psim[:index])
    x=np.linspace(0,1,21)
    y=d(x)
    mean=(x*y).sum()
    m2=((x-mean)**2*y).sum()
    m3=((x-mean)**3*y).sum()
    return m2,m3

if __name__=='__main__':
    gaussian_fit=scipy.io.loadmat('figure_code/gaussian_fit_with_statistics.mat')['gaussian']
    #poisson_fit=scipy.io.loadmat('figure_code/poisson_fit.mat2')['poisson']
    downsample=0.3
    scale=False
    psim=shelve.open('library_300','r')['downsample_'+str(downsample)].todense()
    moments=np.zeros((216000,4))
    moments = np.zeros((psim.shape[0], 4))
    m1=np.sum(np.arange(psim.shape[1])[None,:]*psim,axis=1)
    moments[:, 0] = np.sum(np.arange(psim.shape[1])[None,:] * psim, axis=1)
    moments[:, 1] = np.sum(((np.arange(psim.shape[1])[None,:] - m1[:,None]) ** 2) * psim, axis=1)
    moments[:, 2] = np.sum(((np.arange(psim.shape[1])[None,:] - m1[:,None]) ** 3) * psim, axis=1)
    #moments[:,2]=np.sign(moments[:,2])*np.log10(moments[:,2])
    moments[:, 3] = np.sum(((np.arange(psim.shape[1])[None,:] - m1[:,None]) ** 4) * psim, axis=1)
    data = np.zeros((4, 216000, 3))
    g = shelve.open('self_infer/downsample_0.3/library_300_infer', 'r')
    for i in tqdm(range(216000)):
        # data[:,i,:]=find_MLE_CI2(g[str(i)],self_infer=True).squeeze()[:,:,2]
        data[:, i, :] = g[str(i)]['CI'].squeeze()[:, :, 2]
    g.close()
    metric = data
    metric=metric/np.array([0.7,2,2])[None,None,:]
    X=np.zeros((216000*4,6))
    X[:,:3]=np.tile(moments[:,:3],reps=(4,1))
    X[:,3]=np.repeat(np.array([2,3,4,5]),repeats=216000)
    #X[:, 4] = np.log10(X[:, 1] / X[:, 0])
    sign=np.sign(X[:,2])
    X[:, 5] = np.log10(X[:, 1] / X[:, 0])
    X[:, 1]=np.log10(X[:,1])
    X[:, 2] = np.log10(np.abs(X[:, 2]))
    X[:, 4] = sign*(X[:, 2] - X[:, 1])
    X[:, 2] = sign*X[:,2]
    Y=metric.reshape(216000*4,3)
    torch.manual_seed(0)
    network_spec = ['10'] * 10
    activation = ['relu'] * (len(network_spec) - 1) + ['relu']
    structure = ''
    if len(activation) != len(network_spec):
        print('incorrect network specification, please check')
        exit(0)
    input = [str(X.shape[1])]
    output = ['3']
    normalize = False
    network_spec = input + network_spec + output
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module=[]
    for i,j in zip(range(len(network_spec)),activation):
        module.append(nn.Linear(int(network_spec[i]),int(network_spec[i+1])))
        if j=='relu':
            structure=structure+'r'
            module.append(nn.ReLU())
        elif j=='sigmoid':
            structure=structure+'s'
            module.append(nn.Sigmoid())
    module.append(nn.Linear(int(network_spec[-2]),int(network_spec[-1])))
    model=nn.Sequential(*module)
    # loss function and optimizer
    if normalize:
        network_spec=network_spec+['normalize']
    if scale:
        network_spec=network_spec+['scaled']
    network_spec=''.join(network_spec)
    if os.path.exists(network_spec + structure+ '_third_order_moment_over_variance_log_2.pt'):
        model=torch.load( network_spec + structure+ '_third_order_moment_over_variance_log_2.pt')
    loss_fn = nn.MSELoss()
    if scale:
        if os.path.exists(network_spec + structure + '_third_order_moment_over_variance_log_2.bin'):
            scaler = load(network_spec + structure + '_third_order_moment_over_variance_log_2.bin')
        else:
            scaler = StandardScaler()
            scaler.fit(X)
        X=scaler.transform(X)
    # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if output[0] == '3':
        X_train, X_test, y_train, y_test,index_train,index_test = train_test_split(X, Y,np.arange(X.shape[0]), train_size=0.7, shuffle=True)
    else:
        X_train, X_test, y_train, y_test,index_train,index_test = train_test_split(X, Y.max(axis=1, keepdims=True),np.arange(X.shape[0]), train_size=0.7,
                                                            shuffle=True)
    model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    n_epochs = 60000  # number of epochs to run
    batch_size = 20000  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    g = shelve.open('neural_net_training_third_order_moment_over_variance_log_2')
    if network_spec+structure in g.keys():
        history=g[network_spec + structure]
    else:
        history = []
    g.close()
    start_time = time()
    for epoch in range(n_epochs):
        model.train()
        with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    print('time taken: {}'.format(time() - start_time))
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.4f" % best_mse)
    print("RMSE: %.4f" % np.sqrt(best_mse))
    model.eval()
    Y_eval = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().detach().numpy()
    fig, axs = plt.subplot_mosaic([['error','error','error','false','false','false'],
                                   ['ksyn','ksyn','koff','koff','kon','kon']],height_ratios=(3,2),figsize=(10,9),layout='constrained')
    axs['error'].plot(history)
    if normalize:
        axs['error'].set_ylim(0, 0.1)
    history[0]=1
    axs['error'].set_ylim(np.min(history)*0.8,0.1)
    axs['error'].set_ylabel('Mean Square Error',fontsize=20)
    axs['error'].set_xlabel('Epoch',fontsize=20)
    #axs['error'].set_title('Prediction Error',fontsize=20)
    axs['error'].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    #fig.suptitle('neuro network prediction',fontsize=25)
    axs['ksyn'].scatter(Y[:, 0], Y_eval[:, 0], alpha=0.002, s=2, c='blue')
    axs['ksyn'].set_title('ksyn',fontsize=20)
    axs['koff'].set_xlabel('Ground truth',fontsize=20)
    axs['ksyn'].set_ylabel('Prediction',fontsize=20)
    if normalize:
        axs['kysn'].set_ylim(0, 1)
        axs['ksyn'].set_xlim(0, 1)
    else:
        axs['ksyn'].set_ylim(Y[:, 0].min(), Y[:, 0].max())
        axs['ksyn'].set_xlim(Y[:, 0].min(), Y[:, 0].max())
    axs['koff'].scatter(Y[:, 1], Y_eval[:, 1], alpha=0.002, s=2, c='blue')
    axs['koff'].set_title('koff',fontsize=20)
    if normalize:
        axs['koff'].set_ylim(0, 1)
        axs['koff'].set_xlim(0, 1)
    else:
        axs['koff'].set_ylim(Y[:, 1].min(), Y[:, 1].max())
        axs['koff'].set_xlim(Y[:, 1].min(), Y[:, 1].max())
    axs['kon'].scatter(Y[:, 2], Y_eval[:, 2], alpha=0.002, s=2, c='blue')
    axs['kon'].set_title('kon',fontsize=20)
    if normalize:
        axs['kon'].set_ylim(0, 1)
        axs['kon'].set_xlim(0, 1)
    else:
        axs['kon'].set_ylim(Y[:, 2].min(), Y[:, 2].max())
        axs['kon'].set_xlim(Y[:, 2].min(), Y[:, 2].max())
    axs['error'].tick_params(axis='both', labelsize=15)
    axs['ksyn'].tick_params(axis='both', labelsize=15)
    axs['koff'].tick_params(axis='both', labelsize=15)
    axs['kon'].tick_params(axis='both',  labelsize=15)
    axs['koff'].set_yticks([])
    axs['kon'].set_yticks([])
    false_positive=np.zeros((3,3))
    low,high=1,2.5
    false_positive[2,2]=np.where((Y_eval.max(axis=1)>high)&(Y_eval.max(axis=1)>high)&(Y.max(axis=1)>high)&(Y.max(axis=1)>high))[0].shape[0]
    false_positive[2,1]=np.where((Y_eval.max(axis=1)>low)&(Y_eval.max(axis=1)<high)&(Y.max(axis=1)>high)&(Y.max(axis=1)>high))[0].shape[0]
    false_positive[2,0]=np.where((Y_eval.max(axis=1)<low)&(Y_eval.max(axis=1)<high)&(Y.max(axis=1)>high)&(Y.max(axis=1)>high))[0].shape[0]
    false_positive[1,2]=np.where((Y_eval.max(axis=1)>high)&(Y_eval.max(axis=1)>high)&(Y.max(axis=1)<high)&(Y.max(axis=1)>low))[0].shape[0]
    false_positive[1,1]=np.where((Y_eval.max(axis=1)>low)&(Y_eval.max(axis=1)<high)&(Y.max(axis=1)<high)&(Y.max(axis=1)>low))[0].shape[0]
    false_positive[1,0]=np.where((Y_eval.max(axis=1)<low)&(Y_eval.max(axis=1)<low)&(Y.max(axis=1)<high)&(Y.max(axis=1)>low))[0].shape[0]
    false_positive[0,2]=np.where((Y_eval.max(axis=1)>high)&(Y_eval.max(axis=1)>high)&(Y.max(axis=1)<low)&(Y.max(axis=1)<low))[0].shape[0]
    false_positive[0,1]=np.where((Y_eval.max(axis=1)>low)&(Y_eval.max(axis=1)<high)&(Y.max(axis=1)<low)&(Y.max(axis=1)<low))[0].shape[0]
    false_positive[0,0]=np.where((Y_eval.max(axis=1)<low)&(Y_eval.max(axis=1)<low)&(Y.max(axis=1)<low)&(Y.max(axis=1)<low))[0].shape[0]
    false_positive_ratio=false_positive/false_positive.sum(axis=0)[None,:]
    axs['false'].imshow(false_positive_ratio, extent=[0, 6, 0, 6])
    axs['false'].set_xticks([1, 3, 5])
    axs['false'].set_yticks([1.2, 3.65, 5.2])
    axs['false'].tick_params(axis='both', which='both', length=0)
    axs['false'].set_yticklabels(['x>{}'.format(high), '{}<x<{}'.format(low,high), 'x<{}'.format(low)],rotation=90,fontsize=15)
    axs['false'].set_xticklabels(['x<{}'.format(low), '{}<x<{}'.format(low,high), 'x>{}'.format(high)],fontsize=15)
    axs['false'].set_ylabel('Ground truth APM', fontsize=20)
    axs['false'].set_xlabel('Prediction APM', fontsize=20)
    axs['false'].text(1, 1, str(int(false_positive[2, 0])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(1, 3, str(int(false_positive[2, 1])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(1, 5, str(int(false_positive[2, 2])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(3, 1, str(int(false_positive[1, 0])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(3, 3, str(int(false_positive[1, 1])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(3, 5, str(int(false_positive[1, 2])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(5, 1, str(int(false_positive[0, 0])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(5, 3, str(int(false_positive[0, 1])), horizontalalignment='center', verticalalignment='center', size=15)
    axs['false'].text(5, 5, str(int(false_positive[0, 2])), horizontalalignment='center', verticalalignment='center', size=15)
    fig.text(0.52, 0.95, 'B',  size=20, weight='bold')
    fig.text(0.02, 0.95, 'A', size=20, weight='bold')
    fig.text(0.02, 0.4, 'C',  size=20, weight='bold')
    fig.text(0.38, 0.4, 'D',  size=20, weight='bold')
    fig.text(0.7, 0.4, 'E',  size=20, weight='bold')
    fig.tight_layout()
    fig.show()
    """
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(history)
    if normalize:
        axs[0, 0].set_ylim(0, 0.01)
    history[0]=1
    axs[0, 0].set_ylim(np.min(history)*0.8,0.01)
    axs[0, 0].set_ylabel('Mean Square Error')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_title('Prediction Error')
    fig.suptitle('network ' + network_spec + structure)
    model.eval()
    y_train_eval = model(X_train).cpu().detach().numpy()
    y_test_eval = model(X_test).cpu().detach().numpy()
    axs[0, 1].scatter(y_train.cpu().numpy()[:, 0], y_train_eval[:, 0], alpha=0.002, s=2, c='blue')
    axs[0, 1].scatter(y_test.cpu().numpy()[:, 0], y_test_eval[:, 0], alpha=0.002, s=2, c='blue')
    axs[0, 1].set_title('ksyn')
    axs[0, 1].set_xlabel('ground truth')
    axs[0, 1].set_ylabel('prediction')
    if normalize:
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].set_xlim(0, 1)
    else:
        axs[0, 1].set_ylim(Y[:, 0].min(), Y[:, 0].max())
        axs[0, 1].set_xlim(Y[:, 0].min(), Y[:, 0].max())
    axs[1, 0].scatter(y_train.cpu().numpy()[:, 1], y_train_eval[:, 1], alpha=0.002, s=2, c='blue')
    axs[1, 0].scatter(y_test.cpu().numpy()[:, 1], y_test_eval[:, 1], alpha=0.002, s=2, c='blue')
    axs[1, 0].set_title('koff')
    if normalize:
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_xlim(0, 1)
    else:
        axs[1, 0].set_ylim(Y[:, 1].min(), Y[:, 1].max())
        axs[1, 0].set_xlim(Y[:, 1].min(), Y[:, 1].max())
    axs[1, 1].scatter(y_train.cpu().numpy()[:, 2], y_train_eval[:, 2], alpha=0.002, s=2, c='blue')
    axs[1, 1].scatter(y_test.cpu().numpy()[:, 2], y_test_eval[:, 2], alpha=0.002, s=2, c='blue')
    axs[1, 1].set_title('kon')
    if normalize:
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_xlim(0, 1)
    else:
        axs[1, 1].set_ylim(Y[:, 2].min(), Y[:, 2].max())
        axs[1, 1].set_xlim(Y[:, 2].min(), Y[:, 2].max())
    fig.tight_layout()
    """
    torch.save(model, network_spec + structure+'_third_order_moment_over_variance_log_2' + '.pt')
    if scale:
        dump(scaler, network_spec+structure+'_third_order_moment_over_variance_log_2.bin', compress=True)
    fig.savefig(network_spec + structure + '_third_order_moment_over_variance_log_2.png')

    g = shelve.open('neural_net_training_history_third_order_moment_over_variance_log_2', writeback=True)
    g[network_spec + structure] = history
    g.close()




