import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy,tqdm,scipy,shelve,os
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Data:
    # Constructor
    def __init__(self, inference_path=None, moments=3, downsample=1.0,normalize=False):
        self.downsample = downsample
        self.moments = moments
        self.normalize=normalize
        self.moment = self.get_moments(inference_path)
        cell = np.repeat([100,1000,10000,100000], repeats=self.moment.shape[0])
        cell = cell.reshape(cell.shape[0], 1)
        self.x = np.hstack((np.tile(self.moment,reps=(4,1)), cell))
        self.x=np.hstack((self.x,np.ones((self.x.shape[0],1))*downsample))
        metric = self.get_ci(inference_path)
        self.metric = metric
        #self.x=self.x[self.exclude,:]
        #self.metric=self.metric[self.exclude,:]
        #self.metric = torch.tensor(metric.max(axis=1))

    # Getter
    def get_moments(self, inference_path):
        self.index_list = []
        try:
            psim = np.array(scipy.io.loadmat(inference_path)['distribution'].todense())
        except:
            psim = np.array(scipy.io.loadmat(inference_path)['distribution'])
        moments = np.zeros((psim.shape[0], self.moments))
        m1=np.sum(np.arange(psim.shape[1])[None, :] * psim, axis=1)
        moments[:,0]=m1
        #moments[:, 0] = np.sum((np.arange(psim.shape[1])[None, :] -m1[:,None]) * psim, axis=1)
        moments[:, 1] = np.sum(((np.arange(psim.shape[1])[None,:] - m1[:,None]) ** 2) * psim, axis=1)
        moments[:, 2] = np.sum(((np.arange(psim.shape[1])[None,:] -m1[:,None]) ** 3) * psim,axis=1)
        return moments

    def get_ci(self, inference_path):
        all = scipy.io.loadmat(inference_path)
        data = all['CI']
        #MLE = np.tile(all['MLE'], (4, 1))
        #CI = np.zeros((self.x.shape[0], 6))
        metric = np.zeros((self.x.shape[0], 3))
        for i in range(data.shape[0]):
            metric[216000*i:216000*(i+1), :] =data[i,:,:]
        return metric

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        # getting data length

    def __len__(self):
        return self.len


if __name__=='__main__':
    torch.manual_seed(0)
    scale=True
    network_spec=['10']*10
    activation = ['relu']*(len(network_spec)-1)+['relu']
    structure=''
    if len(activation) != len(network_spec):
        print('incorrect network specification, please check')
        exit(0)
    input=['4']
    output=['3']
    normalize=True
    network_spec = input + network_spec + output
    data=[]
    for i in [0.3]:
        data.append(Data('self_infer_CI_lib_downsample_{}.mat'.format(i),downsample=i,normalize=normalize))
    X=data[0].x[:,:-1]
    Y=data[0].metric
    for i in range(1,len(data)-1):
        X=np.vstack((X,data[i].x))
        Y=np.vstack((Y,data[i].metric))
    if normalize:
        Y=(Y-Y.min(axis=0)[None,:])/((Y.max(axis=0)-Y.min(axis=0))[None,:])
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
    if os.path.exists(network_spec + structure+ '.pt'):
        model=torch.load( network_spec + structure+ '.pt')
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # train-test split of the dataset
    if output[0]=='3':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y.max(axis=1,keepdims=True), train_size=0.7, shuffle=True)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    # training parameters
    n_epochs = 20000  # number of epochs to run
    batch_size = 40000  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []
    start_time=time()
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
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
    print('time taken: {}'.format(time()-start_time))
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    fig,axs=plt.subplots(2,2)
    axs[0,0].plot(history)
    if normalize:
        axs[0,0].set_ylim(0,0.1)
    axs[0,0].set_ylabel('Mean Square Error')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_title('Prediction Error')
    fig.suptitle('network '+network_spec+structure)
    model.eval()
    y_train_eval=model(X_train).cpu().detach().numpy()
    y_test_eval=model(X_test).cpu().detach().numpy()
    axs[0,1].scatter(y_train.cpu().numpy()[:,0],y_train_eval[:,0],alpha=0.002,s=2,c='blue')
    axs[0,1].scatter(y_test.cpu().numpy()[:, 0], y_test_eval[:, 0], alpha=0.002, s=2,c='blue')
    axs[0,1].set_title('ksyn')
    axs[0,1].set_xlabel('ground truth')
    axs[0,1].set_ylabel('prediction')
    if normalize:
        axs[0,1].set_ylim(0,1)
        axs[0,1].set_xlim(0,1)
    else:
        axs[0,1].set_ylim(Y[:,0].min(),Y[:,0].max())
        axs[0, 1].set_xlim(Y[:, 0].min(), Y[:, 0].max())
    axs[1,0].scatter(y_train.cpu().numpy()[:,1],y_train_eval[:,1],alpha=0.002,s=2,c='blue')
    axs[1,0].scatter(y_test.cpu().numpy()[:, 1], y_test_eval[:, 1], alpha=0.002, s=2,c='blue')
    axs[1,0].set_title('koff')
    if normalize:
        axs[1,0].set_ylim(0,1)
        axs[1,0].set_xlim(0,1)
    else:
        axs[1,0].set_ylim(Y[:,1].min(),Y[:,1].max())
        axs[1,0].set_xlim(Y[:, 1].min(), Y[:, 1].max())
    axs[1,1].scatter(y_train.cpu().numpy()[:,2],y_train_eval[:,2],alpha=0.002,s=2,c='blue')
    axs[1,1].scatter(y_test.cpu().numpy()[:, 2], y_test_eval[:, 2], alpha=0.002, s=2,c='blue')
    axs[1,1].set_title('kon')
    if normalize:
        axs[1,1].set_ylim(0,1)
        axs[1,1].set_xlim(0,1)
    else:
        axs[1,1].set_ylim(Y[:,2].min(),Y[:,2].max())
        axs[1, 1].set_xlim(Y[:, 2].min(), Y[:, 2].max())
    fig.tight_layout()
    torch.save(model, network_spec + structure+ '.pt')
    fig.savefig(network_spec+structure+'.png')

    g=shelve.open('neural_net_training_history',writeback=True)
    g[network_spec+structure]=history
    g.close()
    """
    dummy_input = torch.randn(1, int(input), requires_grad=True).to(device)
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "test.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file   # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['moments and cell size'],  # the model's input names
                      output_names=['precision metric'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    """
