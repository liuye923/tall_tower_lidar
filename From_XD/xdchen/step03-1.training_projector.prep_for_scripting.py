#!/usr/bin/env python
import numpy as np
import xarray as xr
import scipy.io as sio
import datetime as dt
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


rootdir = '/raid1/chen423/puget/data/papers/PS-SCL/'
datadir = '/raid1/chen423/puget/data/reanalysis/ERA5/SCL/'


# ## 1. construct model


class decoder(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, train, projection_dim, in_features):
        super(decoder, self).__init__()
        self.train = train
        self.n_features = in_features
        self.projection_dim = projection_dim
        
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )        
        
        # freeze the parameters that are not re-trained
        if self.train==False:
            for param in self.projector.parameters():
                param.requires_grad = False
    

    def forward(self, x_i, x_j):
        z_i = self.projector(x_i.type(torch.FloatTensor).cuda())
        z_j = self.projector(x_j.type(torch.FloatTensor).cuda())

        del x_i, x_j
        return z_i, z_j



def save_model(outfile, model):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), outfile)
    else:
        torch.save(model.state_dict(), outfile)


# ## 2. construct train and testing dataloaders


class TrainDataset(Dataset):

    def __init__(self, root_dir, my_vstring, my_length, my_nt_sp=180):
        self.root_dir = root_dir
        self.nt_sp = my_nt_sp
        self.length = my_length
        eyear = 1981 + self.length-1
        infile = self.root_dir + 'ResNet18_output/%s_input/v%d/agg_%dyear/ResNet18_output.v%d.%d-%d.mat' % (humidvar, 
                                                                                                            my_vstring, self.length,
                                                                                                            my_vstring, 1981, eyear)
        print('loading training data from file %s ...' % infile)
        self.inds = sio.loadmat(infile)['ResNet18output']

    def __len__(self):
        nt = self.inds.shape[0]-self.nt_sp
        return int(nt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pick the corresponding "future" hour using a decaying linear probability
        dt = int(np.random.choice(np.arange(1,self.nt_sp+1), p=np.linspace(2/self.nt_sp, 0, self.nt_sp)))
        
        sample_raw = self.inds[idx]
        sample_pair = self.inds[int(idx+dt)]
        

        return sample_raw, sample_pair



class TestDataset(Dataset):

    def __init__(self, root_dir, my_vstring, my_length, my_nt_sp=180):
        self.root_dir = root_dir
        self.nt_sp = my_nt_sp
        self.length = my_length
        if self.length==10:
            eyear = 1991 + self.length-1
            infile = self.root_dir + 'ResNet18_output/%s_input/v%d/agg_%dyear/ResNet18_output.v%d.%d-%d.mat' % (humidvar, 
                                                                                                                my_vstring, self.length,
                                                                                                                my_vstring, 1991, eyear)
        print('loading test data from file %s ...' % infile)
        self.inds = sio.loadmat(infile)['ResNet18output']

    def __len__(self):
        nt = self.inds.shape[0]-self.nt_sp
        return int(nt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pick the corresponding "future" hour using a decaying linear probability
        dt = int(np.random.choice(np.arange(1,self.nt_sp+1), p=np.linspace(2/self.nt_sp, 0, self.nt_sp)))
        
        sample_raw = self.inds[idx]
        sample_pair = self.inds[int(idx+dt)]
        

        return sample_raw, sample_pair


# ## 3. construct loss


def SCLloss(my_x, my_y, my_temperature=0.5):
    '''
    my_x and my_y has a one-to-one pair. So there are in total N*N pairs. In these N*N, the diagonal pairs are positive,
     and the rest are negative. So we want to maximum diagonal while suppressing the rest.
    '''
    ns = my_x.shape[0]
    # use broadcasting to achieve pairwise cos. Note my_y.t() operation and dimension handling
    cos_matrix = torch.nn.functional.cosine_similarity(my_x[:,:,None], my_y.t()[None,:,:])/my_temperature
    similarity_matrix = torch.exp(cos_matrix)


    loss = torch.tensor([0.0], requires_grad=True)
    for i in np.arange(ns):
        loss = loss -1*torch.log(similarity_matrix[i,i]/(torch.sum(similarity_matrix[i,:])-similarity_matrix[i,i]))
        loss = loss -1*torch.log(similarity_matrix[i,i]/(torch.sum(similarity_matrix[:,i])-similarity_matrix[i,i]))
    
    loss = loss/(2*ns)
    
    return loss


# ## 4. train the model

def train(train_loader, model, optimizer, my_epoch):
    
    loss_epoch = 0
    loss_history = []
    
    for step, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        x_i = data[0].type(torch.FloatTensor).cuda(non_blocking=True)
        x_j = data[1].type(torch.FloatTensor).cuda(non_blocking=True)
        del data
        z_i, z_j = decoder(x_i, x_j)
        del x_i, x_j
        loss = SCLloss(z_i.cpu(), z_j.cpu())
        del z_i, z_j
        loss.backward()
        optimizer.step()
        
        if step%10==0:
            print('    training loss (epoch %3d, batch #%3d): %.6f' % (my_epoch, step, loss.item()))
        
        loss_epoch += loss.item()
        loss_history.append(loss.item())
        
    loss_epoch = loss_epoch/(step+1)
    
    print('  training loss (epoch %3d): %.6f' % (my_epoch, loss_epoch))
    return loss_epoch, loss_history



def test(test_loader, model, my_epoch):
    
    loss_epoch = 0
    
    for step, data in enumerate(test_loader):
        
        optimizer.zero_grad()
        x_i = data[0].type(torch.FloatTensor).cuda(non_blocking=True)
        x_j = data[1].type(torch.FloatTensor).cuda(non_blocking=True)
        del data
        z_i, z_j = decoder(x_i, x_j)
        del x_i, x_j
        loss = SCLloss(z_i.cpu(), z_j.cpu())
        del z_i, z_j

        loss_epoch += loss.item()
    
    loss_epoch = loss_epoch/(step+1)
        
    print('  testing loss (epoch %3d): %.6f' % (my_epoch, loss_epoch))
    
    return loss_epoch




my_length = 10     # years of data used for training
my_nt_sp = 96      # hours that will be considered as the same system. I.e., duration of random sampling in constructing positive pair
batch_size = 256     #
lr = 3e-04

nepochs = 100        # how many epochs we want to run
sepoch = 101          # starting epoch. Used for model re-loading

humidvar = 'RH'



model_lib_dir = rootdir + 'model_lib/%s_input/training_%dyear_%dnt_%dbatch/lr%.0e/' % (humidvar, my_length, my_nt_sp, batch_size, lr)    
 
print(model_lib_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('will be using device ', device)



in_features = 512
projection_dim = 256

decoder = decoder(True, projection_dim=projection_dim, in_features=in_features)

if sepoch==1:
    param_file = model_lib_dir + '../par.init.tar'
    print('load params from %s ' % param_file)
    decoder.load_state_dict(torch.load(param_file, map_location='cpu'))
elif sepoch>1:
    param_file = model_lib_dir + 'par.ubuntu.epoch%03d.tar' % (sepoch - 1)
    print('load params from %s ' % param_file)
    decoder.load_state_dict(torch.load(param_file, map_location='cpu'))

decoder = decoder.to(device)



train_dataset = TrainDataset(root_dir=rootdir, my_vstring=1, my_length=my_length, my_nt_sp=my_nt_sp)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TestDataset(root_dir=rootdir, my_vstring=1, my_length=my_length, my_nt_sp=my_nt_sp)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

nstep = len(train_dataloader)
print('nstep = %d' % nstep)

full_loss_hist_training = np.zeros(nstep*nepochs)
full_loss_training = np.zeros(nepochs)
full_loss_testing = np.zeros(nepochs)

optimizer = torch.optim.Adam(decoder.projector.parameters(), lr=lr)  # TODO: LARS

count = 0
for epoch in np.arange(sepoch, sepoch+nepochs):
    
    print(epoch, dt.datetime.now())
    
    subloss_train, subloss_train_full = train(train_dataloader, decoder, optimizer, epoch)
    subloss_test = test(test_dataloader, decoder, epoch)
    
    full_loss_hist_training[(count*nstep):(count*nstep+nstep)] = subloss_train_full
    full_loss_training[count] = subloss_train
    full_loss_testing[count] = subloss_test
    
    count += 1
    if epoch % 1 ==0:
        
        param_file = model_lib_dir + 'par.ubuntu.epoch%03d.tar' % (epoch)
        print('write params to %s ...' % param_file)
        torch.save(decoder.state_dict(), param_file)
        
        loss_file = model_lib_dir + 'loss.ubuntu.epoch%03d.mat' % (epoch)
        print('write loss to %s ...' % loss_file)
        sio.savemat(loss_file, {'subloss_train':subloss_train, 'subloss_testing':subloss_test,
                                'subloss_train_full':subloss_train_full, 'epoch':epoch, 'batch_size':batch_size, 'lr':lr})


