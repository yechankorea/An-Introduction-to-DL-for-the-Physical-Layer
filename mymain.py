# An introduction to Deep learning for the Physical

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import argparse
import copy
from autoencoder import AutoEncoder

parser = argparse.ArgumentParser(
                    prog = 'Introduction DL for the Physical Layer',
                    description = 'customized code',
                    epilog = 'Text at the bottom of help')
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--snr_db',type=float,default=7,help='training snr dB')
parser.add_argument('--train_num_msgs',type=float,default=8000,help='training num msgs')
parser.add_argument('--test_num_msgs',type=float,default=50000,help='testing num msgs')
parser.add_argument('--test', type=int, default=1)      # 0: train / 1 :test
parser.add_argument('--plot', type=int, default=1)      # 0: constellation / 1: ber curve
parser.add_argument('--weights', type=str,default= 'MY_MODEL_M=4_n=2_TrainSNR=7.pth')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 파라미터 세팅=======================================================================
n = args.n
k = args.k
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
lr = args.lr
SNR_dB = args.snr_db
train_num_msgs = args.train_num_msgs
test_num_msgs = args.test_num_msgs

SNR = round(10**(SNR_dB/10),4)     # snr_db -> snr 소수점 4자리까지
M = 2**k    # input dimension
R = k/n   # channel rate

MODEL_PATH = f"MY_MODEL_M={M}_n={n}_TrainSNR={SNR_dB}.pth"

#Dataset=================================================================================================
train_labels = torch.randint(0,M,(train_num_msgs,)).long() # msg 0~M-1 까지의 메세지를 num_msgs만큼 생성 ex) [1,0,2,0,1,3,0,.......,1,2,3]
train_data = F.one_hot(train_labels, num_classes = M)

test_labels = torch.randint(0,M,(test_num_msgs,)).long() # msg 0~M-1 까지의 메세지를 num_msgs만큼 생성 ex) [1,0,2,0,1,3,0,.......,1,2,3]
test_data = F.one_hot(test_labels, num_classes = M)

train_dataset = Data.TensorDataset(train_data, train_labels) # (data_tensor=train_data, target_tensor=train_labels)
test_dataset = Data.TensorDataset(test_data,test_labels) # (data_tensor=test_data, target_tensor=test_labels)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=test_num_msgs, shuffle=True, num_workers=2)
# =================================================================================================

model = AutoEncoder(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# =================================================================================================
def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# =================================================================================================
train_losses = []
test_losses = []

def train(model=model):


    saved_model={}
    for epoch in range(args.epochs):
        print('\nepoch:{}'.format(epoch))

        for step, (x, y) in enumerate(train_loader):
            train_loss = 0
            best_train_loss = 1
            # x = torch.tensor([x],dtype=torch.long())
            # y = torch.tensor([y], dtype=torch.long())
            x = x.to(torch.float32)
            x = x.to(device)
            y = y.to(device)
            model.train()
            pred = model(x)
            loss = criterion(pred, y)

            train_loss += loss.item()
            train_losses.append(train_loss)

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients


        print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            saved_model= copy.deepcopy(model.state_dict())
            torch.save({'model':saved_model}, MODEL_PATH)


def test(aplot):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        model.eval()
        if aplot==0:
            '''
            constellation plot
            '''
            test_labels = torch.linspace(0, M - 1, steps=M).long().to(device)
            test_data = torch.sparse.torch.eye(M, device=torch.device('cuda')).index_select(dim=0, index=test_labels)
            x = model.encode_signal(test_data)
            x = F.normalize(x, dim=1)
            plt.figure()
            plot_data = x.cpu().detach().numpy()
            plt.scatter(plot_data[:, 0], plot_data[:, 1], label='Autoencoder({},{})'.format(n, k))
            plt.axis((-2.5, 2.5, -2.5, 2.5))
            plt.legend(loc='upper right', ncol=1)
            plt.grid()
            plt.show()
        else:
            '''
            ber curve
            '''
            SNR_range = list(frange(-5, 8, 0.5))
            ber = [None] * len(SNR_range)
            for i in range(0, len(SNR_range)):
                SNR = 10.0 ** (SNR_range[i] / 10.0)
                error_msgs = 0
                for step, (x, y) in enumerate(test_loader):
                    x = x.to(torch.float32)
                    x = x.to(device)
                    y = y.to(device)
                    x = model.encode_signal(x)
                    x = model.AWGN_var(x,SNR)
                    pred = model.decode_signal(x)
                    _ , pred_final = torch.max(pred,axis=1)
                    error_msgs += (pred_final != y).sum().item()
                    ber[i] = error_msgs/ test_num_msgs
                    print('SNR:', SNR_range[n], 'BER:', ber[i])
            ## ploting ber curve
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(SNR_range, ber, 'bo', label='Autoencoder({},{})'.format(n,k))
            plt.yscale('log')
            plt.xlabel('SNR Range')
            plt.ylabel('Block Error Rate')
            plt.grid()
            plt.legend(loc='upper right', ncol=1)
            plt.show()
if __name__ =="__main__":
    seed_torch(0)
    if args.test == 0:
        train(model)
    else:
        model.load_state_dict(torch.load(args.weights)['model'])
        test(args.plot)