import warnings,torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
#for nongreedy decoding
import torch.nn.functional as F

#ignore 'divide by zero' warnings
warnings.filterwarnings("ignore")

processes = 30
datadir = '/home/u6/kathleencosta/'
wavdir = datadir + 'mhwav/'
validated = 'validated.tsv'
files = 5000
valid = 50
test = 20
batchsize = 30
inputdim = 128
hiddendim = 500
lr = 0.01
epochs = 1000

#use GPU if available
if torch.cuda.is_available():
        device = 'cuda'
else:
        device = 'cpu'
print(f'using {device}',flush=True)

#fully connected node for deepspeech
class FullyConnected(torch.nn.Module):
        def __init__(self,n_feature,n_hidden,dropout,relu_max_clip=20):
                super(FullyConnected,self).__init__()
                self.fc = torch.nn.Linear(n_feature,n_hidden,bias=True)
                self.relu_max_clip = relu_max_clip
                self.dropout = dropout
        def forward(self,x):
                x = self.fc(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.hardtanh(x,0,self.relu_max_clip)
                if self.dropout:
                        x = torch.nn.functional.dropout(
                                x,
                                self.dropout,
                                self.training
                        )
                return x

#deepspeech architecture
class DeepSpeech(torch.nn.Module):
        def __init__(
                self,
                n_feature,
                n_hidden=2048,
                n_class=40
        ):
                dropout=0.0
                super(DeepSpeech,self).__init__()
                self.n_hidden = n_hidden
                self.fc1 = FullyConnected(n_feature,n_hidden,dropout)
                self.fc2 = FullyConnected(n_hidden,n_hidden,dropout)
                self.fc3 = FullyConnected(n_hidden,n_hidden,dropout)
                self.bi_rnn = torch.nn.RNN(
                        n_hidden,
                        n_hidden,
                        num_layers=1,
                        nonlinearity="relu",
                        bidirectional=True
                )
                self.fc4 = FullyConnected(n_hidden,n_hidden,dropout)
                self.out = torch.nn.Linear(n_hidden,n_class)
        def forward(self,x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                x = x.squeeze(1)
                x = x.transpose(0,1)
                x,_ = self.bi_rnn(x)
                x = x[:,:,:self.n_hidden] + x[:,:,self.n_hidden:]
                x = self.fc4(x)
                x = self.out(x)
                x = x.permute(1,0,2)
                x = torch.nn.functional.log_softmax(x,dim=2)
                return x

#map from letters to integers
def addtol2i(s,d):
        for letter in s:
                if letter not in d:
                        #reserve 0 for blank
                        d[letter] = len(d)+1

#get spectrogram and target
def getspec(pair):
        trans = [l2i[let] for let in pair[1]]
        filename = wavdir+pair[0][:-3]+'wav'
        fs,w = wavfile.read(filename)
        res = plt.specgram(
                w,
                NFFT=254,
                Fs=fs,
                noverlap=127
        )
        spec = res[0].T
        return (trans,spec)

#read in list of files
f = open(datadir+validated,'r')
t = f.read()
f.close()

#remove header and empty final line
t = t.split('\n')
t = t[1:-1]

#get filenames,glosses, and character map
l2i = {}
pairs = []
for line in t:
        bits = line.split('\t')
        filename = bits[1]
        gloss = bits[2]
        if gloss[0] == '"': gloss = gloss[1:]
        if gloss[-1] == '"': gloss = gloss[:-1]
        addtol2i(gloss,l2i)
        pairs.append((filename,gloss))

#make reverse map from integers to characters
i2l = {pair[1]:pair[0] for pair in l2i.items()}

#number of output categories (wo/blank!)
outsize = len(i2l)

#make spectrograms in parallel
with mp.Pool(processes) as mypool:
        results = mypool.map(getspec,pairs[:files])

#separate training, validation, test
validset = results[:valid]
testset = results[valid:valid+test]
trainset = results[valid+test:]

#custom dataset
class SpecData(Dataset):
        def __init__(self,d):
                self.labels = [torch.Tensor(pair[0]) for pair in d]
                self.specs = [torch.Tensor(pair[1]) for pair in d]
        def __len__(self):
                return len(self.labels)
        def __getitem__(self,idx):
                spec = self.specs[idx]
                label = self.labels[idx]
                return spec,label

#make datasets
traindata = SpecData(trainset)
testdata = SpecData(testset)
validdata = SpecData(validset)

#items in batch must have same length
def pad(batch):
        (xx,yy) = zip(*batch)
        xlens = [len(x) for x in xx]
        ylens = [len(y) for y in yy]
        xxpad = pad_sequence(
                xx,
                batch_first=True,
                padding_value=0
        )
        yypad = pad_sequence(
                yy,
                batch_first=True,
                padding_value=0
        )
        return xxpad,yypad,xlens,ylens

#make dataloaders
trainloader = DataLoader(
        traindata,
        batch_size=batchsize,
        collate_fn=pad,
        shuffle=True
)
validloader = DataLoader(
        validdata,
        batch_size=batchsize,
        collate_fn=pad,
        shuffle=True
)
#batch = 1 for test
testloader = DataLoader(
        testdata,
        batch_size=1,
        shuffle=False
)

asr = DeepSpeech(
        n_feature=inputdim,
        n_hidden=hiddendim,
        n_class=outsize+1
).to(device)
lossfunc = nn.CTCLoss(
        #zero_infinity=True,
        reduction='mean'
)
opt = optim.SGD(
        asr.parameters(),
        lr=lr
)

#train
for epoch in range(epochs):
        i = 0
        epochloss = []
        for inp,outp,inlens,outlens in trainloader:
                asr.zero_grad()
                inp = inp.to(device)
                pred = asr(inp)
                loss = lossfunc(
                        pred.transpose(1,0),
                        outp,
                        inlens,
                        outlens
                )
                loss.backward()
                opt.step()
                epochloss.append(
                        loss.detach().cpu().numpy()
                )
                i += 1
        elossmean = np.mean(epochloss)
        print(f'epoch {epoch} loss: {elossmean}',flush=True)
        #validate
        with torch.no_grad():
                validloss = []
                for inp,outp,inlens,outlens in validloader:
                        inp = inp.to(device)
                        pred = asr(inp)
                        loss = lossfunc(
                                pred.transpose(1,0),
                                outp,
                                inlens,
                                outlens
                        )
                        validloss.append(
                                loss.detach().cpu().numpy()
                        )
                print(
                        f'\tvalid loss: {np.mean(validloss)}',
                        flush=True
                )

#nongreedy decoding
with torch.no_grad():
    for inp, outp in testloader:
        for i in outp[0]:
            print(i2l[int(i)], end='', flush=True)
        print(flush=True)

        inp = inp.to(device)
        outp = outp.to(device)
        pred = asr(inp)
        pred = F.log_softmax(pred, dim=2)

        pred_seq = pred.cpu().detach().numpy().argmax(axis=2)

        newres = []
        for i, p in enumerate(pred_seq[0]):
            if p != 0 and (i == 0 or p != pred_seq[0][i - 1]):
                newres.append(i2l[p])

        print(f'"{"".join(newres)}"', end='\n\n', flush=True)
