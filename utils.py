import torch
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TimeDataset(Dataset):
    def __init__(self, TIME_SEQS, EVENT_SEQS, FEATS):
        self.TIME_SEQS = [torch.FloatTensor(seq).cuda() for seq in TIME_SEQS]
        self.EVENT_SEQS = [torch.LongTensor(seq).cuda() for seq in EVENT_SEQS]
        self.FEATS = [torch.FloatTensor(seq).cuda() for seq in FEATS]
        
    def __len__(self):
        return len(self.TIME_SEQS)
    
    def __getitem__(self, idx):
        return self.TIME_SEQS[idx], self.EVENT_SEQS[idx], self.FEATS[idx],

    
def read_eventseries(path):

    activities = pkl.load(open(path,'rb'))
    news_num = activities[0][2]['events'][0]
    events_set = []
    for _,user_state,seq,wait_time in activities:
        events_set.append(seq)

    return events_set


def read_timeseries(path):

    activities = pkl.load(open(path,'rb'))
    news_num = activities[0][2]['events'][0]
    events_set = []
    for _,user_state,seq,wait_time in activities:
        
        events = []
        for i in range(len(seq['events'])):
            if seq['events'][i] == news_num:
                events.append((seq['times'][i],1))
            else:
                events.append((seq['times'][i],0))
        events_set.append(events)

    return events_set


def read_feats(path):

    with open(path,'rb') as f:
        feats = pkl.load(f)

    return feats


def generate_sequence(timeseries, seq_len, feats, log_mode=False):

    ## For the case that Each time_sequence has different length of time-series data.
    TIME_SEQS = []
    EVENT_SEQS = []

    for seq in timeseries:

        if not log_mode:
            times = [t for (t, e) in seq]
            times = [0] + np.diff(times).tolist()
            events = [e for (t, e) in seq]
            #seq_feats = feats[idx:idx+seq_len]
            #feat = [feat for feat in seq_feats]
            TIME_SEQS.append(times)
            EVENT_SEQS.append(events)

        else:
            times = [t for (t, e) in seq]
            mu = np.mean(times)
            std = np.std(times)
            times = (times-mu)/std
            times = [0] + np.diff(times).tolist()
            events = [e for (t, e) in seq]
            TIME_SEQS.append(times)
            EVENT_SEQS.append(events)

    

    return TIME_SEQS, EVENT_SEQS, feats



def plt_lmbda(timeseries, model, feats, seq_len, log_mode=False, dt=0.01, lmbda0=0.2, alpha=0.8, beta=1.0):

    lmbda_dict = dict()
    pred_dict = dict()
    t_span = np.arange(start=timeseries[0][0], stop=timeseries[-1][0]+dt, step=dt)

    


    lmbda_dict[0] = np.zeros(t_span.shape)


    for t, e in timeseries:
        target = (t_span > t)
        lmbda_dict[0][target] += alpha*np.exp(-beta*(t_span[target]-t))
    lmbda_dict[0] += lmbda0

    
    pred_dict[0] = np.zeros(len(timeseries)-seq_len+1)
    test_timeseq, test_eventseq = generate_sequence(timeseries, feats, seq_len, log_mode=log_mode)
    _, _, _, pred_dict[0] = model((test_timeseq, test_eventseq))


    plt.plot(t_span, lmbda_dict[0], color='green')
    plt.plot([t for t, e in timeseries][seq_len-1:], np.array(pred_dict[0].detach()), color='olive')
    plt.scatter([t for t, e in timeseries], [-1 for _ in timeseries], color='blue')
    plt.show()
    
    
    
    
    
    
    
def collate_time_series(batch):
    with torch.no_grad():
        
        in_time_seq = pad_sequence([seq[:-1] for seq,_,_ in batch], batch_first=True)

        in_event_seq = pad_sequence([seq[:-1] for _,seq,_ in batch], batch_first=True)
        in_feat_seq = pad_sequence([seq[:-1] for _,_,seq in batch], batch_first=True)
        
        out_time_seq = pad_sequence([seq[1:] for seq,_,_ in batch], batch_first=True)
        out_feat_seq = pad_sequence([seq[1:] for _,_,seq in batch], batch_first=True)
        
        mask = []
        for seq,_,_ in batch:
            mask.append(torch.ones(len(seq)-1).cuda())
        mask = torch.nn.utils.rnn.pad_sequence(mask,batch_first=True)
        mask_copy = torch.zeros_like(mask)
        for i in range(len(batch)):
            seq = batch[i][1]
            for j in range(len(seq)-1):
                event = seq[j+1]
                if event == 0:
                    mask[i,j] = 0
                    mask_copy[i,j] = 1
        
        
    
    return in_time_seq, in_feat_seq, in_event_seq, out_time_seq, out_feat_seq, mask, mask_copy
    
    
    
    
    
    
    
    
    
    
    