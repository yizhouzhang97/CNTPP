
import pickle as pkl
import numpy as np
import tqdm
import torch

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from utils import read_timeseries, read_feats, generate_sequence, plt_lmbda, collate_time_series, TimeDataset, read_eventseries
from module import GTPP

def ATE(h0, h1, time_gap, time_window, lamd_func, p_x_t):
    delta_time = time_window/100
    phi_0 = 0.0
    phi_1 = 0.0
    t_0 = time_gap
    t_1 = torch.zeros_like(time_gap)
    
    u_0 = 0.0 
    u_1 = 0.0
    for i in range(0,100):
        _,_,_,lamd_t0 = lamd_func(h0, t_0)
        _,_,_,lamd_t1 = lamd_func(h1, t_1)
        expect_0 = p_x_t.expect(h0, t_0)
        expect_1 = p_x_t.expect(h1, t_1)
        
        phi_0 += delta_time*expect_0*(lamd_t0.unsqueeze(-1))
        phi_1 += delta_time*expect_1*(lamd_t1.unsqueeze(-1))
        t_0 = t_0 + delta_time
        t_1 = t_1 + delta_time
        u_0 = u_0 + delta_time*lamd_t0
        u_1 = u_1 + delta_time*lamd_t1
   
    u_0 = u_0.unsqueeze(-1)
    u_1 = u_1.unsqueeze(-1)
   
    
    return np.array((phi_0/(u_0)).detach().cpu().numpy()), np.array((phi_1/(u_1)).detach().cpu().numpy())
        
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='exponential_hawkes')
    parser.add_argument("--model", type=str, default='GTPP')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--event_class", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--early_stop", type=bool, default=True)

    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--log_mode", type=bool, default=False)
    parser.add_argument("--clip_value", type=float, default=1e-1)
    config = parser.parse_args()

    path = 'data/syn_data/'

    if config.data == 'exponential_hawkes':
        
        events_seq = read_eventseries(path  + 'dataset.pkl')

        train_data = read_timeseries(path  + 'dataset.pkl')
        val_data = read_timeseries(path  + 'dataset.pkl')
        test_data = read_timeseries(path + 'dataset.pkl')
        
        train_feats = read_feats(path + 'text_user.pkl')
        val_feats = read_feats(path + 'text_user.pkl')
        test_feats = read_feats(path + 'text_user.pkl')



    train_timeseq, train_eventseq, train_feats = generate_sequence(train_data, config.seq_len, train_feats, log_mode=config.log_mode)
    train_loader = DataLoader(TimeDataset(train_timeseq, train_eventseq, train_feats), shuffle=True, batch_size=config.batch_size, collate_fn=collate_time_series)
    val_timeseq, val_eventseq, val_feats = generate_sequence(val_data, config.seq_len, val_feats, log_mode=config.log_mode)
    val_loader = DataLoader(TimeDataset(val_timeseq, val_eventseq, val_feats), shuffle=False, batch_size=25, collate_fn=collate_time_series)

   
    news_num = 120
    model = torch.load('model/trained_model.model')
    print(model)
    h_0_set = {}
    h_1_set = {}
    gap_set = {}
    model.eval()
    
    hidden_state = []
    for batch in val_loader:
        if batch[0].nelement() == 0:
            hidden_state.append([])
        else:
            states = model.hidden_state(batch)
            for state in states:
                hidden_state.append(state)
    if True:
        for seq_idx,seq in enumerate(events_seq):
            for event_idx in range(len(seq['events'])-1):
                event = seq['events'][event_idx]
                if event != news_num:
                    if event not in h_0_set:
                        h_0_set[event] = [hidden_state[seq_idx][event_idx-1]]
                        h_1_set[event] = [hidden_state[seq_idx][event_idx]]
                        gap_set[event] = [seq['times'][event_idx]-seq['times'][event_idx-1]]
                    else:
                        h_0_set[event] += [hidden_state[seq_idx][event_idx-1]]
                        h_1_set[event] += [hidden_state[seq_idx][event_idx]]
                        gap_set[event] += [seq['times'][event_idx]-seq['times'][event_idx-1]]
    
    print(len(hidden_state))
    time_window = 10
    ITE_set = {}
    model.cpu()
    for event in h_0_set:
    
        h_0 = torch.stack(h_0_set[event],dim=0).cpu()
        h_1 = torch.stack(h_1_set[event],dim=0).cpu()
        time_gap = torch.tensor(gap_set[event]).float()
        ITE_set[event] = ATE(h_0, h_1, time_gap, time_window, model.intensity_net.cpu(), model.gmm_net.cpu())
       
        print(ITE_set[event])
    pkl.dump(ITE_set,open('model/ITE.pkl','wb'))
        
    















