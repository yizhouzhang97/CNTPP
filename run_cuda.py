

import numpy as np
import tqdm
import torch
import random
import pickle
import copy

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from utils import read_timeseries, read_feats, generate_sequence, plt_lmbda, collate_time_series, TimeDataset
from module import GTPP



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default='exponential_hawkes')
    parser.add_argument("--model", type=str, default='GTPP')
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--feat_dim", type=int, default=32)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--lstm_layer", type=int, default=3)
    parser.add_argument("--mlp_layer", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--event_class", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prt_evry", type=int, default=1)
    parser.add_argument("--early_stop", type=bool, default=True)
    ## Alpha ??
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--log_mode", type=bool, default=False)
    parser.add_argument("--clip_value", type=float, default=5e-2)
    #config = parser.parse_args()
    config = parser.parse_args([])

    path = 'data/syn_data/'

    if config.data == 'exponential_hawkes':

        train_data = read_timeseries(path  + 'dataset_new.pkl')
        val_data = read_timeseries(path  + 'dataset_new.pkl')
        test_data = read_timeseries(path + 'dataset_new.pkl')
        
        train_feats = read_feats(path + 'text_user_new.pkl')
        val_feats = read_feats(path + 'text_user_new.pkl')
        test_feats = read_feats(path + 'text_user_new.pkl')



    timeseq, eventseq, feats = generate_sequence(train_data, config.seq_len, train_feats, log_mode=config.log_mode)
    
    
    idxs = [i for i in range(len(timeseq))]
    random.shuffle(idxs)
    train_idx = idxs[:int(len(timeseq)*0.8)]
    val_idx = idxs[int(len(timeseq)*0.8):]
    train_timeseq = [timeseq[idx] for idx in train_idx]
    train_eventseq = [eventseq[idx] for idx in train_idx]
    print(len(feats))
    train_feats = [feats[idx] for idx in train_idx]
    
    train_loader = DataLoader(TimeDataset(train_timeseq, train_eventseq, train_feats), shuffle=True, batch_size=config.batch_size, collate_fn=collate_time_series)
    val_timeseq = [timeseq[idx] for idx in val_idx]
    val_eventseq = [eventseq[idx] for idx in val_idx]
    val_feats = [feats[idx] for idx in val_idx]
    val_loader = DataLoader(TimeDataset(val_timeseq, val_eventseq, val_feats), shuffle=False, batch_size=10, collate_fn=collate_time_series)
    pickle.dump(idxs, open('data/syn_data/idxs.pkl','wb'))
    
    model = GTPP(config)
    model.cuda()
    

    best_loss = 1e10
    patients = 0
    tol = 30

    for epoch in range(config.epochs):
        torch.cuda.empty_cache()

        model.train()

        loss1 = loss2 = loss3 = loss4 = loss5 = loss6 = 0

        for batch in train_loader:
        
            nll, log_lmbda, int_lmbda, lmbda, likelihood, mu, scale, pi, nll_t, likelihood_t = model.train_batch(batch,epoch)

            loss1 += nll
            loss2 += log_lmbda
            loss3 += int_lmbda
            loss4 += likelihood
            loss5 += nll_t
            loss6 += likelihood_t
            #break
            


        model.eval()
    
        val_loss, val_log_lmbda, val_int_lmbda, val_likelihood, val_nll_t, val_likelihood_t = 0,0,0,0,0,0
        batch_num = 0

        for batch in val_loader:
           
            if batch[3].nelement() < 1:
                continue
            
            '''try:
                val_loss_tmp, val_log_lmbda_tmp, val_int_lmbda_tmp, _, val_likelihood_tmp, _, _, _, val_nll_t_tmp, val_likelihood_t_tmp = model(batch,epoch)
                #batch_num += 1
                batch_num += 1
                val_loss += val_loss_tmp
                val_log_lmbda += val_log_lmbda_tmp
                val_int_lmbda += val_log_lmbda_tmp
                val_likelihood += val_log_lmbda_tmp
                val_nll_t += val_log_lmbda_tmp
                val_likelihood_t += val_log_lmbda_tmp'''
            
                
                
            if True:
                
                val_loss_tmp, val_log_lmbda_tmp, val_int_lmbda_tmp, _, val_likelihood_tmp, _, _, _, val_nll_t_tmp, val_likelihood_t_tmp = model(batch,epoch)
                
                batch_num += 1
                val_loss += val_loss_tmp.detach()
                val_log_lmbda += val_log_lmbda_tmp.detach()
                val_int_lmbda += val_int_lmbda_tmp.detach()
                val_likelihood += val_likelihood_tmp.detach()
                val_nll_t += val_nll_t_tmp.detach()
                val_likelihood_t += val_likelihood_t_tmp.detach()
            
            

        if epoch > 1:
            if best_loss > val_loss.item():
                torch.save(model,'model/trained_model.model')
                patients = 0
                best_loss = val_loss.item()
            else:
                patients += 1
                if patients >= tol:
                    print("Early Stop")
                    print("epoch", epoch)
                    break

        if epoch % config.prt_evry == 0:
            print("Epochs:{}".format(epoch))
            print("Training Total Negative Log Likelihood:{}   Training Negative Log Likelihood:{}   Log Lambda:{}   Integral Lambda:{}   GMM Likelihood:{}    Reverse NLL:{}    Reverse Likelihood:{}".format((loss1-loss5)*config.batch_size/len(train_timeseq), loss1*config.batch_size/len(train_timeseq), loss2*config.batch_size / len(train_timeseq), loss3*config.batch_size / len(train_timeseq), loss4*config.batch_size / len(train_timeseq), loss5*config.batch_size / len(train_timeseq), loss6*config.batch_size / len(train_timeseq)))
            print("Validation Total Negative Log Likelihood:{}   Validation Negative Log Likelihood:{}   Log Lambda:{}   Integral Lambda:{}   GMM Likelihood:{}    Reverse NLL:{}    Reverse Likelihood:{}".format((val_loss-val_nll_t)/ batch_num, val_loss/ batch_num, val_log_lmbda/ batch_num ,val_int_lmbda/ batch_num,val_likelihood/ batch_num, val_nll_t/ batch_num, val_likelihood_t/ batch_num))
            #plt_lmbda(train_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)
            #plt_lmbda(test_data[0], model=model, seq_len=config.seq_len, log_mode=config.log_mode)


    print("end")
    
















