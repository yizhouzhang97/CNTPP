import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam, SGD, RMSprop, Adagrad
from torch.nn import functional as F
from optimization import BertAdam
#from math as pi

from matplotlib import pyplot as plt

from math import pi, exp, log

from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)
    
    
    @staticmethod
    def backward(ctx, *grad_output):
        grad_input = grad_output[0].clone()
        return grad_input * -ctx.lambd, None
    
    



class GMMNet(nn.Module):

    def __init__(self, config):
        super(GMMNet, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Linear(in_features=config.hid_dim+1, out_features=config.mlp_dim)
        self.module_list = nn.ModuleList([nn.Linear(in_features=config.mlp_dim, out_features=config.mlp_dim) for _ in range(config.mlp_layer-1)])
        self.linear_mu =  nn.Linear(in_features=config.mlp_dim, out_features=config.components*config.feat_dim)
        self.linear_scale =  nn.Linear(in_features=config.mlp_dim, out_features=config.components*config.feat_dim)
        self.linear_pi =  nn.Sequential(nn.Linear(in_features=config.mlp_dim, out_features=config.components), nn.Softmax(-1))
        self.components = config.components
        self.feat_dim = config.feat_dim



    def forward(self, hidden_state, target_time, x):
        
        x = x.unsqueeze(-2)

        for p in self.parameters():
            p.data *= (p.data>=0)

        target_time.requires_grad_(True)
        t = self.linear1(target_time.unsqueeze(dim=-1))

        out = F.tanh(self.linear2(torch.cat([hidden_state, t], dim=-1)))
        for layer in self.module_list:
            out = F.tanh(layer(out))
        mu = torch.stack(self.linear_mu(out).chunk(self.components,dim=-1),dim=-2)
        scale = torch.exp(torch.stack(self.linear_scale(out).chunk(self.components,dim=-1),dim=-2))
        #if scale < 0:
        #scale = scale*(scale>0-0.5)+1e-2
        #scale += 1e-4
        pri = self.linear_pi(out)
        
        prec = 1/scale
        #print(prec)
        #print(self.mu)
        #print(mu.size())
        #print(x.size())

        log_gp = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=-1)
        log_gdet = torch.sum(torch.log(prec), dim=-1)
        
        log_prob_g = -.5 * (self.feat_dim * log(2. * pi) + log_gp) + log_gdet
        weighted_log_prob = torch.log(pri) + log_prob_g
        likelihood = torch.logsumexp(weighted_log_prob, dim=-1)

        return likelihood, mu, scale, pri
    
    def expect(self, hidden_state, target_time):

        for p in self.parameters():
            p.data *= (p.data>=0)

        target_time.requires_grad_(True)
        t = self.linear1(target_time.unsqueeze(dim=-1))

        out = F.tanh(self.linear2(torch.cat([hidden_state, t], dim=-1)))
        for layer in self.module_list:
            out = F.tanh(layer(out))
        mu = torch.stack(self.linear_mu(out).chunk(self.components,dim=-1),dim=-2)
        pri = self.linear_pi(out)
        
        return torch.sum(pri.unsqueeze(-1)*mu, dim=-2)





class IntensityNet(nn.Module):

    def __init__(self, config):
        super(IntensityNet, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Linear(in_features=config.hid_dim+1, out_features=config.mlp_dim)
        self.module_list = nn.ModuleList([nn.Linear(in_features=config.mlp_dim, out_features=config.mlp_dim) for _ in range(config.mlp_layer-1)])
        self.linear3 =  nn.Sequential(nn.Linear(in_features=config.mlp_dim, out_features=1))



    def forward(self, hidden_state, target_time):

        for p in self.parameters():
            p.data *= (p.data>=0)

        target_time.requires_grad_(True)
        t = self.linear1(target_time.unsqueeze(dim=-1))

        out = F.tanh(self.linear2(torch.cat([hidden_state, t], dim=-1)))
        for layer in self.module_list:
            out = F.tanh(layer(out))
        int_lmbda = F.softplus(self.linear3(out)).squeeze(-1)
        mean_int_lmbda = torch.sum(int_lmbda)

        lmbda = grad(mean_int_lmbda, target_time, create_graph=True, retain_graph=True)[0]
        nll = int_lmbda -torch.log((lmbda+1e-10))

        return [nll, torch.log((lmbda+1e-10)), int_lmbda, lmbda]

    
    
    
    
class GTPP(nn.Module):

    def __init__(self, config):

        super(GTPP, self).__init__()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_mode = config.log_mode


        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(input_size=1+config.emb_dim+config.feat_dim,
                            num_layers=config.lstm_layer,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.intensity_net = IntensityNet(config)
        self.gmm_net = GMMNet(config)
        self.intensity_net_t = IntensityNet(config)
        self.gmm_net_t = GMMNet(config)  # predict treatment
        self.set_optimizer(total_step=1)
        self.clip_value = config.clip_value


    def set_optimizer(self, total_step, use_bert=False):
        if use_bert:
            self.set_optimizer = BertAdam(params=self.parameters(),
                                          lr=self.lr,
                                          warmup=0.1,
                                          t_total=total_step)
        else:
            self.set_optimizer = Adam(self.parameters(), lr=self.lr)


    def hidden_state(self, batch):
        in_time_seq, in_feat_seq, in_event_seq, out_time_seq, out_feat_seq, mask, treatment_mask = batch
        
        
        
        
        
        
        in_event_seq = in_event_seq.long()
        #print(in_event_seq)
        #if not self.continuous_feat:
        input_event_feat = self.embedding(in_event_seq)
        #else:
        #    input_feat = event_seq
        emb = self.emb_drop(torch.cat([input_event_feat,in_feat_seq],dim=-1))
        lstm_input = torch.cat([emb, in_time_seq.unsqueeze(-1)], dim=-1)
        #print(lstm_input.size())
        hidden_state, _ = self.lstm(lstm_input)
        return hidden_state
    
    
    def forward(self, batch, lambd):
        in_time_seq, in_feat_seq, in_event_seq, out_time_seq, out_feat_seq, mask, treatment_mask = batch # mask: loss 
        
        
        
        
        
        
        in_event_seq = in_event_seq.long()
        #print(in_event_seq)
        #if not self.continuous_feat:
        input_event_feat = self.embedding(in_event_seq)
        #else:
        #    input_feat = event_seq
        emb = self.emb_drop(torch.cat([input_event_feat,in_feat_seq],dim=-1))
        lstm_input = torch.cat([emb, in_time_seq.unsqueeze(-1)], dim=-1)
        #print(lstm_input.size())
        hidden_state, _ = self.lstm(lstm_input)

        nll, log_lmbda, int_lmbda, lmbda = self.intensity_net(hidden_state, in_time_seq) # lambda(t)
        likelihood, mu, scale, pri = self.gmm_net(hidden_state, out_time_seq, out_feat_seq) # p(x|t)
        
        hidden_state_reverse = GradReverse.apply(hidden_state, lambd) # reverse gradint
        
        nll_t, log_lmbda_t, int_lmbda_t, lmbda_t = self.intensity_net_t(hidden_state_reverse, in_time_seq)
        likelihood_t, mu_t, scale_t, pri_t = self.gmm_net_t(hidden_state_reverse, out_time_seq, out_feat_seq)
        

        nll.data = torch.where(torch.isnan(nll.data), torch.full_like(nll.data, 0), nll.data)
        likelihood.data = torch.where(torch.isnan(likelihood.data), torch.full_like(likelihood.data, 0), likelihood.data)
        log_lmbda.data = torch.where(torch.isnan(log_lmbda.data), torch.full_like(log_lmbda.data, 0), log_lmbda.data)
        int_lmbda.data = torch.where(torch.isnan(int_lmbda.data), torch.full_like(int_lmbda.data, 0), int_lmbda.data)
        
        nll = torch.sum(mask*nll)/torch.sum(mask)
        likelihood = torch.sum(mask*likelihood)/torch.sum(mask)
        log_lmbda = torch.sum(mask*log_lmbda)/torch.sum(mask)
        int_lmbda = torch.sum(mask*int_lmbda)/torch.sum(mask)
        #nll -= likelihood/32 # mean dim
        nll -= likelihood/3 # mean dim

        nll_t.data = torch.where(torch.isnan(nll_t.data), torch.full_like(nll_t.data, 0), nll_t.data)
        likelihood_t.data = torch.where(torch.isnan(likelihood_t.data), torch.full_like(likelihood_t.data, 0), likelihood_t.data)
        log_lmbda_t.data = torch.where(torch.isnan(log_lmbda_t.data), torch.full_like(log_lmbda_t.data, 0), log_lmbda_t.data)
        int_lmbda_t.data = torch.where(torch.isnan(int_lmbda_t.data), torch.full_like(int_lmbda_t.data, 0), int_lmbda_t.data)
        
        
        nll_t = torch.sum(treatment_mask*nll_t)/torch.sum(treatment_mask)
        likelihood_t = torch.sum(treatment_mask*likelihood_t)/torch.sum(treatment_mask)
        log_lmbda_t = torch.sum(treatment_mask*log_lmbda_t)/torch.sum(treatment_mask)
        int_lmbda_t = torch.sum(treatment_mask*int_lmbda_t)/torch.sum(treatment_mask)
        #nll_t -= likelihood_t/32
        nll_t -= likelihood_t/3

        return [nll, log_lmbda, int_lmbda, lmbda, likelihood, mu, scale, pri, nll_t, likelihood_t]


    def train_batch(self, batch, epoch_num):

        self.set_optimizer.zero_grad()
        nll, log_lmbda, int_lmbda, lmbda, likelihood, mu, scale, pri, nll_t, likelihood_t = self.forward(batch, 2*(2/(1+exp(-10*epoch_num))-1))
        loss = nll + nll_t
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_value)
        self.set_optimizer.step()

        return nll.item(), log_lmbda.item(), int_lmbda.item(), lmbda, likelihood.item(), mu, scale, pri, nll_t, likelihood_t





    
class GTPP_non_causal(nn.Module):

    def __init__(self, config):

        super(GTPP_non_causal, self).__init__()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_mode = config.log_mode


        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(input_size=1+config.emb_dim+config.feat_dim,
                            num_layers=config.lstm_layer,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.intensity_net = IntensityNet(config)
        self.gmm_net = GMMNet(config)
        self.set_optimizer(total_step=1)
        self.clip_value = config.clip_value


    def set_optimizer(self, total_step, use_bert=False):
        if use_bert:
            self.set_optimizer = BertAdam(params=self.parameters(),
                                          lr=self.lr,
                                          warmup=0.1,
                                          t_total=total_step)
        else:
            self.set_optimizer = Adam(self.parameters(), lr=self.lr)


    def hidden_state(self, batch):
        in_time_seq, in_feat_seq, in_event_seq, out_time_seq, out_feat_seq, mask, treatment_mask = batch
        
        
        
        
        
        
        in_event_seq = in_event_seq.long()
        #print(in_event_seq)
        #if not self.continuous_feat:
        input_event_feat = self.embedding(in_event_seq)
        #else:
        #    input_feat = event_seq
        emb = self.emb_drop(torch.cat([input_event_feat,in_feat_seq],dim=-1))
        lstm_input = torch.cat([emb, in_time_seq.unsqueeze(-1)], dim=-1)
        #print(lstm_input.size())
        hidden_state, _ = self.lstm(lstm_input)
        return hidden_state
    
    
    def forward(self, batch):
        in_time_seq, in_feat_seq, in_event_seq, out_time_seq, out_feat_seq, mask, treatment_mask = batch
        
        
        
        
        
        
        in_event_seq = in_event_seq.long()
        #print(in_event_seq)
        #if not self.continuous_feat:
        input_event_feat = self.embedding(in_event_seq)
        #else:
        #    input_feat = event_seq
        emb = self.emb_drop(torch.cat([input_event_feat,in_feat_seq],dim=-1))
        lstm_input = torch.cat([emb, in_time_seq.unsqueeze(-1)], dim=-1)
        #print(lstm_input.size())
        hidden_state, _ = self.lstm(lstm_input)

        nll, log_lmbda, int_lmbda, lmbda = self.intensity_net(hidden_state, in_time_seq)
        likelihood, mu, scale, pri = self.gmm_net(hidden_state, out_time_seq, out_feat_seq)
        
        
        
        nll = torch.sum(mask*nll)/torch.sum(mask)
        likelihood = torch.sum(mask*likelihood)/torch.sum(mask)
        log_lmbda = torch.sum(mask*log_lmbda)/torch.sum(mask)
        int_lmbda = torch.sum(mask*int_lmbda)/torch.sum(mask)
        nll -= likelihood/32

        return [nll, log_lmbda, int_lmbda, lmbda, likelihood, mu, scale, pri]


    def train_batch(self, batch):

        self.set_optimizer.zero_grad()
        nll, log_lmbda, int_lmbda, lmbda, likelihood, mu, scale, pri = self.forward(batch)
        loss = nll
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_value)
        self.set_optimizer.step()

        return nll.item(), log_lmbda.item(), int_lmbda.item(), lmbda, likelihood.item(), mu, scale, pri





