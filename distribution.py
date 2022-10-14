import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td



def normal_sample(means, log_scales):
    if means.shape != log_scales.shape:
        raise ValueError("Shapes of means and scales don't match.")
    z = torch.empty(means.shape).normal_(0., 1.)
    return torch.exp(log_scales) * z + means


def normal_logpdf(x, mean, log_scale):
    z = (x - mean) * torch.exp(-log_scale)
    return -log_scale - 0.5 * z.pow(2.0) - 0.5 * np.log(2 * np.pi)


def normal_logcdf(x, mean, log_scale):
    z = (x - mean) * torch.exp(-log_scale)
    return torch.log(0.5 * torch.erf(z / np.sqrt(2)) + 0.5 + 1e-10)

def mixnormal_logpdf(x, log_prior, means, log_scales):
    return torch.logsumexp(
        log_prior + normal_logpdf(x.unsqueeze(-1), means, log_scales),
        dim=-1
    )

def mixnormal_logcdf(x, log_prior, means, log_scales):
    return torch.logsumexp(
        log_prior + normal_logcdf(x.unsqueeze(-1), means, log_scales),
        dim=-1
    )

