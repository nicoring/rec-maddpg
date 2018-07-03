import torch
from torch.distributions.kl import register_kl
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

@register_kl(ExpRelaxedCategorical, ExpRelaxedCategorical)
def kl_relaxedcat_relaxedcat(p, q):
    a0 = p.logits - torch.max(p.logits, dim=1, keepdim=True)[0]
    a1 = q.logits - torch.max(q.logits, dim=1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    ea1 = torch.exp(a1)
    z0 = torch.sum(ea0, dim=1, keepdim=True)
    z1 = torch.sum(ea1, dim=1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)
