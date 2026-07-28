import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.eval_utils import l2_norm

def binarize(T, nb_classes):
    return F.one_hot(T, num_classes=nb_classes).float()


class Proxy_Anchor(nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, label, P, loss="Proxy_AN"):
        T = label
        cos = F.linear(l2_norm(X), l2_norm(P))  
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)   
        num_valid_proxies = max(1, len(with_pos_proxies))   
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies

        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=1)
        neg_term = torch.log(1 + N_sim_sum).sum() / max(1, X.shape[0])
        
        return pos_term + neg_term


class Proxy_NCA(nn.Module):
    def __init__(self, scale=3.0):
        super(Proxy_NCA, self).__init__()
        self.scale = scale

    def forward(self, X, label, P):
        X = l2_norm(X)
        P = l2_norm(P)
        sim = torch.matmul(X, P.T) * self.scale
        return F.cross_entropy(sim, label)


class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos=2.0, scale_neg=40.0, margin=0.1):
        super(MultiSimilarityLoss, self).__init__()
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.margin = margin
        
    def forward(self, X, label, P=None):
        X = l2_norm(X)
        sim_mat = torch.matmul(X, X.T)
        epsilon = 1e-5
        loss = list()
        
        for i in range(X.size(0)):
            pos_pair_ = sim_mat[i][label == label[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]
            
            if len(pos_pair_) < 1 or len(neg_pair_) < 1:
                continue
                
            neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]
            
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
                
            pos_loss = 1.0 / self.scale_pos * torch.log(1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.margin))))
            neg_loss = 1.0 / self.scale_neg * torch.log(1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.margin))))
            loss.append(pos_loss + neg_loss)
            
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(X.device)
        return sum(loss) / len(loss)


class SoftTripleLoss(nn.Module):
    def __init__(self, la=20.0, gamma=0.1, tau=0.2, margin=0.01, dim=512, cNum=100, K=10):
        super(SoftTripleLoss, self).__init__()
        self.la = la
        self.gamma = 1. / gamma
        self.tau = tau
        self.margin = margin
        self.K = K
        self.cNum = cNum
        self.fc = nn.Parameter(torch.Tensor(dim, cNum * K))
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        
    def forward(self, X, label, P=None):
        X = l2_norm(X)
        centers = l2_norm(self.fc.T).T      # [D, C*K]
        sim = torch.matmul(X, centers)      # [B, C*K]
        sim = sim.view(-1, self.cNum, self.K)  # [B, C, K]
        
        prob = F.softmax(sim * self.gamma, dim=2)
        sim_center = torch.sum(prob * sim, dim=2)  # [B, C]
        
        margin_m = torch.zeros_like(sim_center)
        margin_m.scatter_(1, label.view(-1, 1), self.margin)
        sim_center = sim_center - margin_m
        
        loss = F.cross_entropy(sim_center * self.la, label)
        return loss