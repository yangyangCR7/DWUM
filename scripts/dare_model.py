import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.backbone import bn_inception, Resnet50

class DAREBackbone(nn.Module):
    def __init__(self, backbone_type='Resnet50', embed_dim=512, pretrained=True, bn_freeze=True):
        super().__init__()
        
        if backbone_type == 'BNInception':
            self.backbone = bn_inception(pretrained=pretrained, bn_freeze=bn_freeze, Pool=True)
            n_mid = 1024
        else:
            self.backbone = Resnet50(pretrained=pretrained, bn_freeze=bn_freeze, Pool=True)
            n_mid = 2048

        self.emb_head_mu = nn.Sequential(
            nn.LayerNorm(n_mid),
            nn.BatchNorm1d(n_mid),
            nn.Linear(n_mid, embed_dim),
        )

        self.emb_head_logvar = nn.Sequential(
            nn.LayerNorm(n_mid),
            nn.BatchNorm1d(n_mid),
            nn.Linear(n_mid, embed_dim),
        )

        self.dim_mask_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()  
        )

    def forward(self, x):
        v = self.backbone(x)
        mu = self.emb_head_mu(v)
        logvar = self.emb_head_logvar(v)
        dim_mask = self.dim_mask_generator(mu.detach())  # [B, D]
        return mu, logvar, dim_mask


class DAREModel(nn.Module):

    def __init__(self, num_classes, backbone_type='Resnet50', embed_dim=512, bn_freeze=True):
        super().__init__()
        self.encoder = DAREBackbone(backbone_type=backbone_type, embed_dim=embed_dim, bn_freeze=bn_freeze)
        self.embed_dim = embed_dim
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.proxies)

    def get_proxies(self):
        return F.normalize(self.proxies, dim=-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, images, return_distribution=False):
        mu, logvar, dim_mask = self.encoder(images)

        if self.training or return_distribution:
            z = self.reparameterize(mu, logvar)
            return {
                'mu': mu,
                'logvar': logvar,
                'z': F.normalize(z, dim=-1),
                'dim_mask': dim_mask,
            }
        else:
            return F.normalize(mu, dim=-1)

class DAREFeatureExtractor:
    def __init__(self, model):
        self._model = model

    def __call__(self, x):
        return self._model(x, return_distribution=False)

    def parameters(self):
        return iter([])

    def modules(self):
        return iter([])