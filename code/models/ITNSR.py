import torch
import torch.nn as nn
import torch.nn.functional as F
# import math
import models
from models import register
from utils import make_coord
from einops import repeat

@register('ITNSR')
class ITNSR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, embedding_coord=None,
                 embedding_scale=None, local_ensemble=True, feat_unfold=True, scale_token=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.scale_token = scale_token

        self.encoder = models.make(encoder_spec)
        if imnet_spec is not None:
            if self.feat_unfold:
                self.imnet = models.make(imnet_spec,args={'out_dim':self.encoder.out_dim*9*3})
            else:
                self.imnet = models.make(imnet_spec,args={'out_dim':self.encoder.out_dim*3})
        else:
            self.imnet = None
        
        if embedding_coord is not None:
            self.embedding_q = models.make(embedding_coord)
            self.embedding_s = models.make(embedding_scale)
        else:
            self.embedding_q = None
            self.embedding_s = None

        if local_ensemble:
            w = {
                'name': 'mlp',
                'args': {
                    'in_dim': 4,
                    'out_dim': 1,
                    'hidden_list': [256],
                    'act': 'gelu'
                }
            }
            self.Weight = models.make(w)

            score = {
                'name': 'mlp',
                'args': {
                    'in_dim': 2,
                    'out_dim': 1,
                    'hidden_list': [256],
                    'act': 'gelu'
                }
            }
            self.Score = models.make(score)

 

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, scale=None):

        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
        # K
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        # enhance local features
        if self.local_ensemble:
            # v_lst = [(-1,-1),(-1,0),(-1,1),(0, -1), (0, 0), (0, 1),(1, -1),(1, 0),(1,1)]#
            v_lst = [(i,j) for i in range(-1, 2, 2) for j in range(-1, 2, 2)]
            # v_lst = [(-1,0), (1,0), (0,1), (0, -1), (-2,-2), (-2, 2), (2, -2), (2, 2)]
            eps_shift = 1e-6
            preds = []
            for v in v_lst:
                vx = v[0]
                vy = v[1]
                # project to LR field 
                tx = ((feat.shape[-2] - 1) / (1 - scale[:,0,0])).view(feat.shape[0],  1)
                ty = ((feat.shape[-1] - 1) / (1 - scale[:,0,1])).view(feat.shape[0],  1)
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0
                bs, q = coord.shape[:2]
                coord_ = coord.clone()

                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                #Interpolate K to HR resolution  
                value = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #Interpolate K to HR resolution 
                coord_k = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                #calculate relation of Q-K
                if self.embedding_q:
                    Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                    K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                    rel = Q - K
                    
                    rel[:, 0] *= feat.shape[-2]
                    rel[:, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        # scale = scale.view(bs*q,-1)
                        scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                        inp = torch.cat([inp, scale_], dim=-1)

                else:
                    Q, K = coord, coord_k
                    rel = Q - K
                    rel[:, :, 0] *= feat.shape[-2]
                    rel[:, :, 1] *= feat.shape[-1]
                    inp = rel
                    if self.scale_token:
                        scale_ = scale.clone()
                        scale_[:, :, 0] *= feat.shape[-2]
                        scale_[:, :, 1] *= feat.shape[-1]
                        inp = torch.cat([inp, scale_], dim=-1)

                score = repeat(self.Score(rel.view(bs * q, -1)).view(bs, q, -1),'b q c -> b q (repeat c)', repeat=3)
                
                weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
                pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
                
                pred +=score
                preds.append(pred)

            preds = torch.stack(preds,dim=-1)

            ret = self.Weight(preds.view(bs*q*3, -1)).view(bs, q, -1)
        else:
            #V
            bs, q = coord.shape[:2]
            value = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            #K
            coord_k = F.grid_sample(
                feat_coord, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)

            if self.embedding_q:
                Q = self.embedding_q(coord.contiguous().view(bs * q, -1))
                K = self.embedding_q(coord_k.contiguous().view(bs * q, -1))
                rel = Q - K
                
                rel[:, 0] *= feat.shape[-2]
                rel[:, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    # scale = scale.view(bs*q,-1)
                    scale_ = self.embedding_s(scale_.contiguous().view(bs * q, -1))
                    inp = torch.cat([inp, scale_], dim=-1)

            else:
                Q, K = coord, coord_k
                rel = Q - K
                rel[:, :, 0] *= feat.shape[-2]
                rel[:, :, 1] *= feat.shape[-1]
                inp = rel
                if self.scale_token:
                    scale_ = scale.clone()
                    scale_[:, :, 0] *= feat.shape[-2]
                    scale_[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, scale_], dim=-1)
            
            
            weight = self.imnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], 3)
            pred = torch.bmm(value.contiguous().view(bs * q, 1, -1), weight).view(bs, q, -1)
            ret = pred
        
        return ret

    def forward(self, inp, coord, scale):

        self.gen_feat(inp)
        return self.query_rgb(coord, scale)




