import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


def get_sim_matrix(image_features, text_features, logit_scale):
    image_norm = F.normalize(image_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)

    t2i_cosine = text_norm @ image_norm.t()
    i2t_cosine = t2i_cosine.t()
    
    return logit_scale * i2t_cosine, logit_scale * t2i_cosine


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    # return loss
    return loss, (t2i_cosine_theta.clone().detach()+1)/2, labels


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

# class TripletLoss_WRT(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletLoss_WRT, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, inputs, targets):
#         n = inputs.size(0)
        
#         # 计算距离矩阵
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         # dist.addmm_(1, -2, inputs, inputs.t())
#         dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
#         dist = dist.clamp(min=1e-12).sqrt()  # 数值稳定性
        
#         # 对于每个样本，找到正样本和负样本
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
        
#         # 计算损失
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
        
#         return loss, dist_ap, dist_an


class TripletLoss_WRT(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss_WRT, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        

        # pow(x, 2) + pow(y, 2)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        
        # dist - 2 * x * y
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist + (mask.float() * 1e6), dim=1)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss, dist_ap, dist_an


class NEL(nn.Module):
    def __init__(self,distance_mode = "Euclidean"):
        super(NEL, self).__init__()
        self.distance_mode =distance_mode

    def forward(self, specific_tokens,general_tokens, targets):
        if self.distance_mode == "Euclidean":
            dist_mat = pdist_torch(general_tokens, specific_tokens)
            N = targets.size(0)
            # shape [N, N]
            is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())

            dist_ap = dist_mat * is_pos

            dist_euclidean = torch.mean(dist_ap, dim=1)

            loss = -torch.mean(dist_euclidean)
        else:
            general_tokens = F.normalize(general_tokens, p=2, dim=1)
            specific_tokens = F.normalize(specific_tokens, p=2, dim=1)
            dist_mat = torch.matmul(general_tokens,specific_tokens.t())
            N = targets.size(0)
            is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())

            dist_ap = dist_mat * is_pos

            dist_cosine = torch.mean(torch.pow(dist_ap,2),dim = 1)

            loss = torch.mean(dist_cosine)

        return loss

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

if __name__ == '__main__':
    # # 测试数据
    a = torch.randn(32,768)
    print(a)
    b = torch.randn(32,768)
    print(b)
    targets = torch.Tensor([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]).long()

    loss_push = NEL("cosine_similarity")
    print(loss_push(a,b,targets))
