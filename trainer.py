from __future__ import print_function, absolute_import
import time
import torch
import torch.nn as nn
import numpy as np
from utils import AverageMeter
from torch.nn import functional as F
from loss import TripletLoss_WRT, compute_sdm, get_sim_matrix
import os
from loss_PMT.Triplet import TripletLoss
from loss_PMT.MSEL import MSEL
from loss_PMT.DCL import DCL


class Trainer(object):
    def __init__(self, model, cfg):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = nn.CrossEntropyLoss().cuda()
        self.criterion_triple = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
        self.criterion_mse = nn.MSELoss().cuda()
        self.DCL = DCL(num_pos=cfg.NUM_POS, feat_norm='no')
        self.MSEL = MSEL(num_pos=cfg.NUM_POS, feat_norm='no')
        self.criterion_dist = nn.CosineEmbeddingLoss().cuda()
        self.cfg = cfg
        self.T = 4

        self.prototypes = {}
        self.proto_weight = 5.0
    
    def update_prototypes(self, new_prototypes):
        self.prototypes.update(new_prototypes)

    def train(self, epoch, data_loader_train, optimizer, training_phase,
              add_num=0, old_model=None, replay=False,writer = None):
        self.model.train()

        if old_model is not None:
            old_model.eval()
            old_model.freeze_all()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        ce_loss = AverageMeter()
        tri_loss = AverageMeter()
        dcl_loss = AverageMeter()
        msel_loss = AverageMeter()
        goal_loss = AverageMeter()
        losses_total = AverageMeter()
        proto_loss_meter = AverageMeter()

        kd_loss_meter = AverageMeter()
        dist_loss = AverageMeter()

        end = time.time()

        num_old_classes = sum(self.model.class_per_task[:training_phase-1]) if training_phase > 1 else 0

        for batch_idx, (input1, input2, label1, label2) in enumerate(data_loader_train):
            data_time.update(time.time() - end)

            label1 += add_num
            label2 += add_num

            # delete the seq_len dim
            # [B,seq_len,C,H,W] --> [B,C,H,W]
            if len(input1.size()) == 5:
                input1 = input1.squeeze(1)
                input2 = input2.squeeze(1)

            input1 = input1.cuda()
            input2 = input2.cuda()

            label1 = label1.cuda()
            label2 = label2.cuda()
            labels = torch.cat((label1,label2),0)

            inputs = torch.cat([input1,input2])
            scores, features = self.model(inputs)

            score1, score2 = scores.chunk(2,0)
            feature1, feature2 = features.chunk(2,0)

            # base_loss
            loss_dcl = torch.tensor(0.0).cuda()
            loss_msel = torch.tensor(0.0).cuda()

            loss_id = self.criterion_ce(score1, label1) + self.criterion_ce(score2, label2)

            if self.cfg.METHOD == 'PMT':
                if epoch <= self.cfg.PL_EPOCH :
                    loss_tri = self.criterion_triple(feature1, feature1, label1) + self.criterion_triple(feature2, feature2, label2)  # intra
                    loss_goal = loss_id + loss_tri

                else:
                    loss_dcl = self.DCL(features, labels)
                    loss_msel = self.MSEL(features, labels)

                    loss_tri = self.criterion_triple(features, features, labels)

                    loss_goal = loss_id + loss_tri + self.cfg.DCL * loss_dcl + self.cfg.MSEL * loss_msel

            else:
                loss_tri = self.criterion_triple(features, features, labels)
                loss_goal = loss_id + loss_tri

            loss_lwf = torch.tensor(0.0).cuda()  # Logit Distillation
            loss_dist = torch.tensor(0.0).cuda() # Feature Distillation
            loss_topo = torch.tensor(0.0).cuda()

            if old_model is not None and training_phase > 1:
                with torch.no_grad():

                    old_scores, old_features = old_model(inputs, return_feature=True)

                target = torch.ones(features.size(0)).to(features.device)
                loss_dist = self.criterion_dist(features, old_features.detach(), target)

                new_logits_old_cls = scores[:, :num_old_classes]

                old_logits_old_cls = old_scores[:, :num_old_classes]

                log_probs_new = F.log_softmax(new_logits_old_cls / self.T, dim=1)
                probs_old = F.softmax(old_logits_old_cls / self.T, dim=1)
                
                loss_lwf = F.kl_div(log_probs_new, probs_old, reduction='batchmean') * (self.T**2)
                norm_factor = np.sqrt(num_old_classes) if num_old_classes > 0 else 1.0
                loss_lwf = loss_lwf / norm_factor

                loss_topo = self.compute_topology_loss(features, old_features.detach())
            
            # lamba = sum(self.model.class_per_task[:-1]) / sum(self.model.class_per_task)
            
            if training_phase==4:
                proto_weight=1.0
                lwf_weight = 20.0
                dist_weight = 800.0
            else:
                proto_weight = 1.0
                lwf_weight = 10.0
                dist_weight = 40.0
            
            if training_phase == 1:
                loss_total = loss_goal
            else:
                loss_total = loss_goal + lwf_weight * loss_lwf + dist_weight * loss_dist + proto_weight * loss_topo

            ce_loss.update(loss_id.item())
            tri_loss.update(loss_tri.item())
            dcl_loss.update(loss_dcl.item())
            msel_loss.update(loss_msel.item())
            goal_loss.update(loss_goal.item())
            kd_loss_meter.update(loss_lwf.item())
            dist_loss.update(loss_dist.item())
            proto_loss_meter.update(loss_topo.item())
            losses_total.update(loss_total.item())

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            
            if (batch_idx + 1) == len(data_loader_train) or (batch_idx + 1) % (len(data_loader_train) // 4) == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_total {:.3f} ({:.3f})\t'
                      'Loss_goal {:.3f} ({:.3f})\t'
                      'ID {:.3f} ({:.3f})\t'
                      'Tri {:.3f} ({:.3f})\t'
                      'DCL {:.3f} ({:.3f})\t'
                      'MSEL {:.3f} ({:.3f})\t'
                      'KD {:.3f} ({:.3f})\t'
                      'Dist {:.3f} ({:.3f})\t'
                      'Proto_KD {:.6f} ({:.6f})\t'
                      .format(epoch, batch_idx + 1, len(data_loader_train),
                              batch_time.val, batch_time.avg,
                              losses_total.val, losses_total.avg,
                              goal_loss.val, goal_loss.avg,
                              ce_loss.val, ce_loss.avg,
                              tri_loss.val, tri_loss.avg,
                              dcl_loss.val, dcl_loss.avg,
                              msel_loss.val, msel_loss.avg,
                              kd_loss_meter.val, kd_loss_meter.avg,
                              dist_loss.val, dist_loss.avg,
                              proto_loss_meter.val, proto_loss_meter.avg,
                              ))

            if writer is not None:
                writer.add_scalar('total_loss', losses_total.val,batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('id_loss', ce_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('tri_loss', tri_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('dcl_loss', dcl_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('msel_loss', msel_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('KD_loss', kd_loss_meter.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('Dist_loss', dist_loss.val, batch_idx + epoch * len(data_loader_train))
                writer.add_scalar('proto_loss', proto_loss_meter.val, batch_idx + epoch * len(data_loader_train))
                
    def compute_topology_loss(self, features, old_features):

        if not self.prototypes:
            return torch.tensor(0.0).cuda()

        sorted_pids = sorted(self.prototypes.keys())
        proto_rgb = torch.stack([self.prototypes[pid]['rgb'] for pid in sorted_pids]).detach()
        proto_ir = torch.stack([self.prototypes[pid]['ir'] for pid in sorted_pids]).detach()

        proto_rgb = F.normalize(proto_rgb, p=2, dim=1)
        proto_ir = F.normalize(proto_ir, p=2, dim=1)

        features_norm = F.normalize(features, p=2, dim=1)       
        old_features_norm = F.normalize(old_features, p=2, dim=1)

        T_proto = 0.5

        # Student vs RGB Protos
        logits_stu_rgb = torch.matmul(features_norm, proto_rgb.t()) / T_proto
        # Teacher vs RGB Protos
        logits_tea_rgb = torch.matmul(old_features_norm, proto_rgb.t()) / T_proto
        
        loss_rgb = F.kl_div(F.log_softmax(logits_stu_rgb, dim=1), 
                            F.softmax(logits_tea_rgb, dim=1), 
                            reduction='batchmean')

        logits_stu_ir = torch.matmul(features_norm, proto_ir.t()) / T_proto
        logits_tea_ir = torch.matmul(old_features_norm, proto_ir.t()) / T_proto
        
        loss_ir = F.kl_div(F.log_softmax(logits_stu_ir, dim=1), 
                           F.softmax(logits_tea_ir, dim=1), 
                           reduction='batchmean')

        return (loss_rgb + loss_ir) / 2.0 * 1000.0

