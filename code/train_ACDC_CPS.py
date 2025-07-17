import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.loss_helper import get_criterion, compute_unsupervised_loss_by_threshold, compute_ulb_hardness_all
from utils.utils import vote_threshold_label_selection_class_2D

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume1, test_single_volume2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Pseudo_Supervision', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=100000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=1,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--w_confident', type=float,
                    default=50, help='w_confident')
parser.add_argument('--w_confident_label', type=float,
                    default=50, help='w_confident_label')
parser.add_argument('--ratio', type=float,
                    default=0.6, help='ratio')
parser.add_argument('--ratio_label', type=float,
                    default=0.3, help='ratio')
parser.add_argument('--ressetion_factor',type=float,
                    default=3,help='ressetion_factor')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1":32 ,"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1300}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    w_confident = args.w_confident
    w_confident_label = args.w_confident_label
    ratio = args.ratio
    ratio_label = args.ratio_label
    ressetion_factor=args.ressetion_factor

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()

    pixel_thresholds1 = 0.9 * np.ones((args.labeled_bs, 256,256))
    pixel_thresholds2 = 0.9 * np.ones((args.labeled_bs, 256,256))
    pixel_thresholds1 = torch.from_numpy(pixel_thresholds1).cuda() 
    pixel_thresholds2 = torch.from_numpy(pixel_thresholds2).cuda()
    pixel_thresholds_true1 = 0.9 * np.ones((args.labeled_bs, 256,256))
    pixel_thresholds_true2 = 0.9 * np.ones((args.labeled_bs, 256,256))
    pixel_thresholds_true1 = torch.from_numpy(pixel_thresholds_true1).cuda() 
    pixel_thresholds_true2 = torch.from_numpy(pixel_thresholds_true2).cuda()

    cla1_0=0.9 * np.ones((args.labeled_bs))
    cla1_1=0.9* np.ones((args.labeled_bs))
    cla1_2=0.9* np.ones((args.labeled_bs))
    cla1_3=0.9* np.ones((args.labeled_bs))
    cla1_0=torch.from_numpy(cla1_0).cuda()
    cla1_1=torch.from_numpy(cla1_1).cuda()
    cla1_2=torch.from_numpy(cla1_2).cuda()
    cla1_3=torch.from_numpy(cla1_3).cuda()

    cla1_0_true=0.9 * np.ones((args.labeled_bs))
    cla1_1_true=0.9* np.ones((args.labeled_bs))
    cla1_2_true=0.9* np.ones((args.labeled_bs))
    cla1_3_true=0.9* np.ones((args.labeled_bs))
    cla1_0_true=torch.from_numpy(cla1_0_true).cuda()
    cla1_1_true=torch.from_numpy(cla1_1_true).cuda()
    cla1_2_true=torch.from_numpy(cla1_2_true).cuda()
    cla1_3_true=torch.from_numpy(cla1_3_true).cuda()

    cla2_0=0.9 * np.ones((args.labeled_bs))
    cla2_1=0.9* np.ones((args.labeled_bs))
    cla2_2=0.9* np.ones((args.labeled_bs))
    cla2_3=0.9* np.ones((args.labeled_bs))
    cla2_0=torch.from_numpy(cla2_0).cuda()
    cla2_1=torch.from_numpy(cla2_1).cuda()
    cla2_2=torch.from_numpy(cla2_2).cuda()
    cla2_3=torch.from_numpy(cla2_3).cuda()

    cla2_0_true=0.9 * np.ones((args.labeled_bs))
    cla2_1_true=0.9* np.ones((args.labeled_bs))
    cla2_2_true=0.9* np.ones((args.labeled_bs))
    cla2_3_true=0.9* np.ones((args.labeled_bs))
    cla2_0_true=torch.from_numpy(cla2_0_true).cuda()
    cla2_1_true=torch.from_numpy(cla2_1_true).cuda()
    cla2_2_true=torch.from_numpy(cla2_2_true).cuda()
    cla2_3_true=torch.from_numpy(cla2_3_true).cuda()

    di1_f=0.9 * np.ones((args.labeled_bs))
    di1_n=0.9 * np.ones((args.labeled_bs))
    di2_f=0.9 * np.ones((args.labeled_bs))
    di2_n=0.9 * np.ones((args.labeled_bs))
    di1_f=torch.from_numpy(di1_f).cuda()
    di1_n=torch.from_numpy(di1_n).cuda()
    di2_f=torch.from_numpy(di2_f).cuda()
    di2_n=torch.from_numpy(di2_n).cuda()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
   
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
   
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    alpha = 0.8

    writer = SummaryWriter(snapshot_path + '/log')
    
    logging.info("{} iterations per epoch".format(len(trainloader)))
    criterion_u = nn.CrossEntropyLoss(reduction='none')

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    p_threshold=0.95
    iterator = tqdm(range(max_epoch), ncols=70)

    

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            
            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
           
            with torch.no_grad():
                pred1_confidence, pred1_label = outputs_soft1.max(dim=1)
                pred2_confidence, pred2_label = outputs_soft2.max(dim=1)
    
                alpha = 0.5

                class1_0_f=(pred1_label[:args.labeled_bs]==0)
                class1_1_f=(pred1_label[:args.labeled_bs]==1)
                class1_2_f=(pred1_label[:args.labeled_bs]==2)
                class1_3_f=(pred1_label[:args.labeled_bs]==3)
                class1_0_n=(pred1_label[args.labeled_bs:]==0)
                class1_1_n=(pred1_label[args.labeled_bs:]==1)
                class1_2_n=(pred1_label[args.labeled_bs:]==2)
                class1_3_n=(pred1_label[args.labeled_bs:]==3)
                class2_0_f=(pred2_label[:args.labeled_bs]==0)
                class2_1_f=(pred2_label[:args.labeled_bs]==1) 
                class2_2_f=(pred2_label[:args.labeled_bs]==2)
                class2_3_f=(pred2_label[:args.labeled_bs]==3)
                class2_0_n=(pred2_label[args.labeled_bs:]==0)
                class2_1_n=(pred2_label[args.labeled_bs:]==1) 
                class2_2_n=(pred2_label[args.labeled_bs:]==2)
                class2_3_n=(pred2_label[args.labeled_bs:]==3)

                c0_0_f=(pred1_confidence[:args.labeled_bs]*class1_0_f).sum()/(class1_0_f.sum()+1)
                c0_1_f=(pred1_confidence[:args.labeled_bs]*class1_1_f).sum()/(class1_1_f.sum()+1)
                c0_2_f=(pred1_confidence[:args.labeled_bs]*class1_2_f).sum()/(class1_2_f.sum()+1)
                c0_3_f=(pred1_confidence[:args.labeled_bs]*class1_3_f).sum()/(class1_3_f.sum()+1)

                c0_0_n=(pred1_confidence[args.labeled_bs:]*class1_0_n).sum()/(class1_0_n.sum()+1)
                c0_1_n=(pred1_confidence[args.labeled_bs:]*class1_1_n).sum()/(class1_1_n.sum()+1)
                c0_2_n=(pred1_confidence[args.labeled_bs:]*class1_2_n).sum()/(class1_2_n.sum()+1)
                c0_3_n=(pred1_confidence[args.labeled_bs:]*class1_3_n).sum()/(class1_3_n.sum()+1)

                c1_0_f=(pred2_confidence[:args.labeled_bs]*class2_0_f).sum()/(class2_0_f.sum()+1)
                c1_1_f=(pred2_confidence[:args.labeled_bs]*class2_1_f).sum()/(class2_1_f.sum()+1)
                c1_2_f=(pred2_confidence[:args.labeled_bs]*class2_2_f).sum()/(class2_2_f.sum()+1)
                c1_3_f=(pred2_confidence[:args.labeled_bs]*class2_3_f).sum()/(class2_3_f.sum()+1)

                c1_0_n=(pred2_confidence[args.labeled_bs:]*class2_0_n).sum()/(class2_0_n.sum()+1)
                c1_1_n=(pred2_confidence[args.labeled_bs:]*class2_1_n).sum()/(class2_1_n.sum()+1)
                c1_2_n=(pred2_confidence[args.labeled_bs:]*class2_2_n).sum()/(class2_2_n.sum()+1)
                c1_3_n=(pred2_confidence[args.labeled_bs:]*class2_3_n).sum()/(class2_3_n.sum()+1)

                cla1_0 = alpha * c0_0_n + (1 - alpha) * cla1_0
                cla1_1 = alpha * c0_1_n + (1 - alpha) * cla1_1
                cla1_2 = alpha * c0_2_n + (1 - alpha) * cla1_2
                cla1_3 = alpha * c0_3_n + (1 - alpha) * cla1_3
                cla1=[cla1_0,cla1_1,cla1_2,cla1_3]

                cla1_0_true = alpha * c0_0_f + (1 - alpha) * cla1_0_true
                cla1_1_true = alpha * c0_1_f + (1 - alpha) * cla1_1_true
                cla1_2_true = alpha * c0_2_f + (1 - alpha) * cla1_2_true
                cla1_3_true = alpha * c0_3_f + (1 - alpha) * cla1_3_true
                cla1_true=[cla1_0_true,cla1_1_true,cla1_2_true,cla1_3_true]

                cla2_0 = alpha * c1_0_n + (1 - alpha) * cla2_0
                cla2_1 = alpha * c1_1_n + (1 - alpha) * cla2_1
                cla2_2 = alpha * c1_2_n + (1 - alpha) * cla2_2
                cla2_3 = alpha * c1_3_n + (1 - alpha) * cla2_3
                cla2=[cla2_0,cla2_1,cla2_2,cla2_3]

                cla2_0_true = alpha * c1_0_f + (1 - alpha) * cla2_0_true
                cla2_1_true = alpha * c1_1_f + (1 - alpha) * cla2_1_true
                cla2_2_true = alpha * c1_2_f + (1 - alpha) * cla2_2_true
                cla2_3_true = alpha * c1_3_f + (1 - alpha) * cla2_3_true
                cla2_true=[cla2_0_true,cla2_1_true,cla2_2_true,cla2_3_true]

                
                pixel_thresholds1 = alpha * pred1_confidence[args.labeled_bs:] + (1 - alpha) * pixel_thresholds1
                pixel_thresholds2 = alpha * pred2_confidence[args.labeled_bs:] + (1 - alpha) * pixel_thresholds2
               
                pixel_thresholds_true1 = alpha * pred1_confidence[:args.labeled_bs] + (1 - alpha) * pixel_thresholds_true1
                pixel_thresholds_true2 = alpha * pred2_confidence[:args.labeled_bs] + (1 - alpha) * pixel_thresholds_true2
                
            
            different1_confident_true,different2_confident_true, different_noconfident1_true,different_noconfident2_true,same_pred_confident1_true,same_pred_confident2_true,same_pred_noconfident1_true,same_pred_noconfident2_true,back1_true,back2_true = vote_threshold_label_selection_class_2D(outputs1[:args.labeled_bs], outputs2[:args.labeled_bs], cla1_true, cla2_true)
            
            different1_confident,different2_confident, different_noconfident1,different_noconfident2,same_pred_confident1,same_pred_confident2,same_pred_noconfident1,same_pred_noconfident2,back1,back2 = vote_threshold_label_selection_class_2D(outputs1[args.labeled_bs:], outputs2[args.labeled_bs:], cla1,cla2)
            
            sa_co_unlabel =w_confident
            sa_noco_unlabel = w_confident * np.exp(-(ratio) ** ressetion_factor)
            di_co_unlabel = w_confident * np.exp(-(ratio*2) ** ressetion_factor)
            di_noco_unlabel = w_confident * np.exp(-(ratio*3) ** ressetion_factor)
            

            sa_co_label = w_confident_label
            sa_noco_label = w_confident_label * np.exp(-(ratio_label) ** ressetion_factor)
            di_co_label = w_confident_label * np.exp(-(ratio_label*2) ** ressetion_factor)
            di_noco_label = w_confident_label * np.exp(-(ratio_label*3) ** ressetion_factor)
            


            loss_con1_cc = di_co_unlabel* criterion_u(outputs2[args.labeled_bs:], outputs1[args.labeled_bs:].softmax(dim=1).max(dim=1)[1].detach().long()) 
            loss_con2_cc = di_co_unlabel*  criterion_u(outputs1[args.labeled_bs:], outputs2[args.labeled_bs:].softmax(dim=1).max(dim=1)[1].detach().long()) 
            
            loss_dif1=  (loss_con1_cc*different1_confident).sum()/(different1_confident.sum()+1)
            loss_dif2=  (loss_con2_cc*different2_confident).sum()/(different2_confident.sum()+1)
            

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            
            

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            same_confident_loss1 = criterion_u(outputs2[args.labeled_bs:], pseudo_outputs1)*sa_co_unlabel
            same_confident_loss2 = criterion_u(outputs1[args.labeled_bs:], pseudo_outputs2)*sa_co_unlabel

        
            same_confident_loss1 = (same_confident_loss1*same_pred_confident1).sum()/(same_pred_confident1.sum()+1)
            same_confident_loss2 = (same_confident_loss2*same_pred_confident2).sum()/(same_pred_confident2.sum()+1)

            
            pseudo_supervision_co1 = criterion_u(outputs2[args.labeled_bs:], pseudo_outputs1)* sa_noco_unlabel
            pseudo_supervision_co2 = criterion_u(outputs1[args.labeled_bs:], pseudo_outputs2)* sa_noco_unlabel
            pseudo_supervision_co1 = (pseudo_supervision_co1*same_pred_noconfident1).sum()/(same_pred_noconfident1.sum()+1)
            pseudo_supervision_co2 = (pseudo_supervision_co2*same_pred_noconfident2).sum()/(same_pred_noconfident2.sum()+1)

            pseudo_supervision_no1 = criterion_u(outputs2[args.labeled_bs:], pseudo_outputs1)*di_noco_unlabel
            pseudo_supervision_no2 = criterion_u(outputs1[args.labeled_bs:], pseudo_outputs2)*di_noco_unlabel
            pseudo_supervision_no1 = (pseudo_supervision_no1*different_noconfident1).sum()/(different_noconfident1.sum()+1)
            pseudo_supervision_no2 = (pseudo_supervision_no2*different_noconfident2).sum()/(different_noconfident2.sum()+1)

            pseudo_supervision_back1 = criterion_u(outputs2[args.labeled_bs:], pseudo_outputs1)*sa_co_unlabel
            pseudo_supervision_back2 = criterion_u(outputs1[args.labeled_bs:], pseudo_outputs2)*sa_co_unlabel
            pseudo_supervision_back1 = (pseudo_supervision_back1*back1).sum()/(back1.sum()+1)
            pseudo_supervision_back2 = (pseudo_supervision_back2*back2).sum()/(back2.sum()+1)
            

            loss1_po_same = criterion_u(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_co_label
            loss2_po_same = criterion_u(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_co_label
            loss1_po_same = (loss1_po_same*same_pred_confident1_true).sum()/(same_pred_confident1_true.sum()+1)
            loss2_po_same = (loss2_po_same*same_pred_confident2_true).sum()/(same_pred_confident2_true.sum()+1)        

            loss1_po_different =  criterion_u(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*di_co_label
            loss2_po_different =   criterion_u(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*di_co_label
            loss1_po_different = (loss1_po_different*different1_confident_true).sum()/(different1_confident_true.sum()+1)
            loss2_po_different = (loss2_po_different*different2_confident_true).sum()/(different2_confident_true.sum()+1)

            loss1_po_back1 = criterion_u(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_co_label
            loss2_po_back2 = criterion_u(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_co_label
            loss1_po_back1 = (loss1_po_back1*back1_true).sum()/(back1_true.sum()+1)
            loss2_po_back2 = (loss2_po_back2*back2_true).sum()/(back2_true.sum()+1)

            loss1_po_noconfident1 = criterion_u(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_noco_label
            loss2_po_noconfident2 = criterion_u(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())*sa_noco_label
            loss1_po_noconfident1 = (loss1_po_noconfident1*same_pred_noconfident1_true).sum()/(same_pred_noconfident1_true.sum()+1)
            loss2_po_noconfident2 = (loss2_po_noconfident2*same_pred_noconfident2_true).sum()/(same_pred_noconfident2_true.sum()+1)

            loss1_po_nodifferent1 = criterion_u(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())* di_noco_label
            loss2_po_nodifferent2 = criterion_u(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())* di_noco_label
            loss1_po_nodifferent1 = (loss1_po_nodifferent1*different_noconfident1_true).sum()/(different_noconfident1_true.sum()+1)
            loss2_po_nodifferent2 = (loss2_po_nodifferent2*different_noconfident2_true).sum()/(different_noconfident2_true.sum()+1)


            loss_diff1=loss1_po_nodifferent1 +  loss1_po_different + loss_dif1 + pseudo_supervision_no1
            loss_diff2=loss2_po_nodifferent2 +  loss2_po_different + loss_dif2 + pseudo_supervision_no2

            loss_same1=loss1_po_same+same_confident_loss1+pseudo_supervision_co1+pseudo_supervision_back1+loss1_po_back1+loss1_po_noconfident1
            loss_same2=loss2_po_same+same_confident_loss2+pseudo_supervision_co2+pseudo_supervision_back2+loss2_po_back2+loss2_po_noconfident2
            
            
            model1_loss = loss1 +consistency_weight*(loss_same1 + loss_diff1)
            model2_loss = loss2 +consistency_weight*(loss_same2 + loss_diff2)
            loss = model1_loss + model2_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1
            #update_ema_variables(model1, model2, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            

            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume2(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume2(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(1)

    snapshot_path = "../Meta_SSL/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)


