
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils import loc2bbox

#ROI是Region of interest的简写，指的是faster rcnn结构中，经过rpn层后，产生的proposal对应的box框

'''
先看utils文件下几个函数
generate_anchor_base        一个方格生成9个框
_enumerate_shifted_anchor   n*n个方格，生成n*n*9个方格
loc2bbox 

输入
self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale) 对应__call__
其中参数：
        rpn_locs[i],        3*3卷积后1*1卷积得到的特征图               view(anchor_num, -1, 4)
                            
        rpn_fg_scores[i],   3*3卷积后1*1卷积得到的特征图  经过softmax   view(anchor_num, -1, 2)  -只取最后一维(dim=-1)-softmax- view(n,-1)
        anchor     所有anchor
        img_size  图像尺寸
        
通过训练好的第一个模块输出的约20000个anchor的4个位置参数，微调原图中所有anchor，生成20000个ROI
'''
class ProposalCreator():
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        '''
          loc           rpn_locs[i].shape, torch.Size([22500, 4])
          score          rpn_fg_scores[i].shape torch.Size([22500])
          anchor          anchor.shape (22500, 4)
        '''
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)
        if loc.is_cuda:
            anchor = anchor.cuda()

        '''
        将RPN网络预测结果转化成建议框
        利用loc对anchor进行初步调整
        '''
        roi = loc2bbox(anchor, loc)  # 将bbox转换为近似groudtruth的anchor(即rois)
        # print('roi.shape',roi.shape)  roi.shape torch.Size([22500, 4])


        '''防止建议框超出图像边缘'''
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        '''
        建议框的宽高的最小值不可以小于16  确保rois的长宽大于最小阈值
        起始就是过滤  面积小的proposals
        '''
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #print('len(keep)',len(keep))  len(keep) 22188 len(keep) 2239 len(keep) 22413 每一张图片都会赛选出少量违规roi
        #print(keep)    tensor([    0,     1,     2,  ..., 22497, 22498, 22499], device='cuda:0')
        roi = roi[keep, :]
        score = score[keep]  # 对剩下的ROIs进行打分（根据region_proposal_network中rois的预测前景概率）
        #print(score) tensor([0.0603, 0.0012, 0.0005,  ..., 0.0060, 0.0013, 0.0023], device='cuda:0',

        '''
        根据得分进行排序，取出建议框
        得到前 n_pre_nms个proposal的index 。nms前先对proposal的数量进行限制，减少计算量
        n_train_pre_nms=12000,n_test_pre_nms=3000,       
        '''
        order = torch.argsort(score, descending=True)

        if n_pre_nms > 0:
            order = order[:n_pre_nms]
            #print('order',order) order tensor([10950, 10941, 10953,  ..., 21020, 19519,  8649], device='cuda:0')
            #print('order.shape',order.shape)  order.shape torch.Size([12000])
        roi = roi[order, :]    #得到前order个proposal
        '''
        print(roi.shape) torch.Size([12000, 4])
        print(roi)
        tensor([[307.7609, 479.6769, 642.4996, 663.0145],
                [335.2711, 531.3202, 651.6342, 681.4247],
                [341.1168, 533.0464, 655.2969, 681.1409],
                ...,
                [535.8418, 113.6923, 566.1920, 138.5561],
                [551.8418, 113.6923, 582.1920, 138.5561],
                [567.8418, 113.6923, 598.1920, 138.5561]], device='cuda:0',
        '''
        score = score[order]  #得到前order的概率值
        '''
        print(score.shape)torch.Size([12000])
        print(score)
        tensor([9.3939e-01, 9.3585e-01, 9.2673e-01,  ..., 4.4731e-06, 4.4731e-06,
                4.4731e-06], device='cuda:0', grad_fn=<IndexBackward>)
        '''
        '''
        对建议框进行非极大抑制
        n_train_post_nms=600,n_test_post_nms=300,
        '''
        #
        #-----------------------------------#
        keep = nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]  #在nms后也对proposal的数量进行限制，减少计算量
        roi = roi[keep]            #得到nms后前order的概率值
        return roi


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            mode = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        self.feat_stride = feat_stride  # 16
        self.proposal_layer = ProposalCreator(mode)
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        #print('x.shape',x.shape)  x.shape torch.Size([1, 1024, 50, 50])
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)      # 1*1卷积
        #  b_s,channel,w,h -> b_s,w,h,channel  -> anchor_num,-1,4
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #print('rpn_locs.shape', rpn_locs.shape) rpn_locs.shape torch.Size([1, 22500, 4])
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        #b_s,channel,w,h -> b_s,w,h,channel  -> anchor_num,-1,2
        rpn_scores = self.score(x)  # 1*1卷积
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        #print('rpn_scores.shape',rpn_scores.shape) rpn_scores.shape torch.Size([1, 22500, 2])
        
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        #print('rpn_softmax_scores.shape',rpn_softmax_scores.shape) rpn_softmax_scores.shape torch.Size([1, 22500, 2])
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        #print('rpn_fg_scores.shape',rpn_fg_scores.shape) rpn_fg_scores.shape torch.Size([1, 22500])


        '''生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)'''
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        #print(anchor.shape) (22500, 4)

        
        rois = list()
        roi_indices = list()
        for i in range(n):
            '''
            rpn_locs[i].shape, torch.Size([22500, 4]) 
            rpn_fg_scores[i].shape torch.Size([22500])
            anchor.shape (22500, 4)
            '''
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor



def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
