import os

import numpy as np
import scipy.signal
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision.ops import nms

'''
resize成最小边为600
如果800*1200  -> 600*(1200* 6/8)    = 600 *900
    1500*900 -> (1500 * 6/9) *600  = 1000*600
'''
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


'''
#已知源框和目标框求出其位置偏差

由（x,y,x,y）转为（x,y,w,h）  左上角右下角点  -   中心点宽高
eps：非负数的最小值
再用eps代踢可能出现的 0 值   ，因为/height，/width 除数不可为0
最后计算出
'''
def bbox2loc(src_bbox, dst_bbox):
    '''由源框  xyxy-> xywh'''
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height
    '''由调整框  xyxy-> xywh'''
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)
    '''求得调整参数'''
    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

# anchor loc
'''
已知源bbox 和位置偏差dx，dy，dh，dw，求目标框G
在 rpn中 roi = loc2bbox(anchor, loc)  由anchor和loc调整 anchor 得到目标框roi
'''
def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)
    # 先转换为 中心点宽高
    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height
    # 取出调整参数
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]
    # 对 anchor中心点宽高调整
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height
    dst_bbox = torch.zeros_like(loc)
    # 调整后的参数放入 dst_bbox
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox   #Destination_box  缩写为dst_box   目标框


'''
This class encodes and decodes a set of bounding boxes into the representation used for training the regressors.
就是把Boundingbox 坐标转换为最后得到的回归量
'''
class DecodeBox():
    def __init__(self, std, mean, num_classes):
        self.std = std
        self.mean = mean
        self.num_classes = num_classes + 1    

    def forward(self, roi_cls_locs, roi_scores, rois, height, width, nms_iou, score_thresh):
        roi_cls_loc = (roi_cls_locs * self.std + self.mean)
        roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

        # 利用classifier网络的预测结果对建议框进行调整获得预测框
        roi = rois.view((-1, 1, 4)).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
        cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])

        # 防止预测框超出图片范围
        cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]).clamp(min=0, max=width)
        cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]).clamp(min=0, max=height)
        
        prob = F.softmax(roi_scores, dim=-1)

        class_conf, class_pred = torch.max(prob, dim=-1)
        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (class_conf >= score_thresh)
        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        cls_bbox = cls_bbox[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        output = []
        for l in range(1, self.num_classes):
            arg_mask = class_pred == l
            #------------------------------------------#
            #   取出对应的框和置信度
            #------------------------------------------#
            cls_bbox_l = cls_bbox[arg_mask, l, :]
            class_conf_l = class_conf[arg_mask]
            
            if len(class_conf_l) == 0:
                continue
            
            detections_class = torch.cat([cls_bbox_l, torch.unsqueeze(class_pred[arg_mask] - 1, -1).float(), torch.unsqueeze(class_conf_l, -1)], -1)
            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, -1],
                nms_iou
            )
            output.extend(detections_class[keep].cpu().numpy())

        return output

'''
iou
'''
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)



'''
输入只有anchor
'''
class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label>0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):

        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        '''
        调用bbox_iou函数计算anchor与bbox的IOU， ious：（N,K），N为anchor中第N个，K为bbox中第K个
        '''
        # print(anchor.shape) (22500,4)
        # print(bbox.shape) (1, 4) 或者(4, 4)或者。。。
        ious = bbox_iou(anchor, bbox)
        #print(ious.shape) (22500, 1)  只有一个box

        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))


        '''
        获得每一个先验框最对应的真实框  [num_anchors, ]  ,
        若第2个anchor与第一个bbo iou最大， 则值为1 ，22500个anchor对于22500个结果
        若只有一个bbox  则argmax_ious 全为0 len = 22500
        '''
        argmax_ious = ious.argmax(axis=1) # argmax[i]=j 表示第i个anchor与第j个bbox IOU值最大
        #print(argmax_ious.shape)  #(22500,)
        #print(argmax_ious)        #[0 0 3 ... 1 1 0]

        '''
        找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        即 每个anchor与所有bbox iou最大值 ，22500个anchor对于22500个结果 len = 22500
        '''
        max_ious = np.max(ious, axis=1)
        #print(max_ious.shape) #(22500,)
        #print(max_ious) #[0.06020499 0.16725855 0.09227666 ... 0.04929885 0.02398586 0.04176652]

        '''
        获得每一个真实框最对应的先验框[num_gt, ]
        求得每个gt对应最大anchor的编号 一个bbox对应一个结果  ，len = bbox数目
        '''
        gt_argmax_ious = ious.argmax(axis=0)  # 求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,K] ,表示第i个gt根第j个anchorIOU最大 得到的是j
        #print(gt_argmax_ious.shape)  (4,)  此时bbox数目一样为4
        #print(gt_argmax_ious)    [10633 13224 13137  9212]


        #   保证每一个真实框都存在对应的先验框
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious


    '''
    
    '''
    def _create_label(self, anchor, bbox):
        '''
        初始化的时候全部设置为-1
        正样本，label=1 ;负样本，label=0 ;剩下的既不是正样本也不是负样本，不用于最终训练，label=-1
        '''
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)


        '''argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        '''
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        '''
        print('argmax_ious',argmax_ious.shape) (22500,)
        print(argmax_ious) [0 0 2 ... 0 0 0]
        print('max_ious', max_ious.shape)  max_ious (22500,)
        print(max_ious) [0.         0.         0.00055754 ... 0.         0.         0.01003464]
        print('gt_argmax_ious', gt_argmax_ious.shape) v
        print(gt_argmax_ious)  [13240 11953 11376]
        '''
        
        '''
          如果小于门限值则设置为负样本
          如果大于门限值则设置为正样本
          每个真实框至少对应一个先验框
          self.neg_iou_thresh =0.3    self.pos_iou_thresh = 0.7
        '''

        #
        # ----------------------------------------------------- #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        '''
        在pos和neg中各随机选128个正样本和128个负样本，不足则补齐128
        将128个正样本的label设置为1，将128个负样本的label设置为0，剩下的anchors的labels都设为0
        '''
        n_pos = int(self.pos_ratio * self.n_sample) # 按照比例计算出正样本数量，pos_ratio=0.5，n_sample=256
        pos_index = np.where(label == 1)[0] # 得到所有正样本的索引
        # 如果选取出来的正样本数多于预设定的正样本数，则随机抛弃，将那些抛弃的样本的label设为-1
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        '''平衡正负样本，保持总数量为256'''

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        # 随机选择不要的负样本，个数为len(neg_index)-neg_index，label值设为-1
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
'''
ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，经过本ProposalTargetCreator的筛选产生128个用于自身的训练
输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）
'''
class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        '''首先将2000个roi和m个bbox给concatenate了一下成为新的roi（2000+m，4）,把bbox和roi拼接，增加正样本数量'''
        #print(roi.shape) torch.Size([600, 4])
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        #print(bbox.shape) (1, 4)
        #print(roi.shape) (601, 4)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = bbox_iou(roi, bbox)
        # print(iou.shape) (601, 1) 此时batch四则里只有一张图片，只有一个bbox
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            '''按行找到最大值，返回最大值对应的序号以及其真正的IOU。返回的是每个roi与**哪个**bbox的最大，以及最大的iou值'''
            '''获得每一个先验框最对应的真实框[num_roi,]'''
            gt_assignment = iou.argmax(axis=1)
            #print('gt_assignment.shape',gt_assignment.shape) (601,)
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            # 每个roi与对应bbox最大的iou
            max_iou = iou.max(axis=1)
            #print(max_iou)
            #print(max_iou.shape) (601,)

            #---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            # 从1开始的类别序号，给每个类得到真正的label
            gt_roi_label = label[gt_assignment] + 1

        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        #同样的根据iou的最大值将正负样本找出来，pos_iou_thresh=0.5
        '''把iou>0.5的找出来'''
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        #print(pos_index.shape)  (22,)
        #print(pos_index)  [  1   2   8  10  11  28  32  36  45  46  57  91  94 101 103 117 119 156 286 288 292 600]
        # 需要保留的roi个数（满足大于pos_iou_thresh条件的roi与64之间较小的一个）
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            # 找出的样本数目过多就随机丢掉一些
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #-----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        '''neg_iou_thresh_high=0.5，neg_iou_thresh_low=0.0   0<iou<0.5的找出来'''
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        #print('neg_index.shape',neg_index.shape) (578,)
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image  #负样本数 = 128 -正样本数
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))  #需要保留的roi个数（满足大于0小于neg_iou_thresh_hi条件的roi与64之间较小的一个
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        #---------------------------------------------------------#
        #   sample_roi      [n_sample, ]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        #---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)  #pos和neg放在一起

        #那么此时输出的128*4的sample_roi就可以去扔到 RoIHead网络里去进行分类与回归了
        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
        #求得 roi 和 gt 的loc差值
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) # 求这128个样本的groundtruth
        '''
        print('keep_inde',keep_index) 128个正负样本  128
        print(gt_assignment[keep_index])  分别对应哪个bbox [0 0 0 0 0 0   。。。 0 0 0 ]
        print('bbox[gt_assignment[keep_index]',bbox[gt_assignment[keep_index]])   分别对应bbox的坐标
        '''
        # ProposalTargetCreator首次用到了真实的21个类的label,且该类最后对loc进行了归一化处理，所以预测时要进行均值方差处理
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 负样本label 设为0
        return sample_roi, gt_roi_loc, gt_roi_label



'''
权重初始化，输入：net-----网络
'''
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    
class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
