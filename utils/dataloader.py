import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
'''
实际调用FRCNNDataset  代码如下
train_dataset   = FRCNNDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
lines就是 2007_train.txt 一共1593条，取0.9训练（1593*0.9=1433） 0.1测试（160）
lines[:num_train]就是取1433条信息，当然顺序打乱   见train.py     135行 np.random.shuffle(lines)
'''
class FRCNNDataset(Dataset):
    def __init__(self, train_lines, shape=[600,600], is_train=True):
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.shape = shape
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
        
    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h,w = self.shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data
            
        # resize image
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data


    def __getitem__(self, index):
        img, y = self.get_random_data(self.train_lines[index], random=self.is_train)
        img = np.transpose(img / 255.0, [2,0,1])
        box = y[:, :4]
        label = y[:, -1]
        return img, box, label

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels

if __name__ =="__main__":
    annotation_path = r'E:\faster-rcnn-bubblllling\faster-rcnn-pytorch-master\2007_train.txt'
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    train_dataset = FRCNNDataset(lines[:num_train], (640,640), is_train=True)
    from torch.utils.data import DataLoader
    gen = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=0, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    for iteration, batch in enumerate(gen):
        imgs, boxes, labels = batch[0], batch[1], batch[2]
        #print(imgs)
        print(imgs.shape)
        print(boxes)
        print(labels)
        break
'''
因为设batch_size =1
print(imgs.shape)   (1, 3, 640, 640)
print(boxes)        [array([[ 96.,  88., 606., 582.]])]
print(labels)       [array([0.])]

只用看这个py文件输入是什么。输出是什么
就够了
输入：   train_lines, shape=[600,600]  txt文件，resize尺寸
输出：  img ，box，labels
'''