import numpy as np

'''
一个方格9个框
对于一个方格，生成个9框
'''
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    '''
    生成 anchor shape为 3*3，4 = 9，4
    anchor_base [[0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]
                 [0. 0. 0. 0.]]
    '''
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            '''
            i=j=0时
            h = 16* 8 * 根号0.5  =90.5
            w = 16* 8 * 根号 1/0.5 =181 
            index = 0
            anchor_base[index, :] = -45.25,-90.5,45.25,90.5
            '''
            '''
            这里np.sqrt(1. / ratios[i])和 np.sqrt(ratios[i])，有一个数学公式
            首先base_size = 16 scale =8,16,32 则框的面积为 128 256 512
            以128^2 为例 求h w  ratio= [0.5,1,2]  
            h/w = ratio  h*w = 128^2 ,则 ratio *w^2 = 128^2  ,  np.sqrt(ratio) * w  = 128  ,w = 128 *  np.sqrt( 1/ratio), h = w*ratio = 128 *  np.sqrt(ratio)
            首先 
            '''
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

'''
nine_anchors = generate_anchor_base()
为9个anchor的左上角右下角坐标 
print(nine_anchors)
[[ -45.254833  -90.50967    45.254833   90.50967 ]= 128^2  面积
 [ -90.50967  -181.01933    90.50967   181.01933 ]  = 256^2
 [-181.01933  -362.03867   181.01933   362.03867 ]      =512^2
 [ -64.        -64.         64.         64.      ]= 128^2
 [-128.       -128.        128.        128.      ]  = 256^2
 [-256.       -256.        256.        256.      ]      =512^2
 [ -90.50967   -45.254833   90.50967    45.254833]= 128^2
 [-181.01933   -90.50967   181.01933    90.50967 ]  = 256^2
 [-362.03867  -181.01933   362.03867   181.01933 ]]     =512^2
'''

'''
38*38个方格 *9 =12996
# 将所有的锚点坐标存储并且将数组转为torch.cudatensor
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)
'''
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    '''
    fear_stride = 16
    shift_x = shift_x [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272
                         288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 528 544 560
                         576 592]
    shift_x.shape, shift_y.shape=(38,38)
    shift 为网格点坐标 shape =(1444,4)
    shift =[[  0   0   0   0]
             [ 16   0  16   0]
             [ 32   0  32   0]
             ...
             [560 592 560 592]
             [576 592 576 592]
             [592 592 592 592]]
    '''
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                      shift_x.ravel(),shift_y.ravel(),), axis=1)
    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]  # 9
    K = shift.shape[0]       #1444
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    #print(anchor.shape)  (1444, 9, 4)
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)
    #  38 *38 为600，600，3图像输入时的特征图尺寸大小
    height, width, feat_stride = 38,38,16
    anchors_all = _enumerate_shifted_anchor(nine_anchors,feat_stride,height,width)
    #  print(np.shape(anchors_all))  (12996, 4)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]

    for i in [108,109,110,111,112,113,114,115,116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)

    plt.show()
