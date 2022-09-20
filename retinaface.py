import os
import cv2
import numpy as np
import onnxruntime
import time
from itertools import product as product
from math import ceil


cfg_mnet={
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
}

class Retinaface():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    def inference(self, img_path):
        # img = cv2.imread(img_path)  # 读取图片
        img=letterbox_image(img_path,(640,640))
        or_img=np.array(img,np.uint8)
        img = img.astype(dtype=np.float32)
        img -= np.array((104, 117, 123),np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR2RGB和HWC2CHW
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)
        return pred, or_img

# 得到anchors
class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']
        self.steps      = cfg['steps']
        #---------------------------#
        #   图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output_np=np.array(anchors).reshape(-1,4)
        return output_np
# 填充灰条，实现resize
def letterbox_image(image, size):
    ih, iw, _   = np.shape(image)
    w, h        = size
    scale       = min(w/iw, h/ih)
    nw          = int(iw*scale)
    nh          = int(ih*scale)

    image       = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image
# 边框坐标解码
def decode(loc, priors, variances):
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
# 关键点解码
def decode_landm(pre, priors, variances):
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms

def draw(image,box_data):  #画框
    boxes=box_data[...,:4].astype(np.int32) #取整方便画框
    scores=box_data[...,4]

    for box, score in zip(boxes, scores):
        top, left, right, bottom = box
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        print(score)
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image,'score:{}'.format(score),
                    (top, left ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def pynms(dets, thresh): #非极大抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 ) * (x2 - x1 )
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] #置信度从大到小排序（下标）

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 计算相交面积
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 )  # 当两个框不想交时x22 - x11或y22 - y11 为负数，
                                           # 两框不相交时把相交面积置0
        h = np.maximum(0, y22 - y11 )  #retianface坐标为小数，不能加1

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)#计算IOU

        idx = np.where(ious <= thresh)[0]  #IOU小于thresh的框保留下来
        index = index[idx + 1]
        # print(index)

    return keep

def filter_box(org_box,conf_thres,iou_thres): #过滤掉无用的框
    conf = org_box[..., 4] > conf_thres #删除置信度小于conf_thres的BOX
    # print(conf)
    box = org_box[conf == True]
    output = []
    curr_cls_box = np.array(box)
    curr_out_box = pynms(curr_cls_box,iou_thres) #经过非极大抑制后输出的BOX下标
    for k in curr_out_box:
        output.append(curr_cls_box[k])  #利用下标取出非极大抑制后的BOX
    output = np.array(output)
    return output

# 人脸识别
def face_de(model_path,img):
    onnx_path=model_path
    model=Retinaface(onnx_path)
    data,or_img=model.inference(img)
    output_1=np.array(data[0]).squeeze()
    output_2=np.array(data[1]).squeeze()
    output_3=np.array(data[2]).squeeze()
    anchors=Anchors(cfg_mnet, image_size=(640, 640)).get_anchors()
    boxes = decode(output_1, anchors, cfg_mnet['variance'])
    landms = decode_landm(output_3, anchors, cfg_mnet['variance'])
    conf=output_2[:,1:2]
    boxs_conf=np.concatenate((boxes,conf,landms),-1)
    boxs_conf=filter_box(boxs_conf,0.5,0.5)
    return boxs_conf,or_img
    # boxs_conf=boxs_conf.tolist()
    # if boxs_conf:
    #     # print(bool(boxs_conf))
    #     boxs_conf=np.array(boxs_conf)
    #     boxs_conf[:, :4] = boxs_conf[:, :4] * 640
    #     boxs_conf[:,5:]=boxs_conf[:,5:]*640
    #     # return boxs_conf , or_img
    #     draw(or_img,boxs_conf)
    # cv2.imshow('re',or_img)
    # cv2.waitKey(0)

retinaface_mode='./sim_retinaface.onnx'


# 在本文件内进行代码调试
def curr_face_de(model_path,img):
    onnx_path = model_path
    model = Retinaface(onnx_path)
    data, or_img = model.inference(img)
    output_1 = np.array(data[0]).squeeze()
    output_2 = np.array(data[1]).squeeze()
    output_3 = np.array(data[2]).squeeze()
    anchors = Anchors(cfg_mnet, image_size=(640, 640)).get_anchors()
    boxes = decode(output_1, anchors, cfg_mnet['variance'])
    landms = decode_landm(output_3, anchors, cfg_mnet['variance'])
    conf = output_2[:, 1:2]
    boxs_conf = np.concatenate((boxes, conf, landms), -1)
    boxs_conf = filter_box(boxs_conf, 0.5, 0.5)
    boxs_conf=boxs_conf.tolist()
    if boxs_conf:
        # print(bool(boxs_conf))
        boxs_conf=np.array(boxs_conf)
        boxs_conf[:, :4] = boxs_conf[:, :4] * 640
        boxs_conf[:,5:]=boxs_conf[:,5:]*640
        # return boxs_conf , or_img
        draw(or_img,boxs_conf)
    cv2.imshow('re',or_img)
    # cv2.waitKey(0)


if __name__=="__main__":
    # 单张图片测试
    # img=cv2.imread('./1.jpg')
    # curr_face_de(retinaface_mode,img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 视频人脸识别测试
    cap = cv2.VideoCapture(0)
    ref,frame=cap.read()
    print(frame.shape)
    if not ref:
        raise ValueError('未能正确读取摄像头')
    while(True):
        ref,frame=cap.read()
        if not ref:
            break
        # cv2.imshow('re',frame)
        curr_face_de(retinaface_mode,frame)
        cv2.waitKey(1)
    cap.release()



#     onnx_path='./sim_retinaface.onnx'
#     model=Retinaface(onnx_path)
#     data,or_img=model.inference('./1.jpg')
#     output_1=np.array(data[0]).squeeze()
#     output_2=np.array(data[1]).squeeze()
#     output_3=np.array(data[2]).squeeze()
#     anchors=Anchors(cfg_mnet, image_size=(1280, 1280)).get_anchors()
#     boxes = decode(output_1, anchors, cfg_mnet['variance'])
#     landms = decode_landm(output_3, anchors, cfg_mnet['variance'])
#     conf=output_2[:,1:2]
#     boxs_conf=np.concatenate((boxes,conf,landms),-1)
#     boxs_conf=filter_box(boxs_conf,0.5,0.5)
#     scale=[640,640,640,640]
#     boxs_conf[:, :4] = boxs_conf[:, :4] * 1280
#     boxs_conf[:,5:]=boxs_conf[:,5:]*1280
#     draw(or_img,boxs_conf)
#     cv2.imshow('re',or_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # print(boxs_conf)
