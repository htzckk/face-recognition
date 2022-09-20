import os
import cv2
import numpy as np
import onnxruntime
import time

class Insightface():
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
        img = cv2.imread(img_path)  # 读取图片
        # print(img.shape)
        or_img = cv2.resize(img, (112, 112))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img=img-127.5
        img /= 127.5
        img = np.expand_dims(img, axis=0)
        # print(img)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img
    def otherinfer(self,img):
        or_img = cv2.resize(img, (112, 112))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img = img - 127.5
        img /= 127.5
        img = np.expand_dims(img, axis=0)
        # print(img)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img

# L2正则化和距离计算
def face_compare(face1,face2):
    norm1=np.linalg.norm(face1)
    face1=face1/norm1
    norm2=np.linalg.norm(face2)
    face2=face2/norm2
    diff=np.subtract(face1,face2)  #对两个矩阵进行减法运算
    dist=np.sum(np.square(diff),1) #
    return dist

def face_embeding(insightface_model,face):
    model=Insightface(insightface_model)
    embeding,face=model.otherinfer(face)
    return embeding,face

insightface_model='./webface_r50.onnx'
if __name__=="__main__":
    model=Insightface(insightface_model)
    output,re_img=model.inference('./liu_face2.jpg')
    output2,imgsda=model.inference('./liu_face.jpg')
    dis=face_compare(output2,output)
    print(dis)
    # print(output.shape)
    # print(output.dtype)

