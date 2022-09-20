import os
import cv2
import numpy as  np
from retinaface import Retinaface,face_de
from insightface import Insightface

retinaface_model='./sim_retinaface.onnx'
insightface_model = './webface_r50.onnx'
know_path= 'know_persons_face'
unkonw_path='./unknow_persons'

# L2正则化和距离计算
def face_compare(face1,face2):
    norm1=np.linalg.norm(face1)
    face1=face1/norm1
    norm2=np.linalg.norm(face2)
    face2=face2/norm2
    diff=np.subtract(face1,face2)  #对两个矩阵进行减法运算
    dist=np.sum(np.square(diff),1) #
    return dist

# 编码整个人脸文件夹
def get_embeding(dir_path,insightface_model_path):
    names=[]
    embedings=[]
    konw_list=os.listdir(dir_path)
    model = Insightface(insightface_model_path)
    for person in konw_list:
        person_path=know_path + '/'+ person
        name=person.split('.')[0]
        names.append(name)
        embeding,or_img=model.inference(person_path)
        embedings.append(embeding)
    return embedings,names

def draw(image,box_data,names):  #画框
    boxes=box_data[...,:4].astype(np.int32) #取整方便画框

    for box, name in zip(boxes, names):
        left, top, right, bottom = box
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(left, top, right, bottom))
        print(name)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image,'name:{}'.format(name),
                    (left, top ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
#   单张人脸编码
def face_embeding(insightface_model,face):
    model=Insightface(insightface_model)
    embeding,face=model.otherinfer(face)
    return embeding,face

def face_rg(retina_moel,insight_model,img,know_embedings,know_names):
    boxes,or_img=face_de(retina_moel,img)
    boxes=boxes.tolist()  #转为列表
    if boxes:  #判断是否检测到人脸
        boxes=np.array(boxes)
        boxes = boxes[..., :4]
        boxes[:,:4]=boxes[:,:4]*640
        boxes=boxes.astype(np.int32)
        out_names=[]
        # print(boxes)
        for box in boxes:
            left, top, right, bottom = np.maximum(box,0)
            face = or_img[top:bottom, left:right]
            if face.shape>=(10,10,3):   #大于该值的人脸进行识别
                embeding, face_img = face_embeding(insight_model, face)
                dis = []
                for konw in know_embedings:
                    distance = face_compare(embeding, konw)
                    dis.append(distance)
                index = np.argmin(dis)
                value = np.min(dis)
                if value < 1.4:
                    out_name = know_names[index]
                else:
                    out_name = 'unknow'
                out_names.append(out_name)
        draw(or_img,boxes,out_names)
    cv2.imshow('re',or_img)


if __name__=="__main__":
    know_embeding,know_name=get_embeding(know_path,insightface_model)
    #图片测试
    # img=cv2.imread('./1.jpg')
    # face_rg(retinaface_model, insightface_model, img, know_embeding, know_name)
    # cv2.waitKey(0)


    cap = cv2.VideoCapture(0)
    ref,frame=cap.read()
    print(frame.shape)
    if not ref:
        raise ValueError('未能正确读取摄像头')
    while(True):
        ref,frame=cap.read()
        if not ref:
            break
        face_rg(retinaface_model, insightface_model, frame, know_embeding, know_name)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()