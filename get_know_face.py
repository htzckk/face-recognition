import os
import cv2
import numpy as  np
from retinaface import Retinaface,face_de
from insightface import Insightface
person_dir='./person_imgs'
retina_model='./sim_retinaface.onnx'
face_dir='./know_persons_face'

# def get_person_img_lis(persondir):
#     out_list=[]
#     img_list=os.listdir(persondir)
#     for img in img_list:
#         img_path=persondir+'/'+img
#         out_list.append(img_path)
#     return out_list

def get_face(person_dir,retinaface_model,face_save_dir):
    img_list=os.listdir(person_dir)
    for img in img_list:
        name=img.split('.')[0]
        # print(name)
        img_path=person_dir+'/'+img
        img_data=cv2.imread(img_path)
        boxes,or_img=face_de(retinaface_model,img_data)
        boxes = boxes.tolist()  # 转为列表
        if boxes:  # 判断是否检测到人脸
            boxes = np.array(boxes)
            boxes = boxes[..., :4]
            boxes[:, :4] = boxes[:, :4] * 640
            boxes = boxes.astype(np.int32)
            for box in boxes:
                left, top, right, bottom = np.maximum(box, 0)
                face = or_img[top:bottom, left:right]
                cv2.imwrite(face_save_dir+'/'+name+'.jpg',face)
    cv2.destroyAllWindows()
if __name__=='__main__':
    get_face(person_dir,retina_model,face_dir)