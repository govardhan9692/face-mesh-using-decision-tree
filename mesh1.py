import cv2
import mediapipe as mp
import itertools
import math
import numpy as np
import cvzone
import csv
from threading import Thread
from cvzone.FaceMeshModule import FaceMeshDetector
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
co=0
ab=0



LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))

RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
# print(RIGHT_EYE_INDEXES)

FACEMESH_FACE_OVAL_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
# print((FACEMESH_FACE_OVAL_INDEXES))

FACEMESH_TESSELATION_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_TESSELATION)))
# print(FACEMESH_TESSELATION_INDEXES)

'''left_eye = []
right_eye = []
face_oval = []

F_T_I = []
Distance = []
Distance2 = []

xl_coord = []  ### LEFT EYE  X-C0ORDINATES
yl_coord = []  ### LEFT EYE  Y-C0ORDINATES
zl_coord = []  ### LEFT EYE  Z-C0ORDINATES
L_EYE = [xl_coord, yl_coord, zl_coord]

xr_coord = []  ### RIGHT EYE X-C0ORDINATES
yr_coord = []  ### RIGHT EYE Y-C0ORDINATES
zr_coord = []  ### RIGHT EYE   Z-C0ORDINATES
R_EYE = [xr_coord, yr_coord, zr_coord]'''

def cam_distance():
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # Drawing
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # # Finding the Focal Length
        # d = 50
        # f = (w*d)/W
        # print(f)

        # Finding distance
        f = 800
        d = (W * f) // w
        #c=int(d)
        #print(c)
        ds.append(d)


def face_dis():


    for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
        for RIGHT_EYE in RIGHT_EYE_INDEXES:
            right_eye.append(face_landmarks.landmark[RIGHT_EYE])

        for FACEMESH_TESSELATION_INDEX in FACEMESH_TESSELATION_INDEXES:
            F_T_I.append(face_landmarks.landmark[FACEMESH_TESSELATION_INDEX])

        #print(F_T_I[1].x)
        point_x=F_T_I[1].x
        #print(point_x)
        point_y=F_T_I[1].y
        point_z=F_T_I[1].z
        axis_x, axis_y, axis_z = 0, 0, 1
        dot_product = point_x * axis_x + point_y * axis_y + point_z * axis_z

        point_magnitude = math.sqrt(point_x ** 2 + point_y ** 2 + point_z ** 2)
        axis_magnitude = math.sqrt(axis_x ** 2 + axis_y ** 2 + axis_z ** 2)
        cosine = dot_product / (point_magnitude * axis_magnitude)
        angle_degrees = math.degrees(math.acos(cosine))
        #print(angle_degrees)









        #print(angle_degrees)
        ### TOP TO BOTTOM
        '''for i in range(len(top_bottom)-1):
            Top_to_Bottom.append(math.sqrt((F_T_I[top_bottom[i]].x - F_T_I[top_bottom[i+1]].x) ** 2 + (F_T_I[top_bottom[i]].y - F_T_I[top_bottom[i+1]].y) ** 2 + (
                    F_T_I[top_bottom[i]].z - F_T_I[top_bottom[i+1]].z) ** 2))'''

        for i in range(len(nose_len) - 1):
            Top_to_Bottom.append(math.sqrt((F_T_I[nose_len[i]].x - F_T_I[nose_len[i + 1]].x) ** 2 + (
                        F_T_I[nose_len[i]].y - F_T_I[nose_len[i + 1]].y) ** 2 + (
                                                   F_T_I[nose_len[i]].z - F_T_I[nose_len[i + 1]].z) ** 2))

        '''Top_to_Bottom.append(math.sqrt((F_T_I[168].x - F_T_I[6].x) ** 2 + (F_T_I[168].y - F_T_I[6].y) ** 2 + (
                F_T_I[168].z - F_T_I[6].z) ** 2))'''
        '''Distance.append(math.sqrt((F_T_I[162].x - F_T_I[152].x) ** 2 + (F_T_I[162].y - F_T_I[152].y) ** 2 + (
                F_T_I[162].z - F_T_I[152].z) ** 2))'''

        '''for i,j in zip(outer2,outer3):
            Distance2.append(math.sqrt((F_T_I[i].x - F_T_I[j].x) ** 2 + (F_T_I[i].y - F_T_I[j].y) ** 2 + (
                    F_T_I[i].z - F_T_I[j].z) ** 2))'''

        '''for i, j in zip(left_corner1, left_corner2):
            Distance3.append(math.sqrt((F_T_I[i].x - F_T_I[j].x) ** 2 + (F_T_I[i].y - F_T_I[j].y) ** 2 + (
                    F_T_I[i].z - F_T_I[j].z) ** 2))
        for i, j in zip(t_b, t_b2):
            Distance3.append(math.sqrt((F_T_I[i].x - F_T_I[j].x) ** 2 + (F_T_I[i].y - F_T_I[j].y) ** 2 + (
                    F_T_I[i].z - F_T_I[j].z) ** 2))'''

        ### LEFT_TO_RIGHT
        for i in range(len(left_to_right)-1):
            Left_to_Right.append (math.sqrt((F_T_I[left_to_right[i]].x - F_T_I[left_to_right[i+1]].x) ** 2 + (F_T_I[left_to_right[i]].y - F_T_I[left_to_right[i+1]].y) ** 2 + (
                    F_T_I[left_to_right[i]].z - F_T_I[left_to_right[i+1]].z) ** 2))

        ### LEFT_TO_RIGHT1
        for i in range(len(left_to_right1)-1):
            Left_to_Right1.append(math.sqrt((F_T_I[left_to_right1[i]].x - F_T_I[left_to_right1[i+1]].x) ** 2 + (F_T_I[left_to_right1[i]].y - F_T_I[left_to_right1[i+1]].y) ** 2 + (
                    F_T_I[left_to_right1[i]].z - F_T_I[left_to_right1[i+1]].z) ** 2))

        ### LEFT_TO_RIGHT2
        for i in range(len(left_to_right1) - 1):
            Left_to_Right2.append(math.sqrt((F_T_I[left_to_right2[i]].x - F_T_I[left_to_right2[i + 1]].x) ** 2 + ( F_T_I[left_to_right2[i]].y - F_T_I[left_to_right2[i + 1]].y) ** 2 + (
                    F_T_I[left_to_right2[i]].z - F_T_I[left_to_right2[i + 1]].z) ** 2))

        ### OUTER_1
        for i in range(len(outer1) - 1):
            Outer_1.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer1[i + 1]].x) ** 2 + (
                        F_T_I[outer1[i]].y - F_T_I[outer1[i + 1]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer1[i + 1]].z) ** 2))

        ### OUTER_2
        for i in range(len(outer2) - 1):
            Outer_2.append(math.sqrt((F_T_I[outer2[i]].x - F_T_I[outer2[i + 1]].x) ** 2 + (
                    F_T_I[outer2[i]].y - F_T_I[outer2[i + 1]].y) ** 2 + (F_T_I[outer2[i]].z - F_T_I[outer2[i + 1]].z) ** 2))

        ### outer_3
        for i in range(len(outer3) - 1):
            Outer_3.append(math.sqrt((F_T_I[outer3[i]].x - F_T_I[outer3[i + 1]].x) ** 2 + (
                    F_T_I[outer3[i]].y - F_T_I[outer3[i + 1]].y) ** 2 + (F_T_I[outer3[i]].z - F_T_I[outer3[i + 1]].z) ** 2))



        ### OUTER_1 AND OUTER_2
        for i in range(36):
            Outer1_Outer2_dis.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer2[i]].x) ** 2 + (
                        F_T_I[outer1[i]].y - F_T_I[outer2[i]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer2[i]].z) ** 2))



        ### OUTER_1 AND OUTER_3
        for i in range(36):
            Outer1_Outer3_dis.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer3[i]].x) ** 2 + (
                    F_T_I[outer1[i]].y - F_T_I[outer3[i]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer3[i]].z) ** 2))


        ### OUTER_2 AND OUTER_3
        for i in range(36):
            Outer2_Outer3_dis.append(math.sqrt((F_T_I[outer2[i]].x - F_T_I[outer3[i]].x) ** 2 + (
                    F_T_I[outer2[i]].y - F_T_I[outer3[i]].y) ** 2 + (F_T_I[outer2[i]].z - F_T_I[outer3[i]].z) ** 2))


        ### NOSE_OUTER1
        for i in range(len(nose_outer1)-1):
           Nose_outer1.append(math.sqrt((F_T_I[nose_outer1[i]].x - F_T_I[nose_outer1[i+1]].x) ** 2 + (
                    F_T_I[nose_outer1[i]].y - F_T_I[nose_outer1[i+1]].y) ** 2 + (F_T_I[nose_outer1[i]].z - F_T_I[nose_outer1[i+1]].z) ** 2))

        ### NOSE_OUTER2
        for i in range(len(nose_outer2)-1):
           Nose_outer2.append(math.sqrt((F_T_I[nose_outer2[i]].x - F_T_I[nose_outer2[i + 1]].x) ** 2 + (
                    F_T_I[nose_outer2[i]].y - F_T_I[nose_outer2[i + 1]].y) ** 2 + (F_T_I[nose_outer2[i]].z - F_T_I[nose_outer2[i + 1]].z) ** 2))

        ### NOSE_OUTER3
        for i in range(len(nose_outer3)-1):
           Nose_outer3.append(math.sqrt((F_T_I[nose_outer3[i]].x - F_T_I[nose_outer3[i + 1]].x) ** 2 + (
                    F_T_I[nose_outer3[i]].y - F_T_I[nose_outer3[i + 1]].y) ** 2 + (F_T_I[nose_outer3[i]].z - F_T_I[nose_outer3[i + 1]].z) ** 2))

        ### NOSE_LENGTH
        for i in range(len(nose_len)-1):
            Nose_len.append(math.sqrt((F_T_I[nose_len[i]].x - F_T_I[nose_len[i + 1]].x) ** 2 + (
                    F_T_I[nose_len[i]].y - F_T_I[nose_len[i + 1]].y) ** 2 + (F_T_I[nose_len[i]].z - F_T_I[nose_len[i + 1]].z) ** 2))

        ## Nose_w1
        for i in range(len(nose_w1)-1):
            Nose_w1.append(math.sqrt((F_T_I[nose_w1[i]].x - F_T_I[nose_w1[i + 1]].x) ** 2 + (
                    F_T_I[nose_w1[i]].y - F_T_I[nose_w1[i + 1]].y) ** 2 + (F_T_I[nose_w1[i]].z - F_T_I[nose_w1[i + 1]].z) ** 2))

        ##Nose_w2
        for i in range(len(nose_w2)-1):
            Nose_w2.append(math.sqrt((F_T_I[nose_w2[i]].x - F_T_I[nose_w2[i + 1]].x) ** 2 + (
                    F_T_I[nose_w2[i]].y - F_T_I[nose_w2[i + 1]].y) ** 2 + (F_T_I[nose_w2[i]].z - F_T_I[nose_w2[i + 1]].z) ** 2))
        ##Nose_w3
        for i in range(len(nose_w3)-1):
            Nose_w3.append(math.sqrt((F_T_I[nose_w3[i]].x - F_T_I[nose_w3[i + 1]].x) ** 2 + (
                    F_T_I[nose_w3[i]].y - F_T_I[nose_w3[i + 1]].y) ** 2 + (F_T_I[nose_w3[i]].z - F_T_I[nose_w3[i + 1]].z) ** 2))
        ##Nose_w4
        for i in range(len(nose_w4)-1):
            Nose_w4.append(math.sqrt((F_T_I[nose_w4[i]].x - F_T_I[nose_w4[i + 1]].x) ** 2 + (
                    F_T_I[nose_w4[i]].y - F_T_I[nose_w4[i + 1]].y) ** 2 + (F_T_I[nose_w4[i]].z - F_T_I[nose_w4[i + 1]].z) ** 2))
        ##Nose_w5
        for i in range(len(nose_w5)-1):
            Nose_w5.append(math.sqrt((F_T_I[nose_w5[i]].x - F_T_I[nose_w5[i + 1]].x) ** 2 + (
                    F_T_I[nose_w5[i]].y - F_T_I[nose_w5[i + 1]].y) ** 2 + (F_T_I[nose_w5[i]].z - F_T_I[nose_w5[i + 1]].z) ** 2))
        ##Nose_w6
        for i in range(len(nose_w6)-1):
            Nose_w6.append(math.sqrt((F_T_I[nose_w6[i]].x - F_T_I[nose_w6[i + 1]].x) ** 2 + (
                    F_T_I[nose_w6[i]].y - F_T_I[nose_w6[i + 1]].y) ** 2 + (F_T_I[nose_w6[i]].z - F_T_I[nose_w6[i + 1]].z) ** 2))

        ##Nose_len
        for i in range(len(nose_len)-1):
            Nose_len.append(math.sqrt((F_T_I[nose_len[i]].x - F_T_I[nose_len[i + 1]].x) ** 2 + (
                    F_T_I[nose_len[i]].y - F_T_I[nose_len[i + 1]].y) ** 2 + (F_T_I[nose_len[i]].z - F_T_I[nose_len[i + 1]].z) ** 2))



        ### OUTER_2 AND OUTER_3
        '''for i in range(36):
            Distance.append(math.sqrt((F_T_I[outer2[i]].x - F_T_I[outer3[j]].x) ** 2 + (
                    F_T_I[outer2[i]].y - F_T_I[outer3[j]].y) ** 2 + (F_T_I[outer2[i]].z - F_T_I[outer3[j]].z) ** 2))'''
        return angle_degrees











#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
move_amount=0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the video properties

detector = FaceMeshDetector(maxFaces=1)

filename= "data3.csv"
face_coord=[]
ds=[]
zoom=0.4
count=0
mo=0
so=0

#print(face_coord)
with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    image1 = cv2.flip(image, 1)

    #img, faces = detector.findFaceMesh(image, draw=False)
    #zoom1(zoom,image)


    if zoom:
        h, w = image1.shape[:2]
        # print(h,w)
        cx, cy = w / 2, h / 2
        fx, fy = 1 / zoom, 1 / zoom
        M = cv2.getRotationMatrix2D((cx, cy), 0, fx)
        frame = cv2.warpAffine(image1, M, (w, h))
        img, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        # Drawing
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # # Finding the Focal Length
        # d = 70
        # f = (w*d)/W
        # print(f)

        # Finding distance
        f = 840
        d = (W * f) // w
        #print(int(d))
        if d >= 40:
            zoom -= 0.0044
            b = zoom
        #elif d in range(68, 74):
            #zoom = b
        elif d <= 40:
            zoom += 0.0044

    face2=cv2.ellipse(img=image1,center=(300,225),axes=(150,200),angle=180,startAngle=0,endAngle=360,color=(0,255,0),thickness=4,lineType=16)







                                                                                                                                    #cvzone.putTextRect(image, f'Depth: {int(d)}cm',

    left_eye = []
    right_eye = []
    face_oval = []



    F_T_I = []

    Top_to_Bottom=[]
    Left_to_Right=[]
    Left_to_Right1=[]
    Left_to_Right2=[]



    Right_corner_dis = []
    Left_corner_dis = []


    Outer_1=[]
    Outer_2 = []
    Outer_3 = []

    Outer1_Outer2_dis=[]
    Outer1_Outer3_dis=[]
    Outer2_Outer3_dis=[]
    ds = []


    Nose_outer1=[]
    Nose_outer2=[]
    Nose_outer3=[]
    Nose_len=[]
    Nose_w1=[]
    Nose_w2=[]
    Nose_w3=[]
    Nose_w4=[]
    Nose_w5=[]
    Nose_w6=[]

    Outer3_Nose_r1=[]
    Outer3_Nose_r2=[]
    Outer3_Nose_r3=[]
    Outer3_Nose_r4=[]
    Outer3_Nose_l1=[]
    Outer3_Nose_l2=[]
    Outer3_Nose_l3=[]
    Outer3_Nose_l4=[]

    Eye_to_Outer3_r1=[]
    Eye_to_Outer3_r2=[]
    Eye_to_Outer3_r3=[]
    Eye_to_Outer3_r4=[]
    Eye_to_Outer3_r5=[]
    Eye_to_Outer3_r6=[]
    Eye_to_Outer3_l1=[]
    Eye_to_Outer3_l2=[]
    Eye_to_Outer3_l3=[]
    Eye_to_Outer3_l4=[]
    Eye_to_Outer3_l5=[]
    Eye_to_Outer3_l6=[]






    xl_coord = []  ### LEFT EYE  X-C0ORDINATES
    yl_coord = []  ### LEFT EYE  Y-C0ORDINATES
    zl_coord = []  ### LEFT EYE  Z-C0ORDINATES
    L_EYE = [xl_coord, yl_coord, zl_coord]

    xr_coord = []  ### RIGHT EYE X-C0ORDINATES
    yr_coord = []  ### RIGHT EYE Y-C0ORDINATES
    zr_coord = []  ### RIGHT EYE   Z-C0ORDINATES
    R_EYE = [xr_coord, yr_coord, zr_coord]

    right_corner1=[10,109,67,103,54,21,162,127,234,93,132,54,172,136,150,149,176,148,152]
    #right_corner2=[109,67,103,54,21,162,127,234,93,132,54,172,136,150,149,176,148,152]

    left_corner1=[10,338,297,332,284,251,289,356,454,323,361,288,397,365,379,378,400,377,152]
    #left_corner2=[338,297,332,284,251,289,356,454,323,361,288,397,365,379,378,400,377,152]

    top_bottom=[10,151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175,152]
    #top_bottom2 = [151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175,152]
    left_to_right=[93,137,123,50,101,100,47,114,188,122,6,351,412,343,277,329,330,280,352,366,323]
    left_to_right1=[132,177,147,187,205,36,142,126,217,174,196,197,419,399,437,355,371,266,425,411,376,401,361]
    left_to_right2=[136,135,214,216,206,203,129,49,131,134,51,5,281,363,360,279,358,423,426,436,434,364,365]

    outer1=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
    outer2=[151,337,299,333,298,301,368,264,447,366,401,435,367,364,394,395,369,396,175,171,140,170,169,135,138,215,177, 137,227,34,139,71,68,104,69,108,151]
    outer3=[9,336,296,334,293,300,383,372,345,352,376,433,416,434,430,431,262,428,199,208,32,211,210,214,192,213,147,123,116,143,156,70,63,105,66,107,9]

    nose_outer1=[168,417,413,414,398,382,341,453,357,343,437,420,360,344,438,309,250,462,370,94,141,242,20,79,218,115,131,198,217,114,128,233,112,155,173,190,189,193,168]
    nose_outer2=[413,465,412,399,456,363,440,457,461,434,19,125,241,237,220,134,236,174,188,245,189,193,168,417,413]
    nose_outer3=[417,351,419,248,281,275,274,354,94,125,44,45,51,3,196,122,193,168,417]
    nose_len=[168,6,197,195,5,4,1]
    nose_w1=[114,188,122,6,351,412,343]
    nose_w2=[217,174,196,197,419,399,437]
    nose_w3=[198,236,3,195,248,456,420]
    nose_w4=[131,134,51,5,281,363,360]
    nose_w5=[115,220,45,4,275,440,344]
    nose_w6=[218,237,44,1,274,457,438]


    outer3_nose_r1=[116,117,118,119,120,121,128]
    outer3_nose_r2=[123,50,101,100,47,114]
    outer3_nose_r3=[147,187,305,36,142,126,217]
    outer3_nose_r4=[192,207,206,203,129,49]
    outer3_nose_l1=[345,346,347,348,349,350,357]
    outer3_nose_l2=[352,280,330,329,277,343]
    outer3_nose_l3=[376,411,425,266,371,355,437]
    outer3_nose_l4=[416,427,426,423,358,279,360]


    eye_to_outer3_r1=[130,226,35,143]
    eye_to_outer3_r2=[25,31,111,116]
    eye_to_outer3_r3=[110,228,117,123]
    eye_to_outer3_r4=[24,229,118,50,187,192]
    eye_to_outer3_r5=[23,230,119,101,205,207,214]
    eye_to_outer3_r6=[22,231,120,100,36,206,216,212,210]
    eye_to_outer3_l1=[359,446,265,372]
    eye_to_outer3_l2=[255,261,340,345]
    eye_to_outer3_l3=[339,448,346,352]
    eye_to_outer3_l4=[254,449,347,280,411,416]
    eye_to_outer3_l5=[253,450,348,330,425,427,434]
    eye_to_outer3_l6=[252,451,349,329,266,426,436,432,430]









    #scale=2)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.


    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = face_mesh.process(frame)







    landmark_name = "LEFT_EYE_INNER"

    # Draw the face mesh annotations on the image.
    frame.flags.writeable = True
    image2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    landmark= results.multi_face_landmarks
    if landmark:
      for face_landmarks in results.multi_face_landmarks:

        mp_drawing.draw_landmarks(
            image=image2,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image2,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image2,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())



        #print("distance from cam:", distance," And face distance=",Distance)


        '''threads=[
        Thread(target=face_dis())]

        for thread in threads:
            thread.start()

        # Wait until both Func1 and Func2 have finished
        for thread in threads:
            thread.join()
        #for i, j in zip(Distance, Distance2):
            #print(ds,i,j)
        #print(Distance2)'''

        #print(mo)


            #mo+=1
    #print("mo",mo)
        target = face_dis()
        t2 = round(target, 2)
        print(t2)
        '''if t2<97.45:
            translation_matrix = np.float32([[1, 0, -ab], [0, 1, 0]])
            translated_frame = cv2.warpAffine(image, translation_matrix, (frame.shape[1], frame.shape[0]))




            cv2.imshow('MediaPipe Face Mesh', translated_frame)
            cv2.waitKey(1)
            ab+=1'''







        #
        # Define the translation matrix to move the frame left

            # Display the frames side by side'''













        with open(filename, mode='a', newline='') as csv_file:

            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)
            sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum5 = 0
            sum6 = 0
            sum7 = 0
            sum8 = 0
            sum9 = 0
            sum10=0
            sum11= 0
            sum12= 0
            sum13= 0
            sum14= 0
            sum15= 0
            sum16= 0
            sum17= 0
            sum18= 0
            sum19= 0
            sum20= 0

            #co=0
            #for i,j,k,l in zip(Top_to_Bottom,Outer_1,Outer_2,Outer_3):
            for a,b,c,d1,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t in zip(Top_to_Bottom,Left_to_Right,Left_to_Right1,Left_to_Right2,Outer_1, Outer_2, Outer_3, Outer1_Outer2_dis, Outer1_Outer3_dis, Outer2_Outer3_dis, Nose_outer1, Nose_outer2, Nose_outer3, Nose_len, Nose_w1, Nose_w2, Nose_w3, Nose_w4, Nose_w5, Nose_w6):

                #Top_to_Bottom, Left_to_Right, Left_to_Right1, Left_to_Right2, Outer_1, Outer_2, Outer_3, Outer1_Outer2_dis, Outer1_Outer3_dis, Outer2_Outer3_dis, Nose_outer1, Nose_outer2, Nose_outer3, Nose_len, Nose_w1, Nose_w2, Nose_w3, Nose_w4, Nose_w5, Nose_w6):


                #Nose_len, Nose_w1, Nose_w2,
                '''Left_corner_dis,Outer_1, Outer_2,Outer_3, Outer1_Outer2_dis, Outer2_Outer3_dis, Outer1_Outer3_dis'''
                sum1+=a
                sum1=round(sum1,3)

                sum2+=b
                sum2=round(sum2,3)

                sum3+=c
                sum3=round(sum3,3)
                #print(sum3)

                sum4+=d1
                sum4 = round(sum4, 3)

                sum5+=e
                sum5 = round(sum5, 3)

                sum6+=f
                sum6 = round(sum6,3)

                sum7+=g
                sum7 = round(sum7,3)

                sum8+=h
                sum8 = round(sum8,3)

                sum9+=i
                sum9 = round(sum9,3)

                sum10 += j
                sum10 = round(sum10,3)

                sum11 += k
                sum11 = round(sum11,3)

                sum12 += l
                sum12 = round(sum12,3)

                sum13 += m
                sum13 = round(sum13,3)

                sum14 += n
                sum14= round(sum14, 3)

                sum15+= o
                sum15= round(sum15, 3)

                sum16+= p
                sum16= round(sum16, 3)

                sum17+= q
                sum17= round(sum17, 3)

                sum18+= r
                sum18= round(sum18, 3)

                sum19+=s
                sum19 = round(sum19, 3)

                sum20 += t
                sum20 = round(sum20, 3)





                #csv_writer.writerow([d,i,j,k,l,m,n,])
                #flipped_frame = cv2.flip(image, 1)
            #co=0
            #cv2.putText(image, {str(t2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            #cv2.putText(image2, f'keep face between 97.45 to 97.55', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image2, str(t2), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
            if d==40 and t2>97.0 and t2<97.90:
                # '''and t2>97.45 and t2<97.55'''

                # data = pd.read_csv('data1.csv')
                # d1 = data.head(1500)

                # # Separate the features and the target variable
                # X = d1.drop(['V'], axis=1)

                # # Separate the features and the target variable

                # y = data['V']

                # # Split the data into training and testing sets
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # # Create a decision tree classifier
                # clf = DecisionTreeClassifier()

                # # Train the classifier on the training data
                # clf.fit(X_train, y_train)

                # # Predict new data
                # # new_data = pd.read_csv('data1.csv')  # Replace 'new_data.csv' with your new dataset
                # predictions = clf.predict([[t2,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15,sum16,sum17,sum18,sum19,sum20]])
                # # Print the predictions
                # print(predictions)


                # cv2.putText(image2, str(predictions), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if co<500:
                    cv2.putText(image2, str(co), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    #time.sleep(5)
                    csv_writer.writerow([t2,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15,sum16,sum17,sum18,sum19,sum20,'Ethan'])
                    print(t2,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15,sum16,sum17,sum18,sum19,sum20)

                    #print(t2,sum1)
                    co+=1
                    #if co==50:
                     #   time.sleep(5)

                #print(co)




            #Write the data to the CSV file
            ''''sum1=0
            sum2=0
            sum3=0                                          
            sum4=0
            sum5=0
            sum6=0
            for i,j,k,l,m,n in zip(Top_to_Bottom,Right_corner_dis,Left_corner_dis,Outer_1,Outer_2,Outer_3):
                sum1+=i
                sum2+=j
                sum3+=k
                sum4+=l
                sum5+=m
                sum6+=n
                #csv_writer.writerow([d,i,j,k,l,m,n])





                csv_writer.writerow([d,sum1,sum2,sum3,sum4,sum5,sum6])

                print(d,sum1,sum2,sum3,sum4,sum5,sum6)'''




                #csv_writer.writerow([ds,i])

                #csv_writer.writerows(i)

















        ''''# Define two points in the face mesh with x,y,z coordinates
        point1 = [F_T_I[356].x, F_T_I[356].y, F_T_I[356].z]
        point2 = [F_T_I[152].x, F_T_I[152].y, F_T_I[152].z]

        # Compute the vector between the two points
        a = [point2[i] - point1[i] for i in range(3)]

        # Define another vector for reference
        b = [0.0, 1.0, 0.0]

        # Compute the dot product of vectors a and b
        dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        # Compute the magnitudes of vectors a and b
        magnitude_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        magnitude_b = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

        # Compute the angle between vectors a and b
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        theta = math.acos(cos_theta)

        # Print the result in degrees
        print("The angle between the two points is {:.2f} degrees".format(math.degrees(theta)))'''






















        '''print("\n")
        # print(len(F_T_I))
        for i in range(468):  # RIGHT EYE RANGEE
            for j in range(468):

                    Distance.append(
                        math.sqrt((F_T_I[i].x - F_T_I[j].x) ** 2 + (F_T_I[i].y - F_T_I[j].y) ** 2 + (
                                F_T_I[i].z - F_T_I[j].z) ** 2))

        #print(Distance)

        sum1 = sum(Distance)
        print("The average is:", float(sum1))'''







        '''for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
            for FACE_OVAL_INDEX in FACEMESH_FACE_OVAL_INDEXES[:37]:
                face_oval.append(face_landmarks.landmark[FACE_OVAL_INDEX])
        #print(face_oval[3])
        #print(face_oval[0])
        distance = math.sqrt((face_oval[3].x - face_oval[0].x) ** 2 + (face_oval[3].y - face_oval[0].y) ** 2 + (
                face_oval[3].z - face_oval[0].z) ** 2)

        print(distance)'''




        '''for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:20]:
                left_eye.append(face_landmarks.landmark[LEFT_EYE_INDEX])
                a = left_eye[0].x
                b = left_eye[0].y
                c = left_eye[0].z
            for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:20]:                                              
                right_eye.append(face_landmarks.landmark[RIGHT_EYE_INDEX])
                a = right_eye[0].x
                b = right_eye[0].y
                c = right_eye[0].z                                      #LEFT_EYE AND RIGHT_EYE

                # print(face_mesh[LEFT_EYE_INDEXES][0])
        # print("\n")
        #print(left_eye[6])
        print("\n")
        #print(right_eye[1])
        #
        print("\n")
        distance = math.sqrt((right_eye[1].x - left_eye[6].x) ** 2 + (right_eye[1].y - left_eye[6].y) ** 2 + (
                    right_eye[1].z - left_eye[6].z) ** 2)
        print(distance)'''







        '''for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:20]:
                left_eye.append(face_landmarks.landmark[LEFT_EYE_INDEX])
                a = left_eye[0].x
                b = left_eye[0].y
                c = left_eye[0].z

                xl_coord.append(a)
                yl_coord.append(b)
                zl_coord.append(c)
            # print(xl_coord)
            # print(yl_coord)
            # print(zl_coord)
            # b=left_eye[0].y
            # c=left_eye[0].z
            print("\n")
            for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:20]:
                right_eye.append(face_landmarks.landmark[RIGHT_EYE_INDEX])
                a = right_eye[0].x
                b = right_eye[0].y
                c = right_eye[0].z

                xr_coord.append(a)
                yr_coord.append(b)
                zr_coord.append(c)
            #print(R_EYE)
            # print(xr_coord)
            # print(yr_coord)
            # print(zr_coord)
            #print("\n")


            def euclidean_distance(vector1, vector2):
                squared_diff_sum = 0
                for i in range(16):
                    squared_diff_sum += (vector1[0][i] - vector2[0][i]) ** 2 + (vector1[1][i] - vector2[1][i]) ** 2 + (
                                vector1[2][1] - vector2[2][i]) ** 2
                print(math.sqrt(squared_diff_sum))

                #print("\n")


            euclidean_distance(L_EYE, R_EYE)'''





        '''for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:20]:
                face_coord.append(face_landmarks.landmark[LEFT_EYE_INDEX])

                print("xcoordinates==", (face_coord[0]).x)'''
        ''''landmark_list = results.multi_face_landmarks[0].landmark
        x_left_eye = landmark_list[168].x * image.shape[1]  # convert normalized x-coordinate to pixel value
        x_right_eye = landmark_list[374].x * imag
        e.shape[1]  # convert normalized x-coordinate to pixel value

        print("x-coordinate of the left eye:", x_left_eye)
        print("x-coordinate of the right eye:", x_right_eye)
    # Flip the image horizontally for a selfie-view display.'''''





    #cv2.imshow('MediaPipe Face Mesh', translated_frame)
    #cv2.imshow('Frames', np.hstack((frame_left, frame_right)))
    cv2.imshow('Frames',image2)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()