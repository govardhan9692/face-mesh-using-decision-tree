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
import pickle
import joblib
import collections
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
co=0
ab=0

LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))

RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))

FACEMESH_FACE_OVAL_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))

FACEMESH_TESSELATION_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_TESSELATION)))

def cam_distance():
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 800
        d = (W * f) // w
        ds.append(d)

def face_dis():
    for face_no, face_landmarks in enumerate(results.multi_face_landmarks):
        for RIGHT_EYE in RIGHT_EYE_INDEXES:
            right_eye.append(face_landmarks.landmark[RIGHT_EYE])

        for FACEMESH_TESSELATION_INDEX in FACEMESH_TESSELATION_INDEXES:
            F_T_I.append(face_landmarks.landmark[FACEMESH_TESSELATION_INDEX])

        point_x=F_T_I[1].x
        point_y=F_T_I[1].y
        point_z=F_T_I[1].z
        axis_x, axis_y, axis_z = 0, 0, 1
        dot_product = point_x * axis_x + point_y * axis_y + point_z * axis_z

        point_magnitude = math.sqrt(point_x ** 2 + point_y ** 2 + point_z ** 2)
        axis_magnitude = math.sqrt(axis_x ** 2 + axis_y ** 2 + axis_z ** 2)
        cosine = dot_product / (point_magnitude * axis_magnitude)
        angle_degrees = math.degrees(math.acos(cosine))

        for i in range(len(nose_len) - 1):
            Top_to_Bottom.append(math.sqrt((F_T_I[nose_len[i]].x - F_T_I[nose_len[i + 1]].x) ** 2 + (
                        F_T_I[nose_len[i]].y - F_T_I[nose_len[i + 1]].y) ** 2 + (
                                                   F_T_I[nose_len[i]].z - F_T_I[nose_len[i + 1]].z) ** 2))

        for i in range(len(left_to_right)-1):
            Left_to_Right.append (math.sqrt((F_T_I[left_to_right[i]].x - F_T_I[left_to_right[i+1]].x) ** 2 + (F_T_I[left_to_right[i]].y - F_T_I[left_to_right[i+1]].y) ** 2 + (
                    F_T_I[left_to_right[i]].z - F_T_I[left_to_right[i+1]].z) ** 2))

        for i in range(len(left_to_right1)-1):
            Left_to_Right1.append(math.sqrt((F_T_I[left_to_right1[i]].x - F_T_I[left_to_right1[i+1]].x) ** 2 + (F_T_I[left_to_right1[i]].y - F_T_I[left_to_right1[i+1]].y) ** 2 + (
                    F_T_I[left_to_right1[i]].z - F_T_I[left_to_right1[i+1]].z) ** 2))

        for i in range(len(left_to_right1) - 1):
            Left_to_Right2.append(math.sqrt((F_T_I[left_to_right2[i]].x - F_T_I[left_to_right2[i + 1]].x) ** 2 + ( F_T_I[left_to_right2[i]].y - F_T_I[left_to_right2[i + 1]].y) ** 2 + (
                    F_T_I[left_to_right2[i]].z - F_T_I[left_to_right2[i + 1]].z) ** 2))

        for i in range(len(outer1) - 1):
            Outer_1.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer1[i + 1]].x) ** 2 + (
                        F_T_I[outer1[i]].y - F_T_I[outer1[i + 1]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer1[i + 1]].z) ** 2))

        for i in range(len(outer2) - 1):
            Outer_2.append(math.sqrt((F_T_I[outer2[i]].x - F_T_I[outer2[i + 1]].x) ** 2 + (
                    F_T_I[outer2[i]].y - F_T_I[outer2[i + 1]].y) ** 2 + (F_T_I[outer2[i]].z - F_T_I[outer2[i + 1]].z) ** 2))

        for i in range(len(outer3) - 1):
            Outer_3.append(math.sqrt((F_T_I[outer3[i]].x - F_T_I[outer3[i + 1]].x) ** 2 + (
                    F_T_I[outer3[i]].y - F_T_I[outer3[i + 1]].y) ** 2 + (F_T_I[outer3[i]].z - F_T_I[outer3[i + 1]].z) ** 2))

        for i in range(36):
            Outer1_Outer2_dis.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer2[i]].x) ** 2 + (
                        F_T_I[outer1[i]].y - F_T_I[outer2[i]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer2[i]].z) ** 2))

        for i in range(36):
            Outer1_Outer3_dis.append(math.sqrt((F_T_I[outer1[i]].x - F_T_I[outer3[i]].x) ** 2 + (
                    F_T_I[outer1[i]].y - F_T_I[outer3[i]].y) ** 2 + (F_T_I[outer1[i]].z - F_T_I[outer3[i]].z) ** 2))

        for i in range(36):
            Outer2_Outer3_dis.append(math.sqrt((F_T_I[outer2[i]].x - F_T_I[outer3[i]].x) ** 2 + (
                    F_T_I[outer2[i]].y - F_T_I[outer3[i]].y) ** 2 + (F_T_I[outer2[i]].z - F_T_I[outer3[i]].z) ** 2))

        for i in range(len(nose_outer1)-1):
           Nose_outer1.append(math.sqrt((F_T_I[nose_outer1[i]].x - F_T_I[nose_outer1[i+1]].x) ** 2 + (
                    F_T_I[nose_outer1[i]].y - F_T_I[nose_outer1[i+1]].y) ** 2 + (F_T_I[nose_outer1[i]].z - F_T_I[nose_outer1[i+1]].z) ** 2))

        for i in range(len(nose_outer2)-1):
           Nose_outer2.append(math.sqrt((F_T_I[nose_outer2[i]].x - F_T_I[nose_outer2[i + 1]].x) ** 2 + (
                    F_T_I[nose_outer2[i]].y - F_T_I[nose_outer2[i + 1]].y) ** 2 + (F_T_I[nose_outer2[i]].z - F_T_I[nose_outer2[i + 1]].z) ** 2))

        for i in range(len(nose_outer3)-1):
           Nose_outer3.append(math.sqrt((F_T_I[nose_outer3[i]].x - F_T_I[nose_outer3[i + 1]].x) ** 2 + (
                    F_T_I[nose_outer3[i]].y - F_T_I[nose_outer3[i + 1]].y) ** 2 + (F_T_I[nose_outer3[i]].z - F_T_I[nose_outer3[i + 1]].z) ** 2))

        for i in range(len(nose_len)-1):
            Nose_len.append(math.sqrt((F_T_I[nose_len[i]].x - F_T_I[nose_len[i + 1]].x) ** 2 + (
                    F_T_I[nose_len[i]].y - F_T_I[nose_len[i + 1]].y) ** 2 + (F_T_I[nose_len[i]].z - F_T_I[nose_len[i + 1]].z) ** 2))

        for i in range(len(nose_w1)-1):
            Nose_w1.append(math.sqrt((F_T_I[nose_w1[i]].x - F_T_I[nose_w1[i + 1]].x) ** 2 + (
                    F_T_I[nose_w1[i]].y - F_T_I[nose_w1[i + 1]].y) ** 2 + (F_T_I[nose_w1[i]].z - F_T_I[nose_w1[i + 1]].z) ** 2))

        for i in range(len(nose_w2)-1):
            Nose_w2.append(math.sqrt((F_T_I[nose_w2[i]].x - F_T_I[nose_w2[i + 1]].x) ** 2 + (
                    F_T_I[nose_w2[i]].y - F_T_I[nose_w2[i + 1]].y) ** 2 + (F_T_I[nose_w2[i]].z - F_T_I[nose_w2[i + 1]].z) ** 2))

        for i in range(len(nose_w3)-1):
            Nose_w3.append(math.sqrt((F_T_I[nose_w3[i]].x - F_T_I[nose_w3[i + 1]].x) ** 2 + (
                    F_T_I[nose_w3[i]].y - F_T_I[nose_w3[i + 1]].y) ** 2 + (F_T_I[nose_w3[i]].z - F_T_I[nose_w3[i + 1]].z) ** 2))

        for i in range(len(nose_w4)-1):
            Nose_w4.append(math.sqrt((F_T_I[nose_w4[i]].x - F_T_I[nose_w4[i + 1]].x) ** 2 + (
                    F_T_I[nose_w4[i]].y - F_T_I[nose_w4[i + 1]].y) ** 2 + (F_T_I[nose_w4[i]].z - F_T_I[nose_w4[i + 1]].z) ** 2))

        for i in range(len(nose_w5)-1):
            Nose_w5.append(math.sqrt((F_T_I[nose_w5[i]].x - F_T_I[nose_w5[i + 1]].x) ** 2 + (
                    F_T_I[nose_w5[i]].y - F_T_I[nose_w5[i + 1]].y) ** 2 + (F_T_I[nose_w5[i]].z - F_T_I[nose_w5[i + 1]].z) ** 2))

        for i in range(len(nose_w6)-1):
            Nose_w6.append(math.sqrt((F_T_I[nose_w6[i]].x - F_T_I[nose_w6[i + 1]].x) ** 2 + (
                    F_T_I[nose_w6[i]].y - F_T_I[nose_w6[i + 1]].y) ** 2 + (F_T_I[nose_w6[i]].z - F_T_I[nose_w6[i + 1]].z) ** 2))

        return angle_degrees

cap = cv2.VideoCapture(0)
move_amount=0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = FaceMeshDetector(maxFaces=1)

# Add variables to track prediction history and accuracy metrics
prediction_history = collections.deque(maxlen=30)  # Store last 30 predictions
confidence_history = collections.deque(maxlen=30)  # Store last 30 confidence values
stability_score = 0
last_prediction = None
prediction_start_time = None
current_streak = 0
longest_streak = 0

filename= "data3.csv"
face_coord=[]
ds=[]
zoom=0.4
count=0
mo=0
so=0

# Load the model and label encoder at startup for better performance
try:
    try:
        clf = joblib.load('decision_tree_model.pkl')
        le = joblib.load('label_encoder.pkl')
    except:
        with open('decision_tree_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
    print("Model and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    clf = None
    le = None

with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    image1 = cv2.flip(image, 1)

    if zoom:
        h, w = image1.shape[:2]
        cx, cy = w / 2, h / 2
        fx, fy = 1 / zoom, 1 / zoom
        M = cv2.getRotationMatrix2D((cx, cy), 0, fx)
        frame = cv2.warpAffine(image1, M, (w, h))
        img, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
        f = 840
        d = (W * f) // w
        if d >= 40:
            zoom -= 0.0044
            b = zoom
        elif d <= 40:
            zoom += 0.0044

    face2=cv2.ellipse(img=image1,center=(300,225),axes=(150,200),angle=180,startAngle=0,endAngle=360,color=(0,255,0),thickness=4,lineType=16)

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

    xl_coord = []  
    yl_coord = []  
    zl_coord = []  
    L_EYE = [xl_coord, yl_coord, zl_coord]

    xr_coord = []  
    yr_coord = []  
    zr_coord = []  
    R_EYE = [xr_coord, yr_coord, zr_coord]

    right_corner1=[10,109,67,103,54,21,162,127,234,93,132,54,172,136,150,149,176,148,152]

    left_corner1=[10,338,297,332,284,251,289,356,454,323,361,288,397,365,379,378,400,377,152]

    top_bottom=[10,151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175,152]
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

    if not success:
      print("Ignoring empty camera frame.")
      continue

    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame)

    landmark_name = "LEFT_EYE_INNER"

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

        target = face_dis()
        t2 = round(target, 2)
        print(t2)
        
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0
        sum8 = 0
        sum9 = 0
        sum10 = 0
        sum11 = 0
        sum12 = 0
        sum13 = 0
        sum14 = 0
        sum15 = 0
        sum16 = 0
        sum17 = 0
        sum18 = 0
        sum19 = 0
        sum20 = 0
        
        for a,b,c,d1,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t in zip(
            Top_to_Bottom,Left_to_Right,Left_to_Right1,Left_to_Right2,
            Outer_1, Outer_2, Outer_3, Outer1_Outer2_dis, Outer1_Outer3_dis, 
            Outer2_Outer3_dis, Nose_outer1, Nose_outer2, Nose_outer3, Nose_len, 
            Nose_w1, Nose_w2, Nose_w3, Nose_w4, Nose_w5, Nose_w6):
            
            sum1+=a
            sum2+=b
            sum3+=c
            sum4+=d1
            sum5+=e
            sum6+=f
            sum7+=g
            sum8+=h
            sum9+=i
            sum10+=j
            sum11+=k
            sum12+=l
            sum13+=m
            sum14+=n
            sum15+=o
            sum16+=p
            sum17+=q
            sum18+=r
            sum19+=s
            sum20+=t
        
        sum1=round(sum1,3)
        sum2=round(sum2,3)
        sum3=round(sum3,3)
        sum4=round(sum4,3)
        sum5=round(sum5,3)
        sum6=round(sum6,3)
        sum7=round(sum7,3)
        sum8=round(sum8,3)
        sum9=round(sum9,3)
        sum10=round(sum10,3)
        sum11=round(sum11,3)
        sum12=round(sum12,3)
        sum13=round(sum13,3)
        sum14=round(sum14,3)
        sum15=round(sum15,3)
        sum16=round(sum16,3)
        sum17=round(sum17,3)
        sum18=round(sum18,3)
        sum19=round(sum19,3)
        sum20=round(sum20,3)
        
        cv2.putText(image2, f"Angle: {t2}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
        
        if clf is not None and le is not None:
            input_features = [[t2, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9,
                              sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17, sum18, sum19, sum20]]
            
            # Get prediction
            y_pred = clf.predict(input_features)
            
            # Get prediction probabilities for confidence level
            probabilities = clf.predict_proba(input_features)[0]
            
            # Add a confidence threshold to determine high confidence predictions
            CONFIDENCE_THRESHOLD = 90.0  # Using 90% threshold
            
            # Find the confidence for the predicted class
            predicted_class_idx = y_pred[0]
            confidence = probabilities[predicted_class_idx] * 100
            
            # Update prediction history
            prediction_history.append(predicted_class_idx)
            confidence_history.append(confidence)
            
            # Calculate similarity scores (how much does this person match each known class)
            similarity_scores = {}
            for idx, class_name in enumerate(le.classes_):
                similarity_scores[class_name] = probabilities[idx] * 100
            
            # Calculate and sort similarities for easier viewing
            sorted_similarities = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
            
            # If confidence is below threshold, mark as unknown
            is_unknown = confidence < CONFIDENCE_THRESHOLD
            
            # Get the actual predicted class name directly from the label encoder
            predicted_name = le.classes_[predicted_class_idx]
            
            # Calculate stability score (how consistent the predictions are)
            if prediction_history:
                most_common = collections.Counter(prediction_history).most_common(1)[0]
                most_common_prediction = most_common[0]
                most_common_count = most_common[1]
                stability_score = (most_common_count / len(prediction_history)) * 100
                
            # Track continuous correct prediction streaks
            if last_prediction is None:
                last_prediction = predicted_class_idx
                prediction_start_time = datetime.now()
                current_streak = 1
            elif last_prediction == predicted_class_idx and not is_unknown:
                current_streak += 1
                if current_streak > longest_streak:
                    longest_streak = current_streak
            else:
                last_prediction = predicted_class_idx
                prediction_start_time = datetime.now()
                current_streak = 1
                
            # Calculate prediction duration
            prediction_duration = 0
            if prediction_start_time:
                prediction_duration = (datetime.now() - prediction_start_time).total_seconds()
            
            # Determine confidence level label and color
            if confidence >= CONFIDENCE_THRESHOLD:
                confidence_label = "HIGH MATCH"
                confidence_color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 70:
                confidence_label = "MODERATE MATCH"
                confidence_color = (0, 165, 255)  # Orange for moderate confidence
            elif confidence >= 40:
                confidence_label = "LOW MATCH"
                confidence_color = (42, 42, 165)  # Red-orange for low confidence
            else:
                confidence_label = "VERY LOW MATCH"
                confidence_color = (0, 0, 255)  # Red for very low confidence
            
            # Display name and confidence on screen
            if confidence >= CONFIDENCE_THRESHOLD:
                # High confidence display
                display_name = f"{predicted_name}"
            else:
                # Lower confidence display - still showing the name
                display_name = f"{predicted_name}?"
            
            cv2.putText(image2, display_name, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, confidence_color, 2)
            
            cv2.putText(image2, f"{confidence:.1f}% - {confidence_label}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
            
            # Show accuracy metrics
            cv2.putText(image2, f"Stability: {stability_score:.1f}%", (10, 130), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add a dedicated Similarity Index section - clearer presentation of how much
            # the person resembles each known individual
            cv2.putText(image2, "SIMILARITY INDEX:", (10, 220), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            y_sim = 250
            for name, sim_value in sorted_similarities:
                # Determine color based on similarity value
                if sim_value > 80:
                    sim_color = (0, 255, 0)  # Green for high similarity
                    sim_text = "STRONG MATCH"
                elif sim_value > 60:
                    sim_color = (0, 255, 255)  # Yellow for moderate similarity
                    sim_text = "MODERATE MATCH"
                elif sim_value > 30:
                    sim_color = (0, 165, 255)  # Orange for weak similarity
                    sim_text = "WEAK MATCH"
                else:
                    sim_color = (0, 0, 255)  # Red for very weak similarity
                    sim_text = "MINIMAL MATCH"
                
                # Display detailed similarity score with interpretation
                cv2.putText(image2, f"{name}: {sim_value:.1f}% - {sim_text}", 
                          (10, y_sim), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, sim_color, 2)
                y_sim += 30
            
            # Show detailed similarity scores for all classes (bar chart)
            y_pos = 80
            for class_name, similarity in similarity_scores.items():
                # Visual indicator of match level - bar chart
                bar_length = int(similarity * 1.5)  # Scale factor for visibility
                bar_color = (0, 255, 0) if similarity > CONFIDENCE_THRESHOLD else (0, 165, 255)
                
                cv2.rectangle(image2, 
                             (frame_width - 280, y_pos - 15),
                             (frame_width - 280 + bar_length, y_pos - 5),
                             bar_color, -1)
                
                cv2.putText(image2, f"{class_name}: {similarity:.1f}%", 
                          (frame_width - 280, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (255, 255, 255), 2)
                y_pos += 30
            
            # Show threshold line in the similarity bars
            threshold_x = frame_width - 280 + int(CONFIDENCE_THRESHOLD * 1.5)
            cv2.line(image2, 
                    (threshold_x, 65), 
                    (threshold_x, y_pos - 40), 
                    (0, 0, 255), 2)
            cv2.putText(image2, "Threshold", (threshold_x - 40, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display streak information
            cv2.putText(image2, f"Streak: {current_streak} frames", (10, 160), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image2, f"Duration: {prediction_duration:.1f}s", (10, 190), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            print(f"Predicted: {predicted_name} ({confidence:.1f}%), Govardhan similarity: {similarity_scores.get('Govardhan', 0):.1f}%, Sai similarity: {similarity_scores.get('Sai', 0):.1f}%")

    cv2.imshow('Frames', image2)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()