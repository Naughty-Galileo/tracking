# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

current_pos = None
tl = None
br = None
# 鼠标事件
def get_rect(im, title='get_rect'):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def on_mouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, on_mouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))
    return (tl, br)


# hsv阈值，便于进行轮廓判断及轨迹绘制，需要根据运动目标的颜色自己进行调整
min_hsv_bound = (0, 0, 221)
max_hsv_bound = (180, 30, 255)


#状态向量
stateSize = 6
#观测向量
measSize = 4
coutrSize = 0
kf = cv2.KalmanFilter(stateSize,measSize,coutrSize)
state = np.zeros(stateSize, np.float32)#[x,y,v_x,v_y,w,h],簇心位置，速度，高宽
meas = np.zeros(measSize, np.float32)#[z_x,z_y,z_w,z_h]
procNoise = np.zeros(stateSize, np.float32)

#状态转移矩阵
cv2.setIdentity(kf.transitionMatrix)#生成单位矩阵
# [1 0 dT 0  0 0]
# [0 1 0  dT 0 0]
# [0 0 1  0  0 0]
# [0 0 0  1  0 0]
# [0 0 0  0  1 0]
# [0 0 0  0  0 1]
#观测矩阵
# [1 0 0 0 0 0]
# [0 1 0 0 0 0]
# [0 0 0 0 1 0]
# [0 0 0 0 0 1]
kf.measurementMatrix = np.zeros((measSize,stateSize),np.float32)
kf.measurementMatrix[0,0]=1.0
kf.measurementMatrix[1,1]=1.0
kf.measurementMatrix[2,4]=1.0
kf.measurementMatrix[3,5]=1.0

#预测噪声
# [Ex 0 0 0 0 0]
# [0 Ey 0 0 0 0]
# [0 0 Ev_x 0 0 0]
# [0 0 0 Ev_y 0 0]
# [0 0 0 0 Ew 0]
# [0 0 0 0 0 Eh]
cv2.setIdentity(kf.processNoiseCov)
kf.processNoiseCov[0,0] = 1e-2
kf.processNoiseCov[1,1] = 1e-2
kf.processNoiseCov[2,2] = 5.0
kf.processNoiseCov[3,3] = 5.0
kf.processNoiseCov[4,4] = 1e-2
kf.processNoiseCov[5,5] = 1e-2

#测量噪声
cv2.setIdentity(kf.measurementNoiseCov)
# for i in range(len(kf.measurementNoiseCov)):
#     kf.measurementNoiseCov[i,i] = 1e-1

video_cap = cv2.VideoCapture('./video/car5.mp4')



# 视频输出
fps = video_cap.get(cv2.CAP_PROP_FPS) #获得视频帧率，即每秒多少帧
size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# videoWriter = cv2.VideoWriter('./video/new_green.mp4' ,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
ticks = 0 
i=0
found = True
notFoundCount = 0
prePointCen = [] #存储小球中心点位置
meaPointCen = []

ret, frame = video_cap.read()
a1,a2 = get_rect(frame, title='get_rect') # 手动选框
r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]  # 手动选框
center_x = c+w/2
center_y = r+h/2

# prePointCen.append((int(center_x),int(center_y)))
for i in range(len(kf.errorCovPre)):
    kf.errorCovPre[i,i] = 1
state[0] = center_x
state[1] = center_y
state[2] = 0.5
state[3] = 0.1
state[4] = w
state[5] = h
kf.statePost = state
ticks = float(cv2.getTickCount()) 


while(True):
    ret, frame = video_cap.read()
    if ret is False:
        break

    precTick = ticks  
    ticks = float(cv2.getTickCount())  
    res = frame.copy()
    dT = float((ticks - precTick)/cv2.getTickFrequency()) 

    #预测得到的小球位置
    kf.transitionMatrix[0,2] = dT
    kf.transitionMatrix[1,3] = dT

    state = kf.predict()
    width = state[4]
    height = state[5]
    x_left = state[0] - width/2 #左上角横坐标
    y_left = state[1] - height/2  #左上角纵坐标
    x_right = state[0] + width/2
    y_right = state[1] + height/2

    center_x = state[0]
    center_y = state[1]
    prePointCen.append((int(center_x),int(center_y)))
    cv2.circle(res, (int(center_x),int(center_y)),2,(255,0,0),-1)
    cv2.rectangle(res, (int(x_left),int(y_left)), (int(x_right),int(y_right)),255,2)

    # #根据颜色二值化得到的小球位置
    # frame = cv2.GaussianBlur(frame, (5,5), 3.0, 3.0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # rangeRes = cv2.inRange(frame, min_hsv_bound,max_hsv_bound)
    # kernel = np.ones((3, 3), np.uint8)
    # # 腐蚀膨胀
    # rangeRes = cv2.erode(rangeRes, kernel, iterations=2)
    # rangeRes = cv2.dilate(rangeRes, kernel, iterations=2)
    # # cv2.imshow("Threshold", rangeRes)
    # cv2.waitKey(1)
    # contours = cv2.findContours(rangeRes.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2]
    # #检测轮廓，只检测最外围轮廓，保存物体边界上所有连续的轮廓点到contours向量内
    # balls = []
    # ballsBox = []
    # for i in range(len(contours)):
    #     x, y, w, h = cv2.boundingRect(np.array(contours[i]))

    #     ratio = float(w/h)
    #     if(ratio > 1.0):
    #         ratio = 1.0 / ratio
    #     if(ratio > 0.75 and w*h>=400):
    #         balls.append(contours[i])
    #         ballsBox.append([x, y, w, h])

    # print( "Balls found:", len(ballsBox))
    # print("\n")

    # for i in range(len(balls)):
    #     # 绘制轮廓
    #     cv2.drawContours(res, balls, i, (20,150,20),1)
    #     cv2.rectangle(res,(ballsBox[i][0],ballsBox[i][1]),(ballsBox[i][0]+ballsBox[i][2],ballsBox[i][1]+ballsBox[i][3]),(0,255,0),2) #二值化得到小球边界

    #     center_x = ballsBox[i][0] + ballsBox[i][2] / 2
    #     center_y = ballsBox[i][1] + ballsBox[i][3] / 2

    #     meaPointCen.append((int(center_x),int(center_y)))
    #     cv2.circle(res,(int(center_x),int(center_y)), 2, (20,150,20) ,-1)

    #     name = "(" + str(center_x) + "," + str(center_y) + ")"
    #     cv2.putText(res, name, (int(center_x) + 3, int(center_y) - 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (20,150,20), 2)
    
    n = len(prePointCen)
    for i in range(1, n):
        # print(i)
        # if prePointCen[i-1] is None or prePointCen[i] is None:
        #     continue
         #  注释掉的这块是为了绘制能够随时间先后慢慢消失的追踪轨迹，但是有一些小错误
        # 计算所画小线段的粗细
        # thickness = int(np.sqrt(64 / float(n - i + 1))*2.5)
        # print(thickness)
        # 画出小线段
        # cv2.line(res, prePointCen[i-1], prePointCen[i], (0, 0, 255), thickness)
        cv2.line(res, prePointCen[i-1], prePointCen[i], (0,0,255), 1, 4)

    # if(len(balls) == 0):
    #     notFoundCount += 1
    #     print("notFoundCount",notFoundCount)
    #     print("\n")

    #     if notFoundCount >= 100:
    #         found = False

    # else:
    #     #测量得到的物体位置
    #     notFoundCount = 0
    #     meas[0] = ballsBox[0][0] + ballsBox[0][2] / 2
    #     meas[1] = ballsBox[0][1] + ballsBox[0][3] / 2
    #     meas[2] = float(ballsBox[0][2])
    #     meas[3] = float(ballsBox[0][3])

    #     #第一次检测
    #     if not found:
            # for i in range(len(kf.errorCovPre)):
            #     kf.errorCovPre[i,i] = 1
            # state[0] = meas[0]
            # state[1] = meas[1]
            # state[2] = 0
            # state[3] = 0
            # state[4] = meas[2]
            # state[5] = meas[3]

            # kf.statePost = state
            # found = True

    #     else:
    # kf.correct(meas) #Kalman修正

    #         print('rr',res.shape)
    #         print("Measure matrix:", meas)
    cv2.imshow("Tracking", res)

    cv2.waitKey(10)
    # videoWriter.write(res)

