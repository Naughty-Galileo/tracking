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


def my_camshift(src):
    if src.endswith('.mp4'):
        cap = cv2.VideoCapture(src)
        # 读取摄像头第一帧图像
        ret, frame = cap.read()

        # 初始化位置窗口
        #r,h,c,w = 250,90,400,125 # simply hardcoded the values
        # r,h,c,w=15,370,319,87

        # 初始位置
        a1,a2 = get_rect(frame, title='get_rect') # 手动选框
        r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]  # 手动选框
        track_window = (c,r,w,h)

        # 设置所要跟踪的ROI
        roi = frame[r:r+h, c:c+w]
        
        # RGB转HSV
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # inRange函数设置亮度阈值
        # 去除低亮度的像素点的影响, 将低于和高于阈值的值设为0
        # eg. mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

        # 计算直方图
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        
        # 归一化 这里采用线性归一化
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # 设置终止条件
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        while(1):
            ret ,frame = cap.read()

            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # 反向投影
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)

                # Draw it on image
                x,y,w,h = track_window
                cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                cv2.imshow('img2', frame)

                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break
                #else:
                # cv2.imwrite(chr(k)+".jpg",img2)

            else:
                break

        cv2.destroyAllWindows()
        cap.release()
    
    else:
        index = 0
        for file in os.listdir(src):
            img = cv2.imread(os.path.join(src, file))

            if index==0:
                a1,a2 = get_rect(img, title='get_rect') # 手动选框
                r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]  # 手动选框
                track_window = (c,r,w,h)

                # 设置所要跟踪的ROI
                roi = img[r:r+h, c:c+w]
                
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

                # 设置终止条件
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            
            index += 1

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            x,y,w,h = track_window
            cv2.rectangle(img, (x,y), (x+w,y+h), 255,2)
            cv2.imshow('img', img)

            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break


if __name__=='__main__':
    # src = './video/car5.mp4'
    src = 'E:/data/OTB100/Walking/img'
    my_meanshift(src)    #  E:/data/OTB100/Walking/img