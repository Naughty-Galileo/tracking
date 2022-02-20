import numpy as np
import cv2
import os
# from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


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



        
def my_klt(src):
    tracks = []
    frame_idx = 0
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

         # 设置所要跟踪的ROI
        roi = frame[r:r+h, c:c+w]
        roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(roi_gray)
        mask[:] = 255
        p = cv2.goodFeaturesToTrack(roi_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(c+x, r+y)])
                prev_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        while True:
            _ret, frame = cap.read()
            if _ret is not True:
                return
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray

                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                d = abs(p1-p0).reshape(-1, 2).max(-1)
                good = d > 1

                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > 15:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

            # if frame_idx % 5 == 0 and len(tracks) == 0:

            #     mask = np.zeros_like(frame_gray)
            #     mask[:] = 255
            #     for x, y in [np.int32(tr[-1]) for tr in tracks]:
            #         cv2.circle(mask, (x, y), 5, 0, -1)
            #     p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            #     if p is not None:
            #         for x, y in np.float32(p).reshape(-1, 2):
            #             tracks.append([(x, y)])


            frame_idx += 1
            prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(20)
            if ch == 27:
                break
    else:
        for file in os.listdir(src):
            img = cv2.imread(os.path.join(src, file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vis = img.copy()
            
            if frame_idx == 0:
                # 初始位置
                a1,a2 = get_rect(img, title='get_rect') # 手动选框
                r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]  # 手动选框

                # 设置所要跟踪的ROI
                roi = img[r:r+h, c:c+w]
                roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

                mask = np.zeros_like(roi_gray)
                mask[:] = 255
                p = cv2.goodFeaturesToTrack(roi_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(c+x, r+y)])
                        prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame_idx += 1

            if len(tracks) > 0:
                img0, img1 = prev_gray, img_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                d = abs(p1-p0).reshape(-1, 2).max(-1)
                good = d > 1

                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > 15:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))


            if frame_idx % 5 == 0 and len(tracks) == 0:
                # 初始位置
                a1,a2 = get_rect(img, title='lk_track') # 手动选框
                r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]  # 手动选框

                ch = cv2.waitKey(20)
                if ch == 27:
                    return 

                # 设置所要跟踪的ROI
                roi = img[r:r+h, c:c+w]
                roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
                
                mask = np.zeros_like(roi_gray)
                mask[:] = 255
                p = cv2.goodFeaturesToTrack(roi_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(c+x, r+y)])

            frame_idx += 1
            prev_gray = img_gray
            cv2.imshow('lk_track', vis)
            ch = cv2.waitKey(20)

            if ch == 27:
                break


if __name__ == '__main__':
    src = "E:/data/OTB100/Walking/img"
    # src = './video/car5.mp4'
    my_klt(src)