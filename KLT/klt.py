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


global img_init
global point1, point2
global tracks
tracks = []
def select_on_mouse(event, x, y, flags, param):
    global point1, point2
    global img_init
    img2 = img_init.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('lk_track', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('lk_track', img2)

    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('lk_track', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])

        cut_img = img_init[min_y:min_y+height, min_x:min_x+width]
        img_gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
        mask[:] = 255
        p = cv2.goodFeaturesToTrack(img_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(min_x+x, min_y+y)])



class App:
    def __init__(self, video_src, img_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []

        if video_src != "":
            self.cam = cv2.VideoCapture(video_src)
            self.is_video = True
        else:

            self.img_src = img_src
            self.is_video = False

        self.frame_idx = 0
        
    def run(self):
        if self.is_video:
            while True:
                _ret, frame = self.cam.read()
                if _ret is not True:
                    return
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray

                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                    d = abs(p1-p0).reshape(-1, 2).max(-1)
                    good = d > 1

                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

                ch = cv2.waitKey(30)
                if ch == 27:
                    break
        else:
            for file in os.listdir(self.img_src):
                img = cv2.imread(os.path.join(self.img_src, file))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                vis = img.copy()
                
                if self.frame_idx == 0:
                    global img_init
                    global tracks
                    img_init = img.copy()
                    cv2.namedWindow('lk_track')
                    cv2.setMouseCallback('lk_track', select_on_mouse)
                    cv2.imshow('lk_track', img_init)
                    cv2.waitKey()

                    self.tracks = tracks
                    self.prev_gray = img_gray
                    print(self.tracks)
                    for points in self.tracks:
                        for x,y in points:
                            cv2.circle(img_init, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.imshow('lk_track', img_init)
                    cv2.waitKey()


                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, img_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                    d = abs(p1-p0).reshape(-1, 2).max(-1)
                    good = d > 1

                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))


                if self.frame_idx % self.detect_interval == 0 and len(self.tracks) == 0:
                    # mask = np.zeros_like(img_gray)
                    # mask[:] = 255
                    # for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    #     cv2.circle(mask, (x, y), 5, 0, -1)
                    # p = cv2.goodFeaturesToTrack(img_gray, mask = mask, **feature_params)
                    # if p is not None:
                    #     for x, y in np.float32(p).reshape(-1, 2):
                    #         self.tracks.append([(x, y)])
                    img_init = img.copy()
                    cv2.namedWindow('lk_track')
                    cv2.setMouseCallback('lk_track', select_on_mouse)
                    cv2.imshow('lk_track', img_init)
                    cv2.waitKey()

                    self.tracks = tracks
                    self.prev_gray = img_gray
                    for points in self.tracks:
                        for x,y in points:
                            cv2.circle(img_init, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.imshow('lk_track', img_init)
                    cv2.waitKey()

                self.frame_idx += 1
                self.prev_gray = img_gray
                cv2.imshow('lk_track', vis)
                ch = cv2.waitKey(30)

                if ch == 27:
                    break

def main():
    video_src = ""
    img_src = "E:/data/OTB100/Walking/img"
    
    demo = App(video_src, img_src)
    demo.run()
    print('Done')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()