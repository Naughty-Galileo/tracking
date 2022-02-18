import numpy as np
import cv2 as cv
import os
# from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


class App:
    def __init__(self, video_src, img_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []

        if video_src != "":
            self.cam = cv.VideoCapture(video_src)
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
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

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
                        cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)
                    p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv.imshow('lk_track', vis)

                ch = cv.waitKey(30)
                if ch == 27:
                    break
        else:
            for file in os.listdir(self.img_src):
                img = cv.imread(os.path.join(self.img_src, file))
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                vis = img.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, img_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

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
                        cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(img_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)
                    p = cv.goodFeaturesToTrack(img_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

                self.frame_idx += 1
                self.prev_gray = img_gray
                cv.imshow('lk_track', vis)
                ch = cv.waitKey(30)
                if ch == 27:
                    break

def main():
    video_src = ""
    img_src = "E:/data/OTB100/Walking/img"
    
    App(video_src, img_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()