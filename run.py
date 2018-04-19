import cv2
import numpy as np
import sys
from common import anorm2, draw_str

prev = None
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

tracks = []
track_len = 10
detect_interval = 5
frame_idx = 0
zero_diff_counter = 1
win_size_ratio = 0.25

if len(sys.argv) <2:
    print("Usage : python run.py <path_to_video>")
    exit()
else:
    cap = cv2.VideoCapture(sys.argv[1])

while cap.isOpened():
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        diff_len = np.sum(d>0.8)
        if diff_len>=3:
            zero_diff_counter =1
            win_size_ratio = 0.25
        elif diff_len==0:
            zero_diff_counter*=1.2
        good = d < 1
        new_tracks = []        
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        if zero_diff_counter>200:
            draw_str(vis, (1020, 20), "STOP")
            win_size_ratio = 0.01

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[0:int(win_size_ratio*mask.shape[0]),:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])


    frame_idx += 1
    prev_gray = frame_gray
    cv2.imshow('Visteon_Dashboard_Motion_Detection', vis)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
       break
  
cap.release()
cv2.destroyAllWindows()
