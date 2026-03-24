import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    print(i, ok)
    cap.release()