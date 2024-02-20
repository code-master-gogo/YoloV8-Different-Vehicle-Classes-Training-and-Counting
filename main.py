import cv2
import math
import datetime
from ultralytics import YOLO
from sort import *
import numpy as np

cap = cv2.VideoCapture("traffic.mp4")

model = YOLO("best.pt")

classNames = ["Auto", "Bike", "Bus", "Car", "Lorry", "Tempo", "Tractor", "car"]

fps_start = datetime.datetime.now()
fps = 0
total_frames = 0

tracker1 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker2 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker3 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker4 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker5 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker6 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracker7 = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [300, 350, 1019, 350]

motorbikeCount = []
carCount = []
busCount = []
autoCount = []
lorryCount = []
tempoCount = []
tractorCount = []

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020,500))
    results = model(frame, stream=True)

    detections1 = np.empty((0, 5))
    detections2 = np.empty((0, 5))
    detections3 = np.empty((0, 5))
    detections4 = np.empty((0, 5))
    detections5 = np.empty((0, 5))
    detections6 = np.empty((0, 5))
    detections7 = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "Bike" and conf > 0.3:
                currentArray1 = np.array([x1, y1, x2, y2, conf])
                detections1 = np.vstack((detections1, currentArray1))
            elif currentClass == "Car" and conf > 0.3:
                currentArray2 = np.array([x1, y1, x2, y2, conf])
                detections2 = np.vstack((detections2, currentArray2))
            elif currentClass == "Auto" and conf > 0.3:
                currentArray3 = np.array([x1, y1, x2, y2, conf])
                detections3 = np.vstack((detections3, currentArray3))
            elif currentClass == "Bus" and conf > 0.3:
                currentArray4 = np.array([x1, y1, x2, y2, conf])
                detections4 = np.vstack((detections4, currentArray4))
            elif currentClass == "Lorry" and conf > 0.3:
                currentArray5 = np.array([x1, y1, x2, y2, conf])
                detections5 = np.vstack((detections5, currentArray5))
            elif currentClass == "Tempo" and conf > 0.3:
                currentArray6 = np.array([x1, y1, x2, y2, conf])
                detections6 = np.vstack((detections6, currentArray6))
            elif currentClass == "Tractor" and conf > 0.3:
                currentArray7 = np.array([x1, y1, x2, y2, conf])
                detections7 = np.vstack((detections7, currentArray7))

    resultsTracker1 = tracker1.update(detections1)
    resultsTracker2 = tracker2.update(detections2)
    resultsTracker3 = tracker3.update(detections3)
    resultsTracker4 = tracker4.update(detections4)
    resultsTracker5 = tracker5.update(detections5)
    resultsTracker6 = tracker6.update(detections6)
    resultsTracker7 = tracker7.update(detections7)

    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

    for result1 in resultsTracker1:
        x3, y3, x4, y4, idm = result1
        x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
        # cv2.putText(frame, f'{int(idm)}', (x3, y4), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Motorbike', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        wm, hm = x4 - x3, y4 - y3
        mx, my = x3 + wm // 2, y3 + hm // 2

        if limits[0] < mx < limits[2] and limits[1] - 7 < my < limits[1] + 7:
            if motorbikeCount.count(idm) == 0:
                motorbikeCount.append(idm)
                cv2.circle(frame, (mx, my), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    for result2 in resultsTracker2:
        x5, y5, x6, y6, idc = result2
        x5, y5, x6, y6 = int(x5), int(y5), int(x6), int(y6)
        cv2.rectangle(frame, (x5, y5), (x6, y6), (0, 0, 255), 2)
        # cv2.putText(frame, f'{int(idc)}', (x5, y6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Car', (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        wc, hc = x6 - x5, y6 - y5
        cx, cy = x5 + wc // 2, y5 + hc // 2

        if limits[0] < cx < limits[2] and limits[1] - 7 < cy < limits[1] + 7:
            if carCount.count(idc) == 0:
                carCount.append(idc)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
    
    for result3 in resultsTracker3:
        x7, y7, x8, y8, ida = result3
        x7, y7, x8, y8 = int(x7), int(y7), int(x8), int(y8)
        cv2.rectangle(frame, (x7, y7), (x8, y8), (255, 0, 0), 2)
        # cv2.putText(frame, f'{int(ida)}', (x7, y8), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Auto', (x7, y7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        wt, ht = x8 - x7, y8 - y7
        tx, ty = x7 + wt // 2, y7 + ht // 2

        if limits[0] < tx < limits[2] and limits[1] - 7 < ty < limits[1] + 7:
            if autoCount.count(ida) == 0:
                autoCount.append(ida)
                cv2.circle(frame, (tx, ty), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    for result4 in resultsTracker4:
        x9, y9, x10, y10, idb = result4
        x9, y9, x10, y10 = int(x9), int(y9), int(x10), int(y10)
        cv2.rectangle(frame, (x9, y9), (x10, y10), (255, 192, 203), 2)
        # cv2.putText(frame, f'{int(idb)}', (x9, y10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Bus', (x9, y9), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 192, 203), 1)
        wb, hb = x10 - x9, y10 - y9
        bx, by = x9 + wb // 2, y9 + hb // 2

        if limits[0] < bx < limits[2] and limits[1] - 7 < by < limits[1] + 7:
            if busCount.count(idb) == 0:
                busCount.append(idb)
                cv2.circle(frame, (bx, by), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
    
    for result5 in resultsTracker5:
        x11, y11, x12, y12, idl = result5
        x11, y11, x12, y12 = int(x11), int(y11), int(x12), int(y12)
        cv2.rectangle(frame, (x11, y11), (x12, y12), (255, 0, 127), 2)
        # cv2.putText(frame, f'{int(idl)}', (x11, y12), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Lorry', (x11, y11), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 127), 1)
        wl, hl = x12 - x11, y12 - y11
        lx, ly = x11 + wl // 2, y11 + hl // 2

        if limits[0] < lx < limits[2] and limits[1] - 7 < ly < limits[1] + 7:
            if lorryCount.count(idl) == 0:
                lorryCount.append(idl)
                cv2.circle(frame, (lx, ly), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    for result6 in resultsTracker6:
        x13, y13, x14, y14, idt = result6
        x13, y13, x14, y14 = int(x13), int(y13), int(x14), int(y14)
        cv2.rectangle(frame, (x13, y13), (x14, y14), (0, 127, 255), 2)
        # cv2.putText(frame, f'{int(idt)}', (x13, y14), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Tempo', (x13, y13), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 127, 255), 1)
        wt, ht = x14 - x13, y14 - y13
        tx, ty = x13 + wt // 2, y13 + ht // 2

        if limits[0] < tx < limits[2] and limits[1] - 7 < ty < limits[1] + 7:
            if tempoCount.count(idt) == 0:
                tempoCount.append(idt)
                cv2.circle(frame, (tx, ty), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)
    
    for result7 in resultsTracker7:
        x15, y15, x16, y16, idtr = result7
        x15, y15, x16, y16 = int(x15), int(y15), int(x16), int(y16)
        cv2.rectangle(frame, (x15, y15), (x16, y16), (255, 255, 0), 2)
        # cv2.putText(frame, f'{int(idtr)}', (x15, y16), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Tractor', (x15, y15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
        wtr, htr = x16 - x15, y16 - y15
        trx, tyr = x15 + wtr // 2, y15 + htr // 2

        if limits[0] < trx < limits[2] and limits[1] - 7 < tyr < limits[1] + 7:
            if tractorCount.count(idtr) == 0:
                tractorCount.append(idtr)
                cv2.circle(frame, (trx, tyr), 2, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    total_frames = total_frames + 1

    fps_end = datetime.datetime.now()
    time_diff = fps_end - fps_start
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    fps_text = f"FPS: {fps:.2f}"
    
    cv2.putText(frame, fps_text, (800, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f'Total: {len(motorbikeCount) + len(carCount) + len(autoCount) + len(busCount) + len(lorryCount) + len(tempoCount) + len(tractorCount)}', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(frame, f'Motorbikes: {len(motorbikeCount)}', (40, 65), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    cv2.putText(frame, f'Cars: {len(carCount)}', (40, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, f'Autos: {len(autoCount)}', (40, 115), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
    cv2.putText(frame, f'Buses: {len(busCount)}', (40, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 192, 203), 1)
    cv2.putText(frame, f'Lorry: {len(lorryCount)}', (40, 165), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 127), 1)
    cv2.putText(frame, f'Tempo: {len(tempoCount)}', (40, 190), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 127, 255), 1)
    cv2.putText(frame, f'Tractor: {len(tractorCount)}', (40, 215), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)
    cv2.imshow("Application", frame)
    cv2.waitKey(1)