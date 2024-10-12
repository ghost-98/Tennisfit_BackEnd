import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLOv10
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import draw_pose
import csv


fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load an official or custom model
model = YOLOv10('yolov10n.pt')  # Load an official Detect model

# Perform tracking with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker

# Open the video file
video_path = "./video/v3.mp4"
video = cv2.VideoCapture(video_path)

focused_tracking_id = ''
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
wait_msec = int(1 / 30 * 1000)

pose_fps = 0
standard_fps_rate = 50
record_type = 0
player_name = 'Tsitsipas'
carmera_viewpoint = 'B'  # 'B', 'F', 'R', 'L'
swing_type = ['Stand', 'Forehand', 'Backhand', 'BackSlice', 'ForeVolley', 'BackVolley', 'Smash', 'Serve']
video_type = 'match'  # 'train', 'match'
dataset = []
label_data = []
isRecord = False
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # Loop through the video frames
    while video.isOpened():
        # Read a frame from the video
        success, frame = video.read()

        if success:

            # Yolov8
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=0) # yolov8 학습 데이터에서 class 0번이 사람이다

            # Get the boxes and track IDs
            boxes = results[0].boxes
            if boxes.id is None:
                continue
            track_ids = results[0].boxes.id.int().cpu().tolist()

            if not focused_tracking_id or int(focused_tracking_id) not in track_ids: # tracking 할 대상을 찾지 못했을 때
                # Visualize the results on the frame
                frame = results[0].plot()
            else: # tracking 할 대상을 찾았을 떄
                track_id_index = track_ids.index(int(focused_tracking_id))
                x1, y1, x2, y2 = np.array(boxes.xyxy[track_id_index], dtype=np.int64)
                label = f'id : {focused_tracking_id} | {boxes.conf[track_id_index]:.2}'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
                color = (0, 0, 255)  # -- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.rectangle(frame, (x1-3, y1), (x1+text_size[0][0], y1-text_size[0][1]-10), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

                crop_obj = frame[y1:y2, x1:x2] # focused tracking box object 잘라내기

                # mediapipe 동작 skeleten 입히기
                # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
                crop_obj.flags.writeable = False
                crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
                crop_results = pose.process(crop_obj)

                # 포즈 주석을 이미지 위에 그립니다.
                crop_obj.flags.writeable = True
                crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    crop_obj,
                    crop_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                frame[y1:y2, x1:x2] = crop_obj

                if isRecord:
                    print(pose_fps)
                    cv2.rectangle(frame, (5, 5), (width - 5, height - 5), color=(0, 0, 255), thickness=3)
                    cv2.circle(frame, (30, 40), radius=10, color=(0, 0, 255), thickness=-1)
                    cv2.putText(frame, 'REC', (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=3)
                    pose_fps += 1
                    for k in range(33):
                        if k in [1, 2, 3, 4, 5, 6, 9, 10]:
                            continue
                        dataset.append(crop_results.pose_world_landmarks.landmark[k].visibility)
                        dataset.append(crop_results.pose_world_landmarks.landmark[k].x)
                        dataset.append(crop_results.pose_world_landmarks.landmark[k].y)
                        dataset.append(crop_results.pose_world_landmarks.landmark[k].z)
                    label_data.append(dataset)
                    dataset = []
                    if pose_fps == standard_fps_rate:
                        if len(label_data) != 60:
                            for _ in range(60 - len(label_data)):
                                label_data.append([0] * 100)
                        try:  # dataset 저장하기
                            arr = np.load(f'./temp_{video_type}/{player_name}_{carmera_viewpoint}_{swing_type[record_type]}_{video_type}_dataset.npy')
                            append = np.append(arr, np.array(label_data).reshape(1, 60, 100), axis=0)
                        except FileNotFoundError:
                            append = np.array(label_data).reshape(1, 60, 100)
                        pose_fps = 0
                        np.save(f'./temp_{video_type}/{player_name}_{carmera_viewpoint}_{swing_type[record_type]}_{video_type}_dataset', append)
                        label_data.clear()
                        isRecord = False

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", frame)

            while not focused_tracking_id or int(focused_tracking_id) not in track_ids:
                temp_frame = frame.copy()
                info = f'Choose The ID You Want To Track(ESC => Initialize Inputs) : {focused_tracking_id}'
                cv2.putText(temp_frame, info, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
                cv2.imshow("YOLOv8 Tracking", temp_frame)
                tracking_id = cv2.waitKey()
                if tracking_id == 27 or tracking_id == 8:
                    focused_tracking_id = ''
                else:
                    focused_tracking_id += chr(tracking_id)
                print(f'id : {focused_tracking_id}')

            key = cv2.waitKey(wait_msec)
            if key == 27:
                break
            elif key == ord('p'):
                focused_tracking_id = ''
            elif key == ord(' '):
                standard_fps_rate = pose_fps + 1
            elif key == ord('0'):
                isRecord = not isRecord
                record_type = 0
                standard_fps_rate = 25
            elif key == ord('1'):
                isRecord = not isRecord
                record_type = 1
                standard_fps_rate = 60
            elif key == ord('2'):
                isRecord = not isRecord
                record_type = 2
                standard_fps_rate = 60
            elif key == ord('3'):
                isRecord = not isRecord
                record_type = 3
                standard_fps_rate = 60
            elif key == ord('4'):
                isRecord = not isRecord
                record_type = 4
                standard_fps_rate = 25
            elif key == ord('5'):
                isRecord = not isRecord
                record_type = 5
                standard_fps_rate = 25
            elif key == ord('6'):
                isRecord = not isRecord
                record_type = 6
                standard_fps_rate = 60
            elif key == ord('7'):
                isRecord = not isRecord
                record_type = 7
                standard_fps_rate = 60
        else:
            # Break the loop if the end of the video is reached
            break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()