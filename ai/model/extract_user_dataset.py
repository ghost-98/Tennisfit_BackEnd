import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLOv10
from IPython.display import display, Image


def extract_user_dataset(video_path):
    mp_pose = mp.solutions.pose

    model = YOLOv10('../yolov10n.pt')

    video = cv2.VideoCapture(video_path)
    focused_tracking_id = ''
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    user_dataset = []
    dataset = []
    sub_dataset = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        # Loop through the video frames
        while video.isOpened():
            # Read a frame from the video
            success, frame = video.read()

            if success:
                results = model.track(frame, persist=True, classes=0)

                # Get the boxes and track IDs
                boxes = results[0].boxes
                if boxes.id is None:
                    continue
                track_ids = results[0].boxes.id.int().cpu().tolist()

                if not focused_tracking_id or focused_tracking_id not in track_ids:  # tracking 할 대상을 찾지 못했을 때
                    for i in track_ids:
                        x1, y1, x2, y2 = np.array(boxes.xyxy[track_ids.index(i)], dtype=np.int64)
                        print(f'id : {i}')
                        _, encoded_image = cv2.imencode('.png', frame[y1:y2, x1:x2])
                        display(Image(data=encoded_image))
                    print('choose tracking object id. If you can\'t find your object, press "n"')
                    focused_tracking_id = input()
                    if focused_tracking_id == 'n':
                        focused_tracking_id = ''
                        continue
                    else:
                        focused_tracking_id = int(focused_tracking_id)
                    print(f'id : {focused_tracking_id}')
                else:  # tracking 할 대상을 찾았을 떄
                    track_id_index = track_ids.index(int(focused_tracking_id))
                    x1, y1, x2, y2 = np.array(boxes.xyxy[track_id_index], dtype=np.int64)
                    label = f'id : {focused_tracking_id} | {boxes.conf[track_id_index]:.2}'

                    crop_obj = frame[y1:y2, x1:x2]  # focused tracking box object 잘라내기

                    # mediapipe 동작 skeleten 입히기
                    crop_obj.flags.writeable = False
                    crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
                    crop_results = pose.process(crop_obj)
                    for k in range(33):
                        if k in [1, 2, 3, 4, 5, 6, 9, 10]:
                            continue
                        sub_dataset.append(crop_results.pose_world_landmarks.landmark[k].visibility)
                        sub_dataset.append(crop_results.pose_world_landmarks.landmark[k].x)
                        sub_dataset.append(crop_results.pose_world_landmarks.landmark[k].y)
                        sub_dataset.append(crop_results.pose_world_landmarks.landmark[k].z)
                    dataset.append(sub_dataset)
                    sub_dataset = []

                cv2.waitKey(int(1 / 30 * 1000))
            else:
                break

    video.release()
    for i in range(len(dataset) - 59):
        user_dataset.append(dataset[i:i + 60])

    return user_dataset


if __name__ == '__main__':
    dataset = extract_user_dataset('../video/v5.mp4')
    dataset = np.array(dataset)
    print(dataset.shape)
    np.save('../video/v5.npy', dataset)
