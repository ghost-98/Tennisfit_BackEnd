from ultralytics import YOLOv10
import base64
import cv2
import numpy as np
import mediapipe as mp


# 이 모듈이 호출될 때 프레임과 id를 받도록 수정
def process_video(video_path, frame_info, prev_dataset=None):
    yolo_model = YOLOv10('../yolov10n.pt')
    prev_dataset = prev_dataset or []  # 이전 데이터가 없다면 빈 리스트로 초기화

    mp_pose = mp.solutions.pose

    video = cv2.VideoCapture(video_path)
    if frame_info[1] != None:
        focused_tracking_id = frame_info[1]
    else:
        focused_tracking_id = ''

    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    image_list = []
    dataset = prev_dataset
    sub_dataset = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        # Loop through the video 'frames'
        while video.isOpened():  # 한프레임당 계속 반복
            # Read a frame from the video
            success, frame = video.read()

            # 비디오 프레임을 성공적으로 읽었을 때만 진행
            if success:
                results = yolo_model.track(frame, persist=True, classes=0)  # 현재 프레임에서 객체 탐지

                # 추적된 객체의 박스와 박스의 ID
                boxes = results[0].boxes
                if boxes.id is None:  # 추적된 객체 없으면 프레임 넘김
                    continue
                track_ids = results[0].boxes.id.int().cpu().tolist()  # 프레임에서 추적된 객체의 id 리스트
                if frame_info and video.get(cv2.CAP_PROP_POS_FRAMES) < frame_info[0]:  # 추적할 객체 id 받아오면 그 전 프레임 continue 해서 넘기는 코드
                    continue
                # 추적된 객체는 있으나 사용자 객체가 없어서 0을 받은 경우
                if focused_tracking_id == 0:
                    focused_tracking_id = ""
                    continue

                # Tracking 할 대상을 찾지 못했을 때
                if not focused_tracking_id or focused_tracking_id not in track_ids:  # fti - 추적중인 객체 / ti - 감지된 객체
                    for i in track_ids:  # 현재 프레임의 추적된 객체의 id를 붙인 이미지를 생성
                        x1, y1, x2, y2 = np.array(boxes.xyxy[track_ids.index(i)], dtype=np.int64)
                        print(f'id : {i}')
                        _, encoded_image = cv2.imencode('.png', frame[y1:y2, x1:x2])
                        encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
                        image_list.append([encoded_image_str, i])

                    # 현재 프레임과 이미지 리스트 반환 (외부에서 재선택할 수 있도록)
                    return {'frame': int(video.get(cv2.CAP_PROP_POS_FRAMES)), 'encoded_images': image_list, 'dataset': dataset}  # 이미지 리스트는 감지된 모든 객체

                # 객체 추출
                track_id_index = track_ids.index(int(focused_tracking_id))
                x1, y1, x2, y2 = np.array(boxes.xyxy[track_id_index], dtype=np.int64)

                crop_obj = frame[y1:y2, x1:x2]  # focused tracking box object 잘라내기

                # mediapipe 동작 스켈레톤 입히기
                crop_obj.flags.writeable = False
                crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
                crop_results = pose.process(crop_obj)
                if crop_results.pose_world_landmarks is None:
                    continue
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

    # 60 프레임 단위로 사용자 데이터셋 생성
    user_dataset = []
    for i in range(len(dataset) - 59):
        user_dataset.append(dataset[i:i + 60])

    # 영상 끝나면 최종 데이터 반환
    return {'dataset': user_dataset}


if __name__ == '__main__':
    dataset1 = process_video('../videos/videos/testvideo2.mp4', [])
    dataset1 = dataset1['dataset']
    dataset2 = process_video('../videos/videos/testvideo2.mp4', [1, 1], dataset1)
    dataset2 = dataset2['dataset']
    dataset3 = process_video('../videos/videos/testvideo2.mp4', [338, 6], dataset2)
    dataset3 = dataset3['dataset']
    dataset4 = process_video('../videos/videos/testvideo2.mp4', [416, 9], dataset3)
    dataset4 = dataset4['dataset']
    dataset5 = process_video('../videos/videos/testvideo2.mp4', [473, 14], dataset4)
    dataset5 = dataset5['dataset']
    dataset6 = process_video('../videos/videos/testvideo2.mp4', [588, 16], dataset5)
    dataset6 = dataset6['dataset']
    print(np.array(dataset6).shape)


'''# 이 모듈이 호출될 때 프레임과 id를 받도록 수정
def process_video(video_path, start_frame=None, selected_id=None, prev_dataset=None):
    yolo_model = YOLOv10('./yolov10n.pt')
    prev_dataset = prev_dataset or []  # 이전 데이터가 없다면 빈 리스트로 초기화

    mp_pose = mp.solutions.pose

    video = cv2.VideoCapture(video_path)
    focused_tracking_id = selected_id  # 외부에서 받은 ID를 사용
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    # 만약 시작 프레임이 있으면 해당 프레임으로 이동
    if start_frame is not None:
        if selected_id == 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

    image_list = []
    dataset = prev_dataset  # 이전 데이터 이어받기
    sub_dataset = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        # Loop through the video 'frames'
        while video.isOpened():  # 한프레임당 계속 반복
            # Read a frame from the video
            success, frame = video.read()

            # 비디오 프레임을 성공적으로 읽었을 때만 진행
            if success:
                results = yolo_model.track(frame, persist=True, classes=0)  # 현재 프레임에서 객체 탐지

                # 추적된 객체의 박스와 박스의 ID
                boxes = results[0].boxes
                if boxes.id is None:
                    continue
                track_ids = results[0].boxes.id.int().cpu().tolist()  # 프레임에서 추적된 객체의 id 리스트

                # Tracking 할 대상을 찾지 못했을 때
                if not focused_tracking_id or focused_tracking_id not in track_ids:  # fti - 추적중인 객체 / ti - 감지된 객체
                    for i in track_ids:  # 현재 프레임의 추적된 객체의 id를 붙인 이미지를 생성
                        x1, y1, x2, y2 = np.array(boxes.xyxy[track_ids.index(i)], dtype=np.int64)
                        print(f'id : {i}')
                        _, encoded_image = cv2.imencode('.png', frame[y1:y2, x1:x2])
                        encoded_image_str = base64.b64encode(encoded_image).decode('utf-8')
                        image_list.append(encoded_image_str)

                    # 현재 프레임과 이미지 리스트 반환 (외부에서 재선택할 수 있도록)
                    return {'frame': video.get(cv2.CAP_PROP_POS_FRAMES), 'encoded_images': image_list, 'dataset': dataset}  # 이미지 리스트는 감지된 모든 객체

                else:  # tracking 할 대상을 찾았을 때
                    track_id_index = track_ids.index(int(focused_tracking_id))
                    x1, y1, x2, y2 = np.array(boxes.xyxy[track_id_index], dtype=np.int64)

                    crop_obj = frame[y1:y2, x1:x2]  # focused tracking box object 잘라내기

                    # mediapipe 동작 스켈레톤 입히기
                    crop_obj.flags.writeable = False
                    crop_obj = cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB)
                    crop_results = pose.process(crop_obj)
                    if crop_results.pose_world_landmarks is None:
                        continue
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

    # 60 프레임 단위로 사용자 데이터셋 생성
    user_dataset = []
    for i in range(len(dataset) - 59):
        user_dataset.append(dataset[i:i + 60])

    # 영상 끝나면 최종 데이터 반환
    return {'dataset': user_dataset}'''