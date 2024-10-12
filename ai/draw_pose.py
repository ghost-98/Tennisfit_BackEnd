import numpy as np
import matplotlib.pyplot as plt
import joblib
from mpl_toolkits.mplot3d import Axes3D


def plot_world_landmarks(
        plt,
        ax,
        landmarks,
        visibility_th=0.5,
):
    landmark_point = []

    for i in range(25):
        print(f'index : {i}')
        k = i * 4
        print(f'visibility : {landmarks[k]}')
        print(f'x : {landmarks[k+1]}')
        print(f'y : {landmarks[k+2]}')
        print(f'z : {landmarks[k+3]}')
        print()
        landmark_point.append(
            [landmarks[k], (landmarks[k+1], landmarks[k+2], landmarks[k+3])])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 1, 2, 9, 10]
    right_arm_index_list = [3, 5, 7, 9, 11, 13]
    left_arm_index_list = [4, 6, 8, 10, 12, 14]
    right_body_side_index_list = [3, 15, 17, 19, 21, 23]
    left_body_side_index_list = [4, 16, 18, 20, 22, 24]
    shoulder_index_list = [3, 4]
    waist_index_list = [15, 16]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

    plt.pause(.001)

    return

if __name__ == '__main__':
    similarity_model = joblib.load('model/model_info/similarity_model.joblib')

    centroids = similarity_model.cluster_centers_
    print(centroids[9].shape)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    shape1_3d = np.load("./variable_dataset/Federer_B_Forehand_dataset.npy")
    shape1_3d = shape1_3d[0]
    shape2_3d = np.load("./video/test_video.npy")
    shape2_3d = shape2_3d[363]
    plot_world_landmarks(plt, ax, shape2_3d[0])