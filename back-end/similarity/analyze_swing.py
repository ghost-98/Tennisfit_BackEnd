import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .similarity_model import *


def analyze_swing(np_file_path):
    # PyTorch가 GPU(cuda) 사용할수 있는지 확인
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('GPU 사용 가능')
    else:
        device = 'cpu'
        print('GPU 사용 불가')

    # 모델 초기화(인스턴스) 및 해당 모델을 GPU/CPU로 이동해서 해당 환경에서 사용하도록
    classification_model = ClassificationModel().to(device)
    forehand_model = Swing016Model().to(device)
    backhand_model = Swing016Model().to(device)
    serve_model = Swing016Model().to(device)
    backslice_model = Swing2Model().to(device)
    forevolley_model = Swing34Model().to(device)
    backvolley_model = Swing34Model().to(device)
    smash_model = Swing5Model().to(device)
    anomaly_detection_model = LSTMAutoEncoder(input_dim=100, latent_dim=50, sequence_length=60, num_layers=6).to(device)

    # 모델 불러오기, 가중치 할당, 경로 ./로 하면 안되나
    classification_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/classification_model', map_location=device))
    forehand_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_forehand', map_location=device))
    backhand_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_backhand', map_location=device))
    serve_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_serve', map_location=device))
    backslice_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_backslice', map_location=device))
    forevolley_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_forevolley', map_location=device))
    backvolley_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_backvolley', map_location=device))
    smash_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/similarity_model_smash', map_location=device))
    anomaly_detection_model.load_state_dict(
        torch.load('../back-end/similarity/model_info/anomaly_detection_model', map_location=device))

    print("모델 불러오기 성공")

    test_dataset = np.load(np_file_path)  # numpy 파일 읽기
    test_data = []
    anomaly_detection_data = []

    # pytorch 모델은 학습모드와 평가모드 있음. eval()은 평가모드로 전환
    classification_model.eval()
    forehand_model.eval()
    backhand_model.eval()
    serve_model.eval()
    smash_model.eval()
    backslice_model.eval()
    forevolley_model.eval()
    backvolley_model.eval()
    anomaly_detection_model.eval()

    # 데이터 전처리
    for data in test_dataset:
        test_data.append({'key': 8, 'value': data})
        anomaly_detection_data.append(data)

    test_data = TestDataset(test_data)
    test_data = DataLoader(test_data)
    anomaly_detection_data = AnomalyDetectionDataset(anomaly_detection_data)
    anomaly_detection_data = DataLoader(anomaly_detection_data)

    # Classification, Similarity 계산
    test_data_list = []

    for data, label in test_data:
        data = data.to(device)
        data_list = []
        with torch.no_grad():
            classification_result = classification_model(data)
            classification_result = F.softmax(classification_result, dim=1)
            out_result, out = torch.max(classification_result, 1)

            # 스윙 결과 결정
            if out.item() == 0:
                swing_result = forehand_model(data)
                data_list.append('Forehand')
            elif out.item() == 1:
                swing_result = backhand_model(data)
                data_list.append('Backhand')
            elif out.item() == 2:
                swing_result = backslice_model(data)
                data_list.append('Backslice')
            elif out.item() == 3:
                swing_result = forevolley_model(data)
                data_list.append('ForeVolley')
            elif out.item() == 4:
                swing_result = backvolley_model(data)
                data_list.append('BackVolley')
            elif out.item() == 5:
                swing_result = smash_model(data)
                data_list.append('Smash')
            elif out.item() == 6:
                swing_result = serve_model(data)
                data_list.append('Serve')

            swing_result = F.softmax(swing_result, dim=1).squeeze().cpu().numpy()  # numpy 배열로 반환
            data_list.append(swing_result)

            test_data_list.append(data_list)

    # 비정상 점수 계산
    anomaly_detection_list = []

    with torch.no_grad():
        for _data in anomaly_detection_data:
            data = _data.to(device)
            predict_values = anomaly_detection_model(data)
            loss = F.l1_loss(predict_values[0], predict_values[1], reduce=False)
            loss = loss.mean(dim=1).cpu().numpy()
            anomaly_detection_list.append(loss)

    anomaly_detection_list = np.concatenate(anomaly_detection_list, axis=0)

    # Reconstruction Error의 평균과 Covariance 계산
    mean = np.mean(anomaly_detection_list, axis=0)
    std = np.cov(anomaly_detection_list.T)

    anomaly_calculator = Anomaly_Calculator(mean, std)

    anomaly_scores = []
    for temp_loss in anomaly_detection_list:
        temp_score = anomaly_calculator(temp_loss)
        anomaly_scores.append(temp_score)

    result = []
    frame_cnt = 0
    threshold = 0.0025
    is_detection = False

    for out_index in range(len(anomaly_scores)):
        if anomaly_scores[out_index] < threshold:
            if frame_cnt == 0:
                result.append(test_data_list[out_index])
                is_detection = True
        if is_detection:
            if frame_cnt < 71:
                frame_cnt += 1
            else:
                frame_cnt = 0
                is_detection = False

    # 유사도 계산
    similarity_cnt = np.zeros(8)
    total_similarity = np.zeros(10)
    forehand_similarity = np.zeros(10)
    backhand_similarity = np.zeros(10)
    backslice_similarity = np.zeros(10)
    forevolley_similarity = np.zeros(10)
    backvolley_similarity = np.zeros(10)
    smash_similarity = np.zeros(10)
    serve_similarity = np.zeros(10)

    for info in result:
        total_similarity += np.array(info[1])
        similarity_cnt[0] += 1
        if info[0] == 'Forehand':
            forehand_similarity += np.array(info[1])
            similarity_cnt[1] += 1
        elif info[0] == 'Backhand':
            backhand_similarity += np.array(info[1])
            similarity_cnt[2] += 1
        elif info[0] == 'Backslice':
            backslice_similarity += np.array(info[1])
            similarity_cnt[3] += 1
        elif info[0] == 'ForeVolley':
            forevolley_similarity += np.array(info[1])
            similarity_cnt[4] += 1
        elif info[0] == 'BackVolley':
            backvolley_similarity += np.array(info[1])
            similarity_cnt[5] += 1
        elif info[0] == 'Smash':
            smash_similarity += np.array(info[1])
            similarity_cnt[6] += 1
        elif info[0] == 'Serve':
            serve_similarity += np.array(info[1])
            similarity_cnt[7] += 1

    final_result = []

    cnt = 0  # x_similarity는 스윙의 유사도 비교횟수
    for info in [total_similarity, forehand_similarity, backhand_similarity, backslice_similarity, forevolley_similarity, backvolley_similarity, smash_similarity, serve_similarity]:
        temp_arr = []
        if similarity_cnt[cnt] != 0:
            info_avg = info / similarity_cnt[cnt]
        else:
            info_avg = info
        cnt += 1
        for _ in range(10):
            max_value = round(np.max(info_avg) * 100, 1)
            max_index = np.argmax(info_avg)
            players = ['Federer', 'Nadal_inv', 'Djokovic', 'Sinner', 'Tsitsipas', 'Zverev', 'Murray', 'Alcaraz', 'Rune', 'Shapovalov_inv']
            temp_arr.append([players[max_index], max_value])
            info_avg[max_index] = -1
        final_result.append(temp_arr)

    return final_result  # 둘다 numpy 배열임