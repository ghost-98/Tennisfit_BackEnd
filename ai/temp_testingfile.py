# 테스트 데이터 개수 출력
import numpy as np

ar = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_Forehand_train_dataset.npy')
ar1 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_Backhand_train_dataset.npy')
ar2 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_BackSlice_train_dataset.npy')
ar3 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_ForeVolley_train_dataset.npy')
ar4 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_BackVolley_train_dataset.npy')
#ar5 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_Smash_train_dataset.npy')
ar6 = np.load('./variable_dataset/testing_data/Sinner/train/Sinner_B_Serve_train_dataset.npy')

ar7 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_Forehand_match_dataset.npy')
ar8 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_Backhand_match_dataset.npy')
ar9 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_BackSlice_match_dataset.npy')
ar10 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_ForeVolley_match_dataset.npy')
ar11 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_BackVolley_match_dataset.npy')
#ar12 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_Smash_match_dataset.npy')
ar13 = np.load('./variable_dataset/testing_data/Sinner/match/Sinner_B_Serve_match_dataset.npy')

print(ar.shape[0])
print(ar1.shape[0])
print(ar2.shape[0])
print(ar3.shape[0])
print(ar4.shape[0])
#print(ar5.shape[0])
print(ar6.shape[0])

print(ar7.shape[0])
print(ar8.shape[0])
print(ar9.shape[0])
print(ar10.shape[0])
print(ar11.shape[0])
#print(ar12.shape[0])
print(ar13.shape[0])

