import numpy as np
import random


# dataset 제일 뒷열 지우기
def delete_last_data(arr):
    return np.delete(arr, -1, axis=0)


# dataset load
def load_dataset(data_type, game_type, name):
    if data_type == 'sampling':
        if game_type == 'train':
            t0 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Forehand_sampling_{game_type}_dataset.npy')
            t1 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Backhand_sampling_{game_type}_dataset.npy')
            t2 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_BackSlice_sampling_{game_type}_dataset.npy')
            t3 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_ForeVolley_sampling_{game_type}_dataset.npy')
            t4 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_BackVolley_sampling_{game_type}_dataset.npy')
            t5 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Smash_sampling_{game_type}_dataset.npy')
            t6 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Serve_sampling_{game_type}_dataset.npy')
        elif game_type == 'test':
            t0 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Forehand_sampling_{game_type}_dataset.npy')
            t1 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Backhand_sampling_{game_type}_dataset.npy')
            t2 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_BackSlice_sampling_{game_type}_dataset.npy')
            t3 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_ForeVolley_sampling_{game_type}_dataset.npy')
            t4 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_BackVolley_sampling_{game_type}_dataset.npy')
            t5 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Smash_sampling_{game_type}_dataset.npy')
            t6 = np.load(f'./variable_dataset/{game_type}ing_data/{name}/{name}_B_Serve_sampling_{game_type}_dataset.npy')
    elif data_type == 'temp':
        t0 = np.load(f'./temp_{game_type}/{name}/{name}_B_Forehand_{game_type}_dataset.npy')
        t1 = np.load(f'./temp_{game_type}/{name}/{name}_B_Backhand_{game_type}_dataset.npy')
        t2 = np.load(f'./temp_{game_type}/{name}/{name}_B_BackSlice_{game_type}_dataset.npy')
        t3 = np.load(f'./temp_{game_type}/{name}/{name}_B_ForeVolley_{game_type}_dataset.npy')
        t4 = np.load(f'./temp_{game_type}/{name}/{name}_B_BackVolley_{game_type}_dataset.npy')
        t5 = np.load(f'./temp_{game_type}/{name}/{name}_B_Smash_{game_type}_dataset.npy')
        t6 = np.load(f'./temp_{game_type}/{name}/{name}_B_Serve_{game_type}_dataset.npy')
    else:
        if game_type == 'total':
            t0 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Forehand_dataset.npy')
            t1 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Backhand_dataset.npy')
            t2 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_BackSlice_dataset.npy')
            t3 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_ForeVolley_dataset.npy')
            t4 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_BackVolley_dataset.npy')
            t5 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Smash_dataset.npy')
            t6 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Serve_dataset.npy')
        else:
            t0 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Forehand_{game_type}_dataset.npy')
            t1 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Backhand_{game_type}_dataset.npy')
            t2 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_BackSlice_{game_type}_dataset.npy')
            t3 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_ForeVolley_{game_type}_dataset.npy')
            t4 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_BackVolley_{game_type}_dataset.npy')
            t5 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Smash_{game_type}_dataset.npy')
            t6 = np.load(f'./variable_dataset/{game_type}/{name}/{name}_B_Serve_{game_type}_dataset.npy')

    return t0, t1, t2, t3, t4, t5, t6


# dataset 확인 'data_type : temp, confirmed, sampling' 'game_type : train, match, total' 'name : 선수 이름'
def check_dataset(data_type, game_type, name):
    t0, t1, t2, t3, t4, t5, t6 = load_dataset(data_type=data_type, game_type=game_type, name=name)

    if data_type == 'temp':
        print(f'{name} {game_type} 임시 데이터')
    else:
        if game_type == 'total':
            print(f'{name} 최종 데이터')
        else:
            print(f'{name} {game_type} 확정 데이터')

    print(f'forehand : {t0.shape[0]}')
    print(f'backhand : {t1.shape[0]}')
    print(f'backslice : {t2.shape[0]}')
    print(f'forevolley : {t3.shape[0]}')
    print(f'backvolley : {t4.shape[0]}')
    print(f'smash : {t5.shape[0]}')
    print(f'serve : {t6.shape[0]}')
    print(f'합계 : {t0.shape[0] + t1.shape[0] + t2.shape[0] + t3.shape[0] + t4.shape[0] + t5.shape[0] + t6.shape[0]}')
    print()


# dataset 합치기
def sum_train_match_dataset(name):
    t0 = np.load(f'./variable_dataset/train/{name}/{name}_B_Forehand_train_dataset.npy')
    t1 = np.load(f'./variable_dataset/train/{name}/{name}_B_Backhand_train_dataset.npy')
    t2 = np.load(f'./variable_dataset/train/{name}/{name}_B_BackSlice_train_dataset.npy')
    t3 = np.load(f'./variable_dataset/train/{name}/{name}_B_ForeVolley_train_dataset.npy')
    t4 = np.load(f'./variable_dataset/train/{name}/{name}_B_BackVolley_train_dataset.npy')
    t5 = np.load(f'./variable_dataset/train/{name}/{name}_B_Smash_train_dataset.npy')
    t6 = np.load(f'./variable_dataset/train/{name}/{name}_B_Serve_train_dataset.npy')

    t7 = np.load(f'./variable_dataset/match/{name}/{name}_B_Forehand_match_dataset.npy')
    t8 = np.load(f'./variable_dataset/match/{name}/{name}_B_Backhand_match_dataset.npy')
    t9 = np.load(f'./variable_dataset/match/{name}/{name}_B_BackSlice_match_dataset.npy')
    t10 = np.load(f'./variable_dataset/match/{name}/{name}_B_ForeVolley_match_dataset.npy')
    t11 = np.load(f'./variable_dataset/match/{name}/{name}_B_BackVolley_match_dataset.npy')
    t12 = np.load(f'./variable_dataset/match/{name}/{name}_B_Smash_match_dataset.npy')
    t13 = np.load(f'./variable_dataset/match/{name}/{name}_B_Serve_match_dataset.npy')

    s0 = np.append(t0, t7, axis=0)
    s1 = np.append(t1, t8, axis=0)
    s2 = np.append(t2, t9, axis=0)
    s3 = np.append(t3, t10, axis=0)
    s4 = np.append(t4, t11, axis=0)
    s5 = np.append(t5, t12, axis=0)
    s6 = np.append(t6, t13, axis=0)

    np.save(f'./variable_dataset/total/{name}/{name}_B_Forehand_dataset.npy', s0)
    np.save(f'./variable_dataset/total/{name}/{name}_B_Backhand_dataset.npy', s1)
    np.save(f'./variable_dataset/total/{name}/{name}_B_BackSlice_dataset.npy', s2)
    np.save(f'./variable_dataset/total/{name}/{name}_B_ForeVolley_dataset.npy', s3)
    np.save(f'./variable_dataset/total/{name}/{name}_B_BackVolley_dataset.npy', s4)
    np.save(f'./variable_dataset/total/{name}/{name}_B_Smash_dataset.npy', s5)
    np.save(f'./variable_dataset/total/{name}/{name}_B_Serve_dataset.npy', s6)


# 좌우 반전
def left_right_inversion_dataset(name):
    t0 = np.load(f'./variable_dataset/total/{name}/{name}_B_Forehand_dataset.npy')
    t1 = np.load(f'./variable_dataset/total/{name}/{name}_B_Backhand_dataset.npy')
    t2 = np.load(f'./variable_dataset/total/{name}/{name}_B_BackSlice_dataset.npy')
    t3 = np.load(f'./variable_dataset/total/{name}/{name}_B_ForeVolley_dataset.npy')
    t4 = np.load(f'./variable_dataset/total/{name}/{name}_B_BackVolley_dataset.npy')
    t5 = np.load(f'./variable_dataset/total/{name}/{name}_B_Smash_dataset.npy')
    t6 = np.load(f'./variable_dataset/total/{name}/{name}_B_Serve_dataset.npy')

    def left_right_inversion(arr):
        for i in arr:
            for j in i:
                for k in range(12):
                    left = 4 * (2 * k + 1)
                    right = 4 * (2 * (k + 1))
                    # 확률
                    j[left + 0], j[right + 0] = j[right + 0], j[left + 0]
                    # x좌표
                    j[left + 1], j[right + 1] = j[right + 1], j[left + 1]
                    # y좌표
                    j[left + 2], j[right + 2] = j[right + 2], j[left + 2]
                    # z좌표
                    j[left + 3], j[right + 3] = j[right + 3], j[left + 3]

    left_right_inversion(t0)
    left_right_inversion(t1)
    left_right_inversion(t2)
    left_right_inversion(t3)
    left_right_inversion(t4)
    left_right_inversion(t5)
    left_right_inversion(t6)

    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_Forehand_dataset.npy', t0)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_Backhand_dataset.npy', t1)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_BackSlice_dataset.npy', t2)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_ForeVolley_dataset.npy', t3)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_BackVolley_dataset.npy', t4)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_Smash_dataset.npy', t5)
    np.save(f'./variable_dataset/total/{name}_inv/{name}_inv_B_Serve_dataset.npy', t6)


# sampling 사용시 함수 내부의 수치 조절해야 함.
def sampling_dataset(name):
    t0, t1, t2, t3, t4, t5, t6 = load_dataset('confirmed', 'train', name)
    t7, t8, t9, t10, t11, t12, t13 = load_dataset('confirmed', 'match', name)

    def sampling(swing, train_dataset, test_dataset, train_cnt, test_cnt):
        np.random.shuffle(train_dataset)
        np.random.shuffle(test_dataset)
        train0, test0 = train_dataset[:train_cnt], train_dataset[train_cnt:]
        train1, test1 = test_dataset[:test_cnt], test_dataset[test_cnt:]
        sum_train = np.append(train0, train1, axis=0)
        sum_test = np.append(test0, test1, axis=0)
        print(sum_train.shape)
        print(sum_test.shape)
        np.save(f'./variable_dataset/training_data/{name}/{name}_B_{swing}_sampling_train_dataset', sum_train)
        np.save(f'./variable_dataset/testing_data/{name}/{name}_B_{swing}_sampling_test_dataset', sum_test)

    sampling('Forehand', t0, t7, 114, 118)
    sampling('Backhand', t1, t8, 120, 102)
    sampling('BackSlice', t2, t9, 46, 53)
    sampling('ForeVolley', t3, t10, 77, 23)
    sampling('BackVolley', t4, t11, 63, 37)
    sampling('Smash', t5, t12, 38, 15)
    sampling('Serve', t6, t13, 45, 57)


if __name__ == '__main__':
    # dataset 개수 확인
    # 선수 임시 데이터 확인
    # check_dataset('temp', 'train', 'Rune')

    # 선수 확정 데이터 확인
    # check_dataset('confirmed', 'match', 'Federer')
    # check_dataset('confirmed', 'train', 'Federer')
    # check_dataset('confirmed', 'total', 'Federer')
    #
    # check_dataset('confirmed', 'match', 'Nadal')
    # check_dataset('confirmed', 'train', 'Nadal')
    # check_dataset('confirmed', 'total', 'Nadal')
    #
    # check_dataset('confirmed', 'match', 'Djokovic')
    # check_dataset('confirmed', 'train', 'Djokovic')
    # check_dataset('confirmed', 'total', 'Djokovic')
    #
    # check_dataset('confirmed', 'match', 'Sinner')
    # check_dataset('confirmed', 'train', 'Sinner')
    # check_dataset('confirmed', 'total', 'Sinner')
    #
    # check_dataset('confirmed', 'match', 'Tsitsipas')
    # check_dataset('confirmed', 'train', 'Tsitsipas')
    # check_dataset('confirmed', 'total', 'Tsitsipas')
    #
    # check_dataset('confirmed', 'match', 'Zverev')
    # check_dataset('confirmed', 'train', 'Zverev')
    # check_dataset('confirmed', 'total', 'Zverev')
    #
    # check_dataset('confirmed', 'match', 'Murray')
    # check_dataset('confirmed', 'train', 'Murray')
    # check_dataset('confirmed', 'total', 'Murray')
    #
    # check_dataset('confirmed', 'match', 'Alcaraz')
    # check_dataset('confirmed', 'train', 'Alcaraz')
    # check_dataset('confirmed', 'total', 'Alcaraz')
    #
    # check_dataset('confirmed', 'match', 'Rune')
    # check_dataset('confirmed', 'train', 'Rune')
    # check_dataset('confirmed', 'total', 'Rune')
    #
    # check_dataset('confirmed', 'match', 'Shapovalov')
    # check_dataset('confirmed', 'train', 'Shapovalov')
    # check_dataset('confirmed', 'total', 'Shapovalov')

    # 선수 sampling 데이터 확인
    check_dataset('sampling', 'train', 'Federer')
    check_dataset('sampling', 'test', 'Federer')
    check_dataset('sampling', 'train', 'Nadal_inv')
    check_dataset('sampling', 'test', 'Nadal_inv')
    check_dataset('sampling', 'train', 'Djokovic')
    check_dataset('sampling', 'train', 'Sinner')
    check_dataset('sampling', 'test', 'Sinner')
    check_dataset('sampling', 'train', 'Tsitsipas')
    check_dataset('sampling', 'test', 'Tsitsipas')
    check_dataset('sampling', 'train', 'Zverev')
    check_dataset('sampling', 'train', 'Murray')
    check_dataset('sampling', 'test', 'Murray')
    check_dataset('sampling', 'train', 'Alcaraz')
    check_dataset('sampling', 'train', 'Rune')
    check_dataset('sampling', 'train', 'Shapovalov_inv')
    check_dataset('sampling', 'test', 'Shapovalov_inv')

    # dataset 합치기
    # sum_train_match_dataset('Nadal')
    # sum_train_match_dataset('Sinner')
    # sum_train_match_dataset('Tsitsipas')

    # 좌우 반전
    # left_right_inversion_dataset('Nadal')
