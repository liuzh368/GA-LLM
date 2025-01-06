import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm




import numpy as np


def compute_time_weight(time_diff, lambda_value=0.001, max_time_diff=1000000):
    """
    计算时间衰减权重，通过缩放时间差，避免过大的时间差导致溢出。
    :param time_diff: 当前时间与轨迹时间戳之间的差值（秒）
    :param lambda_value: 衰减系数，默认值0.001
    :param max_time_diff: 限制最大时间差，默认值1000000
    :return: 时间权重
    """
    # 将时间差缩放为天
    time_diff_in_days = time_diff / (60 * 60 * 24)

    # 限制时间差的最大值，避免指数溢出
    if time_diff_in_days > max_time_diff:
        return 0.0  # 超过最大时间差时直接返回0权重
    else:
        return np.exp(-lambda_value * time_diff_in_days)

def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, is_test=False):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    qa_pairs = []

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]

            # Get the start time of the current trajectory
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()

            # 当前轨迹，权重 1.0
            current_trajectory_data = user_trajectory_data.iloc[:-1]

            # 用户当前时刻之前的全部轨迹，权重 0.7
            if historical_data is not None:
                user_historical_trajectory = historical_data[
                    (historical_data['UserId'] == user) &
                    (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                ].tail(600 - current_trajectory_data.shape[0])
            else:
                user_historical_trajectory = user_data[
                    (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                ].tail(600 - current_trajectory_data.shape[0])

            # 相似轨迹，权重 0.3（仅在非测试数据中使用）
            if not is_test and str(traj_id) in kqt.keys():
                top200 = kqt[str(traj_id)]  # 获取对应的 top200 列表
                if historical_data is not None:
                    historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)
                    matching_historical_traj = historical_traj_ids.isin(top200)
                    matched_historical_data = historical_data[matching_historical_traj].tail(200 - current_trajectory_data.shape[0])
                else:
                    main_traj_ids = main_data['pseudo_session_trajectory_id'].astype(str)
                    matching_main_traj = main_traj_ids.isin(top200)
                    matched_historical_data = main_data[matching_main_traj].tail(200 - current_trajectory_data.shape[0])
                similar_trajectory_data = matched_historical_data
            else:
                similar_trajectory_data = pd.DataFrame()  # 测试数据不使用相似轨迹

            # 当前时间
            current_time = user_trajectory_data['UTCTimeOffsetEpoch'].max()

            # 构建问题 prompt
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            for i, row in current_trajectory_data.iterrows():
                time_weight = compute_time_weight(current_time - row['UTCTimeOffsetEpoch'], lambda_value=0.001)
                question_parts.append(
                    f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']} (Weight={1.0 * time_weight:.3f})."
                )

            # 添加历史轨迹信息（第二重要）
            if not user_historical_trajectory.empty:
                question_parts.append("There is also historical data:")
                for _, row in user_historical_trajectory.iterrows():
                    time_weight = compute_time_weight(current_time - row['UTCTimeOffsetEpoch'], lambda_value=0.001)
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']} (Weight={0.7 * time_weight:.3f})."
                    )

            # 添加相似轨迹（最不重要，且只在训练数据中使用）
            if not similar_trajectory_data.empty:
                question_parts.append("There is also similar trajectory data:")
                for _, row in similar_trajectory_data.iterrows():
                    time_weight = compute_time_weight(current_time - row['UTCTimeOffsetEpoch'], lambda_value=0.001)
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']} (Weight={0.3 * time_weight:.3f})."
                    )

            # 最终问题
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # 构建答案
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # 保存QA对
            qa_pairs.append((question, answer))
    return qa_pairs

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")

    # Parse the arguments
    args = parser.parse_args()

    # Your processing code here
    print(f"Processing dataset: {args.dataset_name}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'
    # Read the data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample.csv')
    kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args, is_test=False)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args, is_test=True)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_nyc_merged_weight.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)


    # Save the test QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt_200items_nyc_merged_weight.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()

