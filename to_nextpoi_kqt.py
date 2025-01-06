import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm


def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, use_sim=False):
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

            num_traj = user_trajectory_data.shape[0]

            # 初始化三个独立的 DataFrame 分别存储不同类型的历史数据
            similar_historical_data = pd.DataFrame()  # 相似轨迹
            early_historical_data = pd.DataFrame()    # 早期历史轨迹

            # 处理早期历史轨迹
            if historical_data is not None:
                early_historical_data = historical_data[
                    (historical_data['UserId'] == user) &
                    (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                ].tail(600)
            else:
                early_historical_data = user_data[
                    (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                ].tail(600)

            # 判断早期历史轨迹是否大于200条，如果是则不使用相似轨迹
            if len(early_historical_data) > 200:
                use_sim = False

            # 处理相似轨迹 (如果 use_sim 为 True 且历史数据未超过200条)
            if use_sim and str(traj_id) in kqt.keys():
                top200 = kqt[str(traj_id)]  # 获取对应的 top200 列表

                if historical_data is not None:
                    historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)
                    matching_historical_traj = historical_traj_ids.isin(top200)
                    matched_historical_data = historical_data[matching_historical_traj]

                    if matched_historical_data.shape[0] >= 200:
                        similar_historical_data = matched_historical_data.tail(200)
                    else:
                        similar_historical_data = matched_historical_data

            # 重置索引
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # 开始拼接问题部分，凸显重要性顺序：1. 相似轨迹 > 2. 早期历史轨迹 > 3. 当前轨迹（最重要的放最后）
            question_parts = []

            # 1. 处理相似轨迹（最不重要，放最前）
            if use_sim and not similar_historical_data.empty:
                question_parts.append(f"<question>: There is historical trajectory data similar to user {user}:")
                for _, row in similar_historical_data.iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )

            # 2. 处理早期历史轨迹（次重要）
            if not early_historical_data.empty:
                question_parts.append(f"There is also earlier, history trajectory data for user {user}:")
                for _, row in early_historical_data.iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )

            # 3. 处理当前轨迹（最重要，放最后）
            question_parts.append(f"The following is the current trajectory of user {user}:")
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                question_parts.append(
                    f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                )

            # 拼接最终的问题字符串
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # 生成答案
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # 添加 QA 对到列表
            qa_pairs.append((question, answer))

    return qa_pairs


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode)
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

    # Add arguments for dataset name and use_sim flag
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    parser.add_argument("-use_sim", type=str, choices=["True", "False"], default="False",
                        help="Whether to use similar trajectories (True/False)")

    # Parse the arguments
    args = parser.parse_args()

    # Convert use_sim argument to a boolean
    use_sim = args.use_sim == "True"

    # Your processing code here
    print(f"Processing dataset: {args.dataset_name}")
    print(f"Using similar trajectories: {use_sim}")
    path = f'../datasets/{args.dataset_name}/preprocessed/'

    # Read the data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample.csv')

    kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')

    # Generate the QA pairs, passing use_sim as a parameter
    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args, use_sim=use_sim)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args, use_sim=use_sim)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_nyc_merged_resort3.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # Save the test QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt_200items_nyc_merged_merged_resort3.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()