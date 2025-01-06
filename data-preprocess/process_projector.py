import pandas as pd
import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm
from collections import defaultdict

def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, use_sim=False):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    qa_pairs = []

    # Dictionary to track the number of POI records in different ranges
    poi_count_bins = defaultdict(int)

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

            # 当当前轨迹数量大于200时，跳过早期历史轨迹和相似轨迹
            if num_traj <= 600:
                # 处理早期历史轨迹
                if historical_data is not None:
                    early_historical_data = historical_data[
                        (historical_data['UserId'] == user) &
                        (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600) # 原论文是600，但是我这里添加了部分token，导致600个签到记录总token数会超过32768
                else:
                    early_historical_data = user_data[
                        (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600) # 原论文是600，但是我这里添加了部分token，导致600个签到记录总token数会超过32768

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

            # 如果当前轨迹签到数据超过500条，只保留最近的500条签到记录
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500  # 只保留最近的500个签到数据

            # 开始拼接问题部分，凸显重要性顺序：1. 相似轨迹 > 2. 早期历史轨迹 > 3. 当前轨迹（最重要的放最后）
            # 在 question_parts 开头添加 "<question>: "，确保无论是否使用相似轨迹，问题开头都有 "<question>:"
            question_parts = ["<question>: "]

            # 当当前轨迹数量大于200时，只拼接当前轨迹
            if num_traj > 200:
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )
            else:
                # 1. 处理相似轨迹（最不重要，放最前）
                num_similar_traj = 0
                if use_sim and not similar_historical_data.empty:
                    question_parts.append(f"There is historical trajectory data similar to user {user}:")
                    for _, row in similar_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                        )
                    num_similar_traj = len(similar_historical_data)

                # 2. 处理早期历史轨迹（次重要）
                num_early_traj = 0
                if not early_historical_data.empty:
                    question_parts.append(f"There is earlier history trajectory data for user {user}:")
                    for _, row in early_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                        )
                    num_early_traj = len(early_historical_data)

                # 3. 处理当前轨迹（最重要，放最后）
                num_current_traj = len(user_trajectory_data) - 1  # 当前轨迹数量
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )

            # 拼接最终的问题字符串
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Based on this data, predict the location user {user} will visit next. At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit a POI. The token for this POI is <POI {user_trajectory_data.iloc[-1]['PoiId']}>. Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # 生成答案
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # 汇总签到记录数量
            total_traj_count = num_similar_traj + num_early_traj + num_current_traj

            # 根据签到数量确定所属的区间，并将该条记录计入相应区间
            bin_index = (total_traj_count // 100) * 100  # 例如将325条签到计入300-400区间
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

            # 添加 QA 对到列表
            qa_pairs.append((question, answer))

    return qa_pairs, poi_count_bins

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

def print_summary(poi_count_bins, dataset_name):
    print(f"\nSummary of POI records usage for {dataset_name} dataset:")
    for bin_range, count in sorted(poi_count_bins.items()):
        print(f"{bin_range} POI records: {count} items")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process dataset names.")

    # Add an argument for the dataset name
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    parser.add_argument("-use_sim", type=str, choices=["True", "False"], default="False",
                        help="Whether to use similar trajectories (True/False)")

    # Parse the arguments
    args = parser.parse_args()

    # Convert use_sim argument to a boolean
    use_sim = args.use_sim == "True"

    # 处理数据集路径
    path = f'../datasets/{args.dataset_name}/preprocessed/'

    # 读取数据
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample.csv')
    kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')
    # kqt1 = jload(f'{path}train_key_top200.json')
    # kqt2 = jload(f'{path}test_key_top200.json')

    # 生成训练数据的问答对及签到记录统计
    qa_pairs_train, poi_count_bins_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args, use_sim=use_sim)
    qa_pairs_test, poi_count_bins_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args, use_sim=use_sim)

    # 将训练问答对保存为 JSON 文件
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_{args.dataset_name}_projector_without_sim_new.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # 输出签到记录区间统计结果
    print_summary(poi_count_bins_train, "train")
    print_summary(poi_count_bins_test, "test")

    # 保存测试数据为 TXT 文件
    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_projector_without_sim_new.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')

if __name__ == "__main__":
    main()