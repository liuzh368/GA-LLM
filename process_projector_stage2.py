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

import pandas as pd
import json
import argparse
import io
import sys
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

            # Initialize three separate DataFrames for different types of historical data
            similar_historical_data = pd.DataFrame()  # similar trajectory
            early_historical_data = pd.DataFrame()    # earlier historical trajectory

            # Skip early and similar trajectories if current trajectory has more than 200 records
            if num_traj <= 600:
                # Process earlier historical trajectory
                if historical_data is not None:
                    early_historical_data = historical_data[
                        (historical_data['UserId'] == user) &
                        (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600)
                else:
                    early_historical_data = user_data[
                        (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600)

                # If earlier history has more than 200 records, skip similar trajectories
                if len(early_historical_data) > 200:
                    use_sim = False

                # Process similar trajectories (if use_sim is True and earlier history is less than 200)
                if use_sim and str(traj_id) in kqt.keys():
                    top200 = kqt[str(traj_id)]
                    if historical_data is not None:
                        historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)
                        matching_historical_traj = historical_traj_ids.isin(top200)
                        matched_historical_data = historical_data[matching_historical_traj]
                        similar_historical_data = matched_historical_data.tail(200)

            # Reset index for current trajectory
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # If current trajectory has more than 500 records, keep only the last 500
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500

            # Construct question in order of importance: 1. Similar trajectory > 2. Earlier history > 3. Current trajectory
            question_parts = ["<question>: "]

            # When current trajectory has more than 200 records, only include current trajectory
            if num_traj > 200:
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for _, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited a POI, the token for this POI is <POI {row['PoiId']}>."
                    )
            else:
                # 1. Process similar trajectory (least important, listed first)
                num_similar_traj = 0
                if use_sim and not similar_historical_data.empty:
                    question_parts.append(f"There is historical trajectory data similar to user {user}:")
                    for _, row in similar_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited a POI, the token for this POI is <POI {row['PoiId']}>."
                        )
                    num_similar_traj = len(similar_historical_data)

                # 2. Process earlier historical trajectory (moderately important)
                num_early_traj = 0
                if not early_historical_data.empty:
                    question_parts.append(f"There is earlier history trajectory data for user {user}:")
                    for _, row in early_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited a POI, the token for this POI is <POI {row['PoiId']}>."
                        )
                    num_early_traj = len(early_historical_data)

                # 3. Process current trajectory (most important, listed last)
                num_current_traj = len(user_trajectory_data) - 1
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for _, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited a POI, the token for this POI is <POI {row['PoiId']}>."
                    )

            # Finalize the question string
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Based on this data, predict the location user {user} will visit next. " \
                        f"At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit a POI. " \
                        f"The token for this POI is <POI {user_trajectory_data.iloc[-1]['PoiId']}>. Which POI id will user {user} visit? " \
                        f"Note that POI id is an integer in the range from 0 to {value}."

            # Generate the answer
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # Track the count of trajectory records
            total_traj_count = num_similar_traj + num_early_traj + num_current_traj
            bin_index = (total_traj_count // 100) * 100
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

            # Append QA pair to the list
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
    with open(f'{path}train_qa_pairs_kqt_200items_{args.dataset_name}_projector_without_sim_stage2.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # 输出签到记录区间统计结果
    print_summary(poi_count_bins_train, "train")
    print_summary(poi_count_bins_test, "test")

    # 保存测试数据为 TXT 文件
    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_projector_without_sim_stage2.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')

if __name__ == "__main__":
    main()