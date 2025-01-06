import pandas as pd
import json
import argparse
import io
import sys
import math
from tqdm import tqdm
from collections import defaultdict
import re


def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, use_sim=False):
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])
    qa_pairs = []
    poi_count_bins = defaultdict(int)

    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()
            num_traj = user_trajectory_data.shape[0]

            similar_historical_data = pd.DataFrame()
            early_historical_data = pd.DataFrame()
            if num_traj <= 600:
                if historical_data is not None:
                    early_historical_data = historical_data[
                        (historical_data['UserId'] == user) &
                        (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                        ].tail(600)
                else:
                    early_historical_data = user_data[
                        (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600)

                if len(early_historical_data) > 200:
                    use_sim = False
                if use_sim and str(traj_id) in kqt.keys():
                    top200 = kqt[str(traj_id)]
                    if historical_data is not None:
                        historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)
                        matching_historical_traj = historical_traj_ids.isin(top200)
                        matched_historical_data = historical_data[matching_historical_traj]
                        if matched_historical_data.shape[0] >= 200:
                            similar_historical_data = matched_historical_data.tail(200)
                        else:
                            similar_historical_data = matched_historical_data

            user_trajectory_data.reset_index(drop=True, inplace=True)
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500

            question_parts = ["<question>: "]
            if num_traj > 200:
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )
            else:
                if use_sim and not similar_historical_data.empty:
                    question_parts.append(f"There is historical trajectory data similar to user {user}:")
                    for _, row in similar_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                        )

                if not early_historical_data.empty:
                    question_parts.append(f"There is also earlier, history trajectory data for user {user}:")
                    for _, row in early_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                        )

                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']}, the token for this POI is <POI {row['PoiId']}>, which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}."
                    )

            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            answer_poi_id = user_trajectory_data.iloc[-1]['PoiId']
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {answer_poi_id}."

            # 用正则表达式检查并替换question中的answer中的POI id
            question = re.sub(rf"\bPOI id {answer_poi_id}\b", "POI id <MASK>", question)

            total_traj_count = len(similar_historical_data) + len(early_historical_data) + len(user_trajectory_data) - 1
            bin_index = (total_traj_count // 100) * 100
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

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
    parser = argparse.ArgumentParser(description="Process dataset names.")
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    parser.add_argument("-use_sim", type=str, choices=["True", "False"], default="False",
                        help="Whether to use similar trajectories (True/False)")
    args = parser.parse_args()
    use_sim = args.use_sim == "True"

    path = f'../datasets/{args.dataset_name}/preprocessed/'
    test_data = pd.read_csv(f'{path}test_sample_v2.csv')
    kqt2 = jload(f'{path}test_key_top200.json')
    # kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')
    qa_pairs_test, poi_count_bins_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=test_data, args=args,
                                                           use_sim=use_sim)
    qa_dict_test = [{"question": q, "answer": a} for q, a in qa_pairs_test]

    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_llm_mask.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')

    print_summary(poi_count_bins_test, "test")


if __name__ == "__main__":
    main()