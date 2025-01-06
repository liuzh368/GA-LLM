import pandas as pd
import json
import argparse
import io
import sys
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

            # Initialize DataFrames for different types of historical data
            similar_historical_data = pd.DataFrame()  # similar trajectory
            early_historical_data = pd.DataFrame()    # earlier history trajectory

            # Skip similar and early history if trajectory exceeds 200
            if num_traj <= 600:
                if historical_data is not None:
                    early_historical_data = historical_data[
                        (historical_data['UserId'] == user) &
                        (historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)
                    ].tail(600)

                if len(early_historical_data) > 200:
                    use_sim = False

                if use_sim and str(traj_id) in kqt.keys():
                    top200 = kqt[str(traj_id)]
                    historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)
                    matched_historical_data = historical_data[historical_traj_ids.isin(top200)]
                    similar_historical_data = matched_historical_data.tail(200)

            # Reset index
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # Truncate current trajectory data to last 500 records if exceeding limit
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500

            # Start constructing the question parts
            question_parts = ["<question>: "]

            # Construct prompt based on different trajectory data types
            if num_traj > 200:
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for _, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} "
                        f"(token is <POI {row['PoiId']}>, coordinates token is <GPS {row['PoiId']}>, "
                        f"a {row['PoiCategoryName']} with category id {row['PoiCategoryId']})."
                    )
            else:
                # 1. Similar trajectory data
                if use_sim and not similar_historical_data.empty:
                    question_parts.append(f"There is historical trajectory data similar to user {user}:")
                    for _, row in similar_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} "
                            f"(token is <POI {row['PoiId']}>, coordinates token is <GPS {row['PoiId']}>, "
                            f"a {row['PoiCategoryName']} with category id {row['PoiCategoryId']})."
                        )

                # 2. Early historical trajectory data
                if not early_historical_data.empty:
                    question_parts.append(f"There is earlier history trajectory data for user {user}:")
                    for _, row in early_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} "
                            f"(token is <POI {row['PoiId']}>, coordinates token is <GPS {row['PoiId']}>, "
                            f"a {row['PoiCategoryName']} with category id {row['PoiCategoryId']})."
                        )

                # 3. Current trajectory data
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for _, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} "
                        f"(token is <POI {row['PoiId']}>, coordinates token is <GPS {row['PoiId']}>, "
                        f"a {row['PoiCategoryName']} with category id {row['PoiCategoryId']})."
                    )

            # Final question structure
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            next_poi_id = user_trajectory_data.iloc[-1]['PoiId']
            question = " ".join(question_parts)
            question += (
                f" Based on this data, predict the location user {user} will visit next. "
                f"At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit a POI. "
                f"The token for this POI is <POI {next_poi_id}>. Which POI id will user {user} visit? "
                f"Note that POI id is an integer in the range from 0 to {value}."
            )

            # Generate the answer
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {next_poi_id}."

            # Record count for bin tracking
            total_traj_count = len(similar_historical_data) + len(early_historical_data) + (len(user_trajectory_data) - 1)
            bin_index = (total_traj_count // 100) * 100
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

            # Add QA pair to the list
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
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    parser.add_argument("-use_sim", type=str, choices=["True", "False"], default="False",
                        help="Whether to use similar trajectories (True/False)")
    args = parser.parse_args()

    use_sim = args.use_sim == "True"

    # Define path
    path = f'../datasets/{args.dataset_name}/preprocessed/'

    # Read data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample.csv')
    kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')

    # Generate QA pairs
    qa_pairs_train, poi_count_bins_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args, use_sim=use_sim)
    qa_pairs_test, poi_count_bins_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args, use_sim=use_sim)

    # Save train QA pairs to JSON
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_{args.dataset_name}_projector_gps_poi_without_sim_stage2.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # Print summary
    print_summary(poi_count_bins_train, "train")
    print_summary(poi_count_bins_test, "test")

    # Save test data to TXT
    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_projector_gps_poi_without_sim_stage2.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')

if __name__ == "__main__":
    main()