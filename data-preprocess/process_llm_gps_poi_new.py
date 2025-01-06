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

            # Initialize dataframes for different types of historical data
            early_historical_data = pd.DataFrame()  # Early history trajectory

            # Process early historical trajectory data
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

            # Reset index
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # Truncate to latest 500 records if necessary
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500

            # Calculate total records
            num_early_traj = len(early_historical_data)
            num_current_traj = len(user_trajectory_data) - 1
            total_traj_count = num_early_traj + num_current_traj

            # Adjust records to 400 if total exceeds
            if total_traj_count > 320:
                remaining_count = 320

                # Prioritize current trajectory data
                final_current_data = user_trajectory_data.iloc[-min(num_current_traj, remaining_count):].to_dict(
                    'records')
                remaining_count -= len(final_current_data)

                # Add remaining from early historical data
                final_early_data = early_historical_data.iloc[-min(num_early_traj, remaining_count):].to_dict('records')
            else:
                final_early_data = early_historical_data.to_dict('records')
                final_current_data = user_trajectory_data.to_dict('records')

            # Assemble the question parts in the correct order
            question_parts = ["<question>: "]

            # Early history trajectory data
            if final_early_data:
                question_parts.append(f"There is history trajectory data for user {user}:")
                for record in final_early_data:
                    question_parts.append(
                        f"At {record['UTCTimeOffset']}, user {user} visited POI id {record['PoiId']}. "
                        f"The POI token is <POI {record['PoiId']}>, located at coordinates ({record['Latitude']:.6f}, {record['Longitude']:.6f}) with the coordinates token <GPS {record['PoiId']}>. "
                        f"It is a {record['PoiCategoryName']} with category ID {record['PoiCategoryId']}."
                    )

            # Current trajectory data
            question_parts.append(f"The following is the current trajectory of user {user}:")
            for record in final_current_data[:-1]:
                question_parts.append(
                    f"At {record['UTCTimeOffset']}, user {user} visited POI id {record['PoiId']}. "
                    f"The POI token is <POI {record['PoiId']}>, located at coordinates ({record['Latitude']:.6f}, {record['Longitude']:.6f}) with the coordinates token <GPS {record['PoiId']}>. "
                    f"It is a {record['PoiCategoryName']} with category ID {record['PoiCategoryId']}."
                )

            # Compile the final question and answer
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # Bin counting for POI records
            bin_index = (total_traj_count // 100) * 100
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

            # Append QA pair
            qa_pairs.append((question, answer))

    return qa_pairs, poi_count_bins


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode)
    return f


def jload(f, mode="r"):
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
    train_data = pd.read_csv(f'{path}train_sample_v2.csv')
    test_data = pd.read_csv(f'{path}test_sample_v2.csv')
    # kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    # kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')
    kqt1 = jload(f'{path}train_key_top200.json')
    kqt2 = jload(f'{path}test_key_top200.json')

    qa_pairs_train, poi_count_bins_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data,
                                                             args=args, use_sim=use_sim)
    qa_pairs_test, poi_count_bins_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args,
                                                           use_sim=use_sim)

    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_{args.dataset_name}_llm_without_sim_gps_poi_new2.json',
              'w') as json_file:
        json.dump(qa_dict_train, json_file)

    print_summary(poi_count_bins_train, "train")
    print_summary(poi_count_bins_test, "test")

    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_llm_without_sim_gps_poi_new2.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()