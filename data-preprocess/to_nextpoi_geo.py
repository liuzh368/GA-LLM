import pandas as pd
import json
import argparse
import io
import sys
import math
from tqdm import tqdm


def calculate_distance_and_bearing(lat1, lon1, lat2, lon2):
    """Calculate the distance in meters and bearing in degrees between two latitude/longitude points."""
    # Earth radius in meters
    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula for distance
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    # Calculate bearing
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360  # Normalize to 0-360 degrees

    return distance, bearing


def describe_relative_position(distance, bearing):
    """Convert distance and bearing into a descriptive direction statement."""
    # Determine the direction description based on the bearing
    if 0 <= bearing < 45 or 315 <= bearing < 360:
        direction = "north-east"
    elif 45 <= bearing < 135:
        direction = "east"
    elif 135 <= bearing < 225:
        direction = "south"
    else:
        direction = "west"

    return f"located {int(distance)} meters to the {direction} (bearing {int(bearing)} degrees) from the previous POI"


def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None):
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
            # if kqt and 'traj_id' in kqt.keys():
            #     top200 = kqt['traj_id']
            #     # Fetch historical data before the start of the current trajectory
            #     if historical_data is not None:
            #         user_historical_data = historical_data[
            #             (str(historical_data['pseudo_session_trajectory_id']) in top200)].tail(300 - num_traj)
            #     else:
            #         user_historical_data = main_data[(str(main_data['pseudo_session_trajectory_id']) in top200)].tail(
            #             300 - num_traj)
            if str(traj_id) in kqt.keys():
                top200 = kqt[str(traj_id)]  # 获取对应的 top200 列表

                if historical_data is not None:
                    # 提取 historical_data 中的伪会话轨迹ID列，并转换为字符串类型
                    historical_traj_ids = historical_data['pseudo_session_trajectory_id'].astype(str)

                    # 使用 isin() 方法来判断哪些轨迹ID在 top200 中
                    matching_historical_traj = historical_traj_ids.isin(top200)

                    # 根据匹配结果筛选出相关的行
                    matched_historical_data = historical_data[matching_historical_traj]

                    # 从匹配的历史数据中取出200 - num_traj条最新的历史数据
                    if matched_historical_data.shape[0] >= (200 - num_traj):
                        user_historical_data = matched_historical_data.tail(200 - num_traj)
                    else:
                        user_historical_data = matched_historical_data
                else:
                    # 提取 main_data 中的伪会话轨迹ID列，并转换为字符串类型
                    main_traj_ids = main_data['pseudo_session_trajectory_id'].astype(str)

                    # 使用 isin() 方法来判断哪些轨迹ID在 top200 中
                    matching_main_traj = main_traj_ids.isin(top200)

                    # 根据匹配结果筛选出相关的行
                    matched_main_data = main_data[matching_main_traj]

                    # 从匹配的历史数据中取出200 - num_traj条最新的历史数据
                    if matched_main_data.shape[0] >= (200 - num_traj):
                        user_historical_data = matched_main_data.tail(200 - num_traj)
                    else:
                        user_historical_data = matched_main_data
            else:
                if historical_data is not None:
                    user_historical_data = historical_data[(historical_data['UserId'] == user) & (
                                historical_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)].tail(
                        600 - num_traj)
                else:
                    user_historical_data = user_data[
                        (user_data['UTCTimeOffsetEpoch'] < start_time_of_current_traj)].tail(600 - num_traj)
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # Create the question based on the current trajectory (excluding the last entry) and historical data
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            prev_lat, prev_lon = None, None  # Variables to store previous latitude and longitude
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                if i == 0:
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}, location at ({row['Latitude']}, {row['Longitude']}).")
                else:
                    # Calculate relative position from the previous POI
                    distance, bearing = calculate_distance_and_bearing(prev_lat, prev_lon, row['Latitude'],
                                                                       row['Longitude'])
                    relative_position = describe_relative_position(distance, bearing)
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}, {relative_position}.")
                prev_lat, prev_lon = row['Latitude'], row['Longitude']  # Update previous coordinates

            # Process historical data with relative positions
            if not user_historical_data.empty:
                question_parts.append("There is also historical data:")
                historical_prev_lat, historical_prev_lon = None, None  # Variables to store previous latitude and longitude of historical data
                for j, row in user_historical_data.iterrows():
                    if historical_prev_lat is None:  # First historical POI, keep the coordinates
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}, location at ({row['Latitude']}, {row['Longitude']}).")
                    else:
                        # Calculate relative position from the previous historical POI
                        historical_distance, historical_bearing = calculate_distance_and_bearing(historical_prev_lat,
                                                                                                 historical_prev_lon,
                                                                                                 row['Latitude'],
                                                                                                 row['Longitude'])
                        historical_relative_position = describe_relative_position(historical_distance,
                                                                                  historical_bearing)
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} which is a {row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}, {historical_relative_position}.")

                    historical_prev_lat, historical_prev_lon = row['Latitude'], row[
                        'Longitude']  # Update historical coordinates

            # Create the final question string
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            question += f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # Form the answer based on the last entry of the current trajectory
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {user_trajectory_data.iloc[-1]['PoiId']}."

            # Append the question-answer pair to the list
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
    train_data = pd.read_csv(f'{path}train_sample_v2.csv')
    test_data = pd.read_csv(f'{path}test_sample_v2.csv')
    kqt1 = jload(f'{path}train_key_top200.json')
    kqt2 = jload(f'{path}test_key_top200.json')
    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args)
    qa_pairs_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_ca_changed_geo.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)


    # Save the test QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt_200items_ca_changed_geo.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')


if __name__ == "__main__":
    main()

