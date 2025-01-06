import pandas as pd
import json
import argparse
import io
import sys
import math
from tqdm import tqdm
from collections import defaultdict

def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, use_sim=False):
    # 按照 UserId、pseudo_session_trajectory_id 和 UTCTimeOffsetEpoch 对数据进行排序
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # 用于存储问答对的列表
    qa_pairs = []

    # 用于跟踪不同范围内的 POI 记录数量
    poi_count_bins = defaultdict(int)

    # 遍历每个用户
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # 根据 'pseudo_session_trajectory_id' 遍历每个用户的轨迹
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]

            # 获取当前轨迹的开始时间
            start_time_of_current_traj = user_trajectory_data['UTCTimeOffsetEpoch'].min()

            num_traj = user_trajectory_data.shape[0]

            # 初始化用于存储不同类型历史数据的 DataFrame
            similar_historical_data = pd.DataFrame()  # 相似轨迹
            early_historical_data = pd.DataFrame()    # 早期历史轨迹

            # 当当前轨迹数量小于等于600时，处理早期历史轨迹和相似轨迹
            if num_traj <= 600:
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

                # 判断是否使用相似轨迹
                if len(early_historical_data) > 200:
                    use_sim = False

                # 处理相似轨迹
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

            # 如果当前轨迹超过500条记录，只保留最后500条
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500

            # 开始拼接问题部分
            question_parts = ["<question>: "]

            if num_traj > 200:
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} with the coordinates token <GPS {row['PoiId']}>."
                    )
            else:
                # 1. 处理相似轨迹
                if use_sim and not similar_historical_data.empty:
                    question_parts.append(f"There is historical trajectory data similar to user {user}:")
                    for _, row in similar_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} with the coordinates token <GPS {row['PoiId']}>."
                        )

                # 2. 处理早期历史轨迹
                if not early_historical_data.empty:
                    question_parts.append(f"There is earlier history trajectory data for user {user}:")
                    for _, row in early_historical_data.iterrows():
                        question_parts.append(
                            f"At {row['UTCTimeOffset']}, user {row['UserId']} visited POI id {row['PoiId']} with the coordinates token <GPS {row['PoiId']}>."
                        )

                # 3. 处理当前轨迹
                question_parts.append(f"The following is the current trajectory of user {user}:")
                for i, row in user_trajectory_data.iloc[:-1].iterrows():
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} with the coordinates token <GPS {row['PoiId']}>."
                    )

            # 拼接最终的问题字符串
            question = " ".join(question_parts)
            value = {'NYC': 4981, 'TKY': 7833, 'CA': 9690}[args.dataset_name.upper()]
            next_poi_id = user_trajectory_data.iloc[-1]['PoiId']  # 使用最后一个 POI id

            # 修改最后的问题部分
            question += f" Based on this data, predict the location user {user} will visit next. " \
                        f"At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit a location. " \
                        f"The coordinates token for this location is <GPS {next_poi_id}>. " \
                        f"Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value}."

            # 生成答案（保持不变）
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {next_poi_id}."

            # 汇总签到记录数量
            total_traj_count = (len(similar_historical_data) + len(early_historical_data) + (len(user_trajectory_data) - 1))

            # 根据签到数量确定所属的区间，并将该条记录计入相应区间
            bin_index = (total_traj_count // 100) * 100
            if bin_index <= 800:
                poi_count_bins[f"{bin_index}-{bin_index + 100}"] += 1
            else:
                poi_count_bins["800+"] += 1

            # 添加 QA 对到列表
            qa_pairs.append((question, answer))

    return qa_pairs, poi_count_bins

def _make_r_io_base(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode)
    return f

def jload(f, mode="r"):
    """加载 JSON 文件为字典。"""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def print_summary(poi_count_bins, dataset_name):
    print(f"\n{dataset_name} 数据集的 POI 记录使用情况汇总：")
    for bin_range, count in sorted(poi_count_bins.items()):
        print(f"{bin_range} 个 POI 记录：{count} 条")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="处理数据集名称。")

    # 添加数据集名称的参数
    parser.add_argument("-dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="数据集名称（例如：ca、nyc、tky）")
    parser.add_argument("-use_sim", type=str, choices=["True", "False"], default="False",
                        help="是否使用相似轨迹（True/False）")

    # 解析参数
    args = parser.parse_args()

    # 将 use_sim 参数转换为布尔值
    use_sim = args.use_sim == "True"

    # 处理数据集路径
    path = f'../datasets/{args.dataset_name}/preprocessed/'

    # 读取数据
    train_data = pd.read_csv(f'{path}train_sample_v2.csv')
    test_data = pd.read_csv(f'{path}test_sample_v2.csv')
    # kqt1 = jload(f'{path}train_key_top35_llama2-longlora.json')
    # kqt2 = jload(f'{path}test_key_top35_llama2-longlora.json')
    kqt1 = jload(f'{path}train_key_top200.json')
    kqt2 = jload(f'{path}test_key_top200.json')

    # 生成训练数据的问答对及签到记录统计
    qa_pairs_train, poi_count_bins_train = generate_qa_pairs(train_data, kqt=kqt1, historical_data=train_data, args=args, use_sim=use_sim)
    qa_pairs_test, poi_count_bins_test = generate_qa_pairs(test_data, kqt=kqt2, historical_data=train_data, args=args, use_sim=use_sim)

    # 将训练问答对保存为 JSON 文件
    qa_dict_train = [{"question": q, "answer": a} for q, a in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt_200items_{args.dataset_name}_projector_gps_without_sim_stage2.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # 输出签到记录区间统计结果
    print_summary(poi_count_bins_train, "train")
    print_summary(poi_count_bins_test, "test")

    # 保存测试数据为 TXT 文件
    with open(f'{path}test_qa_pairs_kqt_200items_{args.dataset_name}_projector_gps_without_sim_stage2.txt', 'w') as txt_file:
        for q, a in qa_pairs_test:
            txt_file.write(q + a + '\n')

if __name__ == "__main__":
    main()