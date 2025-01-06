import pandas as pd


def create_gps_mapping(csv_file_path, output_file_path):
    # 读取 CSV 文件
    data = pd.read_csv(csv_file_path)

    # 按 PoiId 分组，计算经纬度的平均值
    unique_pois = data.groupby('PoiId').agg({'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()

    # 将 PoiId 转换为整数类型，去掉小数点
    unique_pois['PoiId'] = unique_pois['PoiId'].astype(int)

    # 重命名 'PoiId' 列为 'GPS_id'
    unique_pois.rename(columns={'PoiId': 'GPS_id'}, inplace=True)

    # 将结果保存为 CSV 文件
    unique_pois.to_csv(output_file_path, index=False)
    print(f"GPS 映射表已保存到 {output_file_path}")


# 使用示例
csv_file_path = '/home/liuzhao/demo/LLM4POI/datasets/tky/preprocessed/train_sample.csv'
output_file_path = '/home/liuzhao/demo/LLM4POI/datasets/tky/preprocessed/tky_gps_mapping.csv'

create_gps_mapping(csv_file_path, output_file_path)