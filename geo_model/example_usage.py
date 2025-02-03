# example_usage.py

from F_geo_encoder import GPSPositionalEncoder

if __name__ == '__main__':
    # 初始化模型
    encoder = GPSPositionalEncoder()

    # 输入经纬度
    latitude = 40.64508
    longitude = -74.78452

    # 生成位置编码
    pex = encoder(latitude, longitude)

    # 输出结果
    print("输出向量形状:", pex.shape)
    print("输出向量:", pex)