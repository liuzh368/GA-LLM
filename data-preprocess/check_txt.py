file_path = "/home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/test_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi.txt"

# 打开文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    line_count = sum(1 for line in file)

print(f"The file contains {line_count} lines.")