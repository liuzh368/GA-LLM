from transformers import AutoTokenizer

# 模型路径
model_path = "/home/liuzhao/demo/LLM4POI/model"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试句子
sentence = "At 2012-05-04 08:27:53, user 1 visited POI 3313 with token <POI 3313>. The location is at <GPS 3313>."

# 编码句子并获取 token 数量
tokens = tokenizer.tokenize(sentence)
num_tokens = len(tokens)

print(f"句子: {sentence}")
print(f"Tokens: {tokens}")
print(f"Token 数量: {num_tokens}")