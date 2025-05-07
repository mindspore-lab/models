import json

# 读取文件内容
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 写入新的JSON文件
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 将b格式的数据转换为a格式
def convert_b_to_a(b_data):
    a_data = []
    
    for entry in b_data:
        image_id = entry['image_id'].split('/')[-1]  # 提取图片文件名
        question = entry['question']
        responses = entry['response']
        scores = entry['scores']
        
        conversations = []
        
        # 将问题和每个回答配对为人类和GPT的对话
        for response in responses:
            conversations.append([
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": response}
            ])
        
        # 生成a格式的条目，并将scores直接加到外层
        a_data.append({
            "id": image_id.split('.')[0],  # 假设id是去掉扩展名后的文件名
            "image": image_id,
            "scores": scores,  # 将scores作为并列的字段
            "conversations": conversations
        })
    
    return a_data

# 主函数
def main():
    # 读取b文件
    b_file = '/Users/chen/Code/幻觉/Multimodal-Hallucination/generate_data_v2/data/llama3-8b-instruct-fp16/make_up_train_data_desc/easy-5k.json'  # 假设b文件名为'b.json'
    b_data = read_json(b_file)
    
    # 转换格式
    a_data = convert_b_to_a(b_data)
    
    # 写入a文件
    a_file = '/Users/chen/Code/幻觉/mindformers/assert/rrhf-v-5k.json'  # 假设目标文件名为'a.json'
    write_json(a_data, a_file)
    print(f"转换完成，结果已保存至 {a_file}")

# 运行脚本
if __name__ == "__main__":
    main()
