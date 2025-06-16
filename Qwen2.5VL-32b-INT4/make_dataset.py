import os
import json

def generate_training_data(input_folder_path, output_json_path):
    """
    遍历文件夹中的图片和同名txt文件，生成JSON格式的训练数据。

    Args:
        input_folder_path (str): 包含图片和txt文件的文件夹路径。
        output_json_path (str): 生成的JSON文件的保存路径。
    """
    training_data_list = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff') # 支持的图片格式

    # 确保输入文件夹存在
    if not os.path.isdir(input_folder_path):
        print(f"错误：文件夹 '{input_folder_path}' 不存在。")
        return

    for filename in os.listdir(input_folder_path):
        # 检查是否是图片文件
        if filename.lower().endswith(image_extensions):
            image_name_without_ext, _ = os.path.splitext(filename)
            txt_filename = image_name_without_ext + ".txt"
            txt_filepath = os.path.join(input_folder_path, txt_filename)

            # 检查同名的txt文件是否存在
            if os.path.exists(txt_filepath):
                try:
                    with open(txt_filepath, 'r', encoding='utf-8') as f:
                        description = f.read().strip()
                except Exception as e:
                    print(f"读取文件 '{txt_filepath}' 时出错: {e}")
                    continue # 跳过这个文件

                json_image_path = os.path.join(input_folder_path, filename).replace("\\", "/") # 保证路径分隔符为 /

                conversation_entry = {
                    "image": json_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n这个人醉酒了吗？"
                        },
                        {
                            "from": "gpt",
                            "value": f"{description}\n醉酒了"
                        }
                    ]
                }
                training_data_list.append(conversation_entry)
            else:
                print(f"警告：图片 '{filename}' 对应的文本文件 '{txt_filename}' 未找到。")

    # 将结果写入JSON文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(training_data_list, outfile, ensure_ascii=False, indent=4)
        print(f"训练数据已成功生成到 '{output_json_path}'")
    except Exception as e:
        print(f"写入JSON文件 '{output_json_path}' 时出错: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    folder_path = "./醉酒"
    # folder_path = "./pic"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "training_output.json")

    # 3. 调用函数生成数据
    generate_training_data(folder_path, output_file)

