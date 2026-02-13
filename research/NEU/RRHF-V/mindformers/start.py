import argparse
import subprocess
from c2net.context import prepare, upload_output
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description="Start SH Script with Arguments")
    args = parser.parse_args()

    c2net_context = prepare()
    code_path = c2net_context.code_path
    data_path = c2net_context.dataset_path + "/data"
    pretrain_model_path = c2net_context.pretrain_model_path + "/qwen-vl-base-ms"
    
    image_path = data_path + "/media/vann/81749903-f9f0-4935-97d4-3b6d291bb054/qwen-vl-cgq/dataset/llava/train2014"
    text_path = data_path + "/media/vann/81749903-f9f0-4935-97d4-3b6d291bb054/qwen-vl-cgq/dataset/llava/detail_23k.json"
    output_path = c2net_context.output_path
    
    print(f"dataset_path:{data_path}")
    print(f"pretrain_model_path:{pretrain_model_path}")
    
#     os.environ["CODE_PATH"] = code_path
#     os.environ["DATA_PATH"] = data_path
#     os.environ["PRETRAIN_MODEL_PATH"] = pretrain_model_path
#     os.environ["IMAGE_PATH"] = image_path
#     os.environ["TEXT_PATH"] = text_path
#     os.environ["OUTPUT_PATH"] = output_path

    upload_output()

    def load_and_replace_yaml(yaml_file, replacements):
        # Load YAML content
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        # Replace placeholders with key-value pairs
        def replace_placeholders(obj):
            if isinstance(obj, str):
                # If the string contains ${}, replace it with corresponding value from replacements dict
                for key, value in replacements.items():
                    obj = obj.replace(f"${{{key}}}", value)
                    # print(f"key, value:{key, value}")
                return obj
            elif isinstance(obj, dict):
                return {key: replace_placeholders(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [replace_placeholders(item) for item in obj]
            return obj

        return replace_placeholders(config)

    # Define the replacements dictionary
    replacements = {
        "OUTPUT_PATH": output_path,
        "TEXT_PATH": text_path,
        "IMAGE_PATH": image_path
    }

    # Example usage
    yaml_file = 'research/qwenvl/finetune_qwenvl_9.6b_bf16.yaml'  # Replace with your YAML file path
    modified_config = load_and_replace_yaml(yaml_file, replacements)

    # Optionally, write the modified config to a new YAML file
    new_yaml_file = os.path.splitext(yaml_file)[0] + '_new.yaml'

    # 使用新文件名写入修改后的配置
    with open(new_yaml_file, 'w') as file:
        yaml.dump(modified_config, file)
    print(f"YAML 文件已成功写入到 {new_yaml_file}！")
    
    # 1.安装MindSpore
#     install_mindspore = [
#         "pip", "install", "https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.4.10/MindSpore/unified/aarch64/mindspore-2.4.10-cp39-cp39-linux_aarch64.whl",
#         "--trusted-host", "ms-release.obs.cn-north-4.myhuaweicloud.com",
#         "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
#     ]
#     result_mindspore = subprocess.run(install_mindspore, capture_output=True, text=True)
#     # 打印安装MindSpore的输出
#     print("安装MindSpore - 标准输出:")
#     print(result_mindspore.stdout)
#     print("安装MindSpore - 标准错误:")
#     print(result_mindspore.stderr)
    
#     # 2.安装依赖
#     install_requirements = ["pip", "install", "-r", "requirements.txt"]
#     result_requirements = subprocess.run(install_requirements, capture_output=True, text=True)
#     # 打印安装依赖的输出
#     print("安装依赖 - 标准输出:")
#     print(result_requirements.stdout)
#     print("安装依赖 - 标准错误:")
#     print(result_requirements.stderr)


    # 3.执行你的Python脚本
    command = [
        "python", "research/qwenvl/run_qwenvl.py",
        "--config", 'research/qwenvl/finetune_qwenvl_9.6b.yaml',
        "--run_mode", "finetune",
        "--load_checkpoint", pretrain_model_path,
        "--use_parallel", "False",
        "--auto_trans_ckpt", "False",
        "--vocab_file", "qwen.tiktoken",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # 打印执行Python脚本的输出
    print("运行Python脚本 - 标准输出:")
    print(result.stdout)
    print("运行Python脚本 - 标准错误:")
    print(result.stderr)

    
if __name__ == "__main__":
    main()




