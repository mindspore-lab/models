import mindspore as ms
import mindspore.dataset.vision.py_transforms as py_vision
import numpy as np
import os
import argparse
import importlib
import mindspore.ops as ops

def load_model_module(soc_type):
    soc_type = soc_type.lower()
    if soc_type in ['310', 'ascend310']:
        module_name = "310model"
    else:
        module_name = "model"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"找不到模型文件: {module_name}.py")

def load_model(model_path, use_fp16=False):
    model_module = load_model_module("Ascend310" if use_fp16 else "Ascend910")
    model = model_module.MobileNetV3()
    
    # 强制转换参数类型
    if use_fp16:
        print("[INFO] 强制模型参数为float16")
        for param in model.get_parameters():
            param.set_dtype(ms.float16)
        model.to_float(ms.float16)
    
    # 加载预训练权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    param_dict = ms.load_checkpoint(model_path)
    not_loaded = ms.load_param_into_net(model, param_dict, strict_load=False)
    if not_loaded:
        print(f"[WARNING] 未加载参数: {not_loaded}")
    model.set_train(False)
    return model

def preprocess_image(image_path, image_size=(224, 224), use_fp16=False):
    transform = [
        py_vision.Decode(),
        py_vision.Resize(image_size),
        py_vision.CenterCrop(image_size),
        py_vision.ToTensor(),
        py_vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    try:
        with open(image_path, 'rb') as f:
            img = f.read()
            for t in transform:
                img = t(img)
        dtype = ms.float16 if use_fp16 else ms.float32
        return ms.Tensor(img, dtype=dtype).expand_dims(0)
    except Exception as e:
        raise ValueError(f"预处理失败: {str(e)}")

def predict_batch(model, image_dir, batch_size=8, use_fp16=False):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    stats = {'Normal': 0, 'Tuberculosis': 0}
    results = []
    
    for i in range(0, len(image_files), batch_size):
        batch_tensors = []
        valid_files = []
        for path in image_files[i:i+batch_size]:
            try:
                batch_tensors.append(preprocess_image(path, use_fp16=use_fp16))
                valid_files.append(path)
            except Exception as e:
                print(f"跳过 {os.path.basename(path)}: {str(e)}")
        
        if not batch_tensors:
            continue
            
        batch = ops.concat(batch_tensors, axis=0)
        outputs = model(batch)
        probs = ops.Softmax(axis=1)(outputs).asnumpy()
        preds = np.argmax(probs, axis=1)
        
        for idx, path in enumerate(valid_files):
            cls = 'Tuberculosis' if preds[idx] == 1 else 'Normal'
            conf = probs[idx][preds[idx]]
            results.append((os.path.basename(path), cls, f"{conf:.4f}"))
            stats[cls] += 1
    
    return results, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="肺结核X光分类预测")
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径/目录')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--soc', type=str, default='Ascend910', 
                       choices=['310', 'Ascend310', '910', 'Ascend910'])
    args = parser.parse_args()
    
    use_fp16 = args.soc.lower() in ['310', 'ascend310']
    model = load_model(args.model, use_fp16)
    
    if os.path.isfile(args.input):
        try:
            tensor = preprocess_image(args.input, use_fp16=use_fp16)
            output = model(tensor)
            prob = ops.Softmax(axis=1)(output).asnumpy()[0]
            pred = np.argmax(prob)
            print(f"\n预测结果: {['Normal', 'Tuberculosis'][pred]}, 置信度: {prob[pred]:.4f}")
        except Exception as e:
            print(f"推理失败: {str(e)}")
    elif os.path.isdir(args.input):
        results, stats = predict_batch(model, args.input, args.batch_size, use_fp16)
        print("\n详细结果:")
        for name, cls, conf in results:
            print(f"{name:<25} {cls:<15} {conf:<10}")
        print("\n统计:")
        total = sum(stats.values())
        for cls, count in stats.items():
            print(f"{cls}: {count} ({count/total*100:.1f}%)")
        print(f"总计: {total}")
    else:
        print("错误: 输入路径无效")
