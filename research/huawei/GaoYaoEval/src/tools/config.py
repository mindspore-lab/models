import os

# --- Model URLs ---
EXTRACT_ANSWER_MODEL_URL = 'http://10.243.134.97:59080/qwen/v1/chat/completions'
EXTRACT_ANSWER_MODEL_NAME = 'qwen2.5-72b-int4'

OPEN_QUESTION_MODEL_URL = 'http://10.243.134.97:59080/qwen/v1/chat/completions'
OPEN_QUESTION_MODEL_NAME = 'qwen2.5-72b-int4'

COMET_URL = 'http://10.155.106.90:20102/eval_comet'

# --- Sampling Configurations (Reference: Qwen3 Technical Report) ---
SAMPLING_CONFIGS = {
    # 思考模式默认配置
    "thinking_default": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768
    },
    # 创意写作/WritingBench (Thinking Mode)
    "thinking_creative": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768
    },
    # 非思考模式默认配置
    "non_thinking_default": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768
    },
    # 数学推理扩展 (AIME)
    "thinking_math_extended": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 38912
    }
}
