# --- Model URLs ---
LLM_URL = 'http://{ip}:{port}/deepseek/v1/chat/completions'
LLM_NAME = 'deepseek-v3.1'

COMET_URL = 'http://{ip}:{port}/eval_comet'

# --- Sampling Configurations ---
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
