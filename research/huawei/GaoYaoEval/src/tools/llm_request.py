import requests
import logging
from src.tools.config import EXTRACT_ANSWER_MODEL_NAME, EXTRACT_ANSWER_MODEL_URL, COMET_URL

logger = logging.getLogger('eval_logger')

def send_chat_completion(system_prompt, user_prompt,
                         model_name=EXTRACT_ANSWER_MODEL_NAME,
                         model_url=EXTRACT_ANSWER_MODEL_URL,
                         params=None):
    """Generic Chat Completion Request"""

    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Default generic params, can be overridden by 'params' argument
            "temperature": 0.1
        }
        if params:
            data.update(params)

        res = requests.post(model_url, headers=headers, json=data, timeout=120)
        if res.status_code != 200:
            logger.error(f"Request failed: {res.status_code}: {res.text}")
            return ' '
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM Request failed: {e}")
        return ' '

def send_comet_req(request_id, src_lines, ref_lines, mt_lines):
    """COMET Score Request"""
    try:
        headers = {'Content-Type': 'application/json'}
        param = {"request_id": request_id, "src_lines": src_lines, "ref_lines": ref_lines, "mt_lines": mt_lines}
        res = requests.post(COMET_URL, headers=headers, json=param, timeout=120)
        return res.json()["data"].get("comet_score")
    except Exception as e:
        logger.error(f"Comet Request failed: {e}")
        return ""