import re
from judger_algorithm import is_number
from src.evaluation.config import LANG_TO_ANSWER_PREFIX, NUMBER_TO_CHOICE_M3EXAM
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT, TRUE_OR_FALSE_SYSTEM_PROMPT

# 预编译 Regex
CLEAN_PATTERNS = {
    'unused': re.compile(r'\[unused\d+\]'),
    'think_end': re.compile(r'</think>'),
    'multilingual_choice': [
        r"(?i){}\s*\(?([A-D]|[a-d]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])\)?"
    ]
}

MULTILINGUAL_ANSWER_REGEXES = [
    "Answer\s*:", "Answer\s*:​​​​​​", "答案\s*：", "答案\s*:", "Respuesta\s*:", "Réponse\s*:", # ... (保留原列表所有项)
]

def clean_response(response: str) -> str:
    """清洗模型输出，移除思维链和特殊token"""
    if not response: return ""

    # 移除 thinking process
    if '</think>' in response:
        response = response.split('</think>')[-1]

    # 移除 [unused] tags
    if '[unused16]' in response:
        response = response.split('[unused16]')[-1]
    if '[unused17]' in response:
        response = response.split('[unused17]')[-1]
    if '[unused10]' in response:
        response = response.split('[unused10]')[0]

    return response.strip()

def clean_prompt(instruction: str) -> str:
    """清洗Prompt中的特殊标记"""
    if "[unused9]用户:" in instruction:
        instruction = instruction.split('[unused9]用户:')[-1]
    # ... (保留原逻辑)
    return instruction.strip()

def extract_general_choice(text: str) -> str:
    """提取ABCD选项"""
    match = re.fullmatch(r'\s*\(?\s*([A-D0-4])\s*\)?\s*', text, re.IGNORECASE)
    if match: return match.group(1)

    matchs = re.findall(r'answer:\s*([A-D0-4])', text, re.IGNORECASE)
    if len(matchs) == 1: return matchs[0]

    return text

def normalize_multilingual_choice(text: str) -> str:
    """将多语言选项符号标准化为英文 A-D"""
    return (text.replace("أ", " A").replace("ب", " B").replace("ج", " C").replace("د", " D")
            .replace("অ", " A").replace("ব", " B").replace("ড", " C").replace("ঢ", " D")
            .replace("Ａ", " A").replace("Ｂ", " B").replace("Ｃ", " C").replace("Ｄ", " D")
            .strip())

def normalize_response(response: str) -> str:
    """
    去除响应中的Markdown和LaTeX格式，以防止匹配失败。
    Args:
        response: 需要处理的响应字符串。
    Returns:
        normalized_response: 去除格式后的响应字符串。
    """
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\)", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
        .replace("}", "")
        .replace('based on the provided options', "")
        .replace("Option", "")
        .replace("option", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )


def get_prediction_numbers(prediction_prefix, response):
    """
    从给定的字符串答案中解析出数字。
    Args:
        prediction_prefix: 答案的前缀。
        response: 需要解析的字符串答案。
    Returns:
        prediction_numbers: 提取的数字字符串，如果没有找到合适的前缀或数字，返回空字符串。
    """
    if is_number(response):
        prediction_numbers = response.strip()
    else:
        prediction_numbers = parse_response_answer(response, prediction_prefix)
    return prediction_numbers

def parse_response_answer(answer: str, answer_prefix: str):
    """
    从给定的字符串答案中解析出数字。
    Args:
        answer: 需要解析的字符串答案。
        answer_prefix: 答案的前缀。
    Returns:
        prediction_numbers: 提取的数字字符串，如果没有找到合适的前缀或数字，返回空字符串。
    """
    if answer_prefix.lower() not in answer.lower():
        # 兼容用了其他语言的回答
        for item in LANG_TO_ANSWER_PREFIX.values():
            if item.lower() in answer.lower():
                answer_prefix = item
                break
        if answer_prefix.lower() not in answer.lower():
            return ""
    answer_text = answer.lower().split(answer_prefix.lower())[-1].strip()

    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    prediction_numbers = numbers[-1].rstrip(".") if numbers else ""
    if "." in prediction_numbers:
        prediction_numbers = prediction_numbers.rstrip("0").rstrip(".")
    prediction_numbers = prediction_numbers.replace(",", "")
    return prediction_numbers

def extracted_choice_question_answer(response_text, number_to_choice=NUMBER_TO_CHOICE_M3EXAM):
    # 发送请求
    extracted_answer = send_chat_completion(system_prompt=CHOICE_SYSTEM_PROMPT,
                                                     user_prompt=response_text)

    if is_number(extracted_answer):
        extracted_answer = extracted_answer.strip()
        extracted_answer = number_to_choice.get(extracted_answer)
    return extracted_answer

def extracted_true_or_false_question_answer(response_text):
    # 发送请求
    extracted_answer = send_chat_completion(system_prompt=TRUE_OR_FALSE_SYSTEM_PROMPT,
                                                     user_prompt=response_text)
    return extracted_answer