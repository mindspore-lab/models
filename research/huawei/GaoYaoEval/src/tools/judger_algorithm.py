import re
import json
from src.evaluation.config import TRUE_OR_FALSE_TYPE, CHOICE_TYPE, OPEN_QUESTION_TYPE

def is_valid_choice(text):
    return text and text.lower() in ['a', 'b', 'c', 'd']

def is_number(text: str):
    text = text.strip()
    return bool(re.match(r'^-?\d+(\.\d+)?$', text))

def parse_judge_score(judge_output):
    """解析 open_question 的 JSON 评分"""
    try:
        data = json.loads(judge_output)
        points = data.get("points", [])
        similarity = int(data.get("similarity", 0))

        missing = 0.0
        for p in points:
            status = (p.get("status") or "").strip().lower()
            if status == "missing": missing += 1.0
            elif status == "partially_covered": missing += 0.5

        rubric_score = 100 - (15 * missing)
        return int(max(0, min(100, rubric_score)))
    except:
        # 降级处理
        if m := re.search(r'(\d+)', judge_output):
            return int(m.group(1))
        return 0

def get_compare_tag(res, res_swap):
    if res == 'A':
        if res_swap == 'A':
            finnal_res = 'tie'
        else:
            finnal_res = 'win'
    elif res == 'B':
        if res_swap == 'B':
            finnal_res = 'tie'
        else:
            finnal_res = 'lose'
    else:
        if res_swap == 'C':
            finnal_res = 'tie'
        elif res_swap == 'A':
            finnal_res = 'lose'
        else:
            finnal_res = 'win'
    return finnal_res

def match_true_or_false(text):
    '''匹配判断题'''
    true_or_false_full_regex = r'\s*(正确|错误|对|错|True|False|Verdadero|Falso)\s*'
    return re.fullmatch(true_or_false_full_regex, text, re.IGNORECASE) is not None

def match_choice(text):
    '''匹配选择题'''
    choice_full_regex = r'^[A-G]([、，,\s]+[A-G])*'
    choice_start_regex = r'^\s*([:]|[：]|Correct option:)*\s*[A-G][.\s]+'
    if re.fullmatch(choice_full_regex, text):
        return True
    if re.findall(choice_start_regex, text):
        return True
    return False

def identify_question_type(text):
    if match_choice(text):
        return CHOICE_TYPE
    if match_true_or_false(text):
        return TRUE_OR_FALSE_TYPE
    return OPEN_QUESTION_TYPE


def compare_list(a_list, b_list):
    if len(a_list) != len(b_list):
        return 0
    a_list = sorted(a_list)
    b_list = sorted(b_list)
    for a, b in zip(a_list, b_list):
        a = a.strip()
        b = b.strip()
        if len(a) != 1 or len(b) != 1:
            return -1
        if a.lower() != b.lower():
            return 0
    return 1
