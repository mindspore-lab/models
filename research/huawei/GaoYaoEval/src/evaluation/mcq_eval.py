from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import clean_response, extract_general_choice
from src.tools.llm_request import send_chat_completion
from src.tools.judger_algorithm import is_valid_choice, is_number
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT
from src.evaluation.config import NUMBER_TO_CHOICE_1_BASE

class MultiChoiceExam(BaseEval):
    """
    多选题评测类
    """
    number_to_choice = NUMBER_TO_CHOICE_1_BASE
    def evaluate(self, data_list):
        lang_stats = {"total": {}, "match": {}}

        for item in data_list:
            lang = item.get('language')
            if not lang:
                continue

            lang_stats["total"][lang] = lang_stats["total"].get(lang, 0) + 1

            # GT处理
            gt = extract_general_choice(str(item['gt']))
            if is_number(gt):
                gt = self.number_to_choice.get(gt, gt) # 需引入 mapping

            # Response处理
            resp = clean_response(item.get('response', ''))
            pred = extract_general_choice(resp)

            # 提取失败则调用LLM
            if not is_valid_choice(pred):
                pred = send_chat_completion(CHOICE_SYSTEM_PROMPT, resp)
                pred = extract_general_choice(pred)

            item['prediction'] = pred

            if is_valid_choice(pred) and pred.lower() == gt.lower():
                lang_stats["match"][lang] = lang_stats["match"].get(lang, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        return calc_acc_by_language(lang_stats)
