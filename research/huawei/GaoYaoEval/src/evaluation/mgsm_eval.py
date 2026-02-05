from src.evaluation.base_eval import BaseEval
from src.tools.judger_algorithm import is_number
from src.tools.text_processing import clean_response
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import NUMBER_SYSTEM_PROMPT
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.evaluation.config import LANG_TO_ANSWER_PREFIX
from src.tools.text_processing import get_prediction_numbers

class MGSM(BaseEval):

    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            lang = item.get('language')
            stats["total"][lang] = stats["total"].get(lang, 0) + 1

            gt = str(item['gt']).replace(",", "").strip()
            resp = clean_response(item.get('response', ''))

            # 尝试基于前缀提取数字
            prefix = LANG_TO_ANSWER_PREFIX.get(lang, "Answer")
            # ... (解析逻辑，提取 prediction_numbers)
            if 'response' in item and item['response']:
                response = item['response']
                response = clean_response(response=response)
            else:
                self.badcases.append(item)
                continue
            pred = get_prediction_numbers(prefix, response)

            # 失败则调用 LLM
            if not is_number(pred):
                pred = send_chat_completion(NUMBER_SYSTEM_PROMPT, resp)
            if pred == gt:
                stats["match"][lang] = stats["match"].get(lang, 0) + 1

        return calc_acc_by_language(stats)

