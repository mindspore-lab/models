from src.evaluation.base_eval import BaseEval
from src.tools.metrics_and_report_operator import export_country_acc_result
from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import clean_response, extract_general_choice
from src.tools.llm_request import send_chat_completion
from src.tools.judger_algorithm import is_valid_choice, is_number
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT


class SuperBlend(BaseEval):
    def evaluate(self, data_list):
        all_dict = {} # {lang: {country: count}}
        match_dict = {}

        for item in data_list:
            lang = item['language']
            country = item['country']

            # 初始化字典结构
            if lang not in all_dict: all_dict[lang] = {}
            all_dict[lang][country] = all_dict[lang].get(country, 0) + 1

            # 判分逻辑 (同 MCQ)
            is_correct = ...
            # GT处理
            gt = extract_general_choice(str(item['gt']))
            if is_number(gt):
                gt = self.number_to_choice.get(gt, gt)  # 需引入 mapping

            # Response处理
            resp = clean_response(item.get('response', ''))
            pred = extract_general_choice(resp)

            # 提取失败则调用LLM
            if not is_valid_choice(pred):
                pred = send_chat_completion(CHOICE_SYSTEM_PROMPT, resp)
                pred = extract_general_choice(pred)

            item['prediction'] = pred

            if is_valid_choice(pred) and pred.lower() == gt.lower():
                if lang not in match_dict: match_dict[lang] = {}
                match_dict[lang][country] = match_dict[lang].get(country, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        # 生成特定的 Excel 报告
        return export_country_acc_result(all_dict, match_dict)