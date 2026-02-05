import re
from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import (
    clean_response, extract_general_choice, normalize_multilingual_choice,
    MULTILINGUAL_ANSWER_REGEXES, normalize_response
)
from src.tools.judger_algorithm import is_valid_choice
from src.tools.metrics_and_report_operator import calc_acc_by_language

class MMMLU(BaseEval):
    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            lang = item.get('language', 'unknown')
            stats["total"][lang] = stats["total"].get(lang, 0) + 1

            # GT
            gt = normalize_multilingual_choice(extract_general_choice(item['gt']))

            # Response
            resp_text = clean_response(item.get('response', ''))
            # 去除 Markdown
            clean_resp = resp_text.replace("**", "").replace("$\\boxed{", "")

            pred = extract_general_choice(clean_resp)
            if len(pred) == 1:
                pred = normalize_multilingual_choice(pred)
            else:
                # 复杂 Regex 匹配
                response_text = normalize_response(pred)
                for regex_tmpl in MULTILINGUAL_ANSWER_REGEXES:
                    regex = MMMLU.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(regex_tmpl)
                    matchs = re.finditer(regex, response_text)
                    for match in matchs:
                        # 如果有多个匹配，最后一个会覆盖前面的（相当于取最后一个）
                        extracted_answer = self.normalize_extracted_answer(match.group(1))

            # 判分逻辑同上
            if is_valid_choice(pred) and pred == gt:
                stats["match"][lang] = stats["match"].get(lang, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        return calc_acc_by_language(stats)
