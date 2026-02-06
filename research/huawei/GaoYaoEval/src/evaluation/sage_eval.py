import re
from src.evaluation.base_eval import BaseEval
from src.tools.judger_algorithm import parse_judge_score
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import MISSING_POINTS_SYSTEM_PROMPT, COMMON_SYSTEM_PROMPT
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.tools.judger_algorithm import identify_question_type, compare_list
from src.tools.text_processing import extracted_choice_question_answer, extracted_true_or_false_question_answer
from src.evaluation.config import CHOICE_TYPE, OPEN_QUESTION_TYPE, TRUE_OR_FALSE_TYPE


class SAGE(BaseEval):

    def eval_true_or_false(self, gt, item, language, language_to_match_num_dict, response):
        answer = self.get_gt_true_or_false(gt)
        response = self.clean_response(response=response)
        prediction_numbers = self.get_response_true_or_false(text=response)
        if prediction_numbers == -1:
            response_prediction = extracted_true_or_false_question_answer(response).strip()
            prediction_numbers = self.get_response_true_or_false(text=response_prediction)
            if prediction_numbers == -1:
                item["prediction"] = response_prediction
                self.badcases.append(item)
                return
        if answer == prediction_numbers:
            match_num = language_to_match_num_dict.get(language, 0)
            language_to_match_num_dict[language] = match_num + 1
        else:
            item["prediction"] = prediction_numbers
            self.not_pass.append(item)

    def eval_choice(self, gt, item, language, language_to_match_num_dict, response):
        gt_prediction = extracted_choice_question_answer(gt).strip()
        response_prediction = extracted_choice_question_answer(response).strip()
        gt_list = re.split(r'[、,，。.\n]', gt_prediction)
        prediction_list = re.split(r'[、,，。.\n]', response_prediction)
        gt_list = [s.strip() for s in gt_list]
        prediction_list = [s.strip() for s in prediction_list]
        result = compare_list(a_list=gt_list, b_list=prediction_list)
        item["gt_prediction"] = gt_prediction
        item["prediction"] = response_prediction
        if result == -1:
            self.badcases.append(item)
        elif result == 1:
            match_num = language_to_match_num_dict.get(language, 0)
            language_to_match_num_dict[language] = match_num + 1
        else:
            self.not_pass.append(item)

    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            q_type = identify_question_type(item['gt'])
            lang = item['language']
            resp = item.get('response', '')
            if lang not in stats["match"]:
                stats["match"][lang] = {}
            if q_type == CHOICE_TYPE:
                self.eval_choice(gt, item, lang, stats["match"], resp)
            elif q_type == TRUE_OR_FALSE_TYPE:
                self.eval_true_or_false(gt, item, lang, stats["match"], resp)
            else: # Open Ended
                gt = item['gt']
                prompt = item['prompt']
                # 构造打分 Prompt
                user_msg = MISSING_POINTS_SYSTEM_PROMPT.replace("[question]", prompt)\
                            .replace("[reference]", gt)\
                            .replace("[model_output]", resp)

                judge_out = send_chat_completion(COMMON_SYSTEM_PROMPT, user_msg)
                score = parse_judge_score(judge_out)

                if score >= 50:
                    stats["match"][lang] = stats["match"].get(lang, 0) + 1

        return calc_acc_by_language(stats)
