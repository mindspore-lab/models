from src.evaluation.base_eval import BaseEval
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import WIN_RATE_SYSYTEM_PROMPT
from src.tools.text_processing import clean_response
from src.tools.metrics_and_report_operator import calculate_win_rate
from src.tools.judger_algorithm import get_compare_tag

class AlpacaEval(BaseEval):
    def evaluate(self, data_list):
        lang_stats = {} # {lang: {win:0, tie:0, lose:0}}

        for item in data_list:
            lang = item.get('language')
            # 构造 prompt
            prompt = item['prompt']
            answer = item['gt']
            resp = clean_response(item['response'])

            # 正序比较
            user_prompt_1 = f"[Question]{prompt}[Assist 1]{resp}[Assist 2]{answer}"
            res_1 = send_chat_completion(WIN_RATE_SYSYTEM_PROMPT, user_prompt_1)

            # 逆序比较 (Swap)
            user_prompt_2 = f"[Question]{prompt}[Assist 1]{answer}[Assist 2]{resp}"
            res_2 = send_chat_completion(WIN_RATE_SYSYTEM_PROMPT, user_prompt_2)

            # 判定胜负 (A/B/C)
            final_winner = get_compare_tag(res=res_1, res_swap=res_2)
            lang_stats[lang] = lang_stats.get(lang, {}).get(final_winner, 0) + 1

        calculated_win_rate_dict = {lang: calculate_win_rate(stats)  for lang, stats in lang_stats.items()}

        # 计算胜率
        return calculated_win_rate_dict
