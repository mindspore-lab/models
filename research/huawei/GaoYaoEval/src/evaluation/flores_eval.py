import uuid
from src.evaluation.base_eval import BaseEval
from src.tools.llm_request import send_comet_req
from src.tools.text_processing import clean_response
from src.tools.file_operations import write_jsonl

class Flores(BaseEval):
    def evaluate(self, data_list):
        # 按语言分组
        lang_map = {}
        for item in data_list:
            lang_map.setdefault(item['language'], []).append(item)

        results = {"language": [], "score": []}

        for lang, items in lang_map.items():
            srcs = [x['src_text'] for x in items]
            refs = [x['gt'] for x in items]
            mts = [clean_response(x.get('response', '')) for x in items]

            score = send_comet_req(str(uuid.uuid4()), srcs, refs, mts)

            results["language"].append(lang)
            results["score"].append(score)

            # 保存中间结果
            # write_jsonl(...)

        return dict(zip(results["language"], results["score"]))
