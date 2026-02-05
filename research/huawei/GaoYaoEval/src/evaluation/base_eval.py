import os
from abc import ABC, abstractmethod
from src.tools.file_operations import read_jsonl, write_jsonl, write_excel
from src.log.logging_config import logger

class BaseEval(ABC):
    def __init__(self):
        self.badcases = []
        self.not_pass = []

    def run(self, input_path, output_path, badcase_path, not_pass_path):
        """标准评测流程"""
        logger.info(f"Start Eval: {self.__class__.__name__}")
        data_list = read_jsonl(input_path)

        # 处理逻辑
        metrics = self.evaluate(data_list)

        # 结果输出
        write_excel(output_path, metrics)
        if self.badcases:
            write_jsonl(badcase_path, self.badcases)
        if self.not_pass:
            write_jsonl(not_pass_path, self.not_pass)

        return metrics

    @abstractmethod
    def evaluate(self, data_list):
        """子类需实现具体的评测循环"""
        pass
