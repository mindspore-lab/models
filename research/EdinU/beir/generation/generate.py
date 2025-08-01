from __future__ import annotations

import logging
import os

from tqdm.autonotebook import trange

from ..util import write_to_json, write_to_tsv

logger = logging.getLogger(__name__)


class PassageExpansion:
    def __init__(self, model, **kwargs):
        self.model = model
        self.corpus_exp = {}

    @staticmethod
    def save(output_dir: str, corpus: dict[str, str], prefix: str):
        os.makedirs(output_dir, exist_ok=True)

        corpus_file = os.path.join(output_dir, prefix + "-corpus.jsonl")

        logger.info(f"Saving expanded passages to {corpus_file}")
        write_to_json(output_file=corpus_file, data=corpus)

    def expand(
        self,
        corpus: dict[str, dict[str, str]],
        output_dir: str,
        top_k: int = 200,
        max_length: int = 350,
        prefix: str = "gen",
        batch_size: int = 32,
        sep: str = " ",
    ):
        logger.info(f"Starting to expand Passages with {top_k} tokens chosen...")
        logger.info(f"Params: top_k = {top_k}")
        logger.info(f"Params: passage max_length = {max_length}")
        logger.info(f"Params: batch size = {batch_size}")

        corpus_ids = list(corpus.keys())
        corpus_list = [corpus[doc_id] for doc_id in corpus_ids]

        for start_idx in trange(0, len(corpus_list), batch_size, desc="pas"):
            expansions = self.model.generate(
                corpus=corpus_list[start_idx : start_idx + batch_size],
                max_length=max_length,
                top_k=top_k,
            )

            for idx in range(len(expansions)):
                doc_id = corpus_ids[start_idx + idx]
                self.corpus_exp[doc_id] = {
                    "title": corpus[doc_id]["title"],
                    "text": corpus[doc_id]["text"] + sep + expansions[idx],
                }

        # Saving finally all the questions
        logger.info(f"Saving {len(self.corpus_exp)} Expanded Passages...")
        self.save(output_dir, self.corpus_exp, prefix)


class QueryGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.qrels = {}
        self.queries = {}

    @staticmethod
    def save(
        output_dir: str,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        prefix: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)

        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")

        logger.info(f"Saving Generated Queries to {query_file}")
        write_to_json(output_file=query_file, data=queries)

        logger.info(f"Saving Generated Qrels to {qrels_file}")
        write_to_tsv(output_file=qrels_file, data=qrels)

    def generate(
        self,
        corpus: dict[str, dict[str, str]],
        output_dir: str,
        top_p: int = 0.95,
        top_k: int = 25,
        max_length: int = 64,
        ques_per_passage: int = 1,
        prefix: str = "gen",
        batch_size: int = 32,
        save: bool = True,
        save_after: int = 100000,
    ):
        logger.info(f"Starting to Generate {ques_per_passage} Questions Per Passage using top-p (nucleus) sampling...")
        logger.info(f"Params: top_p = {top_p}")
        logger.info(f"Params: top_k = {top_k}")
        logger.info(f"Params: max_length = {max_length}")
        logger.info(f"Params: ques_per_passage = {ques_per_passage}")
        logger.info(f"Params: batch size = {batch_size}")

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        for start_idx in trange(0, len(corpus), batch_size, desc="pas"):
            size = len(corpus[start_idx : start_idx + batch_size])
            queries = self.model.generate(
                corpus=corpus[start_idx : start_idx + batch_size],
                ques_per_passage=ques_per_passage,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k,
            )

            assert len(queries) == size * ques_per_passage

            for idx in range(size):
                # Saving generated questions after every "save_after" corpus ids
                if len(self.queries) % save_after == 0 and len(self.queries) >= save_after:
                    logger.info(f"Saving {len(self.queries)} Generated Queries...")
                    self.save(output_dir, self.queries, self.qrels, prefix)

                corpus_id = corpus_ids[start_idx + idx]
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                query_set = set([q.strip() for q in queries[start_id:end_id]])

                for query in query_set:
                    count += 1
                    query_id = "genQ" + str(count)
                    self.queries[query_id] = query
                    self.qrels[query_id] = {corpus_id: 1}

        # Saving finally all the questions
        logger.info(f"Saving {len(self.queries)} Generated Queries...")
        self.save(output_dir, self.queries, self.qrels, prefix)

    def generate_multi_process(
        self,
        corpus: dict[str, dict[str, str]],
        pool: dict[str, object],
        output_dir: str,
        top_p: int = 0.95,
        top_k: int = 25,
        max_length: int = 64,
        ques_per_passage: int = 1,
        prefix: str = "gen",
        batch_size: int = 32,
        chunk_size: int = None,
    ):
        logger.info(f"Starting to Generate {ques_per_passage} Questions Per Passage using top-p (nucleus) sampling...")
        logger.info(f"Params: top_p = {top_p}")
        logger.info(f"Params: top_k = {top_k}")
        logger.info(f"Params: max_length = {max_length}")
        logger.info(f"Params: ques_per_passage = {ques_per_passage}")
        logger.info(f"Params: batch size = {batch_size}")

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        queries = self.model.generate_multi_process(
            corpus=corpus,
            pool=pool,
            ques_per_passage=ques_per_passage,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            chunk_size=chunk_size,
            batch_size=batch_size,
        )

        assert len(queries) == len(corpus) * ques_per_passage

        for idx in range(len(corpus)):
            corpus_id = corpus_ids[idx]
            start_id = idx * ques_per_passage
            end_id = start_id + ques_per_passage
            query_set = set([q.strip() for q in queries[start_id:end_id]])

            for query in query_set:
                count += 1
                query_id = "genQ" + str(count)
                self.queries[query_id] = query
                self.qrels[query_id] = {corpus_id: 1}

        # Saving finally all the questions
        logger.info(f"Saving {len(self.queries)} Generated Queries...")
        self.save(output_dir, self.queries, self.qrels, prefix)
