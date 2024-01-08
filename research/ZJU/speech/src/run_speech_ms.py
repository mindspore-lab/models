# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023, Shumin Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, DistilBert)."""

import sys
sys.path.append("../")

import argparse
import glob
import logging
import os
import random

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.dataset import (
    GeneratorDataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler
)
from tqdm import tqdm, trange
from mindnlp.transformers.models.bert import BertConfig,BertTokenizer




from data_utils_ms import convert_examples_to_features, processors,InputFeatures
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from speech_ms import SPEECH
# from speech_distilbert import SPEECH_DistilBert
from mindspore import context
context.set_context(device_target='CPU')
logger = logging.getLogger(__name__)



MODEL_CLASSES = {
    "speech_bert": (BertConfig, SPEECH, BertTokenizer),
}
class MsDataset:
    def __init__(self, *data):
        assert len(data) > 0, "At least one tensor should be provided."
        assert all(isinstance(d, (np.ndarray, list, tuple, ms.Tensor)) for d in data), \
            "All inputs should be either np.ndarray, list, tuple or Tensor."
        assert all(len(d) == len(data[0]) for d in data), "All tensors should have the same length."

        self.data = [self._convert_to_tensor(d) for d in data]
        self.num_samples = len(data[0])

    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return ms.Tensor(data)
        elif isinstance(data, (list, tuple)):
            return ms.Tensor(np.array(data))
        elif isinstance(data, ms.Tensor):
            return data

    def __getitem__(self, index):
        index=int(index)
        return tuple(d[index] for d in self.data)

    def __len__(self):
        return self.num_samples
        

def calculate_scores(preds, labels, dimE, task_type):
    """
        task_type:  "token"; "sent"; "sent_onto", "doc_all"; "doc_temporal"; "doc_causal"; "doc_sub"; "doc_corref"; "doc_joint" 
    """
    if task_type == "token":
        positive_labels = list(range(0, dimE - 1)) 
    elif task_type == "sent": 
        positive_labels = list(range(1, dimE)) 
    elif task_type == "sent_onto": 
        positive_labels = list(range(2, dimE)) 
    elif "doc" in task_type: 
        positive_labels = list(range(1, dimE)) 

    p_micro = precision_score(y_true=labels, y_pred=preds, labels=positive_labels, average='micro') 
    r_micro = recall_score(y_true=labels, y_pred=preds, labels=positive_labels, average='micro')
    f1_micro = f1_score(y_true=labels, y_pred=preds, labels=positive_labels, average='micro')
    
    return p_micro, r_micro, f1_micro


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = 1
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #change
    train_sampler = RandomSampler() if args.local_rank == -1 else DistributedSampler()
    col_names = ['example_id', 'list_input_ids', 'list_input_mask', 'list_segment_ids', 'list_sent_label', 'list_token_labels', 'mat_rel_label', 'mention_size', 'pad_token_label_id']
    train_dataloader = GeneratorDataset(train_dataset,column_names=col_names, sampler=train_sampler)
    train_dataloader = train_dataloader.batch(batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for p in model.get_parameters() if not any(nd in p.name for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in model.get_parameters() if any(nd in p.name for nd in no_decay)], 
            "weight_decay": 0.0
        },
    ]
    scheduler = nn.WarmUpLR(learning_rate=args.learning_rate,warmup_steps=args.warmup_steps)
    optimizer = nn.Adam(optimizer_grouped_parameters, learning_rate=scheduler, eps=args.adam_epsilon)
   

    """Training!"""
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     args.train_batch_size
    #     * args.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    # )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_valid_f1_micro = 0.0
    best_steps = 0
    # model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.set_train()
            # batch = tuple(t for t in batch)
            inputs = {
                "example_id": batch[0],
                "task_name": args.task_name,
                "doc_ere_task_type": args.ere_task_type,  
                "max_mention_size": ms.tensor([args.max_mention_size], dtype=ms.int64),  
                "mention_size": batch[1],
                "pad_token_label_id": batch[2], 
                "input_ids": batch[3].view(-1, args.max_seq_length),
                "attention_mask": batch[4].view(-1, args.max_seq_length),
                "token_type_ids": batch[5].view(-1, args.max_seq_length)
                if args.model_type not in ["xlmroberta"] or (args.model_type.startswith("xlmroberta") is False)
                else None,  # XLM don't use segment_ids
                "labels4token": batch[6],
                "labels4sent": batch[7],
                "mat_rel_label": batch[8],
            }
 
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                None
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                ms.ops.clip_by_norm(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                # model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        # central_task = "doc" # token, sent, doc, doc_temporal_joint, doc_causal_joint, doc_sub_joint, doc_corref_joint
                        central_task = args.central_task
                        if args.ere_task_type == "doc_joint":
                            central_task = "doc_temporal_joint"
                        # central_task = "sent"
                        # central_task = "token"
                        key_metric = "eval_f1_micro_" + central_task

                        if results[key_metric] > best_valid_f1_micro:
                            best_valid_f1_micro = results[key_metric]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, model, tokenizer, test=True)
                                logger.info(
                                    "test f1_micro_" + central_task + ": %s, loss: %s, global steps: %s", 
                                    str(results_test[key_metric]), 
                                    str(results_test["eval_loss"]),
                                    str(global_step),
                                )
                    
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # # add
                    # basic = model.module if hasattr(model, "module") else model
                    # bert_to_save = (basic.bert.module if hasattr(basic.bert, "module") else basic.bert)
                    # tmp = os.path.join(output_dir, args.model_type)
                    # if not os.path.exists(tmp):
                    #     os.makedirs(tmp)
                    # bert_to_save.save_pretrained(tmp)
                    # # add end
                    tokenizer.save_vocabulary(output_dir)
                    ms.save_checkpoint(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to [%s]", output_dir)
                    ms.save_checkpoint(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # ms.save_checkpoint(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, prefix="", test=False, infer=True):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler()
        eval_dataloader = GeneratorDataset(eval_dataset, sampler=eval_sampler)
        eval_dataloader = eval_dataloader.batch(batch_size=args.eval_batch_size)

        """Evaluation!"""
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds_token = None
        preds_sent = None
        preds_doc = None
        preds_doc_temp = None
        preds_doc_causal = None
        preds_doc_sub = None
        preds_doc_corref = None
        out_label4token_ids = None
        out_label4sent_ids = None
        out_label4doc_ids = None
        out_label4doc_temp_ids = None 
        out_label4doc_causal_ids = None  
        out_label4doc_sub_ids = None  
        out_label4doc_corref_ids = None   
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t for t in batch)

            inputs = {
                "example_id": batch[0],
                "task_name": args.task_name,
                "doc_ere_task_type": args.ere_task_type, 
                "max_mention_size": ms.tensor([args.max_mention_size], dtype=ms.int64),  
                "mention_size": batch[1],
                "pad_token_label_id": batch[2], 
                "input_ids": batch[3].view(-1, args.max_seq_length),
                "attention_mask": batch[4].view(-1, args.max_seq_length),
                "token_type_ids": batch[5].view(-1, args.max_seq_length)
                if args.model_type not in ["xlmroberta"] or (args.model_type.startswith("xlmroberta") is False)
                else None,  # XLM don't use segment_ids
                "labels4token": batch[6],
                "labels4sent": batch[7],
                "mat_rel_label": batch[8],
            }
            outputs = model(**inputs)
            if "_ere" not in args.model_type and "_ec" not in args.model_type:
                if args.ere_task_type != "doc_joint":
                    tmp_eval_loss, logits_doc, label_doc, logits_sent, label_sent, logits_token, label_tokens = outputs[:7]
                else:
                    if args.task_name == "maven-ere":
                        tmp_eval_loss, logits_doc_temp, label_doc_temp, logits_doc_causal, label_doc_causal, logits_doc_sub, label_doc_sub, logits_doc_corref, label_doc_corref, logits_sent, label_sent, logits_token, label_tokens = outputs[:13]
                    else:
                        tmp_eval_loss, logits_doc_temp, label_doc_temp, logits_doc_causal, label_doc_causal, logits_doc_sub, label_doc_sub, logits_sent, label_sent, logits_token, label_tokens = outputs[:11]
            elif "_ere" in args.model_type and "_ec" not in args.model_type and "_ed" not in args.model_type:
                if args.ere_task_type != "doc_joint":
                    tmp_eval_loss, logits_doc, label_doc = outputs[:3]
                else:
                    if args.task_name == "maven-ere":
                        tmp_eval_loss, logits_doc_temp, label_doc_temp, logits_doc_causal, label_doc_causal, logits_doc_sub, label_doc_sub, logits_doc_corref, label_doc_corref = outputs[:9]
                    else:
                        tmp_eval_loss, logits_doc_temp, label_doc_temp, logits_doc_causal, label_doc_causal, logits_doc_sub, label_doc_sub = outputs[:7] 
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if "_ere" not in args.model_type: # or, only for ERE task 
                if "_ec" not in args.model_type: # or, only for EC task 
                    # for token-level task
                    if preds_token is None:
                        preds_token = logits_token.detach().cpu().numpy()
                        out_label4token_ids = label_tokens.detach().cpu().numpy()
                    else:
                        preds_token = np.append(preds_token, logits_token.detach().cpu().numpy(), axis=0)
                        out_label4token_ids = np.append(out_label4token_ids, label_tokens.detach().cpu().numpy(), axis=0)
                # for sentence-level task
                if preds_sent is None:
                    preds_sent = logits_sent.detach().cpu().numpy()
                    out_label4sent_ids = label_sent.detach().cpu().numpy()
                else:
                    preds_sent = np.append(preds_sent, logits_sent.detach().cpu().numpy(), axis=0)
                    out_label4sent_ids = np.append(out_label4sent_ids, label_sent.detach().cpu().numpy(), axis=0)
            # for document-level task
            if args.ere_task_type != "doc_joint":
                if preds_doc is None:
                    preds_doc = logits_doc.detach().cpu().numpy()
                    out_label4doc_ids = label_doc.detach().cpu().numpy()
                else:
                    preds_doc = np.append(preds_doc, logits_doc.detach().cpu().numpy(), axis=0)
                    out_label4doc_ids = np.append(out_label4doc_ids, label_doc.detach().cpu().numpy(), axis=0)
            else: 
                if preds_doc_temp is None:
                    preds_doc_temp = logits_doc_temp.detach().cpu().numpy()
                    out_label4doc_temp_ids = label_doc_temp.detach().cpu().numpy()
                else:
                    preds_doc_temp = np.append(preds_doc_temp, logits_doc_temp.detach().cpu().numpy(), axis=0)
                    out_label4doc_temp_ids = np.append(out_label4doc_temp_ids, label_doc_temp.detach().cpu().numpy(), axis=0)

                if preds_doc_causal is None:
                    preds_doc_causal = logits_doc_causal.detach().cpu().numpy()
                    out_label4doc_causal_ids = label_doc_causal.detach().cpu().numpy()
                else:
                    preds_doc_causal = np.append(preds_doc_causal, logits_doc_causal.detach().cpu().numpy(), axis=0)
                    out_label4doc_causal_ids = np.append(out_label4doc_causal_ids, label_doc_causal.detach().cpu().numpy(), axis=0)

                if preds_doc_sub is None:
                    preds_doc_sub = logits_doc_sub.detach().cpu().numpy()
                    out_label4doc_sub_ids = label_doc_sub.detach().cpu().numpy()
                else:
                    preds_doc_sub = np.append(preds_doc_sub, logits_doc_sub.detach().cpu().numpy(), axis=0)
                    out_label4doc_sub_ids = np.append(out_label4doc_sub_ids, label_doc_sub.detach().cpu().numpy(), axis=0)
                
                if args.task_name == "maven-ere":
                    if preds_doc_corref is None:
                        preds_doc_corref = logits_doc_corref.detach().cpu().numpy()
                        out_label4doc_corref_ids = label_doc_corref.detach().cpu().numpy()
                    else:
                        preds_doc_corref = np.append(preds_doc_corref, logits_doc_corref.detach().cpu().numpy(), axis=0)
                        out_label4doc_corref_ids = np.append(out_label4doc_corref_ids, label_doc_corref.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if "_ere" not in args.model_type:
            preds_token = np.argmax(preds_token, axis=1)
            preds_sent = np.argmax(preds_sent, axis=1)
        if args.ere_task_type != "doc_joint": 
            preds_doc = np.argmax(preds_doc, axis=1)
        else:
            preds_doc_temp = np.argmax(preds_doc_temp, axis=1)
            preds_doc_causal = np.argmax(preds_doc_causal, axis=1)
            preds_doc_sub = np.argmax(preds_doc_sub, axis=1)
            if args.task_name == "maven-ere": 
                preds_doc_corref = np.argmax(preds_doc_corref, axis=1)

        if "_ere" not in args.model_type and "_ec" not in args.model_type: # or, only for ECE or EC task
            p_micro_token, r_micro_token, f1_micro_token = calculate_scores(preds_token, out_label4token_ids, len(processors[eval_task]().get_labels4tokens()), "token")
            if args.task_name == "ontoevent-doc":
                p_micro_sent, r_micro_sent, f1_micro_sent = calculate_scores(preds_sent, out_label4sent_ids, len(processors[eval_task]().get_labels4sent()), "sent_onto")
            elif args.task_name == "maven-ere":
                p_micro_sent, r_micro_sent, f1_micro_sent = calculate_scores(preds_sent, out_label4sent_ids, len(processors[eval_task]().get_labels4sent()), "sent")
        elif "_ec" in args.model_type: # or, only for EC task
            p_micro_token = None 
            r_micro_token = None 
            f1_micro_token = None
            if args.task_name == "ontoevent-doc":
                p_micro_sent, r_micro_sent, f1_micro_sent = calculate_scores(preds_sent, out_label4sent_ids, len(processors[eval_task]().get_labels4sent()), "sent_onto")
            elif args.task_name == "maven-ere":
                p_micro_sent, r_micro_sent, f1_micro_sent = calculate_scores(preds_sent, out_label4sent_ids, len(processors[eval_task]().get_labels4sent()), "sent")
            p_micro_doc = None 
            r_micro_doc = None 
            f1_micro_doc = None  
        else: # only for ECE task
            p_micro_token = None 
            r_micro_token = None 
            f1_micro_token = None 
            p_micro_sent = None 
            r_micro_sent = None 
            f1_micro_sent = None 

        if args.ere_task_type == "doc_all": 
            p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, len(processors[eval_task]().get_labels4doc()), args.ere_task_type) 
        else:
            if args.task_name == "ontoevent-doc":
                if args.ere_task_type == "doc_temporal":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+3, args.ere_task_type)
                elif args.ere_task_type == "doc_causal":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+2, args.ere_task_type)
                elif args.ere_task_type == "doc_sub":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+3, args.ere_task_type)
                elif args.ere_task_type == "doc_joint":  
                    p_micro_temporal_joint, r_micro_temporal_joint, f1_micro_temporal_joint  = calculate_scores(preds_doc_temp, out_label4doc_temp_ids, 1+3, "doc_temporal")
                    p_micro_causal_joint, r_micro_causal_joint, f1_micro_causal_joint = calculate_scores(preds_doc_causal, out_label4doc_causal_ids, 1+2, "doc_causal")
                    p_micro_sub_joint, r_micro_sub_joint, f1_micro_sub_joint = calculate_scores(preds_doc_sub, out_label4doc_sub_ids, 1+3, "doc_sub")
            elif args.task_name == "maven-ere":
                if args.ere_task_type == "doc_temporal":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+6, args.ere_task_type)
                elif args.ere_task_type == "doc_causal":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+2, args.ere_task_type)
                elif args.ere_task_type == "doc_sub":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+1, args.ere_task_type)
                elif args.ere_task_type == "doc_corref":  
                    p_micro_doc, r_micro_doc, f1_micro_doc = calculate_scores(preds_doc, out_label4doc_ids, 1+1, args.ere_task_type)
                elif args.ere_task_type == "doc_joint":  
                    p_micro_temporal_joint, r_micro_temporal_joint, f1_micro_temporal_joint  = calculate_scores(preds_doc_temp, out_label4doc_temp_ids, 1+6, "doc_temporal")
                    p_micro_causal_joint, r_micro_causal_joint, f1_micro_causal_joint = calculate_scores(preds_doc_causal, out_label4doc_causal_ids, 1+2, "doc_causal")
                    p_micro_sub_joint, r_micro_sub_joint, f1_micro_sub_joint = calculate_scores(preds_doc_sub, out_label4doc_sub_ids, 1+1, "doc_sub")
                    p_micro_corref_joint, r_micro_corref_joint, f1_micro_corref_joint = calculate_scores(preds_doc_corref, out_label4doc_corref_ids, 1+1, "doc_corref") 
                
        if infer:
            if "_ere" not in args.model_type: # or, only for document-level task
                np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-token.npy"), preds_token)
                np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-sentence.npy"), preds_sent)
            if args.ere_task_type != "doc_joint":   
                np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document.npy"), preds_doc)
            else:
                if args.task_name == "ontoevent-doc":
                    np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document_temporal.npy"), preds_doc_temp)
                    np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document_causal.npy"), preds_doc_causal)
                elif args.task_name == "maven-ere": 
                    np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document_temporal.npy"), preds_doc_temp)
                    np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document_causal.npy"), preds_doc_causal)
                    np.save(os.path.join(eval_output_dir, str(prefix) + "_preds-document_subevent.npy"), preds_doc_sub)

        if args.ere_task_type != "doc_joint":  
            result = {
                "eval_p_micro_token": p_micro_token, "eval_r_micro_token": r_micro_token, "eval_f1_micro_token": f1_micro_token, 
                "eval_p_micro_sent": p_micro_sent, "eval_r_micro_sent": r_micro_sent, "eval_f1_micro_sent": f1_micro_sent, 
                "eval_p_micro_doc": p_micro_doc, "eval_r_micro_doc": r_micro_doc, "eval_f1_micro_doc": f1_micro_doc, 
                "eval_loss": eval_loss
            }
        else: 
            if args.task_name == "ontoevent-doc":
                result = {
                    "eval_p_micro_token": p_micro_token, "eval_r_micro_token": r_micro_token, "eval_f1_micro_token": f1_micro_token, 
                    "eval_p_micro_sent": p_micro_sent, "eval_r_micro_sent": r_micro_sent, "eval_f1_micro_sent": f1_micro_sent, 
                    "eval_p_micro_doc_temporal_joint": p_micro_temporal_joint, "eval_r_micro_doc_temporal_joint": r_micro_temporal_joint, "eval_f1_micro_doc_temporal_joint": f1_micro_temporal_joint, 
                    "eval_p_micro_doc_causal_joint": p_micro_causal_joint, "eval_r_micro_doc_causal_joint": r_micro_causal_joint, "eval_f1_micro_doc_causal_joint": f1_micro_causal_joint, 
                    "eval_loss": eval_loss
                }
            elif args.task_name == "maven-ere":
                result = {
                    "eval_p_micro_token": p_micro_token, "eval_r_micro_token": r_micro_token, "eval_f1_micro_token": f1_micro_token, 
                    "eval_p_micro_sent": p_micro_sent, "eval_r_micro_sent": r_micro_sent, "eval_f1_micro_sent": f1_micro_sent, 
                    "eval_p_micro_doc_temporal_joint": p_micro_temporal_joint, "eval_r_micro_doc_temporal_joint": r_micro_temporal_joint, "eval_f1_micro_doc_temporal_joint": f1_micro_temporal_joint, 
                    "eval_p_micro_doc_causal_joint": p_micro_causal_joint, "eval_r_micro_doc_causal_joint": r_micro_causal_joint, "eval_f1_micro_doc_causal_joint": f1_micro_causal_joint, 
                    "eval_p_micro_doc_sub_joint": p_micro_sub_joint, "eval_r_micro_doc_sub_joint": r_micro_sub_joint, "eval_f1_micro_doc_sub_joint": f1_micro_sub_joint,   
                    "eval_loss": eval_loss
                } 
        
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + "are test: " + str(test)))
            writer.write("model            = %s\n" % str(args.model_name_or_path))
            # writer.write(
            #     "total batch size = %d\n"
            #     % (
            #         args.per_gpu_train_batch_size
            #         * args.gradient_accumulation_steps
            #         * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
            #     )
            # )
            writer.write("train num epochs = %d\n" % args.num_train_epochs)
            writer.write("fp16             = %s\n" % args.fp16)
            writer.write("max seq length   = %d\n" % args.max_seq_length)
            for key in sorted(result.keys()):
                # logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results

def save_input_features_list(features_list, file_path):
    data_dict = {}
    for i, features in enumerate(features_list):
        data_dict[f'example_id_{i}'] = features.example_id
        data_dict[f'mention_size_{i}'] = features.mention_size
        data_dict[f'pad_token_label_id_{i}'] = features.pad_token_label_id
        data_dict[f'list_input_ids_{i}'] = features.list_input_ids
        data_dict[f'list_input_mask_{i}'] = features.list_input_mask
        data_dict[f'list_segment_ids_{i}'] = features.list_segment_ids
        data_dict[f'list_token_labels_{i}'] = features.list_token_labels
        data_dict[f'list_sent_label_{i}'] = features.list_sent_label
        data_dict[f'mat_rel_label_{i}'] = features.mat_rel_label
    np.savez(file_path, **data_dict)
    
def load_input_features_list(file_path):
    loaded_data = np.load(file_path)
    features_list = []
    i = 0
    while f'example_id_{i}' in loaded_data:
        example_id = loaded_data[f'example_id_{i}']
        mention_size = loaded_data[f'mention_size_{i}']
        pad_token_label_id = loaded_data[f'pad_token_label_id_{i}']
        list_input_ids = loaded_data[f'list_input_ids_{i}']
        list_input_mask = loaded_data[f'list_input_mask_{i}']
        list_segment_ids = loaded_data[f'list_segment_ids_{i}']
        list_token_labels = loaded_data[f'list_token_labels_{i}']
        list_sent_label = loaded_data[f'list_sent_label_{i}']
        mat_rel_label = loaded_data[f'mat_rel_label_{i}']
        
        features = InputFeatures(example_id, mention_size, pad_token_label_id,
                                 list_input_ids, list_input_mask, list_segment_ids,
                                 list_token_labels, list_sent_label, mat_rel_label)
        
        features_list.append(features)
        i += 1
    
    return features_list
    
def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "valid"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "Cached_{}_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.max_mention_size),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file+".npz") and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = load_input_features_list(cached_features_file+".npz")
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label4sent_list = processor.get_labels4sent()
        label4token_list = processor.get_labels4tokens()
        label4rel_list = processor.get_labels4doc() 
        if evaluate:
            examples = processor.get_valid_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)      
        logger.info("Training number: %s", str(len(examples)))
        
        features = convert_examples_to_features(
            examples,
            label4token_list,
            label4sent_list,
            label4rel_list,
            args.max_seq_length,
            args.max_mention_size,
            tokenizer,
            cls_token_at_end=bool(args.model_type.startswith("xlnet")), # xlnet has a cls token at the end,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type.startswith("xlnet") else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type.startswith("roberta")), # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type.startswith("xlnet")), # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type.startswith("xlnet") else 0,
            model_name=args.model_name_or_path,
            task_name=args.task_name
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file [%s]", cached_features_file)
            save_input_features_list(features, cached_features_file)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_example_id = ms.tensor([f.example_id for f in features], dtype=ms.int64)
    all_mention_size = ms.tensor([f.mention_size for f in features], dtype=ms.int64)
    all_pad_token_label_id = ms.tensor([f.pad_token_label_id for f in features], dtype=ms.int64) 
    all_list_input_ids = ms.tensor([f.list_input_ids for f in features], dtype=ms.int64)
    all_list_input_mask = ms.tensor([f.list_input_mask for f in features], dtype=ms.int64)
    all_list_segment_ids = ms.tensor([f.list_segment_ids for f in features], dtype=ms.int64)
    all_list_label4token_ids = ms.tensor([f.list_token_labels for f in features], dtype=ms.int64)
    all_list_label4sent_ids = ms.tensor([f.list_sent_label for f in features], dtype=ms.int64)
    all_mat_rel_label_ids = ms.tensor([f.mat_rel_label for f in features], dtype=ms.int64)
    
    # dataset = TensorDataset(all_example_id, all_mention_size, all_pad_token_label_id, all_list_input_ids, all_list_input_mask, all_list_segment_ids, all_list_label4token_ids, all_list_label4sent_ids, all_mat_rel_label_ids)
    
    dataset=MsDataset(all_example_id, all_mention_size, all_pad_token_label_id, all_list_input_ids, all_list_input_mask, all_list_segment_ids, all_list_label4token_ids, all_list_label4sent_ids, all_mat_rel_label_ids)
    return dataset


def main():
    print("start")
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data directory.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    ) # "speech_bert, speech_roberta, speech_distilbert"
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    ) # "bert-base-uncased, roberta-base, distilbert-base-uncased"
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    ) # "ontoevent-doc, maven-ere"
    parser.add_argument(
        "--central_task",
        default=None,
        type=str,
        required=True,
        help="The central task for optimization: " + 
            "token, " + "sent, " + "doc",
    ) # "token, sent, doc"
    parser.add_argument(
        "--ere_task_type",
        default=None,
        type=str,
        required=True,
        help="The type of doc ere task to train selected in the list: " + 
            "doc_all, " + "doc_joint, " + "doc_temporal, " +  "doc_causal, " +  "doc_sub, " +  "doc_corref" + 
            "doc_all: for \"All Joint\" in the paper; " + 
            "doc_joint: for each ERE subtask +joint; " + 
            "doc_temporal/doc_causal/doc_sub: for each ERE subtask only",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_mention_size",
        default=50,
        type=int,
        help="The maximum size of event mentions in on document. The event mention size of each document should not larger than it, documents shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set") 
    # Note that for maven-ere datasets, we only evaluate on the valid set, thus "--do_test" should be dismissed for experiments on maven-ere 
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd

    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1
    # args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s,  16-bits training: %s",
        args.local_rank,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare for task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label4token_list = processor.get_labels4tokens()
    num_labels4token = len(label4token_list) 
    label4sent_list = processor.get_labels4sent()
    num_labels4sent = len(label4sent_list)
    label4doc_list = processor.get_labels4doc()
    num_labels4doc = len(label4doc_list) 

    # Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels4token,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print(config)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("config")
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("config")

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    """Training"""
    import time
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 ):
    #if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to [%s]", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        ms.save_checkpoint(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)

    """Evaluation"""
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + ".ckpt", recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: [%s]", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            result = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if best_steps:
        logger.info("best steps of eval f1 is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
