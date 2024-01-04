# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


import json
import codecs
import logging
import os
from typing import List

import tqdm
from mindspore import context
context.set_context(device_target='CPU')
# from transformers import PreTrainedTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, DistilBertTokenizer, CamembertTokenizer, XLMRobertaTokenizer
from mindnlp.transformers.models.bert import BertTokenizer 
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, mention_size, list_tokens, list_triggerL, list_triggerR, list_token_labels=None, list_sent_label=None, mat_rel_label=None):
        """Constructs a InputExample.
        Args:
            example_id: str. unique id for the example.
            mention_size: int. the quantity of event mentions in one doc 
            tokens: list of [list of tokens]. 
            triggerL: list of int. beginning position of the trigger
            triggerR: list of int. endding position of the trigger
            token_label: (Optional) string. list of (the label of the token list). This should be specified for train and valid examples, but not for test examples.
            sent_label: (Optional) string. list of (the label of the sentence). This should be specified for train and valid examples, but not for test examples.
            rel_label: (Optional) string. list of (the label of the relation between sentence pairs). This should be specified for train and valid examples, but not for test examples.
        """
        self.example_id = example_id
        self.mention_size = mention_size
        self.list_tokens = list_tokens
        self.list_triggerL = list_triggerL
        self.list_triggerR = list_triggerR
        self.list_token_labels = list_token_labels
        self.list_sent_label = list_sent_label
        self.mat_rel_label = mat_rel_label 


class InputFeatures(object):
    def __init__(self, example_id, mention_size, pad_token_label_id, list_input_ids, list_input_mask, list_segment_ids, list_token_labels, list_sent_label, mat_rel_label):
        self.example_id = example_id
        self.mention_size = mention_size
        self.pad_token_label_id = pad_token_label_id
        self.list_input_ids = list_input_ids
        self.list_input_mask = list_input_mask
        self.list_segment_ids = list_segment_ids
        self.list_token_labels = list_token_labels
        self.list_sent_label = list_sent_label
        self.mat_rel_label = mat_rel_label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the valid set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels4tokens(self):
        """Gets the list of token labels for this data set."""
        raise NotImplementedError()
    
    def get_labels4sent(self):
        """Gets the list of sentence labels for this data set."""
        raise NotImplementedError()
    
    def get_labels4doc(self):
        """Gets the list of sentence pair labels for this data set."""
        raise NotImplementedError()


class OntoEventProcessor(DataProcessor):
    """Processor for the OntoEvent data set."""
    
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'OntoEvent-Doc/event_dict_on_doc_train.json'), "train")

    def get_valid_examples(self, data_dir):
        logger.info("LOOKING AT {} valid".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'OntoEvent-Doc/event_dict_on_doc_valid.json'), "valid")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'OntoEvent-Doc/event_dict_on_doc_test.json'), "test")

    def get_labels4sent(self):
        file_path = ONTOEVENT_LABEL_PATH 
        data = json2dicts(file_path)
        list_label4sent = [key for key in data[0].keys()]
        list_label4sent.insert(0, "None") 
        return list_label4sent
    
    def get_labels4tokens(self):
        list_label4token = self.get_labels4sent() # id: 0 ~ |E|, 0 is for None
        list_label4token.append(NAME_NON_TRIGGER) # id: |E|+1
        list_label4token.append(NAME_PADDING) # id: |E|+2 
        return list_label4token

    def get_labels4doc(self):
        list_label4doc = [key for key in relation_map_ontoevent.keys()]
        list_label4doc.insert(0, NAME_NO_RELATION) 
        return list_label4doc
    
    def create_examples(self, file_path, set_type):
        """Creates examples for the training and valid sets."""
        examples = []
        data = json2dicts(file_path)[0]

        for doc_id in data.keys():
            dict_doc = data[doc_id]
            mention_size = len(dict_doc["events"])
            list_mention_id = []
            list_tokens = []
            list_triggerL = []
            list_triggerR = []
            list_token_labels = []
            list_sent_label = []
            mat_rel_label = []
             
            for i in range(mention_size):
                mat_rel_label.append([NAME_NO_RELATION]*mention_size)
            dict_rel_pairs = dict_doc['relations']
            for rel in dict_rel_pairs:
                for event_index_pair in dict_rel_pairs[rel]:
                    head_index = event_index_pair[0]
                    tail_index = event_index_pair[1]
                    mat_rel_label[head_index][tail_index] = rel 
        
            for event_instance in dict_doc["events"]:
                list_token_label = [NAME_NON_TRIGGER] * len(event_instance['event_mention_tokens']) 
                sid = event_instance['sent_id']
                if type(event_instance['sent_id'] != str):
                    sid = str(sid)    
                # e_id = "%s-+-%s-+-%s" % (set_type, event_instance['doc_id'], sid)
                e_id = "%s-+-%s-+-%s" % (event_instance['event_type'], event_instance['doc_id'], sid)
                list_mention_id.append(e_id)
                if (type(event_instance['trigger_pos']) == int):
                    triL = event_instance['trigger_pos']
                    triR = triL
                else:
                    triL = event_instance['trigger_pos'][0]
                    triR = event_instance['trigger_pos'][1]
                for i in range(triL, triR): 
                    list_token_label[i] = event_instance['event_type']
                
                list_tokens.append(event_instance['event_mention_tokens'])
                list_triggerL.append(triL) 
                list_triggerR.append(triR)
                list_token_labels.append(list_token_label)
                list_sent_label.append(event_instance['event_type']) 

            examples.append(
                InputExample(
                    example_id=doc_id,
                    mention_size=mention_size,
                    list_tokens=list_tokens,
                    list_triggerL=list_triggerL,
                    list_triggerR=list_triggerR,
                    list_token_labels=list_token_labels,
                    list_sent_label=list_sent_label,
                    mat_rel_label=mat_rel_label,
                )
            )
        return examples
    

class MAVENEREProcessor(DataProcessor):
    """Processor for the MAVENERE data set."""
    
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'MAVEN_ERE/train.jsonl'), "train")

    def get_valid_examples(self, data_dir):
        logger.info("LOOKING AT {} valid".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'MAVEN_ERE/valid.jsonl'), "valid")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'MAVEN_ERE/test.jsonl'), "test")

    def get_labels4sent(self):
        list_label4sent = ["None", "Know", "Warning", "Catastrophe", "Placing", "Causation", "Arriving", "Sending", "Protest", "Preventing_or_letting", "Motion", "Damaging", "Destroying", "Death", "Perception_active", "Presence", "Influence", "Receiving", "Check", "Hostile_encounter", "Killing", "Conquering", "Releasing", "Attack", "Earnings_and_losses", "Choosing", "Traveling", "Recovering", "Using", "Coming_to_be", "Cause_to_be_included", "Process_start", "Change_event_time", "Reporting", "Bodily_harm", "Suspicion", "Statement", "Cause_change_of_position_on_a_scale", "Coming_to_believe", "Expressing_publicly", "Request", "Control", "Supporting", "Defending", "Building", "Military_operation", "Self_motion", "GetReady", "Forming_relationships", "Becoming_a_member", "Action", "Removing", "Surrendering", "Agree_or_refuse_to_act", "Participation", "Deciding", "Education_teaching", "Emptying", "Getting", "Besieging", "Creating", "Process_end", "Body_movement", "Expansion", "Telling", "Change", "Legal_rulings", "Bearing_arms", "Giving", "Name_conferral", "Arranging", "Use_firearm", "Committing_crime", "Assistance", "Surrounding", "Quarreling", "Expend_resource", "Motion_directional", "Bringing", "Communication", "Containing", "Manufacturing", "Social_event", "Robbery", "Competition", "Writing", "Rescuing", "Judgment_communication", "Change_tool", "Hold", "Being_in_operation", "Recording", "Carry_goods", "Cost", "Departing", "GiveUp", "Change_of_leadership", "Escaping", "Aiming", "Hindering", "Preserving", "Create_artwork", "Openness", "Connect", "Reveal_secret", "Response", "Scrutiny", "Lighting", "Criminal_investigation", "Hiding_objects", "Confronting_problem", "Renting", "Breathing", "Patrolling", "Arrest", "Convincing", "Commerce_sell", "Cure", "Temporary_stay", "Dispersal", "Collaboration", "Extradition", "Change_sentiment", "Commitment", "Commerce_pay", "Filling", "Becoming", "Achieve", "Practice", "Cause_change_of_strength", "Supply", "Cause_to_amalgamate", "Scouring", "Violence", "Reforming_a_system", "Come_together", "Wearing", "Cause_to_make_progress", "Legality", "Employment", "Rite", "Publishing", "Adducing", "Exchange", "Ratification", "Sign_agreement", "Commerce_buy", "Imposing_obligation", "Rewards_and_punishments", "Institutionalization", "Testing", "Ingestion", "Labeling", "Kidnapping", "Submitting_documents", "Prison", "Justifying", "Emergency", "Terrorism", "Vocalizations", "Risk", "Resolve_problem", "Revenge", "Limiting", "Research", "Having_or_lacking_access", "Theft", "Incident", "Award"]
        return list_label4sent
    
    def get_labels4tokens(self):
        list_label4token = self.get_labels4sent() # id: 0 ~ |E|, 0 is for None
        list_label4token.append(NAME_NON_TRIGGER) # id: |E|+1
        list_label4token.append(NAME_PADDING) # id: |E|+2 
        return list_label4token

    def get_labels4doc(self):
        list_label4doc = [key for key in relation_map_mavenere.keys()]
        list_label4doc.insert(0, NAME_NO_RELATION) 
        return list_label4doc
    
    def create_examples(self, file_path, set_type):
        """Creates examples for the training and valid sets."""
        examples = []
        data = json2dicts(file_path)

        dict_docid2mentionids = {} 

        for dict_doc in data:
            doc_id = dict_doc["id"] 
            mention_size = len(dict_doc['tokens'])
            list_tokens = dict_doc['tokens'] 
            list_triggerL = []
            list_triggerR = []
            list_token_labels = []
            list_sent_label = []
            mat_rel_label = []

            # initiate the mat_rel_label
            # Note that there are no labels between event instance pairs in the test set, thus the preformance calculated on the test set is actually meaningless 
            # We need to dump prediction results and submit them to MAVEN-ERE CodaLab competition or evaluate on the valid set instead 
            for i in range(mention_size):
                mat_rel_label.append([NAME_NO_RELATION]*mention_size)

            list_coref_sentid = [] 
            dict_eid2event = {}
            dict_sid2event = {}
            if set_type != 'test': # relation ground truth of maven-ere are hidden and the format of maven-ere train/valid set are a little different from test set 
                for event_instance in dict_doc['events']:
                    dict_eid2event[event_instance['id']] = {'id': event_instance['id'], 'type': event_instance['type'], 'sent_id': event_instance['mention'][0]['sent_id'], 'offset': event_instance['mention'][0]['offset']} 
                    list_coref_sentid_temp = []
                    for cor_event in event_instance['mention']:
                        list_coref_sentid_temp.append(cor_event['sent_id'])
                        dict_sid2event[cor_event['sent_id']] = {'id': event_instance['id'], 'type': event_instance['type'], 'sent_id': cor_event['sent_id'], 'offset': cor_event['offset'], 'mention_id': cor_event['id']}  
                    list_coref_sentid_temp = list(set(list_coref_sentid_temp)) 
                    for i in range(len(list_coref_sentid_temp) - 1):
                        for j in range(i+1, len(list_coref_sentid_temp)):
                            list_coref_sentid.append([list_coref_sentid_temp[i], list_coref_sentid_temp[j]]) 

                dict_rel_pairs = {}
                dict_rel_pairs.update(dict_doc['temporal_relations'])
                dict_rel_pairs.update(dict_doc['causal_relations'])
                # dict_rel_pairs.update(dict_doc['subevent_relations'])
                dict_rel_pairs['subevent_relations'] = dict_doc['subevent_relations']

                for rel in dict_rel_pairs:
                    for event_id_pair in dict_rel_pairs[rel]:
                        if dict_eid2event.get(event_id_pair[0]) and dict_eid2event.get(event_id_pair[1]): 
                            head_index = dict_eid2event[event_id_pair[0]]['sent_id']
                            tail_index = dict_eid2event[event_id_pair[1]]['sent_id']
                            mat_rel_label[head_index][tail_index] = rel
            
                for event_id_pair in list_coref_sentid:
                    mat_rel_label[event_id_pair[0]][event_id_pair[1]] = NAME_COREF_RELATION
            else:
                for event_instance in dict_doc['event_mentions']:
                    dict_sid2event[event_instance['sent_id']] = {'id': event_instance['id'], 'type': event_instance['type'], 'sent_id': event_instance['sent_id'], 'offset': event_instance['offset'], 'mention_id': event_instance['id']}   

            list_mention_id = []
            for pos in range(mention_size):
                if dict_sid2event.get(pos): 
                    event_instance = dict_sid2event[pos]
                    list_mention_id.append(event_instance['mention_id']) 
                else:
                    event_instance = {'id': '', 'type': "None", 'sent_id': pos, 'offset': [0, 0]}
                    list_mention_id.append(doc_id + "-+-" + str(pos))
 
                list_token_label = [NAME_NON_TRIGGER] * len(list_tokens[pos])
                
                if (type(event_instance['offset']) == int):
                    triL = event_instance['offset']
                    triR = triL
                else:
                    if len(event_instance['offset']) < 2:
                        triL = event_instance['offset'][0]
                        triR = event_instance['offset'][0]
                    else: 
                        triL = event_instance['offset'][0]
                        triR = event_instance['offset'][1]
                for i in range(triL, triR): 
                    list_token_label[i] = event_instance['type']
                
                list_triggerL.append(triL) 
                list_triggerR.append(triR)
                list_token_labels.append(list_token_label) 
                list_sent_label.append(event_instance['type'])

            dict_docid2mentionids[doc_id] = list_mention_id

            examples.append(
                InputExample(
                    example_id=doc_id,
                    mention_size=mention_size,
                    list_tokens=list_tokens,
                    list_triggerL=list_triggerL,
                    list_triggerR=list_triggerR,
                    list_token_labels=list_token_labels,
                    list_sent_label=list_sent_label,
                    mat_rel_label=mat_rel_label,
                )
            )
        dict2json(dict_docid2mentionids, MAVENERE_MENTION_ID_PATH)
        return examples

     
def json2dicts(jsonFile):
        data = []
        with codecs.open(jsonFile, "r", "utf-8") as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
        return data


def dict2json(dic, jsonFile):
    with open(jsonFile, 'w') as outfile: # 'a+'
        json.dump(dic, outfile)
        # outfile.write('\n')
        print("Finishing writing a dict into " + jsonFile)
    
    
def convert_examples_to_features(
    examples: List[InputExample],
    label4token_list: List[str],
    label4sent_list: List[str],
    label4rel_list: List[str], 
    max_length: int,
    max_size: int,
    tokenizer: BertTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    model_name=None,
    task_name=None
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """ 
    
    label4token_map = {label4token: i for i, label4token in enumerate(label4token_list)}
    label4sent_map = {label: i for i, label in enumerate(label4sent_list)}
    label4rel_map = {label: i for i, label in enumerate(label4rel_list)}
    pad_token_label_id = label4token_map[NAME_PADDING]

    list_example_id = set()
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        list_example_id.add(example.example_id)
    example_id_map = {example_id: i+1 for i, example_id in enumerate(list_example_id)} # example_id counts from 1
    dict_exid2docid = {i+1: example_id for i, example_id in enumerate(list_example_id)} # example_id counts from 1
    if task_name == "maven-ere": 
        dict2json(dict_exid2docid, MAVENERE_EXAMPLE_ID_PATH) 
    
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 500 == 0:
            print("-" * 20)
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            print("-" * 20)

        list_input_ids = []
        list_attention_mask = []
        list_token_type_ids = []
        list_label4token_ids = []
        list_label4sent_ids = []
        mat_rel_label_ids = []

        # initial the mat_rel_label_ids
        for i in range(max_size):
            mat_rel_label_ids.append([label4rel_map[NAME_NO_RELATION]]*max_size)  
        mat_size = min(max_size, example.mention_size) 
        for i in range(mat_size):
            for j in range(mat_size):
                rel_name = example.mat_rel_label[i][j] 
                mat_rel_label_ids[i][j] = label4rel_map[rel_name] 

        for i in range(example.mention_size):
            tokens = []
            tokens.extend(example.list_tokens[i])
            label4token_ids = [] 
            for token_name in example.list_token_labels[i]: 
                label4token_ids.append(label4token_map[token_name])
 
            # Account for [CLS] and [SEP] with "-2" and with "-3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_length - special_tokens_count:
                tokens = tokens[:(max_length - special_tokens_count)]
                label4token_ids = label4token_ids[:(max_length - special_tokens_count)]
            
            tokens.append(sep_token)
            label4token_ids.append(pad_token_label_id)
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens.append(sep_token)
                label4token_ids.append(pad_token_label_id) 
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens.append(cls_token)
                label4token_ids.append(pad_token_label_id) 
                token_type_ids.append(cls_token_segment_id) 
            else:
                tokens.insert(0, cls_token)
                label4token_ids.insert(0, pad_token_label_id)
                token_type_ids.insert(0, cls_token_segment_id) 

            input_ids = tokenizer.convert_tokens_to_ids(tokens) 

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                label4token_ids = ([label4token_map[NAME_PADDING]] * padding_length) + label4token_ids 
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                label4token_ids = label4token_ids + ([label4token_map[NAME_PADDING]] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            assert len(label4token_ids) == max_length

            example_id = example_id_map[example.example_id]
            
            list_input_ids.append(input_ids)
            list_attention_mask.append(attention_mask)
            list_token_type_ids.append(token_type_ids)
            list_label4token_ids.append(label4token_ids)
            label4sent_id = label4sent_map[example.list_sent_label[i]]
            list_label4sent_ids.append(label4sent_id)

        # padding or truncation
        if example.mention_size <= max_size: # padding 
            for i in range(example.mention_size, max_size):
                list_input_ids.append([pad_token] * max_length)
                list_attention_mask.append([0 if mask_padding_with_zero else 1] * max_length)
                list_token_type_ids.append([pad_token_segment_id] * max_length)
                list_label4token_ids.append([pad_token_label_id] * max_length)
                list_label4sent_ids.append(len(label4sent_map))
        else: # truncation
            list_input_ids = list_input_ids[:max_size]  
            list_attention_mask = list_attention_mask[:max_size]  
            list_token_type_ids = list_token_type_ids[:max_size] 
            list_label4token_ids = list_label4token_ids[:max_size]  
            list_label4sent_ids = list_label4sent_ids[:max_size]

        assert len(list_input_ids) == max_size
        assert len(list_attention_mask) == max_size
        assert len(list_input_ids) == max_size
        assert len(list_token_type_ids) == max_size
        assert len(list_label4token_ids) == max_size
        assert len(list_label4sent_ids) == max_size
        assert len(mat_rel_label_ids) == max_size
        assert len(mat_rel_label_ids[0]) == max_size 

        features.append(InputFeatures(example_id=example_id, mention_size=example.mention_size, pad_token_label_id=pad_token_label_id, list_input_ids=list_input_ids, list_input_mask=list_attention_mask, list_segment_ids=list_token_type_ids, list_token_labels=list_label4token_ids, list_sent_label=list_label4sent_ids, mat_rel_label=mat_rel_label_ids))

        if ex_index < 2:
                logger.info("**** Example ****")
                logger.info("example_id: {}".format(example.example_id))
                logger.info("mention_size: {}".format(example.mention_size))
                logger.info("pad_token_label_id: {}".format(pad_token_label_id))
                logger.info("list_input_ids: {}".format(" ".join(map(str, list_input_ids))))
                logger.info("list_input_mask: {}".format(" ".join(map(str, list_attention_mask))))
                logger.info("list_segment_ids: {}".format(" ".join(map(str, list_token_type_ids))))
                logger.info("list_token_labels: {}".format(" ".join(map(str, list_label4token_ids))))
                logger.info("list_sent_label: {}".format(" ".join(map(str, list_label4sent_ids))))
                logger.info("mat_rel_label: {}".format(" ".join(map(str, mat_rel_label_ids))))
    return features
    

processors = {"ontoevent-doc": OntoEventProcessor, "maven-ere": MAVENEREProcessor} # other dataset can also be used here

relation_map_ontoevent = {'BEFORE': 1, 'AFTER': 2, 'EQUAL': 3, 'CAUSE': 4, 'CAUSEDBY': 5, 'COSUPER': 6, 'SUBSUPER': 7, 'SUPERSUB': 8}
NAME_COREF_RELATION = "coreference"
relation_map_mavenere = {'BEFORE': 1, 'OVERLAP': 2, 'CONTAINS': 3, 'SIMULTANEOUS': 4, 'BEGINS-ON': 5, 'ENDS-ON': 6, 'CAUSE': 7, 'PRECONDITION': 8, 'subevent_relations': 9, NAME_COREF_RELATION: 10}

NAME_NON_TRIGGER = "Non-Trigger"
NAME_PADDING = "Padding"
NAME_NO_RELATION = "NA"
ONTOEVENT_LABEL_PATH = "./Datasets/OntoEvent-Doc/event_dict_label_data.json"
# # file path for the json data contains all ontoevent event type labels

MAVENERE_EXAMPLE_ID_PATH = "./Datasets/MAVEN_ERE/map_exid_to_docid.json" 
MAVENERE_MENTION_ID_PATH = "./Datasets/MAVEN_ERE/map_docid_to_mentionids.json" 