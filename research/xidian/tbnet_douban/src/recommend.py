# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Inference result aggregator."""

import copy
import warnings


def _reverse_id_maps(id_maps):
    """
    Reverse the key-value position of object id maps.

    Args:
        id_maps(dict[any, dict[any, any]]): Collection of object id maps.

    Returns:
        dict[any, dict[any, any]], collection of the reversed object id maps.
    """
    reversed_maps = dict()
    for obj_type, id_map in id_maps.items():
        if not isinstance(id_map, dict):
            continue
        reversed_map = dict()
        for src_id, intern_id in id_map.items():
            reversed_map[intern_id] = src_id
        reversed_maps[obj_type] = reversed_map
    return reversed_maps


class RelationPath:
    """
    Relation path connects the suggested item and the historical item.

    Args:
        relation1 (str): Source id of relation 1.
        reference (str): Source id of the reference entity.
        relation2 (str): Source id of relation 2.
        hist_item (str): Source id of the historical item.
        rl1_intern_id (int): Internal id of relation 1.
        ref_intern_id (int): Internal id of the reference entity.
        rl2_intern_id (int): Internal id of relation 1.
        hist_intern_id (int): Internal id of the historical item.
        importance (float): Path importance.
    """
    def __init__(self,
                 relation1, reference, relation2, hist_item,
                 rl1_intern_id, ref_intern_id, rl2_intern_id, hist_intern_id,
                 importance):
        self.relation1 = relation1
        self.reference = reference
        self.relation2 = relation2
        self.hist_item = hist_item
        self.rl1_intern_id = rl1_intern_id
        self.ref_intern_id = ref_intern_id
        self.rl2_intern_id = rl2_intern_id
        self.hist_intern_id = hist_intern_id
        self.importance = importance


class Suggestion:
    """
    Recommended item id with relation paths as explanations.

    Args:
        item (str): Source id of the suggested item.
        item_intern_id (int): Internal id of the suggested item.
        score (float): Recommendation score, higher the score, more the item got recommended.
        paths (list[RelationPath]): List of relation paths connect to the historical items, must be in descending order
            by importance.
    """
    def __init__(self, item, item_intern_id, score, paths):
        self.item = item
        self.item_intern_id = item_intern_id
        self.score = score
        # paths must be sorted with importance in descending order
        self.paths = paths


class Recommender:
    """
    TB-Net inference and result aggregation.

    Args:
        network(Cell): TB-Net.
        id_maps(dict[str, dict[str, int]]): Object id maps.
        top_k (int): The number of items to be recommended.

    Inputs:
        item (Tensor): Candidate item IDs, int Tensor in shape of [batch size, ].
        rl1 (Tensor): item-reference relation IDs, int Tensor in shape of [batch size, per-item paths].
        ref (Tensor): Reference object IDs, int Tensor in shape of [batch size, per-item paths].
        rl2 (Tensor): reference-hist_item relation IDs, int Tensor in shape of [batch size, per-item paths].
        hist_item (Tensor): Historical item IDs, int Tensor in shape of [batch size, per-item paths].

    Outputs:
        scores (Tensor): Item recommendation scores, float Tensor in shape of [batch size, ]
        importances (Tensor): Relation paths' importance [0.0, 1.0], float Tensor in shape of
            [batch size, per-item paths].
        item_embs (Tensor): Candidate item embeddings, float Tensor in shape of [batch size, embedding dim]
        rl1_embs (Tensor): item-reference relation embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim, embedding dim].
        ref_embs (Tensor): Reference object embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim].
        rl2_embs (Tensor): reference-hist_item relation embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim, embedding dim].
        hist_item_embs (Tensor): Historical item embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim].

    Supported Platforms:
        ``GPU``
    """
    def __init__(self, network, id_maps, top_k):
        if top_k < 1:
            raise ValueError('top_k is less than 1.')
        self._network = network
        self._r_id_maps = _reverse_id_maps(id_maps)
        self._top_k = top_k
        self._suggestions = []
        self._paths_sorted = False

    def __call__(self, item, rl1, ref, rl2, hist_item):
        """Inference and aggregate the results."""
        result = self._network(item, rl1, ref, rl2, hist_item)
        scores = result[0]
        importances = result[1]
        self._aggregate(item, rl1, ref, rl2, hist_item, scores, importances)
        return result

    def suggest(self):
        """
        Get suggestions.

        Returns:
            list[Suggestion] - Item suggestions.
        """
        if not self._paths_sorted:
            self._sort_paths()
        return copy.deepcopy(self._suggestions)

    def _aggregate(self, item, rl1, ref, rl2, hist_item, scores, importances):
        """Aggregate inference results."""
        items = item.asnumpy()
        relations1 = rl1.asnumpy()
        references = ref.asnumpy()
        relations2 = rl2.asnumpy()
        hist_items = hist_item.asnumpy()
        scores = scores.asnumpy()
        importances = importances.asnumpy()

        batch_size = items.shape[0]

        for i in range(batch_size):
            if self._add(items[i], relations1[i], references[i], relations2[i],
                         hist_items[i], scores[i], importances[i]):
                self._paths_sorted = False

        if len(self._suggestions) > self._top_k:
            self._suggestions = self._suggestions[0:self._top_k]

    def _add(self, item, relations1, references, relations2, hist_items, score, importances):
        """Add a single infer record."""
        if item <= 0:  # unseen item
            return False

        # insert at the appropriate position
        for i, old_suggest in enumerate(self._suggestions):
            if i >= self._top_k:
                return False
            if score > old_suggest.score:
                rec = self._to_suggestion(item, relations1, references, relations2,
                                          hist_items, score, importances)
                self._suggestions.insert(i, rec)
                return True

        # append if has rooms
        if len(self._suggestions) < self._top_k:
            rec = self._to_suggestion(item, relations1, references, relations2,
                                      hist_items, score, importances)
            self._suggestions.append(rec)
            return True

        return False

    def _to_suggestion(self, item, relations1, references, relations2, hist_items, score, importances):
        """Converts a single infer result to an item suggestion."""
        src_item = self._intern2src_id('item', item)
        suggestion = Suggestion(src_item, item, score, [])
        num_paths = importances.shape[0]
        for i in range(num_paths):
            relation1 = self._intern2src_id('relation', relations1[i])
            reference = self._intern2src_id('reference', references[i])
            relation2 = self._intern2src_id('relation', relations2[i])
            hist_item = self._intern2src_id('item', hist_items[i])
            path = RelationPath(relation1, reference, relation2, hist_item,
                                relations1[i], references[i], relations2[i], hist_items[i],
                                importances[i])
            suggestion.paths.append(path)
        return suggestion

    def _sort_paths(self):
        """Sort all item paths."""
        for suggestion in self._suggestions:
            suggestion.paths.sort(key=lambda x: x.importance, reverse=True)
        self._paths_sorted = True

    def _intern2src_id(self, obj_type, intern_id):
        """Map internal id to source id."""
        src_id = self._r_id_maps[obj_type].get(intern_id, None)
        if src_id is None:
            src_id = f'{obj_type}:{intern_id}'
            warnings.warn(f'Source ID of {src_id} was not found in id map.', ResourceWarning)
        return src_id
