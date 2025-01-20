import mindspore as ms
from mindspore import nn

from models.utils.deteval import get_socre_A, get_socre_B, combine_results
from models.utils.extract_textpoint import get_dict

__all__ = ["E2EMetric"]


class E2EMetric(nn.Metric):
    def __init__(
        self,
        mode,
        gt_mat_dir,
        character_dict_path,
        main_indicator="f_score_e2e",
        **kwargs,
    ):
        self.mode = mode
        self.gt_mat_dir = gt_mat_dir
        self.label_list = get_dict(character_dict_path)
        self.max_index = len(self.label_list)
        self.main_indicator = main_indicator
        self.clear()
        self.metric_names = [
            "total_num_gt", 
            "total_num_det",
            "global_accumulative_recall", 
            "hit_str_count", 
            "recall",
            "precision",
            "f_score",
            "seqerr",
            "recall_e2e",
            "precision_e2e",
            "f_score_e2e",
        ]

    def update(self, preds, batch, **kwargs):
        batch_numpy = []
        for item in batch:
            if isinstance(item, ms.Tensor):
                batch_numpy.append(item.numpy())
            else:
                batch_numpy.append(item)
        if self.mode == "A":
            gt_polyons_batch = batch_numpy[0]
            temp_gt_strs_batch = batch_numpy[1][0]
            ignore_tags_batch = batch_numpy[2]
            gt_strs_batch = []

            for temp_list in temp_gt_strs_batch:
                t = ""
                for index in temp_list:
                    if index < self.max_index:
                        t += self.label_list[index]
                gt_strs_batch.append(t)

            for pred, gt_polyons, gt_strs, ignore_tags in zip(
                [preds], gt_polyons_batch, [gt_strs_batch], ignore_tags_batch
            ):
                # prepare gt
                gt_info_list = [
                    {"points": gt_polyon, "text": gt_str, "ignore": ignore_tag}
                    for gt_polyon, gt_str, ignore_tag in zip(
                        gt_polyons, gt_strs, ignore_tags
                    )
                ]
                # prepare det
                e2e_info_list = [
                    {"points": det_polyon, "texts": pred_str}
                    for det_polyon, pred_str in zip(pred["points"], pred["texts"])
                ]
                result = get_socre_A(gt_info_list, e2e_info_list)
                self.results.append(result)
        else:
            img_id = batch[3].asnumpy()[0]
            e2e_info_list = [
                {"points": det_polyon, "texts": pred_str}
                for det_polyon, pred_str in zip(preds["points"], preds["texts"])
            ]
            result = get_socre_B(self.gt_mat_dir, img_id, e2e_info_list)
            self.results.append(result)

    def eval(self):
        metrics = combine_results(self.results)
        self.clear()
        return metrics

    def clear(self):
        self.results = []