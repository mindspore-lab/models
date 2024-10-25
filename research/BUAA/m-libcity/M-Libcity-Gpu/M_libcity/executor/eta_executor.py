import json
import os
import time

import mindspore
from mindspore import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint, Callback

from executor.traffic_state_executor import TrafficStateExecutor
import numpy as np


class ETAExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.output_pred = config.get("output_pred", True)
        self.output_dim = None
        self._scalar = None

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        self.model.eval()

        y_truths = []
        y_preds = []
        test_pred = {}
        for batch in test_dataloader.create_dict_iterator():
            output = self.model.predict(*batch.values())
            y_true = batch['time']
            y_pred = output
            y_truths.append(y_true.asnumpy())
            y_preds.append(y_pred.asnumpy())
            if self.output_pred:
                for i in range(y_pred.shape[0]):
                    uid = batch['uid'][i].astype(mindspore.int32).asnumpy()[0]
                    if uid not in test_pred:
                        test_pred[str(uid)] = {}
                    traj_id = batch['traj_id'][i].astype(mindspore.int32).asnumpy()[0]
                    current_longi = batch['current_longi'][i].asnumpy()
                    current_lati = batch['current_lati'][i].asnumpy()
                    coordinates = []
                    for longi, lati in zip(current_longi, current_lati):
                        coordinates.append((float(longi), float(lati)))
                    traj_len = batch['traj_len'][i].astype(mindspore.int32).asnumpy()[0]
                    start_timestamp = batch['start_timestamp'][i].astype(mindspore.int32).asnumpy()[0]
                    outputs = {}
                    outputs['coordinates'] = coordinates[:traj_len]
                    outputs['start_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime(start_timestamp))
                    outputs['truth'] = float(y_true.asnumpy()[i][0])
                    outputs['prediction'] = float(y_pred.asnumpy()[i][0])
                    test_pred[str(uid)][str(traj_id)] = outputs
        self.model.train()
        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
        if self.output_pred:
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.json'
            with open(os.path.join(self.evaluate_res_dir, filename), 'w') as f:
                json.dump(test_pred, f)
        self.evaluator.clear()
        self.evaluator.collect({'y_true': mindspore.Tensor(y_truths), 'y_pred': mindspore.Tensor(y_preds)})
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result
