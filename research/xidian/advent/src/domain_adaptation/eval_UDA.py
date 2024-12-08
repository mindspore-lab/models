# Copyright 2023 Xidian University
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


import os
import numpy as np
from tqdm import tqdm

from src.utils.func import per_class_iu, fast_hist
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.nets.train_model import WithEvalCellSrc
from PIL import Image


def colorize_mask(mask,palette:np.ndarray):
    palette = list(palette)
    palette = palette + [255 for _ in range((256*3 - len(palette)))]
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def evaluate_domain_adaptation(model, test_loader, cfg,
                               verbose=True,viz=True):

    # eval
    mIoU = eval_model(cfg,model,test_loader,verbose=verbose,viz=viz)
    return mIoU



def eval_model(cfg, model, test_loader, verbose=True,viz=False):
    model.set_train(False)

    label_names = cfg.TEST.LABEL_NAMES
    # eval
    eval_net = WithEvalCellSrc(model)
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))

    index = 0
    for data in tqdm(test_loader.create_dict_iterator(),total=len(test_loader)):
        index = index + 1
        image, label,name = data['data'], data['label'],data['name'].asnumpy()
        pred_main = eval_net(image, label.shape[-2:])[-1]
        output = pred_main.asnumpy()
        output = output.transpose(1, 2, 0)
        output = np.argmax(output, axis=2)
        label = label.asnumpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if verbose and index > 0 and index % 100 == 0:
            print('{:d} : {:0.2f}'.format(
                index, 100 * np.nanmean(per_class_iu(hist))))
        if viz:
            save_path = os.path.join(cfg.TEST.OUTPUT_DIR,name[0])
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            output_viz = colorize_mask(output.squeeze(),np.concatenate(cfg.TEST.LABEL_PALETTE,axis=0))
            output_viz.save(save_path)
            print(f"cnt : {index}")

    inters_over_union_classes = per_class_iu(hist)
    print(f'IoUs: {[(label_name,round(x*100,2)) for label_name, x in zip(label_names,inters_over_union_classes)]}')
    computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
    print('Current mIoU:', computed_miou)
    return computed_miou


def load_checkpoint_for_evaluation(model, checkpoint):
    saved_state_dict = load_checkpoint(checkpoint)
    load_param_into_net(model, saved_state_dict)
    model.set_train(False)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
