import os
import mindspore as ms
from mindspore import dataset as ds
from config import Config
from model import CaptionModel
from read_file import build_data
from evaluation import compute_scores
from module.decoder import greedy_decode, beam_decode

def eval(config, model = None, val_data = None, val_dict = None):

    if model is None:
        model = CaptionModel(config)
        param_dict = ms.load_checkpoint(os.path.join(config.model_save_path, config.ck))
        ms.load_param_into_net(model, param_dict)

        # 读取数据
        print("读取数据")
        ds.config.set_auto_offload(True)
        ds.config.set_enable_autotune(True)
        column_names = ['img', 'caption', 'label', 'img_id']
        val_dict = build_data(config)
        val_data = ds.GeneratorDataset(val_dict, column_names = column_names, shuffle = False)
        val_data = val_data.batch(config.batch_size)
        val_data = val_data.create_dict_iterator()
        print("val data is: ", len(val_dict))
        print("读取数据结束")

    model.set_train(False)

    gts = {}
    res = {}

    decode = greedy_decode if config.decode_method == 'greedy' else beam_decode

    for i, batch in enumerate(val_data):

        img = batch['img']
        caption_index = batch['caption']

        caption_index = caption_index[:, 0, 0].unsqueeze(1)
        caption_index = decode(model, img, caption_index, config)

        pred_str = config.tokenizer.batch_decode(caption_index.tolist(), skip_special_tokens=True)

        bs = img.shape[0]
        for k in range(bs):
            image_id = int(batch['img_id'][k])
            gts[image_id] = val_dict.imgid_to_sentences[image_id]
            res[image_id] = [pred_str[k]]

    score = compute_scores(gts, res)
    print(score[0])

if __name__ == '__main__':
    config = Config(TrainOrVal = 'test')
    with ms._no_grad():
        eval(config)