import mindspore as ms
#from mindvision.engine.callback import ValAccMonitor
#from mindspore.train.callback import TimeMonitor
from mindspore import nn,ops,Tensor
from load_data import get_loader
from model import IDCM_NN
from load_data import get_loader
from evaluate import fx_calc_map_label
from pdb import set_trace
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import numpy as np
from mindspore import context
from mindspore import save_checkpoint, load_checkpoint, export,load_param_into_net




if __name__ == '__main__':

 

    context.set_context(mode=context.GRAPH_MODE,device_target="Ascend")
    dataset = 'pascal'
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 500
    batch_size = 100
    lr = 5e-5
    betas = (0.4, 0.999)
    weight_decay = 0
    epoch_size=300

    print('...Data loading is beginning...')

    train_data_loader,test_data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')

    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],        output_dim=input_data_par['num_class'])
    param_dict = load_checkpoint("./004.ckpt")
    load_param_into_net(model_ft, param_dict)
    input1 = Tensor(np.random.uniform(0.0, 1.0, size=[100,4096]).astype(np.float32))
    input2 = Tensor(np.random.uniform(0.0, 1.0, size=[100,300]).astype(np.float32))
   
    export(model_ft,input1,input2,file_name='bestmodel', file_format='MINDIR')
    t_imgs, t_txts, t_labels = [], [], []
    for imgs, txts, labels in test_data_loader:
        set_trace()
        t_view1_feature, t_view2_feature, _, _ = model_ft(imgs, txts)
        t_imgs.append(t_view1_feature.asnumpy())
        t_txts.append(t_view2_feature.asnumpy())
        t_labels.append(labels.asnumpy())
        t_imgs = np.concatenate(t_imgs)
        t_txts = np.concatenate(t_txts)
        t_labels = np.concatenate(t_labels).argmax(1)
        img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
        txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)               
   
    print(img2text, txt2img)

            

                



