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
import time
from mindspore import context
from mindspore import save_checkpoint, load_checkpoint, export,load_param_into_net


def calc_label_sim(label_1, label_2):
    #set_trace()
    matmul = ops.MatMul(transpose_a=False, transpose_b=True)
    
    Sim = matmul(label_1,label_2)
    #Sim = label_1.float().mm(label_2.float().t())
    return Sim

# def cos(x, y):
#     return x.mm(y.t())


"""
class calc_loss(nn.LossBase):
    def __init__(self):
        super(calc_loss, self).__init__()
    def construct(self,view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
        #set_trace()
        term1 = ((view1_predict - labels_1) ** 2).sum(1).sqrt().mean() + (
                    (view2_predict - labels_2) ** 2).sum(1).sqrt().mean()

        cos = lambda x, y: x.mm(y.t()) / (
            (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
        theta11 = cos(view1_feature, view1_feature)
        theta12 = cos(view1_feature, view2_feature)
        theta22 = cos(view2_feature, view2_feature)
        Sim11 = calc_label_sim(labels_1, labels_1)
        Sim12 = calc_label_sim(labels_1, labels_2)
        Sim22 = calc_label_sim(labels_2, labels_2)
        term21 = ((1 + ops.Exp(theta11)).log() - Sim11 * theta11).mean()
        term22 = ((1 + ops.Exp(theta12)).log() - Sim12 * theta12).mean()
        term23 = ((1 + ops.Exp(theta22)).log() - Sim22 * theta22).mean()
        term2 = term21 + term22 + term23

        term3 = ((view1_feature - view2_feature) ** 2).sum(1).sqrt().mean()

        im_loss = term1 + alpha * term2 + beta * term3
        return im_loss


def cos(x,y):
    matmul = ops.MatMul()
    sqrt = ops.Sqrt()
    transpose = ops.Transpose()
    
    out= matmul(x,transpose(y))/ (matmul(sqrt((x ** 2).sum(1, keepdim=True)),transpose(sqrt((y ** 2).sum(1, keepdim=True))))).clamp(min=1e-6) / 2.
    return out

"""


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
    cast = ops.Cast()
    sqrt = ops.Sqrt()
    matmul = ops.MatMul(transpose_a=False, transpose_b=True)
    op = ops.ReduceSum(keep_dims=True)
    log = ops.Log()
    exp = ops.Exp()

    #term1 = ((view1_predict-cast(labels_1,ms.float32))**2).sum(1).sqrt().mean() + ((view2_predict-cast(labels_1,ms.float32))**2).sum(1).sqrt().mean()
    term1 = (sqrt(((view1_predict - cast(labels_1, ms.float32)) ** 2).sum(1)).mean() + sqrt(((view2_predict - cast(labels_1, ms.float32)) ** 2).sum(1)).mean())

    #cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.

    cos = lambda x, y: matmul(x,y)/ ops.clip_by_value((matmul(sqrt(op(x ** 2,1)),sqrt(op(y ** 2,1)))),1e-6,100000) / 2.
    
    labels_1=cast(labels_1,ms.float32)
    labels_2=cast(labels_2,ms.float32)
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = cast(calc_label_sim(labels_1, labels_1),ms.float32)
    Sim12 = cast(calc_label_sim(labels_1, labels_2),ms.float32)
    Sim22 = cast(calc_label_sim(labels_2, labels_2),ms.float32)
    #set_trace()
    term21 = (log(1+exp(theta11))- Sim11 * theta11).mean()
    term22 = (log(1+exp(theta12)) - Sim12 * theta12).mean()
    term23 = (log(1+exp(theta22)) - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = sqrt(((view1_feature - view2_feature)**2).sum(1)).mean()

    im_loss = term1 + alpha * term2 + beta * term3
    #im_loss = term1 + term2 + term3
    return im_loss

#封装模型、损失、优化器

class CustomTrainOneStepCell(nn.Cell):

    def __init__(self, network, optimizer):

        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network# 定义前向网络
        self.network.set_grad()# 构建反向网络
        self.optimizer = optimizer# 定义优化器
        self.weights = self.optimizer.parameters# 待更新参数
        self.grad = ops.GradOperation(get_by_list=True) # 反向传播获取梯度

    def construct(self, images,texts,labels):
        #set_trace()
        #view1_feature, view2_feature, view1_predict, view2_predict = self.network(image,text)

        #labels=data['labels']
        alpha = 5e-3
        beta = 1e-1
        #set_trace()
        loss=self.network(images,texts,labels)
        grads = self.grad(self.network, self.weights)(images,texts,labels)# 进行反向传播，计算梯度
        loss = F.depend(loss, self.optimizer(grads))# 使用优化器更新权重参数

        return loss

class MyWithLossCell(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, images,texts,labels):
        """连接前向网络和损失函数"""

        imgs = ms.Tensor(data['images'])
        txts = ms.Tensor(data['texts'])

        view1_feature, view2_feature, view1_predict, view2_predict = self.backbone(imgs,txts)
        alpha = 1e-3
        beta = 1e-1
        return self.loss_fn(view1_feature, view2_feature, view1_predict, view2_predict,labels, labels, alpha, beta)

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone
"""
class MyTrainStep(nn.TrainOneStepCell):
   

    def __init__(self, network, optimizer):
      
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self,imgs,txts,labels,alpha,beta):
      
        weights = self.weights
        loss = self.network(imgs,txts)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
"""
class GradWrap(nn.Cell):
    
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(filter(lambda x: x.requires_grad,
            network.get_parameters()))
        self.grad=C.GradOperation(get_by_list=True)
    # 注释2
    def construct(self,data,labels):
        weights = self.weights
        grads = self.grad(self.network, weights)(data,labels)
        return grads

if __name__ == '__main__':

    x1 = Tensor(np.array([[1, 1, 1, 1, 1, 1,2, 2, 2, 2, 2, 2,3, 3, 3, 3, 3, 3],
                         [4, 4, 4, 4, 4, 4,5, 5, 5, 5, 5, 5,6, 6, 6, 6, 6, 6],
                         [7, 7, 7, 7, 7, 7,8, 8, 8, 8, 8, 8,9, 9, 9, 9, 9, 9]]), ms.float32)
    x2 = Tensor(np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
                          [4, 4, 4, 4, 4, 4, 7,7,7,7,8, 5, 6, 6, 6, 6, 6, 6],
                          [7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9]]), ms.float32)
    x3=Tensor(np.array([[1, 1, 1, 1, 1],
                          [4, 4, 4, 4, 4],
                          [7, 7, 7, 7, 7]]), ms.float32)
    x4 = Tensor(np.array([[1, 1, 1, 1, 1],
                          [4, 4,3,3,3],
                          [7, 7, 7, 7, 7]]), ms.float32)
    x5 = Tensor(np.array([[0,0,0,0,0],
                          [0,0,0,0,1],
                          [0,0,0,0,0]]), ms.float32)
    x6 = Tensor(np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0]]), ms.float32)


    context.set_context(mode=context.PYNATIVE_MODE,device_target="Ascend")
    dataset = 'pascal'
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 500
    batch_size = 50
    lr = 5e-5
    #lr = 1e-4
    betas = (0.4, 0.999)
    weight_decay = 0
    epoch_size=300

    print('...Data loading is beginning...')

    train_data_loader,test_data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')

    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class'])
    #param_dict = load_checkpoint("./004.ckpt")
    #load_param_into_net(model_ft, param_dict)
    params_to_update = model_ft.trainable_params()
    #params_to_update = load_checkpoint("./004.ckpt")
    optimizer = nn.Adam(params_to_update, learning_rate=lr,beta1=0.6,beta2=0.999,weight_decay=0.0001)
    print('...Training is beginning...')

    #loss_opeartion = nn.WithLossCell(model_ft, calc_loss)
    #train_network = GradWrap(loss_opeartion)
    #calc_loss=calc_loss()
    model=MyWithLossCell(model_ft, calc_loss)
    train_network=CustomTrainOneStepCell(model,optimizer)
    best_acc = 0.0
    for i in range(epoch_size):
        for phase in ['train', 'test']:
            if phase == 'train':
                clock0=time.time()
                # Set model to training mode
                train_network.set_train()
                running_loss = 0.0
               # set_trace()
                #for imgs, txts, labels in data_loader[phase].create_dict_iterator():

                for data in train_data_loader.create_dict_iterator():
                    #set_trace()
                    loss = train_network(data['images'],data['texts'],data['labels'])

                    #optimizer(grads)
                    #view1_feature, view2_feature, view1_predict, view2_predict = model_ft(imgs, txts)

                    #loss = calc_loss(view1_feature, view2_feature, view1_predict,view2_predict, labels, labels, alpha, beta)

                    running_loss += loss
                #set_trace()
                epoch_loss = running_loss / 800
                clock1=time.time()
                print("batchtime:"+str(clock1-clock0)+"seconds")
            else:
                # Set model to evaluate mode
                train_network.set_train(False)
                t_imgs, t_txts, t_labels = [], [], []
                for imgs, txts, labels in test_data_loader:
                    t_view1_feature, t_view2_feature, _, _ = model_ft(imgs, txts)
                    t_imgs.append(t_view1_feature.asnumpy())
                    t_txts.append(t_view2_feature.asnumpy())
                    t_labels.append(labels.asnumpy())
                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels).argmax(1)
                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

                #set_trace()
                #print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
                print(phase,epoch_loss,img2text, txt2img)

            if phase == 'test' and img2text>0.703 and txt2img>0.716 :
                
                print(img2text, txt2img)
                if (img2text + txt2img) / 2. > best_acc:
                    best_acc = (img2text + txt2img) / 2
                    print('saved')
                    clock2=time.time()
                    print("time:"+str(clock2-clock0)+"seconds")
                    save_checkpoint(model_ft, "./linear.ckpt") 


                



