import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.ops import value_and_grad
from mindspore import ms_function
from mindspore.common.initializer import initializer, HeNormal, HeUniform
import mindcv

from base.base_dataloader import return_dataset
from base.base_net import Predictor_deep


class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling, same as torch.nn.AdaptiveAvgPool2d when output shape is 1
    """

    def __init__(self, keep_dims: bool = False) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def construct(self, x):
        x = ops.mean(x, axis=(2, 3), keep_dims=self.keep_dims)
        return x


class cellCls(nn.Cell):
    """cal the classify loss of G and clssifier"""
    def __init__(self, net_G, net_cls, cls_loss):
        super(cellCls, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.pool = GlobalAvgPooling()
        self.net_cls = net_cls
        self.cls_loss = cls_loss

    def construct(self, x_labeled, target):
        feat_labeled = self.net_G.forward_features(x_labeled)
        feat_labeled = self.pool(feat_labeled)

        out_labeled = self.net_cls(feat_labeled)

        loss_cls = self.cls_loss(out_labeled, target)
        return loss_cls
    
class cellAdv(nn.Cell):
    """cal the adv loss"""
    def __init__(self, net_G, net_cls):
        super(cellAdv, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.pool = GlobalAvgPooling()
        self.net_cls = net_cls
        self.softmax = nn.Softmax()

        
    def construct(self, x_unlabeled, w_adent):
        feat_unlabeled = self.net_G.forward_features(x_unlabeled)
        feat_unlabeled = self.pool(feat_unlabeled)

        out_unlabeled = self.net_cls(feat_unlabeled)

        out_unlabeled = self.softmax(out_unlabeled)
        loss_adent = w_adent * ops.mean(ops.reduce_sum(out_unlabeled * ops.log(out_unlabeled + 1e-5), -1))
        return loss_adent

class TrainOneStep(nn.Cell):
    """自定义训练网络"""

    def __init__(self, cell_cls, cell_adv, optim, optim_G, optim_F, logger):
        super(TrainOneStep, self).__init__(auto_prefix=True)
        self.cell_cls = cell_cls
        self.cell_cls.set_grad()
        self.cell_adv = cell_adv 
        self.cell_adv.set_grad()

        self.optim = optim
        self.weight = self.optim.parameters
        self.cls_gradFN = value_and_grad(cell_cls, grad_position=None, weights=self.weight)

        self.optim_G = optim_G
        self.weight_G = self.optim_G.parameters
        self.advG_gradFN = value_and_grad(cell_adv, grad_position=None, weights=self.weight_G)
        self.optim_F = optim_F
        self.weight_F = self.optim_F.parameters
        self.advF_gradFN = value_and_grad(cell_adv, grad_position=None, weights=self.weight_F)


        self.logger = logger

    def construct(self, x_labeled, target, x_unlabeled, w_adent):
        loss_cls, grad = self.cls_gradFN(x_labeled, target)
        self.optim(grad)

        loss_advF, grad_F =self.advF_gradFN(x_unlabeled, w_adent)
        loss_advG, grad_G = self.advG_gradFN(x_unlabeled, -w_adent)
        self.optim_F(grad_F)
        self.optim_G(grad_G)
        return loss_cls, loss_advF

    
class inferMME(nn.Cell):

    def __init__(self, net_G, net_cls):
        super(inferMME, self).__init__(auto_prefix=False)
        self.net_G = net_G
        self.net_cls = net_cls
        self.net_G.set_train(False)
        self.net_cls.set_train(False)
        self.pool = GlobalAvgPooling()

    def construct(self, data, label):
        feat = self.net_G.forward_features(data)
        feat = self.pool(feat)
        outputs = self.net_cls(feat)
        return outputs, label
    

class Train:
    '''训练流程'''
    
    def __init__(self, args, logger):
        self.args = args
        
        # sld tld means source label dataset, target label dataset
        # tvd tud  ttd means target val/unlabeled/test dataset
#         self.sld, self.tld, self.tvd, self.tud, self.ttd, cls_list = return_dataset(args)
        self.source_dataset, self.target_dataset, self.target_dataset_val, self.target_dataset_unl, self.target_dataset_test, cls_list = return_dataset(args)
        self.log = logger

        if args['net'] == 'vgg16':
            self.net_G = mindcv.vgg16(pretrained=True)
            inc = 4096
            bs = 24
        elif args['net'] == 'resnet34':
            self.net_G = mindcv.resnet34(pretrained=True)
            inc = 512 
            bs = 24
        else:
            assert 0, '{} is not preset architure'.format(self.net)

        # source dataset with label    
        Generatorsource_dataset = ds.GeneratorDataset(source=self.source_dataset, column_names=["data", "label", "path"], num_parallel_workers=8)
        self.sld = Generatorsource_dataset.batch(bs, False).shuffle(256)

        # target dataset
        Generatortarget_dataset = ds.GeneratorDataset(source=self.target_dataset, column_names=["data", "label", "path"], num_parallel_workers=8)
        self.tld = Generatortarget_dataset.batch(min(bs, len(self.target_dataset)), False).shuffle(256)

        # target val dataset
        Generatortarget_dataset_val = ds.GeneratorDataset(source=self.target_dataset_val, column_names=["data", "label", "path"], num_parallel_workers=8)
        self.tvd = Generatortarget_dataset_val.batch(min(bs, len(self.target_dataset_val)), False).shuffle(256)

        # target unlabel dataset
        Generatortarget_dataset_unl = ds.GeneratorDataset(source=self.target_dataset_unl, column_names=["data", "label", "path"], num_parallel_workers=8)
        self.tud = Generatortarget_dataset_unl.batch(bs*2, False).shuffle(256)

        # target test dataset
        Generatortarget_dataset_test = ds.GeneratorDataset(source=self.target_dataset_test, column_names=["data", "label", "path"], num_parallel_workers=8)
        self.ttd = Generatortarget_dataset_test.batch(bs*2, False).shuffle(256)

        # classifier
        self.net_cls = Predictor_deep(num_class=len(cls_list), inc=inc, grl_coeff=1)

        # loss func 
        self.loss_cls = nn.CrossEntropyLoss()

        self.cellCls = cellCls(self.net_G, self.net_cls, self.loss_cls)
        self.cellAdv = cellAdv(self.net_G, self.net_cls)
        G_cls_params = []
        G_params = []
        cls_params = []
        self.cls_lr_lis = self.lr_schedule()
        self.convbn_lr_lis = self.lr_schedule(lr_multi=0.1)
        for param in self.net_G.trainable_params():
            if 'classifier' not in param.name:
                G_params.append(param)
            else:
                G_cls_params.append(param)
        G_dic = [{'params':G_params, 'lr':self.convbn_lr_lis},
                {'params':G_cls_params, 'lr':self.cls_lr_lis}]

        for param in self.net_cls.trainable_params():
                cls_params.append(param)
        cls_dic = [{'params':cls_params, 'lr':self.cls_lr_lis}]

        param_dic = [{'params':G_params, 'lr':self.convbn_lr_lis},
                {'params':G_cls_params, 'lr':self.cls_lr_lis},
                {'params':cls_params, 'lr':self.cls_lr_lis}]

        self.optim_G = nn.SGD(G_dic, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.optim_F = nn.SGD(cls_dic, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.optim = nn.SGD(param_dic, momentum=0.9, weight_decay=0.0005, nesterov=True)
        
        self.TrainMME = TrainOneStep(self.cellCls, self.cellAdv, self.optim, self.optim_G, self.optim_F, self.log)
        self.inferMME = inferMME(self.net_G, self.net_cls)
        
        self.cat_ops = ops.Concat()
        self.squeeze_ops = ops.Squeeze()
        self.data_iter_val = self.tvd.create_tuple_iterator()
        self.data_iter_test = self.ttd.create_tuple_iterator()

    def train(self):
        data_iter_s = self.sld.create_tuple_iterator()
        #  tld means target labeled dataset      
        data_iter_t = self.tld.create_tuple_iterator()
        #  sl means target unlabeled dataset       
        data_iter_t_unl = self.tud.create_tuple_iterator()
        
        if self.args['resume']:
            init_step = self.args['resumeStep'] 
            self.load_model()
        else:
            init_step = 0
        
        best_val_acc = 0
        best_test_acc = 0
        for step in range(init_step, self.args['steps']):
            self.TrainMME.set_train(True)
            #  sld means source labeled dataset       
            data_t = next(data_iter_t)
            data_tu = next(data_iter_t_unl)
            data_s = next(data_iter_s)
            data_labeled = self.squeeze_ops(self.cat_ops((data_t[0], data_s[0])))
            target = self.cat_ops((data_t[1], data_s[1]))
            
            data_unlabeled = self.squeeze_ops(data_tu[0])
            #pred, label = self.inferMME(data_labeled, target)
            loss_cls, loss_adv = self.TrainMME(data_labeled, target, data_unlabeled, w_adent=0.1)
            loss_cls = loss_cls.mean()
            loss_adv = loss_adv.mean()

            self.log.info('step-{}-training metric--cls-lr:{:.4f}--conv-lr{:.4f}--loss_cls:{:.4f}--loss_adent_G:{:.4f}--val-acc:{:.4f}--test-acc{:.4f}'.format(
                step,
                self.cls_lr_lis[step],
                self.convbn_lr_lis[step],
                float(loss_cls), 
                float(loss_adv),
                float(best_val_acc),  
                float(best_test_acc)))  
            
            if (step+1) % self.args['val_interval'] == 0:
                val_acc = self.val(step)                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best = True
                else:
                    best = False
                self.save_model(step, best)
                test_acc = self.test()
                if best:
                    best_test_acc = test_acc


    def val(self, step):
        data_num = len(self.target_dataset_val)
        sample_num = 0
        correct_total_num = 0
        self.inferMME.set_train(False)
        for itx, (img, target, path) in enumerate(self.data_iter_val):
            img = self.squeeze_ops(img)
            pred, _ = self.inferMME(img, target)
            acc, correct_num, batch_size = self.cal_acc(pred, target)
            sample_num += batch_size
            correct_total_num += correct_num
            if sample_num>=data_num:
                break
        acc = correct_total_num/sample_num
        val_str = 'val dataset performance--------acc:{:.4f}-----------'.format(float(acc))
        self.log.info(val_str)
        return acc
        
    def test(self):
        data_num = len(self.target_dataset_test)
        sample_num = 0
        correct_total_num = 0
        self.inferMME.set_train(False)
        for itx, (img, target, path) in enumerate(self.data_iter_test):
            img = self.squeeze_ops(img)
            pred, _ = self.inferMME(img, target)
            acc, correct_num, batch_size = self.cal_acc(pred, target)
            sample_num += batch_size
            correct_total_num += correct_num
            if sample_num>=data_num:
                break
        acc = correct_total_num/sample_num
        test_str = 'test dataset performance--------acc:{:.4f}-----------'.format(float(acc))
        self.log.info(test_str)
        return acc

                
    def save_model(self, step, best=False):
        save_path = self.args['save_path']
        #ms.save_checkpoint(self.net_G, "{}/S{}_G.ckpt".format(save_path, step))
        #ms.save_checkpoint(self.net_cls, "{}/S{}_cls.ckpt".format(save_path, step))
        if best:
            ms.save_checkpoint(self.net_G, "{}/best_G.ckpt".format(save_path))
            ms.save_checkpoint(self.net_cls, "{}/best_cls.ckpt".format(save_path))
            self.log.info('save best ckpt to {}'.format(save_path))

    
    def load_model(self, step=None):
        if not step:
            loadName = 'best'
        else:
            loadName = 'S{}'.format(step)
        save_path = self.args['save_path']
        G_param_dict = ms.load_checkpoint("{}/{}_G.ckpt".format(save_path, loadName))
        G_param_not_load = ms.load_param_into_net(self.net_G, G_param_dict)
 
        cls_param_dict = ms.load_checkpoint("{}/{}_cls.ckpt".format(save_path, loadName))
        cls_param_not_load = ms.load_param_into_net(self.net_cls, cls_param_dict)       
        
    def cal_acc(self, pred, target):
        pred = ops.Argmax(output_type=ms.int32)(pred)
        batch_num = target.shape[0]
        x = ms.numpy.full(batch_num, 1)
        y = ms.numpy.full(batch_num, 0)
        correct_vector = ms.numpy.where(pred == target, x, y)
        correct_num = ops.reduce_sum(correct_vector.astype(ms.float32))
        acc = correct_num/batch_num
        return acc, correct_num, batch_num
    
    def lr_schedule(self, lr_multi=1, gamma=0.0001, power=0.75):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = []
        for iter_num in range(self.args['steps']):
            lr_step = self.args['lr_init'] * (1 + gamma * iter_num) ** (- power)
            lr.append(lr_step * lr_multi)
        return lr