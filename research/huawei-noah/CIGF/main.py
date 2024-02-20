import os
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import mindspore as ms
import mindspore
from mindspore import context, nn, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal
import mindspore.ops as ops
from mindspore import ops as P
from mindspore.ops import functional as F
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time
from tqdm import tqdm
from mindspore import ParameterTuple
from mindspore.common.initializer import initializer, XavierNormal


class Model_all(nn.Cell):
    def __init__(self, adjs):
        super(Model_all, self).__init__()
        self.adjs = adjs
        self.allEmbed = NNs.defineParam('Embed0', [args.user + args.item, args.latdim // 2],requires_grad=True, reg=True)
        self.actFunc = 'leakyRelu'
        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        self.allEmbed_dim = ops.Shape()(self.allEmbed)[1]
        self.all_FC = []
        for beh in range(args.behNum):
            for index in range(1,args.gnn_layer):
                self.FC = FC(self.allEmbed_dim,args.behNum, reg=True, useBias=True,
                                        activation=self.actFunc, name='attention_%d_%d'%(beh,index), reuse=True)
                self.all_FC.append(self.FC)
        
        self.metalat111FC = FC(self.allEmbed_dim*2, args.behNum, reg=True, useBias=True,
                            activation='softmax', name='gate111', reuse=True)
        self.testFC = FC(self.allEmbed_dim*2, 1, reg=True, useBias=True,
                            activation='softmax', name='gate111', reuse=True)

    def construct(self, data):
        self.uids = data[0] 
        self.iids = data[1]

        for beh in range(args.behNum):
            ego_embeddings = self.allEmbed
            all_embeddings = [ego_embeddings]
            for index in range(args.gnn_layer):
                if args.multi_graph == False:
                    symm_embeddings = ops.SparseTensorDenseMatmul()(self.adjs[beh].indices, self.adjs[beh].values, self.adjs[beh].shape, all_embeddings[-1])
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = symm_embeddings
                        all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        
                elif args.multi_graph == True:        
                    if index == 0:
                        symm_embeddings = ops.SparseTensorDenseMatmul()(self.adjs[beh].indices, self.adjs[beh].values, self.adjs[beh].shape, all_embeddings[-1])
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        
                    else:
                        atten = self.all_FC[beh*(args.gnn_layer-1)+index-1](self.allEmbed)
                        temp_embeddings = []
                        for inner_beh in range(args.behNum):
                            neighbor_embeddings = ops.SparseTensorDenseMatmul()(self.adjs[inner_beh].indices, self.adjs[inner_beh].values, self.adjs[inner_beh].shape, symm_embeddings)
                            temp_embeddings.append(neighbor_embeddings)
                        all_temp_embeddings = ops.Stack(axis=1)(temp_embeddings)
                        symm_embeddings = ops.ReduceSum(keep_dims=False)(ops.mul(all_temp_embeddings, ops.expand_dims(atten, -1)), axis=1)
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                            

            all_embeddings = ops.AddN()(all_embeddings)
            self.ulat[beh], self.ilat[beh] = ops.split(all_embeddings, [args.user, args.item])
        self.ulat_merge, self.ilat_merge = ops.AddN()(self.ulat), mindspore.ops.AddN()(self.ilat)

        preds_output = []
        # print('args.behNum',args.behNum)
        for src in range(args.behNum):
        
            uids = self.uids[src]
            iids = self.iids[src]
            src_ulat = ops.gather(self.ulat[src], Tensor(np.array(uids),mindspore.int32), 0)
            src_ilat = ops.gather(self.ilat[src], Tensor(np.array(iids),mindspore.int32), 0)

            input=mindspore.ops.Concat(axis=-1)((src_ulat, src_ilat))
            metalat111 = self.metalat111FC(input)
            w1 = ops.reshape(metalat111, (-1, args.behNum, 1))
            
            exper_info = []
            for index in range(args.behNum):
                exper_info.append(
                    ops.gather(self.ulat[index], Tensor(np.array(uids),mindspore.int32), 0) * ops.gather(self.ilat[index], Tensor(np.array(iids),mindspore.int32), 0))
            predEmbed = mindspore.ops.Stack(axis=2)(exper_info)
            sesg_out = mindspore.ops.reshape(predEmbed @ w1, [-1, args.latdim // 2])

            preds = mindspore.ops.Squeeze()(ops.ReduceSum(keep_dims=False)(sesg_out, axis=-1))
            preds_output.append(preds*args.mult)
        return preds_output
        
    def predict(self, data):
        
        uids = data[0] 
        iids = data[1]
        src_ulat = ops.Gather()(self.ulat[-1], Tensor(np.array(uids),mindspore.int32), 0)
        src_ilat = ops.Gather()(self.ilat[-1], Tensor(np.array(iids),mindspore.int32), 0)
        input=mindspore.ops.Concat(axis=-1)((src_ulat, src_ilat))
        # print('ops.Shape()(input)',ops.Shape()(input))
        metalat111 = self.metalat111FC(input)
        # print('ops.Shape()(metalat111)',ops.Shape()(metalat111))
        w1 = ops.reshape(metalat111, (-1, args.behNum, 1))
        
        exper_info = []
        for index in range(args.behNum):
            exper_info.append(
                ops.Gather()(self.ulat[index], Tensor(np.array(uids),mindspore.int32), 0) * ops.Gather()(self.ilat[index], Tensor(np.array(iids),mindspore.int32), 0))
        predEmbed = mindspore.ops.Stack(axis=2)(exper_info)
        sesg_out = mindspore.ops.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = mindspore.ops.Squeeze()(ops.ReduceSum(keep_dims=False)(sesg_out, axis=-1))

        return preds * args.mult


class Loss_All(nn.LossBase):
    '''
    自定义loss函数
    '''
    def __init__(self):
        super(Loss_All, self).__init__("mean")

 
    def construct(self, out, label):
        self.preLoss = 0
        for src in range(args.behNum):
            preds = out[src]
            label_now = label[src]
            self.preLoss += P.ReduceMean(keep_dims=False)(ops.SigmoidCrossEntropyWithLogits()(preds, label_now))
            if src == args.behNum - 1:
                self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        self.loss = self.preLoss + self.regLoss
        return self.loss


class Recommender:
    def __init__(self, handler):
        self.handler = handler
        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        if args.data == 'beibei':
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR45', 'NDCG45', 'HR50', 'NDCG50', 'HR55', 'NDCG55', 'HR60', 'NDCG60', 'HR65', 'NDCG65', 'HR100', 'NDCG100']
        else:
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35', 'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        self.prepareModel()
        log('Model Prepared')
        

        self.loss = Loss_All()
        self.net = Model_all(self.adjs)
        self.loss_net = CustomWithLossCell(self.net, self.loss)
        self.optimizer = mindspore.nn.Adam(self.net.trainable_params(),learning_rate=0.001)
        self.train_net = TrainOneStepCell(self.loss_net, self.optimizer)

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        params_all = list(self.net.get_parameters())
        print('params_all',params_all)
        for m in self.net.parameters_and_names():
            print('m',m)
        # no_conv_params = list(filter(lambda x: 'conv' not in x.name, params_all))
        no_conv_params = list(filter(lambda x: x.requires_grad, params_all))
        print('no_conv_params',no_conv_params)
        globalStep = mindspore.Parameter(Tensor(2, ms.float32), requires_grad = False)
        learningRate = mindspore.nn.ExponentialDecayLR(args.lr, args.decay, args.decay_step, is_stair=True)(globalStep)
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
        train_time = 0
        test_time = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch()
            t1 = time()
            train_time += t1-t0
            print('Train_time',t1-t0,'Total_time',train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test:                  
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3-t2
                print('Test_time',t3-t2,'Total_time',test_time)
                log(self.makePrint('Test', ep, reses, test))                                                                    

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))


    def create_multiple_adj_mat(self, adj_mat):
        def left_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()

    
    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []

        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []

        for i in range(args.behNum):
            R = self.handler.trnMats[i].tolil()
            
            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])
            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))

            
            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats

        for i in range(args.behNum):
            adj = self.final_trnMats[i]
            idx, data, shape = transToLsts(adj, norm=False)
            self.adjs.append(mindspore.SparseTensor(Tensor(idx), Tensor(data, dtype=ms.float32), tuple(shape)))

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))

        self.train_net.set_train()
        for i in tqdm(range(steps)):    
            self.uids, self.iids, self.label_all= [], [], []       
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]
            for beh in range(args.behNum):
                uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
                sampNum = len(uLocs)// 2
                labels_pos = ops.tile(Tensor(np.array([1.0]), mindspore.float32), (sampNum,))
                labels_neg = ops.tile(Tensor(np.array([0.0]), mindspore.float32), (sampNum,))
                labels = ops.Concat(axis = 0)(([labels_pos, labels_neg]))

                self.uids.append(uLocs)
                self.iids.append(iLocs)
                self.label_all.append(labels)

            data = [self.uids,self.iids]
            loss = self.train_net(data, self.label_all)
            epochLoss += loss 
           
        ret = dict()
        ret['Loss'] = epochLoss / steps
        return ret



    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            rdnNegSet = np.random.permutation(negset)[:99]
            locset = np.concatenate((rdnNegSet, np.array([posloc]))) 
            tstLocs[i] = locset
            for j in range(100):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                cur += 1
        return uLocs, iLocs, temTst, tstLocs


    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2 
        
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in tqdm(range(steps)):
            self.uids, self.iids = [], [] 
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]

            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1])
            self.uids.append(uLocs)
            self.iids.append(iLocs)
            data = [self.uids[-1],self.iids[-1]]
            preds = self.net.predict(data).asnumpy()
            hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 100]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
        
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        # saver = tf.train.Saver()
        mindspore.save_checkpoint(model, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        mindspore.load_checkpoint('Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, value_batch, label_batch):
        logits = self._backbone(value_batch)
        loss = self._loss_fn(logits, label_batch)
        return loss


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)
        self.grad_reducer = F.identity

    def construct(self, value_batch, label_batch):
        weights = self.weights
        loss = self.network(value_batch, label_batch)
        grads = self.grad(self.network, weights)(value_batch, label_batch)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
 
if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text

    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    recom = Recommender(handler)
    recom.run()
