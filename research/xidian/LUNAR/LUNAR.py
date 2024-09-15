import os
from sklearn.metrics import roc_auc_score
import utils
import variables as var
from copy import deepcopy
import mindspore
import mindspore.ops as ops
from mindspore import Tensor, nn, set_seed
# Message passing scheme

class GNN1(nn.Cell):
    def __init__(self, k):
        super(GNN1, self).__init__()
        self.k = k
        self.hidden_size = 256
        self.network = nn.SequentialCell([
            nn.Dense(k, self.hidden_size),
            nn.Tanh(),
            nn.Dense(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dense(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dense(self.hidden_size, 1),
            nn.Sigmoid()
        ])
        self.reshape = ops.Reshape()
        self.gather = ops.Gather()
        self.concat = ops.Concat(axis=1)
    
    def construct(self, x, edge_index, edge_attr):
        row, col = edge_index
        # 消息传递
        messages = self.message(x[row], x[col], edge_attr)
        # 聚合
        out = self.aggregate(messages, col)
        return out

    def message(self, x_i, x_j, edge_attr):
        # 消息是边权重
        return edge_attr

    def aggregate(self, inputs, index):
        # 聚合 k 个消息
        out = self.reshape(inputs, (-1, self.k))
        # 通过网络
        out = self.network(out)
        return out
# GNN

class GNN(nn.Cell):
    def __init__(self, k):
        super(GNN, self).__init__()
        self.k = k
        self.L1 = GNN1(self.k)
    
    def construct(self, data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = ops.Squeeze(1)(out)
        return out

class LossNet(nn.Cell):
    def __init__(self, net, criterion):
        super().__init__()
        self.net = net
        self.loss_fn = criterion
    
    def construct(self,data, train_mask):
        out = self.net(data)
        loss = self.loss_fn(out[train_mask == 1],data.y[train_mask == 1]).sum()
        return loss

def run(train_x,train_y,val_x,val_y,test_x,test_y,dataset,seed,k,samples,train_new_model):  

    # loss function
    criterion = nn.MSELoss(reduction = 'none')    

    # path to save model parameters
    model_path = './saved_models/%s/%d/net_%d.ckpt' %(dataset,k,seed)
    if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path)) 
    
    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon)
    data = utils.build_graph(x, y, dist, idx)
        
    # data = data.to(var.device)                                                                    
    # torch.manual_seed(seed)
    set_seed(seed)
    # net = GNN(k).to(var.device)
    net = GNN(k)
   
    if train_new_model == True:
      
        # optimizer = optim.Adam(net.parameters(), lr = var.lr, weight_decay = var.wd)
        optimizer = nn.Adam(net.trainable_params(), learning_rate=var.lr, weight_decay = var.wd)
        # with torch.no_grad():
            
        #     net.eval()
        #     out = net(data)
        #     loss = criterion(out,data.y)

        #     val_loss = loss[val_mask == 1].mean()
        #     val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())

        #     best_val_score = 0
        net.set_train(False)
        out = net(data)
        loss = criterion(out,data.y)
        # print(loss)
        val_loss = loss[val_mask == 1].mean()
        val_score = roc_auc_score(data.y[val_mask==1].asnumpy(),out[val_mask==1].asnumpy())
        best_val_score = 0

        loss = LossNet(net,criterion)
        trian_net = nn.TrainOneStepCell(loss, optimizer)
        # training
        for epoch in range(var.n_epochs):
            # net.set_train() 
            trian_net.set_train() 
            loss = trian_net(data,train_mask)

            net.set_train(False)
            out = net(data)
            loss = criterion(out,data.y)
            val_loss = loss[val_mask == 1].mean()
            if epoch % 40 ==0:
                print(val_loss)
            val_score = roc_auc_score(data.y[val_mask==1].asnumpy(),out[val_mask==1].asnumpy())
            if val_score >= best_val_score:
                # save model parameters
                # best_dict = {'epoch': epoch,
                #            'model_state_dict': deepcopy(net.state_dict()),
                #            'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                #            'val_loss': val_loss,
                #            'val_score': val_score,
                #            'k': k,}
                mindspore.save_checkpoint(net,model_path)
                best_val_score = val_score
                    
       
        # load best model
        # param_dict = mindspore.load_checkpoint("model.ckpt")
        # net.load_state_dict(best_dict['model_state_dict'])
        
    # if not training a new model, load the saved model
    if train_new_model == False:
        param_dict = mindspore.load_checkpoint(model_path)
        param_not_load, _ = mindspore.load_param_into_net(net, param_dict)
        print(param_not_load)
        # load_dict = torch.load(model_path)
        # net.load_state_dict(load_dict['model_state_dict'])
 
    # testing
    net.set_train(False)
    out = net(data)
    loss = criterion(out,data.y)
    # with torch.no_grad():
    #     net.eval()
    #     out = net(data)
    #     loss = criterion(out,data.y)
       
    # return output for test points
    return out[test_mask==1].asnumpy()
