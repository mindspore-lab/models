import numpy as np
import time
import utils
import variables as var
from sklearn.metrics import roc_auc_score 
import LUNAR
import argparse
import mindspore as ms
def main(args):
    for seed in [0,1,2,3,4]:
        print("Running trial with random seed = %d" %seed)
        #load dataset (without negative samples)
        train_x, train_y, val_x, val_y, test_x, test_y = utils.load_dataset(args.dataset,seed)
        start = time.time()
        test_out = LUNAR.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,seed,args.k,args.samples,args.train_new_model)
        end = time.time()     
        test_y = test_y.asnumpy()
        # test_out = test_out.asnumpy()
        score = 100*roc_auc_score(test_y, test_out)

        print('Dataset: %s \t Samples: %s \t k: %d \t Score: %.4f \t Runtime: %.2f seconds' %(args.dataset,args.samples,args.k,score,(end-start)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'HRSS')
    parser.add_argument("--samples", type = str, default = 'MIXED', help = 'Type of negative samples for training')
    parser.add_argument("--k", type = int, default = 100)
    parser.add_argument("--train_new_model", default=True,action="store_true", help = 'Train a new model vs. load existing model')
    args = parser.parse_args()

    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=5)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU')
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    main(args)
    