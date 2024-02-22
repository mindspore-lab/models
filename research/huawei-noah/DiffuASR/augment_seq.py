# -*- encoding: utf-8 -*-
# here put the import lib
import os
import argparse
import torch
from tqdm import tqdm

from generators.augment_generator import AugmentGenerator
from models.Bert4Rec import Bert4Rec
from utils.utils import set_seed, load_pretrained_model
from utils.logger import AugLogger

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='sasrec',
                    choices=['bert4rec', 'sasrec_reverse', 'random', 'random_seq'], 
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="ml1m", 
                    choices=['ml1m', 'beauty', 'steam', 'video', 'yelp'], 
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="./saved/",
                    help="the path that pretrained model saved in")


# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")

# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

# Augmentation Parameters
parser.add_argument("--aug_num",
                    default=10,
                    type=int,
                    help="the item number of augmentation")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--no_discrete_user",
                    default=False,
                    action="store_true",
                    help="save augment for not discrete users")

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
pretrain_dir_pre = os.path.join("./saved/", args.dataset)
args.pretrain_dir = os.path.join(pretrain_dir_pre, args.pretrain_dir)


def main():

    log_manager = AugLogger(args)  # initialize the log manager
    logger = log_manager.get_logger()    # get the logger

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    # generator is used to manage dataset
    generator = AugmentGenerator(args, logger, device)
    user_num, item_num = generator.get_user_item_num()
    aug_loader = generator.make_augmentloader()

    logger.info("Loading Model: " + args.model_name)
    if args.model_name == 'bert4rec':
        model = Bert4Rec(user_num, item_num, device, args)
    else:
        raise ValueError
    
    if (not args.model_name == 'random') & (not args.model_name == 'random_seq'):
        model = load_pretrained_model(args.pretrain_dir, model, logger, device=device)
    model.to(device)
    model.eval()
    aug_data = []

    for batch in tqdm(aug_loader):

        batch = tuple(t.to(device) for t in batch)
        seq, positions = batch
        seq, positions = seq.long(), positions.long()
        item_indicies = torch.arange(1, item_num+1)    # (1, item_num) to get the item embedding matrix
        item_indicies = item_indicies.repeat(seq.shape[0], 1)   # (bs, item_num)
        item_indicies = item_indicies.to(device).long()
        per_aug_data = torch.empty(0).to(device)

        for _ in range(args.aug_num):
            
            with torch.no_grad():

                logits = - model.predict(seq, item_indicies, positions)

            #aug_item = torch.argsort(logits, descending=True)[:, 0]   # return the index of max score
            aug_item = torch.argsort(logits, descending=False)[:, 0]   # return the index of max score
            aug_item = aug_item + 1
            aug_item = aug_item.unsqueeze(1)    # (bs, 1)
            per_aug_data = torch.cat([aug_item, per_aug_data], dim=1)  # [..., n-3, n-2, n-1]

            # get the next step input
            seq = torch.cat([seq, aug_item], dim=1)[:, 1:]
            positions = torch.cat([positions, (positions[:, -1] + 1).unsqueeze(1)], dim=1)[:, 1:]

        aug_data.append(per_aug_data)

    aug_data = torch.cat(aug_data, dim=0)

    aug_data = aug_data.detach().cpu().numpy()
    generator.save_aug(aug_data)

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()





