# -*- encoding: utf-8 -*-
# here put the import lib
import os
import argparse
import mindspore
from tqdm import tqdm

from generators.diffusion_generator import DiffusionGenerator
from trainers.diffusion_trainer import DiffusionTrainer

from utils.utils import set_seed, load_pretrained_model
from utils.logger import Logger

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='diffusion',
                    choices=['diffusion'], 
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=['yelp'], 
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--guide_model",
                    default="none",
                    type=str,
                    choices=["none", "cg", "cf",],   # cg = classifier guide, cf = classifier free, control = controlnet
                    help="the type of guided model")
parser.add_argument("--noise_model",
                    default="unet",
                    choices=["unet"],
                    type=str,
                    help="the noise prediction model")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="diffusion",
                    help="the path that pretrained model saved in")
parser.add_argument("--demo",
                    default=False,
                    action="store_true",
                    help="whether use the demo dataset")
parser.add_argument("--train",
                    default=False,
                    action="store_true",
                    help="whether run diffusion training")
parser.add_argument("--pretrain_item",
                    default=False,
                    action="store_true",
                    help="whether load the pretrained item embedding")
parser.add_argument("--freeze_item",
                    default=False,
                    action="store_true",
                    help="whether freeze the item embedding during diffusion training")
parser.add_argument("--rec_path",
                    default="bert4rec/normal",
                    type=str,
                    help="the path of well-trained recommendation model")
parser.add_argument("--guide_type",
                    default="item",
                    type=str,
                    choices=["item", "cond", "seq", "bpr"],
                    help="the type of classifier. 'item' uses the first item for guide; \
                    'guide' uses the condition vector for guide. \
                    'seq' uses all items in original sequences for guidance")
parser.add_argument("--classifier_scale",
                    default=1,
                    type=float,
                    help="the scale for classifier guided model")
parser.add_argument("--pref",
                    default=False,
                    action="store_true",
                    help="whether use a preference model")
parser.add_argument("--rounding",
                    default=False,
                    action="store_true",
                    help="whether add rounding loss")
parser.add_argument("--rounding_scale",
                    default=1,
                    type=float,
                    help="the weight for rounding loss")
parser.add_argument("--simple",
                    default=False,
                    action="store_true",
                    help="whether use the simple loss")
parser.add_argument("--clamp",
                    default=False,
                    action="store_true",
                    help="whether use the clamp")
parser.add_argument("--clamp_step",
                    default=10000,
                    type=int,
                    help="the timestep to start using the clamp")

parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")

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
                    default=10,
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

# Other parameters
parser.add_argument("--train_batch_size",
                    default=512,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=100,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
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
                    default=8,
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

# Parameters for Diffusion Model
parser.add_argument("--num_diffusion_timesteps",
                    default=1000,
                    type=int,
                    help="the number of timesteps")
parser.add_argument("--guide",
                    default=False,
                    action="store_true",
                    help="whether add guide vector to the diffusion model")
parser.add_argument("--dm_scheduler",
                    default="linear",
                    choices=["linear", "quad", "sigmoid", "cosine"],
                    type=str,
                    help="the type of scheduler for diffusion steps")


args = parser.parse_args()
set_seed(args.seed) # fix the random seed
base_save_path = os.path.join('./saved/', args.dataset)
#base_save_path = os.path.join(base_save_path, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path
args.pretrain_dir = os.path.join(base_save_path, args.pretrain_dir)
args.rec_path = os.path.join(base_save_path, args.rec_path)


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger = log_manager.get_logger()    # get the logger

    # device = mindspore.device("cuda:"+str(args.gpu_id) if mindspore.cuda.is_available()
    #                       and not args.no_cuda else "cpu")

    # generator is used to manage dataset
    generator = DiffusionGenerator(args, logger)

    trainer = DiffusionTrainer(args, logger, generator)

    if args.train:
        trainer.train()

    else:
        trainer.load_model()
        trainer.augment()



if __name__ == "__main__":

    main()






