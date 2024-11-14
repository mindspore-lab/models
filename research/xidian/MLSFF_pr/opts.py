import argparse

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--net', default='GSFF',help='fc or attnin or trans，use which net')
    parser.add_argument('--dataset', default='syd',help='choose dataset,sydney,rsicd,ucm')
    parser.add_argument('--use_num_attr', type=int, default=1,help='')
    parser.add_argument('--learning_rate', type=float, default=1e-3,help='')
    parser.add_argument('--checkpoint_dir',type=str,
                        default='./stored_weights/',help='the path to save model')  
    parser.add_argument('--log_dir',type=str,default='./log',
                        help='the path to save model')
    parser.add_argument('--pretrained',type=bool,
                        default=False,help='use pretrained model or not')
    parser.add_argument('--joint_model',type=bool,
                        default=False,help='use joint model or not')
    parser.add_argument('--pretrained_model_path',type=str,
                        default='best_para.pkl',
                        help='the path where  stored the pretrained model ')                       
    parser.add_argument('--num_epoches',type=int,default=100,help='num_epochs') 
    parser.add_argument('--eval_every_iters',type=int,default=30,help='') 
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='') 
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--batch_size', type=int, default=5,
                    help='minibatch size')
    parser.add_argument('--use_att', type=bool, default=True,
                    help='')
    parser.add_argument('--use_box', type=bool, default=False,
                    help='minibatch size')
    parser.add_argument('--norm_att_feat', type=bool, default=False,
                    help='minibatch size')
    parser.add_argument('--norm_box_feat', type=bool, default=False,
                    help='minibatch size')
    parser.add_argument('--eval_every_steps',type=int,default=1000,
                        help='how many steps to do an test on val data')
    parser.add_argument('--sample_max', type=int, default=1,
                    help='')
    parser.add_argument('--learning_rate_decay_every', type=int, default=5, 
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.9, 
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0, 
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)') 
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--seq_per_img', type=int, default=5,help='')
    parser.add_argument('--temperature', type=float, default=1.0, help='')
    parser.add_argument('--seq_length', 
                        default=20, type=int, help='max caption length limited')   
    parser.add_argument('--own_img_dir',type=str,
                        default='',
                        help='the path where  stored the own imgs ')
    parser.add_argument('--num_layers', type=int, default=1,                   
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_size', type=int, default=200,                   
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--grad_clip', type=float, default=0.1, 
                    help='clip gradients at this value')
    parser.add_argument('--SEED', 
                        default=7, type=int, help='seed')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=100,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--sg_label_embed_size', type=int, default=128, help='')
    parser.add_argument('--att_hid_size', type=int, default=256,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--drop_prob_lm', type=float, default=0.0,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--seq_per_image', type=int, default=5,
                    help='')
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')
    parser.add_argument('--cls',type=bool,
                        default=True,help='use cls or not')
    parser.add_argument('--AdamW_warmup',default=False,help="")
    parser.add_argument('--AdamW',default=False,help="")
    parser.add_argument('--SGD',default=False,help="")
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')
    args=parser.parse_args()    
    return args
