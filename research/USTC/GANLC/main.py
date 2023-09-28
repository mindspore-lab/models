import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import mindspore.nn as nn
import mindspore
from mindspore import context, amp
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
import mindspore.dataset as ds

# context.set_context(mode=context.PYNATIVE_MODEGRAPH_MODE,device_id=0, device_target="GPU")
context.set_context(mode=context.GRAPH_MODE, device_id=0, device_target="GPU")
mindspore.set_context(pynative_synchronize=True)
from skimage.measure import compare_psnr, compare_ssim
from sklearn import metrics

sys.path.insert(1, './code')
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


from dataloader import train_dataset
from models import generator,backward_warp
from tqdm import tqdm
from ops import *
from models import *

# All arguments. These are the same arguments as in the original TecoGan repo. I might prune them at a later date.

parser = argparse.ArgumentParser()
parser.add_argument('--rand_seed', default=1, type=int, help='random seed')
# Directories
parser.add_argument('--input_dir_LR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--input_dir_len', default=-1, type=int,
                    help='length of the input for inference mode, -1 means all')
parser.add_argument('--input_dir_HR', default='', nargs="?",
                    help='The directory of the input resolution input data, for inference mode')
parser.add_argument('--mode', default='train', nargs="?", help='train, or inference')
parser.add_argument('--output_dir', default="output", help='The output directory of the checkpoint')
parser.add_argument('--output_pre', default='', nargs="?", help='The name of the subfolder for the images')
parser.add_argument('--output_name', default='output', nargs="?", help='The pre name of the outputs')
parser.add_argument('--output_ext', default='jpg', nargs="?", help='The format of the output when evaluating')
parser.add_argument('--summary_dir', default="summary", nargs="?", help='The dirctory to output the summary')
parser.add_argument('--videotype', default=".mp4", type=str, help="Video type for inference output")
parser.add_argument('--inferencetype', default="dataset", type=str, help="The type of input to the inference loop. "
                                                                         "Either video or dataset folder.")

# Models
parser.add_argument('--g_checkpoint',
                    default='./output/pattient0/test_mindspore/generator_99.ckpt',
                    help='If provided, the generator will be restored from the provided checkpoint')
parser.add_argument('--f_checkpoint',
                    default='./output/pattient0/test_mindspore/fnet_99.ckpt',
                    nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
parser.add_argument('--pic_checkpoint',
                    default='./output/pattient0/test_mindspore/discrim_pic_99.ckpt',
                    nargs="?",
                    help='If provided, the discriminator will be restored from the provided checkpoint')
parser.add_argument('--num_resblock', type=int, default=16, help='How many residual blocks are there in the generator')
parser.add_argument('--discrim_resblocks', type=int, default=4, help='Number of resblocks in each resnet layer in the '
                                                                     'discriminator')
parser.add_argument('--discrim_channels', type=int, default=128, help='How many channels to use in the last two '
                                                                      'resnet blocks in the discriminator')
# Models for training
parser.add_argument('--pre_trained_model', type=str2bool, default=False,
                    help='If True, the weight of generator will be loaded as an initial point'
                         'If False, continue the training')

# Machine resources
parser.add_argument('--queue_thread', default=8, type=int,
                    help='The threads of the queue (More threads can speedup the training process.')

parser.add_argument('--RNN_N', default=2, nargs="?", help='The number of the rnn recurrent length')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size of the input batch')
parser.add_argument('--flip', default=True, type=str2bool, help='Whether random flip data augmentation is applied')
parser.add_argument('--random_crop', default=True, type=str2bool, help='Whether perform the random crop')
parser.add_argument('--movingFirstFrame', default=True, type=str2bool,
                    help='Whether use constant moving first frame randomly.')
parser.add_argument('--crop_size', default=512, type=int, help='The crop size of the training image with width')
# Training data settings
parser.add_argument('--input_medical_dir', type=str, default="./data_download/mayo_data_arranged_patientwise0",
                    help='The directory of the video input data, for training')
# The loss parameters

# Training parameters
parser.add_argument('--EPS', default=1e-5, type=float, help='The eps added to prevent nan')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate for the network')
parser.add_argument('--decay_step', default=100, type=int, help='The steps needed to decay the learning rate')
parser.add_argument('--decay_rate', default=0.8, type=float, help='The decay rate of each decay step')
parser.add_argument('--beta', default=0.9, type=float, help='The beta1 parameter for the Adam optimizer')
parser.add_argument('--adameps', default=1e-8, type=float, help='The eps parameter for the Adam optimizer')
parser.add_argument('--max_epochs', default=100, type=int, help='The max epoch for the training')
parser.add_argument('--savefreq', default=10, type=int, help='The save frequence for training')
parser.add_argument('--savepath',
                    default="./output/pattient0/test_mindspore/",
                    type=str, help='The save path for training')
#
# Dst parameters
parser.add_argument('--ratio', default=0.01, type=float, help='The ratio between content loss and adversarial loss')
parser.add_argument('--Dt_mergeDs', default=True, type=str2bool, help='Whether only use a merged Discriminator.')
parser.add_argument('--Dt_ratio_max', default=1.0, type=float, help='The max ratio for the temporal adversarial loss')
parser.add_argument('--D_LAYERLOSS', default=True, type=str2bool, help='Whether use layer loss from D')
scale = 1
args = parser.parse_args()

if args.output_dir is None:
    raise ValueError("The output directory is needed")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.exists(args.summary_dir):
    os.mkdir(args.summary_dir)

if args.mode == "inference":
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    dataset = train_dataset(args)
    dataload = ds.GeneratorDataset(dataset, ["fbp", "phantom", "sinogram"], shuffle=False)
    dataload = dataload.batch(batch_size=1)

    generator_F = generator(1, args=args)
    g_checkpoint = mindspore.load_checkpoint(args.g_checkpoint)
    mindspore.load_param_into_net(generator_F, g_checkpoint)

    fnet = FNet(1)
    f_checkpoint = mindspore.load_checkpoint(args.f_checkpoint)
    mindspore.load_param_into_net(fnet, f_checkpoint)
    PSNR = 0
    SSIM = 0
    pic_position = '/picture_99_fnet/'
    if not os.path.exists(args.savepath + pic_position):
        os.makedirs(args.savepath + pic_position)
    logfile = open(args.savepath + pic_position + 'PSNR.txt', 'w+')
    PSNR_10,SSIM_10=[],[]
    PP10, SS10 = 0, 0
    fnet.set_train(False)
    generator_F.set_train(False)
    generator_F = amp.auto_mixed_precision(generator_F, amp_level="O2")
    fnet = amp.auto_mixed_precision(fnet, amp_level="O2")
    batch_idx=0
    for r_inputs, r_targets, r in dataload:
        batch_idx+=1
        if len(r.shape)==1:
            break
        if batch_idx%235==1:
            PSNR_10.append(PP10/235)
            SSIM_10.append(SS10/235)
            PP10, SS10 = 0, 0
        output_channel = r_inputs.shape[2]
        inputimages = r_inputs.shape[1]
        gen_outputs = []
        gen_warppre = []
        learning_rate = args.learning_rate
        data_n, data_t, data_c, lr_h, lr_w = r_inputs.size()
        _, _, _, hr_h, hr_w = r_targets.size()
        Frame_t_pre = r_inputs[:, 0:-1, :, :, :]
        Frame_t = r_inputs[:, 1:, :, :, :]
        fnet_input = fnet(Frame_t.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w),
                          Frame_t_pre.reshape(data_n * (data_t - 1), data_c, lr_h, lr_w))
        gen_flow = fnet_input
        gen_flow = ops.reshape(gen_flow[:, 0:2],
                                 (data_n, (inputimages - 1), 2, args.crop_size , args.crop_size ))

        input0 = ops.cat(
            (r_inputs[:, 0, :, :, :],
             ops.zeros(size=(1, data_c , args.crop_size, args.crop_size),
                         dtype=mindspore.float32)), axis=1)
        gen_pre_output = generator_F(input0)
        gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], data_c, args.crop_size,
                                             args.crop_size )
        cur_flow = gen_flow[:, 0, :, :, :]
        gen_pre_output_warp = backward_warp(gen_pre_output, cur_flow)

        gen_pre_output_warp = preprocessLr(gen_pre_output_warp)

        gen_pre_output_reshape = ops.reshape(gen_pre_output_warp,
                                               (gen_pre_output_warp.shape[0], data_c,
                                                args.crop_size, args.crop_size))
        inputs = ops.cat((r_inputs[:, 1, :, :, :], gen_pre_output_reshape), axis=1)
        gen_output = generator_F(inputs)
        gen_outputs.append(gen_output)
        gen_outputs = ops.stack(gen_outputs, axis=1)
        name = batch_idx
        r_targets = r_targets[:,1,:,:,:].cpu().detach().numpy().squeeze()#
        r_targets = cut_image(r_targets, vmin=0.0, vmax=1.0)
        gen_outputs = gen_outputs.cpu().detach().numpy().squeeze()
        data_range = np.max(r_targets) - np.min(r_targets)
        fbp_image = r_inputs[:,1,:,:,:].cpu().detach().numpy().squeeze()
        for irange in range(inputimages-1):
            gen_outputs = cut_image(gen_outputs, vmin=0.0, vmax=data_range)
            fbp_image = cut_image(fbp_image, vmin=0.0, vmax=data_range)
            psnr_gan_ar = compare_psnr(r_targets, gen_outputs, data_range=data_range)
            ssim_gan_ar = compare_ssim(r_targets, gen_outputs, data_range=data_range)
            psnr_fbp = compare_psnr(r_targets, fbp_image, data_range=data_range)
            ssim_fbp = compare_ssim(r_targets, fbp_image, data_range=data_range)
            rmse=np.sqrt(metrics.mean_squared_error(r_targets,gen_outputs))
            rmse_fbp = np.sqrt(metrics.mean_squared_error(r_targets, fbp_image))
            PSNR += psnr_gan_ar
            SSIM += ssim_gan_ar
            PP10+=psnr_gan_ar
            SS10+=ssim_gan_ar
            mess = 'batch idx:{}  psnr:{}  ssim:{} \n'.format(batch_idx, psnr_gan_ar, ssim_gan_ar)
            logfile.write(mess)
            print(mess)
            Batch_idx=batch_idx
    batch_idx=Batch_idx
    print('batch idx:', batch_idx)
    PSNR = PSNR / (batch_idx )
    SSIM = SSIM / (batch_idx )
    mess = 'Final average:{}  psnr:{}  ssim:{} \n'.format(batch_idx + 1, PSNR, SSIM)
    print(PSNR_10)
    print(SSIM_10)
    print(mess)
    logfile.write(mess)
    logfile.close()

elif args.mode == "train":
    dataset = train_dataset(args)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    dataload = ds.GeneratorDataset(dataset, ["fbp", "phantom", "sinogram"], shuffle=False)
    dataload = dataload.batch(batch_size=1)
    logfile = open(args.savepath + 'train1.txt', 'w+')
    generator_F = generator(1, args=args)
    fnet = FNet(1)
    discriminator_Pic = discriminator_pic(args=args)
    counter1 = 0.
    counter2 = 0.
    min_gen_loss = np.inf
    tdis_learning_rate = args.learning_rate
    if not args.Dt_mergeDs:
        tdis_learning_rate = tdis_learning_rate * 0.3

    GAN_FLAG = True
    d_scheduler = nn.cosine_decay_lr(min_lr=args.learning_rate / 100, max_lr=args.learning_rate, total_step=args.max_epochs,
                                     step_per_epoch=1, decay_epoch=args.decay_step)
    g_scheduler = nn.cosine_decay_lr(min_lr=args.learning_rate / 100, max_lr=args.learning_rate, total_step=args.max_epochs,
                                     step_per_epoch=1, decay_epoch=args.decay_step)
    d_pic_scheduler = nn.cosine_decay_lr(min_lr=args.learning_rate / 100, max_lr=args.learning_rate,
                                         total_step=args.max_epochs, step_per_epoch=1, decay_epoch=args.decay_step)
    f_optimizer = nn.Adam(fnet.trainable_params(), d_scheduler,
                          beta1=args.beta, beta2=0.999,
                          eps=args.adameps)
    gen_optimizer = nn.Adam(generator_F.trainable_params(), g_scheduler, beta1=args.beta, beta2=0.999,
                            eps=args.adameps)
    discriminator_optimizer = nn.Adam(discriminator_Pic.trainable_params(), d_pic_scheduler,
                                      beta1=args.beta, beta2=0.999, eps=args.adameps)
    # Loading pretrained models and optimizers
    if args.pre_trained_model:
        g_checkpoint = mindspore.load_checkpoint(args.g_checkpoint)
        mindspore.load_param_into_net(generator_F, g_checkpoint)
        f_checkpoint = mindspore.load_checkpoint(args.f_checkpoint)
        mindspore.load_param_into_net(fnet, f_checkpoint)
        pic_checkpoint = mindspore.load_checkpoint(args.pic_checkpoint)
        mindspore.load_param_into_net(discriminator_Pic, pic_checkpoint)
    current_epoch = 0

    inputimages = args.RNN_N
    outputimages= args.RNN_N
    adversarial_loss = nn.BCELoss(reduction='mean')

    def warp_fn(Frame_t, Frame_t_pre):
        data_n,data_c, lr_h, lr_w = Frame_t.shape
        fnet_input = fnet(Frame_t.reshape(data_n , data_c, lr_h, lr_w),
                          Frame_t_pre.reshape(data_n , data_c, lr_h, lr_w))

        gen_flow = fnet_input
        gen_flow = ops.reshape(gen_flow[:, 0:2],
                               (data_n, (inputimages - 1), 2, args.crop_size , args.crop_size))
        input_frames = ops.reshape(Frame_t,
                                   (Frame_t.shape[0] * (inputimages - 1), 1, args.crop_size,
                                    args.crop_size))
        s_input_warp = backward_warp(ops.reshape(Frame_t_pre, (
            Frame_t_pre.shape[0] * (inputimages - 1), 1, args.crop_size, args.crop_size)),
                                     gen_flow[:, 0, :, :,
                                     :])
        warp_loss = nn.MSELoss()(s_input_warp, input_frames)
        return warp_loss, gen_flow


    def gen_fn(Frame_t,Frame_t_pre,gen_flow, targets):
        gen_outputs = []
        input0 = ops.cat(
            (Frame_t_pre,
             ops.zeros(size=(Frame_t_pre.shape[0], 1, args.crop_size, args.crop_size),
                       dtype=mindspore.float32)), axis=1)

        gen_pre_output = generator_F(input0)
        gen_pre_output = gen_pre_output.view(gen_pre_output.shape[0], 1, args.crop_size ,
                                             args.crop_size )
        gen_outputs.append(gen_pre_output)

        cur_flow = gen_flow[:, 0, :, :, :]
        gen_pre_output_warp = backward_warp(gen_pre_output, cur_flow.half())
        gen_pre_output_warp = preprocessLr(gen_pre_output_warp)
        gen_pre_output_reshape = gen_pre_output_warp.view(gen_pre_output_warp.shape[0], 1, args.crop_size,
                                                           args.crop_size)
        inputs = ops.cat((Frame_t, gen_pre_output_reshape), axis=1)
        gen_output = generator_F(inputs)
        gen_outputs.append(gen_output)
        gen_outputs = ops.stack(gen_outputs, axis=1)
        gen_outputs = gen_outputs.view(gen_outputs.shape[0], outputimages, 1, args.crop_size ,
                                       args.crop_size )

        s_gen_output = ops.reshape(gen_outputs[:, outputimages - 1, :, :, :],
                                    (gen_outputs.shape[0] * 1, 1, args.crop_size ,
                                     args.crop_size ))
        s_targets = ops.reshape(targets[:, outputimages - 1, :, :, :], (
            targets.shape[0] * 1, 1, args.crop_size , args.crop_size ))

        dis_fake_output, fake_layer2 = discriminator_Pic(s_gen_output)
        dis_real_output, real_layer2 = discriminator_Pic(s_targets)
        #### Computing the layer loss using the discriminator outputs
        if (args.D_LAYERLOSS):
            Fix_Range = 0.02
            sum_layer_loss = 0
            layer_n = len(real_layer2)
            layer_norm = [12.0, 14.0, 24.0, 100.0]
            for layer_i in range(layer_n):
                real_layer22 = real_layer2[layer_i]
                fake_layer22 = fake_layer2[layer_i]
                layer_diff2 = real_layer22 - fake_layer22
                layer_loss = ops.mean(ops.sum(ops.abs(layer_diff2), dim=[3]))

                scaled_layer_loss = Fix_Range * layer_loss / layer_norm[layer_i]

                sum_layer_loss += scaled_layer_loss

        content_loss = nn.MSELoss()(s_targets, s_gen_output)
        gen_loss = content_loss

        t_adversarial_loss2 = ops.mean(ops.log(1 - dis_fake_output + args.EPS))
        dt_ratio = mindspore.tensor(args.Dt_ratio_max)

        gen_loss += args.ratio * t_adversarial_loss2
        gen_loss += sum_layer_loss * dt_ratio

        return gen_loss, s_gen_output,s_targets


    def disc_fn(s_gen_output,s_targets):
        dis_fake_output, fake_layer2 = discriminator_Pic(s_gen_output)
        dis_real_output, real_layer2 = discriminator_Pic(s_targets)

        t_discrim_fake_loss = ops.log(1 - dis_fake_output + args.EPS)
        t_discrim_real_loss = ops.log(dis_real_output + args.EPS)

        t_discrim_loss = -(ops.mean(t_discrim_fake_loss) + ops.mean(t_discrim_real_loss) + args.EPS)
        return t_discrim_loss


    since = time.time()
    scale = 1
    for e in tqdm(range(current_epoch, args.max_epochs)):
        d_loss = 0.
        g_loss = 0.
        f_loss = 0.
        batch_idx = -1
        for data in dataload:
            batch_idx = batch_idx + 1
            if batch_idx >= 540:
                a = 1
            inputs, targets = data[0], data[1]
            valid = ops.ones((inputs.shape[0], 1), mindspore.float32)
            fake = ops.zeros((inputs.shape[0], 1), mindspore.float32)
            generator_F.set_train()
            discriminator_Pic.set_train()
            fnet.set_train()

            data_n, data_t, data_c ,lr_h, lr_w = inputs.shape
            _, _, _, hr_h, hr_w = targets.shape
            inputimages = args.RNN_N
            outputimages = args.RNN_N

            output_channel = targets.shape[2]

            learning_rate = args.learning_rate
            Frame_t_pre = inputs[:, 0:-1, :, :, :].squeeze(1)
            Frame_t = inputs[:, 1:, :, :, :].squeeze(1)
            generator_F = amp.auto_mixed_precision(generator_F, amp_level="O2")
            discriminator_Pic = amp.auto_mixed_precision(discriminator_Pic, amp_level="O2")
            fnet = amp.auto_mixed_precision(fnet, amp_level="O2")


            grad_warp = mindspore.value_and_grad(warp_fn, None,
                                           f_optimizer.parameters, has_aux=True)
            (warp_loss, gen_flow), grad_w = grad_warp(Frame_t, Frame_t_pre)
            f_optimizer(grad_w)

            grad_gen = mindspore.value_and_grad(gen_fn, None, gen_optimizer.parameters, has_aux=True)
            (gen_loss, s_gen_output,s_targets), grad_g = grad_gen(Frame_t, Frame_t_pre,gen_flow,targets)
            gen_optimizer(grad_g)

            grad_dis = mindspore.value_and_grad(disc_fn, None, discriminator_optimizer.parameters)
            t_discrim_loss, grad_d = grad_dis(s_gen_output, s_targets)
            discriminator_optimizer(grad_d)


            if (gen_loss>1 or gen_loss<0):
                print('g_loss: {} , batch idx:{}'.format(gen_loss, batch_idx))
            f_loss = f_loss + ((1 / (batch_idx + 1)) * (warp_loss - f_loss))

            g_loss = g_loss + ((1 / (batch_idx + 1)) * (gen_loss - g_loss))

            d_loss = d_loss + ((1 / (batch_idx + 1)) * (t_discrim_loss - d_loss))


        # Printing out metrics
        print("Epoch: {}".format(e + 1))
        mess = 'batch idx:{} f_loss{} g_loss:{}  d_loss:{}\n'.format(e,f_loss, g_loss, d_loss,)
        logfile.write(mess)
        print("\nSaving model...")
        # Saving the models
        mindspore.save_checkpoint(generator_F, args.savepath+"generator.ckpt")
        mindspore.save_checkpoint(fnet, args.savepath + "fnet.ckpt")
        mindspore.save_checkpoint(discriminator_Pic, args.savepath + "discrim_pic.ckpt")
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        if e % (args.savefreq ) ==0:
            mindspore.save_checkpoint(generator_F, args.savepath + "generator_{}.ckpt".format( e))
            mindspore.save_checkpoint(fnet, args.savepath + "fnet_{}.ckpt".format( e))
            mindspore.save_checkpoint(discriminator_Pic, args.savepath + "discrim_pic_{}.ckpt".format( e))

    logfile.close()
