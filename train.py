import argparse
import numpy as np
import os
import sys
import time
import shutil
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import Models
import Datasets
from Loss.loss import define_loss, allowed_losses, MSE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import str2bool, AverageMeter, mkdir_if_missing, \
                        Projector, filter_data, define_init_weights, \
                        filter_data, filter_sparse_guidance

from Models.adversarial import ResnetGeneratorCycle

# This speeds up training
torch.backends.cudnn.benchmark = True

# Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--n_training_iterations', type=int, default=1e7, help='Number of iterations for training')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')
parser.add_argument('--mod', type=str, default='mod', choices=Models.allowed_models(), help='Model for use')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True, help='Use GPUs')

parser.add_argument('--nworkers', type=int, default=4, help='num of threads')
parser.add_argument('--input_type', type=str, default='rgb', choices=['depth','rgb'], help='use rgb for rgbdepth')

parser.add_argument('--filter_window', type=int, default=16, help='window w used for the input sparse depth filter')
parser.add_argument('--filter_th', type=float, default=0.5, help='object thickness theta used for the input sparse depth filter')

# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')
parser.add_argument('--max_depth', type=float, default=85.0, help='maximum depth of LIDAR input')
parser.add_argument('--sparse_val', type=float, default=0.0, help='value to encode sparsity with')
parser.add_argument("--train_target", type=str2bool, nargs='?', const=True, default=False, help="whether to add target data (KITTI) during training")
parser.add_argument("--use_image_translation", type=str2bool, nargs='?', const=True, default=True, help="translate CARLA RGB to KITTI style")

# Paths settings
parser.add_argument('--save_path', default='Saved/', help='save path')
parser.add_argument('--data_path_target', default='./Data/Kitti', help='path to target dataset')
parser.add_argument('--data_path_source', default='./Data/Carla', help='path to source dataset')

# Optimizer settings
parser.add_argument('--learning_rate', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orthogonal weights initialisation')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')

# Loss settings
parser.add_argument('--loss_criterion_source', type=str, default='berhu', choices=allowed_losses(), help="loss criterion for source data")
parser.add_argument('--loss_criterion_target', type=str, default='mae', choices=allowed_losses(), help="loss criterion for target data")
parser.add_argument('--print_freq', type=int, default=20, help="print every x iterations")
parser.add_argument('--metric', type=str, default='rmse', choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument('--wlid', type=float, default=0.1, help="weight intermediate maps loss")
parser.add_argument('--wrgb', type=float, default=0.1, help="weight intermediate maps loss")
parser.add_argument('--wpred', type=float, default=1, help="weight main prediction")
parser.add_argument('--wguide', type=float, default=0.1, help="weight intermediate maps loss")
# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True,
                    default=True, help="cudnn optimization active")
parser.add_argument("--seed", type=str2bool, nargs='?', const=True,
                    default=True, help="use seed")
parser.add_argument("--seed_used", type=int, default=1, help="Actual seed used")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')

parser.add_argument('--save_name', default='', type=str, help='Name of the folder where to save the models')
parser.add_argument('--load_path', default='', type=str, help='Path where to the pretrained model from')




def main():
    global args
    args = parser.parse_args()
    if args.num_samples == 0:
        args.num_samples = None

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    
    # Init model
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in)

    if args.mod == 'mod':
        define_init_weights(model, args.weight_init)

    # Load on gpu before passing params to optimizer
    if args.cuda:
        model = model.cuda()
    
    save_id = '{}_{}_{}_{}_batch{}_pretrain{}_wlid{}_wrgb{}_wguide{}_wpred{}_num_samples{}'.\
              format(args.mod, args.loss_criterion_source,
                     args.learning_rate,
                     args.input_type, 
                     args.batch_size,
                     args.load_path!='', args.wlid, args.wrgb, args.wguide, args.wpred, 
                    args.num_samples)

    optimizer = torch.optim.Adam(model.parameters(), 
            lr=args.learning_rate, weight_decay=args.weight_decay)


    # Optional to use different losses
    criterion_source = define_loss(args.loss_criterion_source)
    criterion_target = define_loss(args.loss_criterion_target)

    # INIT KITTI dataset
    print('Load KITTI')
    dataset = Datasets.define_dataset('kitti', args.data_path_target, args.input_type)
    dataset.prepare_dataset()
    train_loader = get_loader(args, dataset, only_train=True)

    # INIT Carla dataset
    print('Load Carla')
    dataset = Datasets.define_dataset('carla', args.data_path_source, args.input_type)
    dataset.prepare_dataset()
    # The sparsification of the data and projection from the LiDAR reference
    # frame to the RGB camera explained in the paper happens in the dataloader
    train_loader_carla = get_loader(args, dataset, is_carla=True, only_train=True)
    train_loader_iter = iter(train_loader)

    # Resume training
    if args.save_name == '':
        args.save_path = os.path.join(args.save_path, save_id)
    else:
        args.save_path = os.path.join(args.save_path, args.save_name)
    if os.path.exists(args.save_path):
        raise Exception('Save path already exists')

    mkdir_if_missing(args.save_path)

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    # Load pretrained state
    if args.load_path != '':
        print("=> loading checkpoint {:s}".format(args.load_path))
        check = torch.load(args.load_path, map_location=lambda storage, loc: storage)['state_dict']
        model.load_state_dict(check)
        
        
    if args.use_image_translation:
        image_trans_net = ResnetGeneratorCycle(3, 3, 64, n_blocks=9)
        state_dict = torch.load('./image_translation_weights.pth')
        image_trans_net.load_state_dict(state_dict)
        image_trans_net.eval()
        if args.cuda:
            image_trans_net = image_trans_net.cuda()
            

    # Start training
    global_step = 0
    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start EPOCH {}".format(epoch + 1))

        # Define container objects
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        score_train_rmse = AverageMeter()
        score_train_mae = AverageMeter()
        metric_train = Metrics(max_depth=args.max_depth)

        # Train model for args.nepochs
        model.train()

        # compute timing
        end = time.time()
        for i, (input, gt, filepath) in tqdm(enumerate(train_loader_carla)):
            # Time dataloader
            data_time.update(time.time() - end)
            loss_extra = 0
            # Put inputs on gpu if possible
            if args.cuda:
                input, gt = input.cuda(), gt.cuda()

            # The LiDAR depths have large regions where no input depth is given
            # We remove all of the GT in the synthetic data where no input information is given
            # in a NxN window around the GT point (we set N=41) to avoid the model trying to estimate
            # depth for areas without any input guidance
            input_depth = input[:,0:1]
            input_depth, gt = filter_data(input_depth, gt, max_depth=args.max_depth)
            input[:,0:1] = input_depth

            ### Load target set (KITTI) data
            if args.train_target:
                try:
                    input_target, gt_target, filepath_t  = next(train_loader_iter)
                except:
                    train_loader_iter = iter(train_loader)
                    input_target, gt_target, filepath_t = next(train_loader_iter)
                
                if args.cuda:
                    input_target, gt_target = input_target.cuda(), gt_target.cuda()
                
            if args.use_image_translation:
                # The CycleGAN model was trained with inputs in the range of [-1, 1]
                with torch.no_grad():
                    rgb_trans = image_trans_net(input[:,1:]/128.5 - 1)
                rgb_trans = 128.5*(rgb_trans + 1)
                rgb_trans = rgb_trans.clamp(0, 255)
                input = torch.cat([input[:,:1], rgb_trans], 1) 
                
            if args.train_target:
                input_joint = torch.cat([input, input_target])
                prediction, lidar_out, precise, guide = model(input_joint, epoch)
                # We separate predictions from the target domain and source domain
                prediction_target, lidar_out_target, precise_target, guide_target = prediction[args.batch_size:], lidar_out[args.batch_size:], precise[args.batch_size:], guide[args.batch_size:]
                prediction, lidar_out, precise, guide = prediction[:args.batch_size], lidar_out[:args.batch_size], precise[:args.batch_size], guide[:args.batch_size]
            else:
                prediction, lidar_out, precise, guide = model(input, epoch)
            
            # We compute the loss for the source domain data
            loss = criterion_source(prediction, gt)
            loss_lidar = criterion_source(lidar_out, gt)
            loss_rgb = criterion_source(precise, gt)
            loss_guide = criterion_source(guide, gt)
            loss = args.wpred*loss + args.wlid*loss_lidar + args.wrgb*loss_rgb + args.wguide*loss_guide

            if args.train_target:
                loss_target = 0
                # We filter the input data for supervision as explained in the paper
                filtered_sparse_data = filter_sparse_guidance(input_target[:,:1], args.filter_window, args.filter_th)
                # We compute the loss for the target domain data
                loss_target += args.wpred*(criterion_target(prediction_target, filtered_sparse_data)) 
                loss_target += args.wlid*(criterion_target(lidar_out_target, filtered_sparse_data)) 
                loss_target += args.wrgb*(criterion_target(precise_target, filtered_sparse_data)) 
                loss_target += args.wguide*(criterion_target(guide_target, filtered_sparse_data)) 

                loss = loss + loss_target

            metric_train.calculate(prediction[:, 0:1].detach(), gt.detach())

            score_train_rmse.update(metric_train.get_metric('rmse'), metric_train.num)
            score_train_mae.update(metric_train.get_metric('mae'), metric_train.num)
            losses.update(loss.item(), input.size(0))

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_time.update(time.time() - end)
            end = time.time()

            global_step += 1

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'RMSE Train {score.val:.4f} ({score.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader_carla), batch_time=batch_time,
                   loss=losses,
                   score=score_train_rmse))

            if global_step == args.n_training_iterations:
                dict_save = {
                        'epoch': epoch + 1,
                        'arch': args.mod,
                        'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                save_checkpoint(dict_save, False, epoch+1, global_step)
                return 1
        print("===> Average RMSE score on training set is {:.4f}".format(score_train_rmse.avg))
        print("===> Average MAE score on training set is {:.4f}".format(score_train_mae.avg))
        dict_save = {
                'epoch': epoch + 1,
                'arch': args.mod,
                'state_dict': model.state_dict(),

            'optimizer': optimizer.state_dict()}
        save_checkpoint(dict_save, False, epoch+1)


def save_checkpoint(state, to_copy, epoch, global_step=0):
    if global_step > 0:
        filepath = os.path.join(args.save_path, 'checkpoint_model_step_{:d}.pth.tar'.format(int(global_step)))
    else:
        filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, filepath)

if __name__ == '__main__':
    main()
