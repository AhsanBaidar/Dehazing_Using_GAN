import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='image-dehazing')
    parser.add_argument('--data_dir', type=str, default='USV_Dataset/Sand_Storm/', help='dataset directory') #change to sand_Storm or Fog
    parser.add_argument('--save_dir', default='Saved_Models', help='data save directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
    parser.add_argument('--exp', default='Dehazing_Model', help='model to select')
    parser.add_argument('--p_factor', type=float, default=0.60, help='perceptual loss factor')
    parser.add_argument('--g_factor', type=float, default=0.40, help='gan loss factor')
    parser.add_argument('--glr', type=float, default=1e-4, help='generator learning rate')
    parser.add_argument('--dlr', type=float, default=0.8e-4, help='discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr_step_size', type=int, default=2000, help='period of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor of learning rate decay')
    parser.add_argument('--patch_gan', type=int, default=30, help='Patch GAN size')
    parser.add_argument('--pool_size', type=int, default=10, help='Buffer size for storing generated samples from G')
    parser.add_argument('--period', type=int, default=1, help='period of printing logs')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')

    return parser.parse_args()
