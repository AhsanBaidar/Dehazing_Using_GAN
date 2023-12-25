import os
from model import Generator, Discriminator
from utils import SaveData, ImagePool
from data import MyDataset
from vgg16 import Vgg16
from arguments import get_args
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm

args = get_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def train(args):
    generator = Generator()
    generator = generator.cuda()
    discriminator = Discriminator()
    discriminator = discriminator.cuda()
    
    l1_loss = nn.L1Loss().cuda()
    l2_loss = nn.MSELoss().cuda()
    bce_loss = nn.BCELoss().cuda()
    
    optimizerG = optim.Adam(generator.parameters(), lr=args.glr)
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.dlr)
    schedulerG = lr_scheduler.StepLR(optimizerG, args.lr_step_size, args.lr_gamma)
    schedulerD = lr_scheduler.StepLR(optimizerD, args.lr_step_size, args.lr_gamma)
    save = SaveData(args.save_dir, args.exp, True)
    save.save_params(args)
    
    dataset = MyDataset(args.data_dir, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=int(args.n_threads))

    real_label = Variable(torch.ones([1, 1, args.patch_gan, args.patch_gan], dtype=torch.float)).cuda()
    fake_label = Variable(torch.zeros([1, 1, args.patch_gan, args.patch_gan], dtype=torch.float)).cuda()

    image_pool = ImagePool(args.pool_size)

    vgg = Vgg16(requires_grad=False)
    vgg.cuda()

    for epoch in range(args.epochs):
        print("* Epoch {}/{}".format(epoch + 1, args.epochs))

        schedulerG.step()
        schedulerD.step()
        d_total_real_loss, d_total_fake_loss, d_total_loss = 0, 0, 0
        g_total_res_loss, g_total_per_loss, g_total_gan_loss, g_total_loss = 0, 0, 0, 0

        generator.train()
        discriminator.train()

        for batch, images in tqdm(enumerate(dataloader)):
            input_image, target_image = images
            input_image = Variable(input_image.cuda())
            target_image = Variable(target_image.cuda())
            output_image = generator(input_image)

            # Update D
            discriminator.requires_grad(True)
            discriminator.zero_grad()

            ## real image
            real_output = discriminator(target_image)
            d_real_loss = bce_loss(real_output, real_label)
            d_real_loss.backward()
            d_real_loss = d_real_loss.data.cpu().numpy()
            d_total_real_loss += d_real_loss

            ## fake image
            fake_image = output_image.detach()
            fake_image = Variable(image_pool.query(fake_image.data))
            fake_output = discriminator(fake_image)
            d_fake_loss = bce_loss(fake_output, fake_label)
            d_fake_loss.backward()
            d_fake_loss = d_fake_loss.data.cpu().numpy()
            d_total_fake_loss += d_fake_loss

            ## loss
            d_total_loss += d_real_loss + d_fake_loss

            optimizerD.step()

            # Update G
            discriminator.requires_grad(False)
            generator.zero_grad()

            ## reconstruction loss
            g_res_loss = l1_loss(output_image, target_image)
            g_res_loss.backward(retain_graph=True)
            g_res_loss = g_res_loss.data.cpu().numpy()
            g_total_res_loss += g_res_loss

            ## perceptual losssave.save_model
            g_per_loss = args.p_factor * l2_loss(vgg(output_image), vgg(target_image))
            g_per_loss.backward(retain_graph=True)
            g_per_loss = g_per_loss.data.cpu().numpy()
            g_total_per_loss += g_per_loss

            ## gan loss
            output = discriminator(output_image)
            g_gan_loss = args.g_factor * bce_loss(output, real_label)
            g_gan_loss.backward()
            g_gan_loss = g_gan_loss.data.cpu().numpy()
            g_total_gan_loss += g_gan_loss

            ## loss
            g_total_loss += g_res_loss + g_per_loss + g_gan_loss

            optimizerG.step()

        d_total_real_loss = d_total_real_loss / (batch + 1)
        d_total_fake_loss = d_total_fake_loss / (batch + 1)
        d_total_loss = d_total_loss / (batch + 1)
        g_total_res_loss = g_total_res_loss / (batch + 1)
        g_total_per_loss = g_total_per_loss / (batch + 1)
        g_total_gan_loss = g_total_gan_loss / (batch + 1)
        g_total_loss = g_total_loss / (batch + 1)

        if epoch % args.period == 0:
            log = "discriminator_loss: {:.5f} \t generator_loss: {:.5f}".format(d_total_loss, g_total_loss)
            print(log)
            save.save_log(log)
            save.save_model(generator, epoch)


if __name__ == '__main__':
    train(args)
