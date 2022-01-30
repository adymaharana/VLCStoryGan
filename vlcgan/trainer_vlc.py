from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import pickle
from tqdm import tqdm
import json

from .miscc.config import cfg
from .miscc.utils import mkdir_p
from .miscc.utils import weights_init
from .miscc.utils import save_story_results, save_model, save_test_samples
from .miscc.utils import KL_loss
from .miscc.utils import compute_discriminator_loss, compute_generator_loss, compute_dual_densecap_loss
from shutil import copyfile
from torchvision.models import vgg16
from densecap.model.densecap import densecap_resnet50_fpn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class GANTrainer(object):
    def __init__(self, cfg, output_dir, ratio=1.0, vocab_size=None):
        if cfg.TRAIN.FLAG:
            output_dir = output_dir + '_r' + str(ratio) + '/'
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            self.test_dir = os.path.join(output_dir, 'Test')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.test_dir)
            copyfile('./cfg/pororo_s1.yml', output_dir + 'pororo_s1.yml')
            copyfile('./cfg/pororo_s2.yml', output_dir + 'pororo_s2.yml')
            copyfile('./model.py', output_dir + 'model.py')
        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        assert cfg.IMG_DISC or cfg.STORY_DISC
        self.use_image_disc = cfg.IMG_DISC
        self.use_story_disc = cfg.STORY_DISC
        self.use_martt = cfg.USE_MARTT

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.cfg = cfg

        if cfg.TRAIN.PERCEPTUAL_LOSS:
            self.perceptual_loss_net = PerceptualLoss()

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from .model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, StoryMarttGAN

        if self.use_martt:
            netG = StoryMarttGAN(self.cfg, self.video_len)
        else:
            netG = StoryGAN(self.cfg, self.video_len)
        netG.apply(weights_init)
        print(netG)

        if self.use_image_disc:
            if self.cfg.DATASET_NAME == 'youcook2':
                use_categories = False
            else:
                use_categories = True

            netD_im = STAGE1_D_IMG(self.cfg, use_categories=use_categories)
            netD_im.apply(weights_init)
            print(netD_im)

            if self.cfg.NET_D != '':
                state_dict = \
                    torch.load(self.cfg.NET_D,
                               map_location=lambda storage, loc: storage)
                netD_im.load_state_dict(state_dict)
                print('Load from: ', self.cfg.NET_D)
        else:
            netD_im = None

        if self.use_story_disc:
            netD_st = STAGE1_D_STY_V2(self.cfg)
            netD_st.apply(weights_init)
            # for m in netD_st.modules():
            #     print(m.__class__.__name__)
            print(netD_st)
        else:
            netD_st = None

        if self.cfg.NET_G != '':
            state_dict = \
                torch.load(self.cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', self.cfg.NET_G)

        if self.cfg.CUDA:
            netG.cuda()
            if self.use_image_disc:
                netD_im.cuda()
            if self.use_story_disc:
                netD_st.cuda()
            if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                self.perceptual_loss_net.loss_network.cuda()

        total_params = sum(p.numel() for p in netD_st.parameters() if p.requires_grad) + sum(
            p.numel() for p in netD_im.parameters() if p.requires_grad) + sum(
            p.numel() for p in netG.parameters() if p.requires_grad)
        print("Total Parameters: %s", total_params)

        return netG, netD_im, netD_st

    def load_dual_model(self):

        if self.cfg.DENSECAP_DUAL:
            with open('../../densecap-pytorch/model_params/config.json', 'r') as f:
                model_args = json.load(f)

            model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                          return_features=False,
                                          feat_size=model_args['feat_size'],
                                          hidden_size=model_args['hidden_size'],
                                          max_len=model_args['max_len'],
                                          emb_size=model_args['emb_size'],
                                          rnn_num_layers=model_args['rnn_num_layers'],
                                          vocab_size=model_args['vocab_size'],
                                          fusion_type=model_args['fusion_type'],
                                          box_detections_per_img=10)

            print("Loading pretrained checkpoint into Dense Captioning Model")
            checkpoint = torch.load(
                '../../densecap-pytorch/model_params/train_all_val_all_bz_2_epoch_10_inject_init.pth.tar')
            model.load_state_dict(checkpoint['model'])

            for p in model.parameters():
                p.requires_grad = False

            if self.cfg.CUDA:
                model.cuda()
        else:
            raise ValueError

        return model

    def sample_real_image_batch(self):
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
        batch_idx, batch = next(self.imagedataset)
        b = batch
        if self.cfg.CUDA:
            for k, v in batch.items():
                if k == 'text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b

    def train(self, imageloader, storyloader, testloader, stage=1):
        self.imageloader = imageloader
        self.imagedataset = None
        netG, netD_im, netD_st = self.load_network_stageI()

        if self.cfg.DENSECAP_DUAL:
            netDual = self.load_dual_model()

        im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        st_real_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(0))
        if self.cfg.CUDA:
            im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()

        generator_lr = self.cfg.TRAIN.GENERATOR_LR
        discriminator_lr = self.cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = self.cfg.TRAIN.LR_DECAY_EPOCH

        if self.use_image_disc:
            im_optimizerD = \
                optim.Adam(netD_im.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        if self.use_story_disc:
            st_optimizerD = \
                optim.Adam(netD_st.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        loss_collector = []

        count = 0
        # save_test_samples(netG, testloader, self.test_dir, epoch=0, mart=self.use_mart)

        # save_test_samples(netG, testloader, self.test_dir)
        for epoch in range(self.max_epoch + 1):
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                if self.use_story_disc:
                    for param_group in st_optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr
                if self.use_image_disc:
                    for param_group in im_optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr

            for i, data in tqdm(enumerate(storyloader, 0)):
                ######################################################
                # (1) Prepare training data
                ######################################################
                im_batch = self.sample_real_image_batch()
                st_batch = data

                im_real_cpu = im_batch['images']
                im_motion_input = im_batch['description'][:, :self.cfg.TEXT.DIMENSION]
                im_content_input = im_batch['content'][:, :, :self.cfg.TEXT.DIMENSION]
                im_real_imgs = Variable(im_real_cpu)
                im_motion_input = Variable(im_motion_input)
                im_content_input = Variable(im_content_input)
                im_labels = Variable(im_batch['labels'])
                if self.use_martt:
                    im_input_ids = Variable(im_batch['input_id'])
                    im_masks = Variable(im_batch['mask'])
                    im_tag_ids = Variable(im_batch['tag_ids'])

                st_real_cpu = st_batch['images']
                st_motion_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION]
                st_content_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION]
                st_texts = st_batch['text']
                st_real_imgs = Variable(st_real_cpu)
                st_motion_input = Variable(st_motion_input)
                st_content_input = Variable(st_content_input)
                st_labels = Variable(st_batch['labels'])
                if self.use_martt:
                    st_input_ids = Variable(st_batch['input_ids'])
                    st_masks = Variable(st_batch['masks'])
                    st_tag_ids = Variable(st_batch['tag_ids'])

                if self.cfg.CUDA:
                    st_real_imgs = st_real_imgs.cuda()
                    im_real_imgs = im_real_imgs.cuda()
                    st_motion_input = st_motion_input.cuda()
                    im_motion_input = im_motion_input.cuda()
                    st_content_input = st_content_input.cuda()
                    im_content_input = im_content_input.cuda()
                    im_labels = im_labels.cuda()
                    st_labels = st_labels.cuda()
                    if self.use_martt:
                        im_input_ids = im_input_ids.cuda()
                        im_masks = im_masks.cuda()
                        im_tag_ids = im_tag_ids.cuda()
                    if self.use_martt:
                        st_input_ids = st_input_ids.cuda()
                        st_masks = st_masks.cuda()
                        st_tag_ids = st_tag_ids.cuda()

                im_motion_input = torch.cat((im_motion_input, im_labels), 1)
                st_motion_input = torch.cat((st_motion_input, st_labels), 2)
                #######################################################
                # (2) Generate fake stories and images
                ######################################################

                if len(self.gpus) > 1:
                    netG = nn.DataParallel(netG)

                if self.use_martt:
                    st_inputs = (st_motion_input, st_content_input, st_input_ids, st_masks, st_tag_ids, st_labels)
                else:
                    st_inputs = (st_motion_input, st_content_input)
                # lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)
                lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar, s_word = netG.sample_videos(*st_inputs)

                if self.use_martt:
                    im_inputs = (im_motion_input, im_content_input, im_input_ids, im_masks, im_tag_ids, im_labels)
                else:
                    im_inputs = (im_motion_input, im_content_input)
                # lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                #    nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)
                lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = netG.sample_images(*im_inputs)

                # print(st_fake.shape, im_fake.shape)

                characters_mu = (st_labels.mean(1) > 0).type(torch.FloatTensor).cuda()
                st_mu = torch.cat(
                    (c_mu, st_motion_input[:, :, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)

                im_mu = torch.cat((im_motion_input, cim_mu), 1)
                ############################
                # (3) Update D network
                ###########################
                if self.use_image_disc:
                    netD_im.zero_grad()
                    im_errD, imgD_loss_report = \
                        compute_discriminator_loss(netD_im, im_real_imgs, im_fake,
                                                   im_real_labels, im_fake_labels, im_labels,
                                                   im_mu, self.gpus, mode='image', dual=False,
                                                   real_captions=None if True else (im_input_ids, im_masks),
                                                   contrastive=self.cfg.IMG_CONTRAST,
                                                   real_caption_embeds=im_motion_input if self.cfg.IMG_CONTRAST else None)
                    im_errD.backward()
                    im_optimizerD.step()
                else:
                    im_errD = torch.tensor(0)
                    imgD_loss_report = {}

                if self.use_story_disc:
                    netD_st.zero_grad()
                    st_errD, stD_loss_report = \
                        compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                                   st_real_labels, st_fake_labels, st_labels,
                                                   st_mu, self.gpus, mode='story', dual=False,
                                                   real_captions=None,
                                                   contrastive=self.cfg.STORY_CONTRAST,
                                                   real_caption_embeds=st_motion_input if self.cfg.STORY_CONTRAST else None)
                    st_errD.backward()
                    st_optimizerD.step()
                else:
                    st_errD = torch.tensor(0)
                    stD_loss_report = {}

                ############################
                # (2) Update G network
                ###########################
                # TODO: Add config parameter for number of generator steps
                for g_iter in range(self.cfg.TRAIN.UPDATE_RATIO):
                    netG.zero_grad()

                    if self.use_martt:
                        st_inputs = (st_motion_input, st_content_input, st_input_ids, st_masks, st_tag_ids, st_labels)
                    else:
                        st_inputs = (st_motion_input, st_content_input)
                    # _, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                    #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)
                    _, st_fake, m_mu, m_logvar, c_mu, c_logvar, s_word = netG.sample_videos(*st_inputs)

                    if self.use_martt:
                        im_inputs = (im_motion_input, im_content_input, im_input_ids, im_masks, im_tag_ids, im_labels)
                    else:
                        im_inputs = (im_motion_input, im_content_input)
                    # _, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                    # nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)
                    _, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = netG.sample_images(*im_inputs)

                    characters_mu = (st_labels.mean(1) > 0).type(torch.FloatTensor).cuda()
                    st_mu = torch.cat(
                        (c_mu, st_motion_input[:, :, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)

                    im_mu = torch.cat((im_motion_input, cim_mu), 1)

                    if self.use_image_disc:
                        im_errG, imG_loss_report = compute_generator_loss(netD_im, im_fake,
                                                                          im_real_labels, im_labels, im_mu, self.gpus,
                                                                          mode='image', dual=False,
                                                                          real_captions=None if True else (im_input_ids, im_masks),
                                                                          contrastive=self.cfg.IMG_CONTRAST,
                                                                          fake_caption_embeds=im_motion_input if self.cfg.IMG_CONTRAST else None)
                        if self.cfg.DENSECAP_DUAL:
                            if self.cfg.CUDA:
                                targets = {'boxes': im_batch['boxes'].cuda(),
                                           'caps': im_batch['caps'].cuda(),
                                           'caps_len': im_batch['caps_len']}

                            else:
                                targets = {'boxes': im_batch['boxes'], 'caps': im_batch['caps'],
                                           'caps_len': im_batch['caps_len']}
                            im_errDual, im_densecap_loss_report = compute_dual_densecap_loss(netDual, im_fake, targets,
                                                                                             self.gpus, self.cfg,
                                                                                             mode='image')

                    else:
                        im_errG = torch.tensor(0)
                        imG_loss_report = {}

                    if self.use_story_disc:
                        st_errG, stG_loss_report = compute_generator_loss(netD_st, st_fake,
                                                                          st_real_labels, st_labels, st_mu, self.gpus,
                                                                          mode='story', dual=False,
                                                                          real_captions=None if True else (st_input_ids, st_masks),
                                                                          contrastive=self.cfg.STORY_CONTRAST,
                                                                          fake_caption_embeds=st_motion_input if self.cfg.STORY_CONTRAST else None)
                        if self.cfg.DENSECAP_DUAL:
                            if self.cfg.CUDA:
                                targets = {'boxes': st_batch['boxes'].cuda(),
                                           'caps': st_batch['caps'].cuda(),
                                           'caps_len': st_batch['caps_len']}
                            else:
                                targets = {'boxes': st_batch['boxes'], 'caps': st_batch['caps'],
                                           'caps_len': st_batch['caps_len']}
                            st_errDual, st_densecap_loss_report = compute_dual_densecap_loss(netDual, st_fake, targets,
                                                                                             self.gpus, self.cfg,
                                                                                             mode='story')
                    else:
                        st_errG = torch.tensor(0)
                        stG_loss_report = {}

                    im_kl_loss = KL_loss(cim_mu, cim_logvar)
                    st_kl_loss = KL_loss(c_mu, c_logvar)
                    kl_loss = im_kl_loss + self.ratio * st_kl_loss
                    if self.use_image_disc and self.use_story_disc:
                        errG_total = im_errG + im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (
                                st_errG + st_kl_loss * self.cfg.TRAIN.COEFF.KL)
                    elif self.use_image_disc:
                        errG_total = im_errG + im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (
                                    st_kl_loss * self.cfg.TRAIN.COEFF.KL)
                    else:
                        errG_total = im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (
                                    st_errG + st_kl_loss * self.cfg.TRAIN.COEFF.KL)

                    if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                        if self.cfg.CUDA:
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu.cuda())
                        else:
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu)
                        errG_total += per_loss

                    if self.cfg.DENSECAP_DUAL:
                        pass
                        # errG_total += st_errDual + im_errDual

                    errG_total.backward()
                    optimizerG.step()

                # delete variables to free space?
                del st_real_imgs, im_real_imgs, st_motion_input, im_motion_input, st_content_input, im_content_input, im_labels, st_labels
                if self.use_martt:
                    del im_input_ids, im_masks, st_input_ids, st_masks

                # if i%20 == 0 and i>0:
                #     save_test_samples(netG, testloader, self.test_dir, epoch, mart=self.use_mart)

                # loss_collector.append([imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report])
                count = count + 1


            end_t = time.time()
            print('''[%d/%d][%d/%d] %s Total Time: %.2fsec'''
                  % (epoch, self.max_epoch, i, len(storyloader), cfg.DATASET_NAME, (end_t - start_t)))

            for loss_report in [imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report]:
                for key, val in loss_report.items():
                    print(key, val)
            if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                print("Perceptual Loss: ", per_loss.data.item())
            if self.cfg.DENSECAP_DUAL:
                for loss_report in [im_densecap_loss_report, st_densecap_loss_report]:
                    for key, val in loss_report.items():
                        print(key, val)

            print('--------------------------------------------------------------------------------')

            if epoch % self.snapshot_interval == 0:
                save_test_samples(netG, testloader, self.test_dir, epoch, martt=self.use_martt)
                save_model(netG, netD_im, netD_st, optimizerG, im_optimizerD, st_optimizerD, epoch, self.model_dir)

        # np.save(os.path.join(self.model_dir, 'losses.npy'), loss_collector)
        with open(os.path.join(self.model_dir, 'losses.pkl'), 'wb') as f:
            pickle.dump(loss_collector, f)
        save_model(netG, netD_im, netD_st, self.max_epoch, self.model_dir)

    def sample(self, testloader, generator_weight_path, out_dir, stage=1):

        netG, _, _ = self.load_network_stageI()
        netG.load_state_dict(torch.load(generator_weight_path)['netG_state_dict'])
        save_test_samples(netG, testloader, out_dir, 60, martt=self.use_martt)