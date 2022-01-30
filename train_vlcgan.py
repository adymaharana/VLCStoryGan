from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import numpy as np
import functools

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from vlcgan.miscc.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/pororo_s1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--dataset', type=str, help='dataset_name', default='pororo')
    parser.add_argument('--checkpoint_file', type=str, help='path to generation checkpoint', default='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.dataset == 'pororo':
        if cfg.USE_MARTT:
            import vlcgan.pororo_data_const as data
        else:
            import vlcgan.pororo_data as data
    else:
        raise ValueError

    if cfg.USE_MARTT:
        from vlcgan.trainer_vlc import GANTrainer
    else:
        from vlcgan.trainer import GANTrainer

    dir_path = args.data_dir
    if any([cfg.USE_MARTT, cfg.USE_MART, cfg.USE_TRANSFORMER]):
        assert cfg.USE_MARTT ^ cfg.USE_MART ^ cfg.USE_TRANSFORMER, "Choose one of Transformer (Non-recurrent) or MART or MARTT"

    random.seed(0)
    torch.manual_seed(0)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(0)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset = TextDataset(cfg.DATA_DIR, 'train',
        #                       imsize=cfg.IMSIZE,
        #                       transform=image_transform)
        #assert dataset
        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))
            vid = torch.stack(vid).permute(1, 0, 2, 3)
            return vid

        video_len = 5
        n_channels = 3
        video_transforms = functools.partial(video_transform, image_transform=image_transforms)

        if args.dataset == 'pororo':
            counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()
            print("----------------------------------------------------------------------------------")
            print("Preparing TRAINING dataset")
            base = data.VideoFolderDataset(dir_path, counter = counter, cache = dir_path, min_len = 4, mode="train")
            if cfg.USE_MARTT:
                storydataset = data.StoryDataset(base, dir_path, video_transforms, cfg.MARTT.num_hidden_layers,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)
                imagedataset = data.ImageDataset(base, dir_path, image_transforms, cfg.MARTT.num_hidden_layers,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)
            else:
                storydataset = data.StoryDataset(base, dir_path, video_transforms,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)
                imagedataset = data.ImageDataset(base, dir_path, image_transforms,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)

        elif args.dataset == 'flintstones':

            print("----------------------------------------------------------------------------------")
            print("Preparing TRAINING dataset")
            base = data.VideoFolderDataset(dir_path, cache = dir_path, min_len = 4, mode="train")
            if cfg.USE_MARTT:
                storydataset = data.StoryDataset(base, video_transforms, cfg.MARTT.num_hidden_layers,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)
                imagedataset = data.ImageDataset(base, image_transforms, cfg.MARTT.num_hidden_layers,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT,
                                                 densecap=cfg.DENSECAP_DUAL)
            else:
                storydataset = data.StoryDataset(base, video_transforms,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL,
                                                 densecap=cfg.DENSECAP_DUAL)
                imagedataset = data.ImageDataset(base, image_transforms,
                                                 return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL,
                                                 densecap=cfg.DENSECAP_DUAL)

        else:
            raise ValueError


        if cfg.USE_MART:
            cfg.MART.raw_glove_path = '/playpen-ssd/adyasha/projects/data/glove.840B.300d.txt'
            if cfg.MART.vocab_glove_path == '':
                cfg.MART.vocab_glove_path = os.path.join(dir_path, 'martgan_embeddings.mat')
            storydataset.vocab.extract_glove(cfg.MART.raw_glove_path, cfg.MART.vocab_glove_path)
            cfg.MART.pretrained_embeddings = cfg.MART.vocab_glove_path
        if cfg.USE_MARTT:
            cfg.MARTT.raw_glove_path = '/playpen-ssd/adyasha/projects/data/glove.840B.300d.txt'
            if cfg.MARTT.vocab_glove_path == '':
                cfg.MARTT.vocab_glove_path = os.path.join(dir_path, 'marttgan_embeddings.mat')
            storydataset.vocab.extract_glove(cfg.MARTT.raw_glove_path, cfg.MARTT.vocab_glove_path)
            cfg.MARTT.pretrained_embeddings = cfg.MARTT.vocab_glove_path

        print('Using config:')
        pprint.pprint(cfg)

        imageloader = torch.utils.data.DataLoader(
            imagedataset, batch_size=cfg.TRAIN.IM_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        storyloader = torch.utils.data.DataLoader(
            storydataset, batch_size=cfg.TRAIN.ST_BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        print("----------------------------------------------------------------------------------")
        print("Preparing VALIDATION dataset")


        test_dir_path = dir_path

        if args.dataset == 'pororo':
            base_test = data.VideoFolderDataset(test_dir_path, counter, test_dir_path, 4, mode="val")
            if cfg.USE_MARTT:
                testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms, cfg.MARTT.num_hidden_layers,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT)
            else:
                testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        else:
            base_test = data.VideoFolderDataset(test_dir_path, test_dir_path, 4, mode="val")
            if cfg.USE_MARTT:
                testdataset = data.StoryDataset(base_test, video_transforms, cfg.MARTT.num_hidden_layers,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT)
            else:
                testdataset = data.StoryDataset(base_test, video_transforms,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=20,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

        # update vocab config parameters for MART
        if cfg.USE_MART:
            cfg.MART.vocab_size = len(storydataset.vocab)
            cfg.MART.max_t_len = storydataset.max_len
            cfg.MART.max_position_embeddings = storydataset.max_len

        # update vocab config parameters for MART
        if cfg.USE_MARTT:
            cfg.MARTT.vocab_size = len(storydataset.vocab)
            cfg.MARTT.max_t_len = storydataset.max_len
            cfg.MARTT.max_position_embeddings = storydataset.max_len

        if cfg.USE_MARTT or cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL:
            cfg.VOCAB_SIZE = len(storydataset.vocab)

        cfg.DATASET_NAME = args.dataset

        algo = GANTrainer(cfg, output_dir, ratio = 1.0)
        algo.train(imageloader, storyloader, testloader, cfg.STAGE)
    else:
        def video_transform(video, image_transform):
            vid = []
            for im in video:
                vid.append(image_transform(im))
            vid = torch.stack(vid).permute(1, 0, 2, 3)
            return vid

        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        video_transforms = functools.partial(video_transform, image_transform=image_transforms)


        test_dir_path = dir_path

        if args.dataset == 'pororo':
            counter = np.load(os.path.join(dir_path, 'frames_counter.npy'), allow_pickle=True).item()
            base_test = data.VideoFolderDataset(test_dir_path, counter, test_dir_path, 4, mode="test")
            if cfg.USE_MARTT:
                testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms, cfg.MARTT.num_hidden_layers,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT)
            else:
                testdataset = data.StoryDataset(base_test, test_dir_path, video_transforms,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        else:
            base_test = data.VideoFolderDataset(test_dir_path, test_dir_path, 4, mode="test")
            if cfg.USE_MARTT:
                testdataset = data.StoryDataset(base_test, video_transforms, cfg.MARTT.num_hidden_layers,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL or cfg.USE_MARTT)
            else:
                testdataset = data.StoryDataset(base_test, video_transforms,
                                                return_caption=cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL)

        if cfg.USE_MART:
            testdataset.init_mart_vocab()
            print("Built vocabulary of %s words" % len(testdataset.vocab))
            if cfg.MART.vocab_glove_path == '':
                cfg.MART.vocab_glove_path = os.path.join(dir_path, 'martgan_embeddings.mat')
            testdataset.vocab.extract_glove(cfg.MART.raw_glove_path, cfg.MART.vocab_glove_path)
            cfg.MART.pretrained_embeddings = cfg.MART.vocab_glove_path

        # update vocab config parameters for MART
        if cfg.USE_MART:
            cfg.MART.vocab_size = len(testdataset.vocab)
            cfg.MART.max_t_len = testdataset.max_len
            cfg.MART.max_position_embeddings = testdataset.max_len
        if cfg.USE_MARTT:
            cfg.MARTT.raw_glove_path = '/playpen-ssd/adyasha/projects/data/glove.840B.300d.txt'
            if cfg.MARTT.vocab_glove_path == '':
                cfg.MARTT.vocab_glove_path = os.path.join(dir_path, 'marttgan_embeddings.mat')
            testdataset.vocab.extract_glove(cfg.MARTT.raw_glove_path, cfg.MARTT.vocab_glove_path)
            cfg.MARTT.pretrained_embeddings = cfg.MARTT.vocab_glove_path

        if cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL:
            cfg.VOCAB_SIZE = len(testdataset.vocab)

        # update vocab config parameters for MART
        if cfg.USE_MARTT:
            cfg.MARTT.vocab_size = len(testdataset.vocab)
            cfg.MARTT.max_t_len = testdataset.max_len
            cfg.MARTT.max_position_embeddings = testdataset.max_len

        if cfg.USE_MARTT or cfg.USE_MART or cfg.IMG_DUAL or cfg.STORY_DUAL:
            cfg.VOCAB_SIZE = len(testdataset.vocab)

        testloader = torch.utils.data.DataLoader(
            testdataset, batch_size=24,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(cfg, output_dir)
        algo.sample(testloader, args.checkpoint_file, os.path.join(output_dir + '_r1.0', 'Eval'), cfg.STAGE)
