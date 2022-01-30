import os, re
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import pickle
from random import randrange

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, im_input_size, mode='train'):

        self.dir_path = folder
        splits = json.load(open(os.path.join(self.dir_path, 'train-val-test_split.json'), 'r'))
        train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
        self.labels = pickle.load(open(os.path.join(folder, 'labels.pkl'), 'rb'))

        if mode == 'train':
            self.ids = train_ids
            self.transform = transforms.Compose([
                Image.fromarray,
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.ids = val_ids if mode == "val" else test_ids
            self.transform = transforms.Compose([
                Image.fromarray,
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, item):

        globalID= self.ids[item]
        arr = np.load(os.path.join(self.dir_path, 'video_frames', globalID + '.npy'))
        n_frames = arr.shape[0]
        im = arr[randrange(n_frames)]
        label = self.labels[globalID]
        return self.transform(im), torch.Tensor(label)

    def __len__(self):
        return len(self.ids)


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, im_input_size,
                 out_img_folder = None,
                 mode='train',
                 video_len = 5):
        self.followings = {}
        self.data_folder = data_folder
        self.labels = pickle.load(open(os.path.join(data_folder, 'labels.pkl'), 'rb'))
        self.video_len = video_len
        min_len = video_len-1

        splits = json.load(open(os.path.join(self.data_folder, 'train-val-test_split.json'), 'r'))
        train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]

        if os.path.exists(os.path.join(data_folder, 'following_cache' + str(video_len-1) +  '.pkl')):
            self.followings = pickle.load(open(os.path.join(data_folder, 'following_cache' + str(video_len-1) + '.pkl'), 'rb'))
        else:
            all_clips = train_ids + val_ids + test_ids
            all_clips.sort()
            for idx, clip in enumerate(tqdm(all_clips, desc="Counting total number of frames")):
                season, episode = int(clip.split('_')[1]), int(clip.split('_')[3])
                has_frames = True
                for c in all_clips[idx+1:idx+min_len+1]:
                    s_c, e_c = int(c.split('_')[1]), int(c.split('_')[3])
                    if s_c != season or e_c != episode:
                        has_frames = False
                        break
                if has_frames:
                    self.followings[clip] = all_clips[idx+1:idx+min_len+1]
                else:
                    continue
            pickle.dump(self.followings, open(os.path.join(self.data_folder, 'following_cache' + str(min_len) + '.pkl'), 'wb'))
            
        self.filtered_followings = {}
        for i, f in self.followings.items():
            #print(f)
            if len(f) == 4:
                self.filtered_followings[i] = f
            else:
                continue
        self.followings = self.filtered_followings

        train_ids = [tid for tid in train_ids if tid in self.followings]
        val_ids = [vid for vid in val_ids if vid in self.followings]
        test_ids = [tid for tid in test_ids if tid in self.followings]

        # print(list(self.followings.keys())[:10])

        if mode == 'train':
            self.ids = train_ids
            self.transform = transforms.Compose([
                # Image.fromarray,
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.ids = val_ids if mode == "val" else test_ids
            self.transform = transforms.Compose([
                # Image.fromarray,
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


        self.out_dir = out_img_folder

    def __getitem__(self, item):

        globalIDs = [self.ids[item]] + self.followings[self.ids[item]]

        images = []
        for idx, globalID in enumerate(globalIDs):
            if self.out_dir:
                im = Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, idx))).convert('RGB')
                images.append(im)
            else:
                arr = np.load(os.path.join(self.data_folder, 'video_frames', globalID + '.npy'))
                n_frames = arr.shape[0]
                im = arr[randrange(n_frames)]
                images.append(np.expand_dims(np.array(im), axis = 0))

        # print([(type(im)) for im in images])

        labels = [self.labels[globalID] for globalID in globalIDs]
        return torch.cat([self.transform(image).squeeze(0) for image in images], dim=0), torch.tensor(np.vstack(labels))

    def __len__(self):
        return len(self.ids)
