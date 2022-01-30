import os, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from random import randrange
from collections import Counter
import json
from parse_utils import indexsublist, get_const_tags, get_tag_weights

unique_characters = ["Wilma", "Fred", "Betty", "Barney", "Dino", "Pebbles", "Mr Slate"]

class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 vocab_file,
                 annotations_file,
                 vocab_from_file=False,
                 unk_word="[UNK]",
                 pad_word="[PAD]"):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            print('Reading vocabulary from %s file!' % self.vocab_file)
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab['word2idx']
                self.idx2word = vocab['idx2word']
            print('Vocabulary successfully loaded from %s file!' % self.vocab_file)
        else:
            print("Building voabulary from scratch")
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({
                    'word2idx': self.word2idx,
                    'idx2word': self.idx2word
                }, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        annotations = json.load(open(self.annotations_file, 'r'))
        print("Tokenizing captions")
        for globalID, d in tqdm(annotations.items()):
            tokens = d["tokens"]
            counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def load_glove(self, filename):
        """ returns { word (str) : vector_embedding (torch.FloatTensor) }
        """
        glove = {}
        with open(filename) as f:
            for line in tqdm(f.readlines()):
                values = line.strip("\n").split(" ")  # space separator
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector
        return glove

    def extract_glove(self, raw_glove_path, vocab_glove_path, glove_dim=300):

        if os.path.exists(vocab_glove_path):
            print("Pre-extracted embedding matrix exists at %s" % vocab_glove_path)
        else:
            # Make glove embedding.
            print("Loading glove embedding at path : {}.\n".format(raw_glove_path))
            glove_full = self.load_glove(raw_glove_path)
            print("Glove Loaded, building word2idx, idx2word mapping.\n")
            idx2word = {v: k for k, v in self.word2idx.items()}

            glove_matrix = np.zeros([len(self.word2idx), glove_dim])
            glove_keys = glove_full.keys()
            for i in tqdm(range(len(idx2word))):
                w = idx2word[i]
                w_embed = glove_full[w] if w in glove_keys else np.random.randn(glove_dim) * 0.4
                glove_matrix[i, :] = w_embed
            print("vocab embedding size is :", glove_matrix.shape)
            torch.save(glove_matrix, vocab_glove_path)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache=None, min_len=4, mode='train'):
        self.lengths = []
        self.followings = {}
        self.dir_path = folder
        self.total_frames = 0

        # train_id, test_id = np.load(self.dir_path + 'train_test_ids.npy', allow_pickle=True, encoding='latin1')
        splits = json.load(open(os.path.join(self.dir_path, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]

        if os.path.exists(cache + 'following_cache' + str(min_len) +  '.npy'):
            self.followings = pickle.load(open(cache + 'following_cache' + str(min_len) + '.pkl', 'rb'))
        else:
            all_clips = train_id + val_id + test_id
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
            pickle.dump(self.followings, open(os.path.join(folder, 'following_cache' + str(min_len) + '.pkl'), 'wb'))
            
        self.filtered_followings = {}
        for i, f in self.followings.items():
            #print(f)
            if len(f) == 4:
                self.filtered_followings[i] = f
            else:
                continue
        self.followings = self.filtered_followings

        train_id = [tid for tid in train_id if tid in self.followings]
        val_id = [vid for vid in val_id if vid in self.followings]
        test_id = [tid for tid in test_id if tid in self.followings]

        if os.path.exists(os.path.join(folder, 'labels.pkl')):
            self.labels = pickle.load(open(os.path.join(folder, 'labels.pkl'), 'rb'))
        else:
            print("Computing and saving labels")
            annotations = json.load(open(os.path.join(folder, 'flintstones_annotations_v1-0.json'), 'r'))
            self.labels = {}
            for sample in annotations:
                sample_characters = [c["entityLabel"].strip().lower() for c in sample["characters"]]
                self.labels[sample["globalID"]] = [1 if c.lower() in sample_characters else 0 for c in unique_characters]
            pickle.dump(self.labels, open(os.path.join(folder, 'labels.pkl'), 'wb'))

        self.embeds = np.load(os.path.join(self.dir_path, "flintstones_use_embeddings.npy"))
        self.sent2idx = pickle.load(open(os.path.join(self.dir_path, 'flintstones_use_embed_idxs.pkl'), 'rb'))

        if mode == 'train':
            self.orders = train_id
        elif mode =='val':
            self.orders = val_id
        elif mode == 'test':
            self.orders = test_id
        else:
            raise ValueError
        print("Total number of clips {}".format(len(self.orders)))
        print("%s, %s, %s stories in training, validation and test" % (len(train_id), len(val_id), len(test_id)))

    def __getitem__(self, item):
        return [self.orders[item]] + self.followings[self.orders[item]]

    def __len__(self):
        return len(self.orders)

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, max_height, return_caption=False, out_dir=None, densecap=False):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.parsed_descriptions = json.load(open(os.path.join(self.dir_path, 'parses.json')))
        self.const_tag2id = json.load(open(os.path.join(self.dir_path, 'const_tag2idx.json')))
        self.transforms = transform
        self.labels = dataset.labels
        self.return_caption = return_caption
        self.max_height = max_height
        self.tag_padding_idx = len(self.const_tag2id)

        if self.return_caption:
            self.init_mart_vocab()
            self.max_len = self.tokenize_descriptions()
            print("Max sequence length = %s" % self.max_len)
        else:
            self.vocab = None
        self.out_dir = out_dir

        self.densecap_dataset = None

    def tokenize_descriptions(self):
        caption_lengths = []
        self.tokenized_descriptions = {}
        for globalID, d in self.parsed_descriptions.items():
            caption_lengths.append(len(d["tokens"]))
        return max(caption_lengths)

    def init_mart_vocab(self):

        vocab_file = os.path.join(self.dir_path, 'martt_vocab.pkl')
        if os.path.exists(vocab_file):
            vocab_from_file = True
        else:
            vocab_from_file = False

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'parses.json'),
                                vocab_from_file=vocab_from_file)

    def save_story(self, output, save_path = './'):
        all_image = []
        images = output['images_numpy']
        texts = output['text']
        for i in range(images.shape[0]):
            all_image.append(np.squeeze(images[i]))
        output = PIL.Image.fromarray(np.concatenate(all_image, axis = 0))
        output.save(save_path + 'image.png')
        fid = open(save_path + 'text.txt', 'w')
        for i in range(len(texts)):
            fid.write(texts[i] +'\n' )
        fid.close()
        return

    def _pad_tag_ids(self, const_tag_ids):
        new_tag_ids = []
        for tags in const_tag_ids:
            if len(tags) < self.max_height:
                new_tags = tags + [self.tag_padding_idx] * (self.max_height - len(tags))
            elif len(tags) > self.max_height:
                new_tags = tags[:(self.max_height-1)] + [tags[-1]]
            else:
                new_tags = tags
            new_tag_ids.append(new_tags)

        if len(new_tag_ids) < self.max_len:
            new_tag_ids = new_tag_ids + [[self.tag_padding_idx] * self.max_height] * (self.max_len - len(new_tag_ids))
        elif len(new_tag_ids) > self.max_len:
            new_tag_ids = new_tag_ids[:self.max_len]
        else:
            pass

        return new_tag_ids

    def _sentence_to_idx(self, sentence_parse):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """

        assert len(sentence_parse["tree"].keys()) == 1
        root_node = list(sentence_parse["tree"].keys())[0]
        tokens, const_tags, tree_mask = get_const_tags(root_node, sentence_parse["tree"][root_node])
        tree_height = tree_mask.shape[-1]
        # const_tag_weights = np.array([get_tag_weights(t, self.const_tag2id) for t in const_tags])
        const_tag_ids = self._pad_tag_ids([[self.const_tag2id[t] for t in tags] for tags in const_tags])
        # print(const_tag_ids)


        tokens_l = len(tokens)
        if tokens_l < self.max_len:
            mask = np.zeros((self.max_len, self.max_len, tree_height))
            # padding_mask_1 = np.zeros((valid_l, self.max_len-valid_l, mask.shape[-1]))
            # padding_mask_2 = np.zeros((self.max_len-valid_l, valid_l, mask.shape[-1]))
            # mask = np.concatenate((np.concatenate((mask, padding_mask_1), axis=0), padding_mask_2), axis=1)
            mask[:tokens_l, :tokens_l, :] = tree_mask
            tokens += [self.vocab.pad_word] * (self.max_len - tokens_l)
            # const_tag_ids = const_tag_ids + [[self.tag_padding_idx]*tree_height]*(self.max_len-tokens_l)
            # const_tag_weights = np.concatenate((const_tag_weights, np.zeros((self.max_len - tokens_l, len(self.const_tag2id)))), axis=0)
        elif tokens_l > self.max_len:
            tokens = tokens[:self.max_len]
            mask = tree_mask[:self.max_len, :self.max_len, :]
            # const_tag_weights = const_tag_weights[:self.max_len, :]
            # const_tag_ids = const_tag_ids[:self.max_len]
        else:
            mask = tree_mask

        if tree_height < self.max_height:
            mask = np.concatenate((np.ones((self.max_len, self.max_len, self.max_height-tree_height)), mask), axis=-1)
            # const_tag_ids = [t + [self.tag_padding_idx]*(self.max_height-len(t)) for t in const_tag_ids]
        elif tree_height > self.max_height:
            filter_indices = list(range(0, self.max_height-1)) + [-1]
            mask = np.take(mask, filter_indices, axis=-1)
            # const_tag_ids = [t[:(self.max_height-1)] + [t[-1]] for t in const_tag_ids]
        else:
            pass

        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in tokens]
        # return input_ids, [np.array(t) for t in const_tag_ids], np.flip(mask, axis=-1)
        return input_ids, const_tag_ids, np.flip(mask, axis=-1)


    def __getitem__(self, item):
        lists = self.dataset[item]
        labels = []
        images = []
        text = []
        input_ids = []
        masks= []
        sent_embeds = []
        const_tag_ids = []
        assert len(lists) == 5
        for idx, globalID in enumerate(lists):
            if self.out_dir:
                im = PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, idx))).convert('RGB')
            else:
                arr = np.load(os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy'))
                n_frames = arr.shape[0]
                im = arr[randrange(n_frames)]
            images.append(np.expand_dims(np.array(im), axis = 0))
            text.append(' '.join(self.parsed_descriptions[globalID]['tokens']))
            labels.append(np.expand_dims(self.labels[globalID], axis = 0))
            sent_embeds.append(np.expand_dims(self.dataset.embeds[self.dataset.sent2idx[globalID]], axis = 0))

            if self.return_caption:
                input_id, const_tag_id, mask = self._sentence_to_idx(self.parsed_descriptions[globalID])
                input_ids.append(np.expand_dims(input_id, axis=0))
                masks.append(np.expand_dims(mask, axis=0))
                # const_tag_ids.append(np.expand_dims(const_tag_id, axis=0))
                const_tag_ids.append(const_tag_id)

        sent_embeds = np.concatenate(sent_embeds, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        images = np.concatenate(images, axis = 0)
        # image is T x H x W x C
        transformed_images = self.transforms(images)
        # After transform, image is C x T x H x W

        sent_embeds = torch.tensor(sent_embeds)
        labels = torch.tensor(np.array(labels).astype(np.float32))

        data_item = {'images': transformed_images, 'text':text, 'description': sent_embeds, 'images_numpy':images, 'labels':labels}

        if self.return_caption:
            input_ids = torch.tensor(np.concatenate(input_ids))
            masks = torch.tensor(np.concatenate(masks, axis=0)).permute(0, 3, 2, 1)
            const_tag_ids = torch.tensor(const_tag_ids)
            data_item.update({'input_ids': input_ids, 'masks': masks, 'tag_ids': const_tag_ids})

        if self.densecap_dataset:
            boxes, caps, caps_len = [], [], []
            for idx, v in enumerate(lists):
                img_id = str(v).replace('.png', '')[2:-1]
                path = img_id + '.png'
                boxes.append(torch.as_tensor([ann['box'] for ann in self.densecap_dataset[path]], dtype=torch.float32))
                caps.append(torch.as_tensor([ann['cap_idx'] for ann in self.densecap_dataset[path]], dtype=torch.long))
                caps_len.append(torch.as_tensor([sum([1 for k in ann['cap_idx'] if k!= 0]) for ann in self.densecap_dataset[path]], dtype=torch.long))
            targets = {
                'boxes': torch.cat(boxes),
                'caps': torch.cat(caps),
                'caps_len': torch.cat(caps_len),
            }
            data_item.update(targets)

        return data_item

    def __len__(self):
        return len(self.dataset.orders)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, max_height, return_caption=False, densecap=False):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.transforms = transform
        self.parsed_descriptions = json.load(open(os.path.join(self.dir_path, 'parses.json')))
        self.const_tag2id = json.load(open(os.path.join(self.dir_path, 'const_tag2idx.json')))
        self.tag_padding_idx = len(self.const_tag2id)
        self.labels = dataset.labels
        self.return_caption = return_caption
        self.max_height = max_height
        self.tag_padding_idx = len(self.const_tag2id)

        if self.return_caption:
            self.init_mart_vocab()
            self.max_len = self.tokenize_descriptions()
            print("Max sequence length = %s" % self.max_len)
        else:
            self.vocab = None

        # if densecap:
        #     self.densecap_dataset = DenseCapDataset(self.dir_path)
        # else:
        self.densecap_dataset = None

    def tokenize_descriptions(self):
        caption_lengths = []
        self.tokenized_descriptions = {}
        for globalID, d in self.parsed_descriptions.items():
            caption_lengths.append(len(d["tokens"]))
        return max(caption_lengths)

    def _pad_tag_ids(self, const_tag_ids):
        new_tag_ids = []
        for tags in const_tag_ids:
            if len(tags) < self.max_height:
                new_tags = tags + [self.tag_padding_idx] * (self.max_height - len(tags))
            elif len(tags) > self.max_height:
                new_tags = tags[:(self.max_height - 1)] + [tags[-1]]
            else:
                new_tags = tags
            new_tag_ids.append(new_tags)

        if len(new_tag_ids) < self.max_len:
            new_tag_ids = new_tag_ids + [[self.tag_padding_idx] * self.max_height] * (self.max_len - len(new_tag_ids))
        elif len(new_tag_ids) > self.max_len:
            new_tag_ids = new_tag_ids[:self.max_len]
        else:
            pass

        return new_tag_ids

    def _sentence_to_idx(self, sentence_parse):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """

        assert len(sentence_parse["tree"].keys()) == 1
        root_node = list(sentence_parse["tree"].keys())[0]
        tokens, const_tags, tree_mask = get_const_tags(root_node, sentence_parse["tree"][root_node])
        tree_height = tree_mask.shape[-1]
        # const_tag_weights = np.array([get_tag_weights(t, self.const_tag2id) for t in const_tags])
        const_tag_ids = self._pad_tag_ids([[self.const_tag2id.get(t, self.const_tag2id['UNK']) for t in tags] for tags in const_tags])

        tokens_l = len(tokens)
        # print(tokens_l, tree_height)
        if tokens_l < self.max_len:
            mask = np.zeros((self.max_len, self.max_len, tree_height))
            # padding_mask_1 = np.zeros((valid_l, self.max_len-valid_l, mask.shape[-1]))
            # padding_mask_2 = np.zeros((self.max_len-valid_l, valid_l, mask.shape[-1]))
            # mask = np.concatenate((np.concatenate((mask, padding_mask_1), axis=0), padding_mask_2), axis=1)
            mask[:tokens_l, :tokens_l, :] = tree_mask
            tokens += [self.vocab.pad_word] * (self.max_len - tokens_l)
            # const_tag_ids = const_tag_ids + [[self.tag_padding_idx]*tree_height]*(self.max_len-tokens_l)
            # const_tag_weights = np.concatenate((const_tag_weights, np.zeros((self.max_len - tokens_l, len(self.const_tag2id)))), axis=0)
        elif tokens_l > self.max_len:
            tokens = tokens[:self.max_len]
            mask = tree_mask[:self.max_len, :self.max_len, :]
            # const_tag_weights = const_tag_weights[:self.max_len, :]
            # const_tag_ids = const_tag_ids[:self.max_len]
        else:
            mask = tree_mask

        if tree_height < self.max_height:
            mask = np.concatenate((np.ones((self.max_len, self.max_len, self.max_height-tree_height)), mask), axis=-1)
            # new_const_tag_ids = [t + [self.tag_padding_idx]*(self.max_height-len(t)) for t in const_tag_ids]

        elif tree_height > self.max_height:
            filter_indices = list(range(0, self.max_height-1)) + [-1]
            mask = np.take(mask, filter_indices, axis=-1)
            # new_const_tag_ids = []
            # for t in const_tag_ids:
            #     if len(t) < self.max_height:
            #         new_const_tag_ids.append(t + [self.tag_padding_idx]*(self.max_height-len(t)))
            #     else:
            #         new_const_tag_ids.append(t[:(self.max_height-1)] + [t[-1]])
        else:
            # new_const_tag_ids = const_tag_ids
            pass

        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in tokens]
        # return input_ids, [np.array(t) for t in new_const_tag_ids], np.flip(mask, axis=-1)
        return input_ids, const_tag_ids, np.flip(mask, axis=-1)


    def init_mart_vocab(self):

        vocab_file = os.path.join(self.dir_path, 'martt_vocab.pkl')
        if os.path.exists(vocab_file):
            vocab_from_file = True
        else:
            vocab_from_file = False

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'parses.json'),
                                vocab_from_file=vocab_from_file)

    def __getitem__(self, item):

        # single image input
        globalID = self.dataset[item][0]
        arr = np.load(os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy'))
        n_frames = arr.shape[0]
        im = arr[randrange(n_frames)]

        image = np.array(im)
        text = ' '.join(self.parsed_descriptions[globalID]["tokens"])
        label = np.array(self.labels[globalID]).astype(np.float32)
        sent_embed = self.dataset.embeds[self.dataset.sent2idx[globalID]]

        input_id = None
        mask = None
        if self.return_caption:
            input_id, const_tag_id, mask = self._sentence_to_idx(self.parsed_descriptions[globalID])
            input_id = np.array(input_id)
            mask = np.array(mask)

        # input ofr conditional vector
        lists = self.dataset[item]
        sent_embeds = []
        for idx, globalID in enumerate(lists):
            sent_embeds.append(np.expand_dims(self.dataset.embeds[self.dataset.sent2idx[globalID]], axis=0))
        sent_embeds = np.concatenate(sent_embeds, axis=0)

        ##
        sent_embeds = torch.tensor(sent_embeds)
        image = self.transforms(image)
        data_item = {'images': image, 'text':text, 'description': sent_embed,
                     'labels':label, 'content': sent_embeds}

        if self.return_caption:
            input_id = torch.tensor(input_id)
            mask = torch.tensor(mask).permute(2, 1, 0)
            const_tag_id = torch.tensor(const_tag_id)
            data_item.update({'input_id': input_id, 'mask': mask, 'tag_ids': const_tag_id})

        if self.densecap_dataset:
            path = globalID + '.png'
            try:
                _ = self.densecap_dataset[path]
            except KeyError:
                shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
                video_len = int(longer / shorter)
                raise KeyError

            boxes = torch.as_tensor([ann['box'] for ann in self.densecap_dataset[path]], dtype=torch.float32)
            caps = torch.as_tensor([ann['cap_idx'] for ann in self.densecap_dataset[path]], dtype=torch.long)
            caps_len = torch.as_tensor([sum([1 for k in ann['cap_idx'] if k!= 0]) for ann in self.densecap_dataset[path]], dtype=torch.long)
            targets = {
                'boxes': boxes,
                'caps': caps,
                'caps_len': caps_len,
            }
            data_item.update(targets)

        return data_item

    def __len__(self):
        return len(self.dataset.orders)