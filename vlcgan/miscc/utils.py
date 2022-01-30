import os
import errno
import numpy as np
import PIL
from copy import deepcopy
from .config import cfg
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # print(context.shape)

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # print(context.shape, query.shape)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL

    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)

    attn = attn * gamma1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

# TODO: Masked cross-entropy loss
def compute_dual_discriminator_loss(netD, img_features, real_captions, gpus, mode='image'):

    if len(gpus) > 1:
        netD = torch.nn.DataParallel(netD)
    real_input_ids, real_masks = real_captions

    if mode == 'story':
        batch_size = img_features.size(0)
        story_len = img_features.size(1)
        mask = torch.ones(batch_size, story_len)
        if len(gpus) > 0:
            mask = mask.cuda()
        outputs = netD.get_captions(img_features, mask, real_input_ids)
    else:
        outputs = netD.get_captions(img_features, real_input_ids)
    # print(outputs.shape)
    # print(real_captions.shape)
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    # criterion = LabelSmoothingLoss(outputs.size(-1), 0.1)

    errDual = criterion(outputs, real_input_ids.view(-1))
    return errDual


def compute_dual_densecap_loss(netDual, imgs, targets, gpus, cfg, mode='image', ):

    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print(mode)
    print(imgs.shape)
    print({k:v.shape for k,v in targets.items()})

    if len(gpus) > 1:
        netDual = torch.nn.DataParallel(netDual)

    if mode == 'image':
        bs, n_channels, h, w = imgs.size()
        imgs = [F.relu(imgs[i].clone(), inplace=True) for i in range(bs)]
        imgs_ = [transform(imgs[i].clone()) for i in range(bs)]
        imgs_ = [F.relu(imgs_[i], inplace=True) for i in range(bs)]
        targets_ = [{'boxes': b, 'caps': c, 'caps_len': c_len} for b, c, c_len in zip(targets['boxes'], targets['caps'], targets['caps_len'])]
        # targets_ = [{'boxes': b, 'caps': c, 'caps_len': c_len} for i in zip(targets['boxes'], targets['caps'], targets['caps_len'])]

    elif mode == 'story':
        bs, n_channel, video_len, h, w = imgs.size()
        imgs = imgs.permute(0, 2, 1, 3, 4).view(bs*video_len, n_channel, h, w)
        imgs_ = [transform(img) for img in imgs]
        targets_ = []

    else:
        raise ValueError
    #
    #     imgs = imgs.view(-1, imgs.size(-3), imgs.size(-2), imgs.size(-1))
    #     targets['boxes'] = targets['boxes'].view(-1, targets['boxes'].size)
    #     targets['boxes'] = targets['boxes'].view(-1, targets['boxes'].size)
    #     targets['boxes'] = targets['boxes'].view(-1, targets['boxes'].size)

    losses = netDual(imgs_, targets_)
    detect_loss = losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                  losses['loss_classifier'] + losses['loss_box_reg']
    caption_loss = losses['loss_caption']

    errDual = cfg.DENSECAP.detect_loss_weight*detect_loss + cfg.DENSECAP.caption_loss_weight*caption_loss

    loss_report = {
        mode + ' Object Detection Loss --> ': detect_loss.data.item(),
        mode + ' Dense Caption Loss --> ': caption_loss.data.item(),
    }

    return errDual, loss_report


def compute_contrastive_loss(netD, img_features, text_features, gpus):

    loss_fct = nn.CrossEntropyLoss()
    # print(img_features.shape, text_features.shape)
    temp = netD.img_pooler(img_features).squeeze()
    # print(temp.shape)
    I_f = netD.img_joint_fc(temp)
    T_f = netD.txt_joint_fc(text_features)

    # normalized features
    img_projections = I_f / I_f.norm(dim=-1, keepdim=True)
    text_projections = T_f / T_f.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = netD.logit_scale.exp()
    logits_per_image = logit_scale * img_projections @ text_projections.t()

    bs = img_features.shape[0]
    labels = torch.arange(bs)
    if gpus:
        labels = labels.cuda()
    loss_i = loss_fct(logits_per_image, labels)
    loss_t = loss_fct(logits_per_image.t(), labels)
    loss = (loss_i + loss_t)/2
    return loss


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions, gpus, mode='image',
                               dual=False,
                               real_captions=None,
                               contrastive=False,
                               real_caption_embeds=None):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()

    real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake), gpus)

    if mode == 'story':
        real_features_st = real_features
        fake_features = fake_features.mean(1).squeeze()
        real_features = real_features.mean(1).squeeze()

    # real pairs
    inputs = (real_features, cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = \
        nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5

    loss_report = {
        mode + ' Fake/Real Discriminator Loss (Real pairs) --> ': errD_real.data.item(),
        mode + ' Fake/Real Discriminator Loss (Wrong pairs) --> ': errD_wrong.data.item(),
        mode + 'Fake/Real Discriminator Loss (Fake pairs) --> ': errD_fake.data.item(),
    }

    # if dual:
    #     if mode == 'story':
    #         errDual = compute_dual_discriminator_loss(netD, real_features_st, real_captions, gpus, mode=mode)
    #     else:
    #         errDual = compute_dual_discriminator_loss(netD, real_features, real_captions, gpus, mode=mode)
    #     # TODO: Add a weight for ErrDual
    #     errD = errD + errDual
    #     loss_report[mode + ' Dual Discriminator Loss --> '] = errDual.data.item()

    if contrastive:
        if mode == "story":
            vid_len, embed_dim = real_caption_embeds.shape[1], real_caption_embeds.shape[2]
            real_caption_embeds = real_caption_embeds.view(-1, vid_len*embed_dim)
        errContrast = compute_contrastive_loss(netD, real_features, real_caption_embeds, gpus)
        loss_report[mode + ' Constrastive Discriminator Loss --> '] = errContrast.data.item()
        errD = errD + errContrast

    if netD.cate_classify is not None:
        # print('Real features shape', real_features.shape)
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, real_features, gpus)
        # print('Categorical logits shape', cate_logits.shape)
        cate_logits = cate_logits.squeeze()
        errD = errD + 1.0 * cate_criterion(cate_logits, real_catelabels)
        acc = get_multi_acc(cate_logits.cpu().data.numpy(), real_catelabels.cpu().data.numpy())
        loss_report[mode + ' Character Classifier Accuracy (Discriminator) --> '] = acc

    return errD, loss_report

def compute_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions, gpus, mode='image',
                           dual=False, real_captions=None, contrastive=False, fake_caption_embeds=None):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    if mode == 'story':
        fake_features_st = fake_features
        fake_features = torch.mean(fake_features, dim=1).squeeze()
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake

    loss_report = {
        mode + ' Fake/Real Generator Loss (Fake pairs) --> ': errD_fake.data.item(),
    }

    if netD.cate_classify is not None:
        # print('Fake features shape', fake_features.shape)
        cate_logits = nn.parallel.data_parallel(netD.cate_classify, fake_features, gpus)
        cate_logits = cate_logits.mean(dim=-1).mean(dim=-1)
        cate_logits = cate_logits.squeeze()
        # print(cate_logits.shape, fake_catelabels.shape)
        errD_fake = errD_fake + 1.0 * cate_criterion(cate_logits, fake_catelabels)
        acc = get_multi_acc(cate_logits.cpu().data.numpy(), fake_catelabels.cpu().data.numpy())
        loss_report[mode + ' Character Classifier Accuracy (Generator) --> '] = acc
    # TODO: Add weight for dual learning loss

    if dual:
        if mode == 'story':
            errDual = compute_dual_discriminator_loss(netD, fake_imgs, real_captions, gpus, mode=mode)
        else:
            errDual = compute_dual_discriminator_loss(netD, fake_imgs, real_captions, gpus, mode=mode)

        errD_fake = errD_fake + errDual
        loss_report[mode + ' Dual Generator Loss --> '] = errDual.data.item()

    if contrastive:
        if mode == "story":
            vid_len, embed_dim = fake_caption_embeds.shape[1], fake_caption_embeds.shape[2]
            fake_caption_embeds = fake_caption_embeds.view(-1, vid_len*embed_dim)
        errContrast_fake = compute_contrastive_loss(netD, fake_features, fake_caption_embeds, gpus)
        loss_report[mode + ' Constrastive Generator Loss --> '] = errContrast_fake.data.item()
        errD_fake = errD_fake + errContrast_fake

    return errD_fake, loss_report


#############################
def weights_init(module):

    initializer_range = 0.02

    """ Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    else:
        classname = module.__class__.__name__
        if classname == "MyInceptionFeatureExtractor":
            pass
        elif classname == 'BertLayerNorm':
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            module.weight.data.normal_(0.0, 0.02)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, texts, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_epoch_%03d.png' % 
            (image_dir, epoch), normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

    if texts is not None:
        fid = open('%s/lr_fake_samples_epoch_%03d.txt' % (image_dir, epoch), 'wb')
        for i in range(num):
            fid.write(str(i) + ':' + texts[i] + '\n')
        fid.close()

##########################\
def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(ground_truth, images, texts, idx, image_dir, epoch, lr = False, upscale = False):
    # print("Generated Images shape: ", images.shape)

    video_len = cfg.VIDEO_LEN
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        gts = []
        for i in range(ground_truth.shape[0]):
            if upscale:
                gts.append(vutils.make_grid(torch.transpose(nn.functional.interpolate(ground_truth[i], scale_factor=2, mode='nearest'), 0, 1), video_len))
            else:
                gts.append(vutils.make_grid(torch.transpose(ground_truth[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        # print("Ground Truth shape, Generated Images shape: ", gts.shape, all_images.shape)
        all_images = np.concatenate([all_images, gts], axis = 1)

    output = PIL.Image.fromarray(all_images)
    if lr:
         output.save('%s/lr_samples_epoch_%03d_%03d.png' % (image_dir, epoch, idx))
    else:
        output.save('%s/fake_samples_epoch_%03d_%03d.png' % (image_dir, epoch, idx))

    if texts is not None:
        fid = open('%s/fake_samples_epoch_%03d_%03d.txt' % (image_dir, epoch, idx), 'w')
        for idx in range(images.shape[0]):
            fid.write(str(idx) + '--------------------------------------------------------\n')
            for i in range(len(texts)):
                fid.write(texts[i][idx] +'\n' )
            fid.write('\n\n')
        fid.close()
    return 

def get_multi_acc(predict, real):
    predict = 1/(1+np.exp(-predict))
    correct = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if real[i][j] == 1 and predict[i][j]>=0.5:
                correct += 1
    acc = correct / float(np.sum(real))
    return acc

def save_model(netG, netD_im, netD_st, optimizerG, im_Optimizer, st_Optimizer, epoch, model_dir):
    torch.save({
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'optimizer_state_dict': optimizerG.state_dict(),
    }, '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    # torch.save(
    #     netG.state_dict(),
    #     '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    if netD_im:
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netD_im.state_dict(),
            'optimizer_state_dict': im_Optimizer.state_dict(),
        }, '%s/netD_im_epoch_%d.pth' % (model_dir, epoch))
        # torch.save(
        #     netD_im.state_dict(),
        #     '%s/netD_im_epoch_last.pth' % (model_dir))
    if netD_st:
        torch.save({
            'epoch': epoch,
            'netG_state_dict': netD_st.state_dict(),
            'optimizer_state_dict': st_Optimizer.state_dict(),
        }, '%s/netD_st_epoch_%d.pth' % (model_dir, epoch))
        # torch.save(
        #     netD_st.state_dict(),
        #     '%s/netD_st_epoch_last.pth' % (model_dir))
    print('Saved G/D models')

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_test_samples(netG, dataloader, save_path, epoch, mart=False, martt=False):
    print('Generating Test Samples...')
    save_images = []
    save_labels = []

    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            #print('Processing at ' + str(i))
            real_cpu = batch['images']
            motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            catelabel = batch['labels']
            real_imgs = Variable(real_cpu)
            motion_input = Variable(motion_input)
            content_input = Variable(content_input)
            # print(batch.keys())
            if mart or martt:
                st_input_ids = Variable(batch['input_ids'])
                st_masks = Variable(batch['masks'])
                if martt:
                    st_tag_ids = Variable(batch['tag_ids'])

            if cfg.CUDA:
                real_imgs = real_imgs.cuda()
                motion_input = motion_input.cuda()
                content_input = content_input.cuda()
                catelabel = catelabel.cuda()
                if mart or martt:
                    st_input_ids = st_input_ids.cuda()
                    st_masks = st_masks.cuda()
                    if martt:
                        st_tag_ids = st_tag_ids.cuda()

            motion_input = torch.cat((motion_input, catelabel), 2)
            #content_input = torch.cat((content_input, catelabel), 2)
            if mart:
                _, fake, _, _, _, _ = netG.sample_videos(motion_input, content_input, st_input_ids, st_masks, catelabel)
            elif martt:
                _, fake, _, _, _, _, _ = netG.sample_videos(motion_input, content_input, st_input_ids, st_masks, st_tag_ids, catelabel)
            else:
                _, fake, _,_,_,_ = netG.sample_videos(motion_input, content_input)
            save_story_results(real_cpu, fake, batch['text'], i, save_path, epoch, upscale=True if fake.shape[-1]==128 else False)
            save_images.append(fake.cpu().data.numpy())
            save_labels.append(catelabel.cpu().data.numpy())

    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images-epoch-%s.npy' % epoch, save_images)
    np.save(save_path + '/labels-epoch-%s.npy' % epoch, save_labels)

# ---------------------------------------------------------------------------------
# DAMSM Loss
# ---------------------------------------------------------------------------------

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    # w12 = torch.sum(x1 * x2, dim)
    # w1 = torch.norm(x1, 2, dim)
    # w2 = torch.norm(x2, 2, dim)
    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    if x1.dim() == 2:
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    print(x1.shape, x2.shape)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    x1_norm = torch.norm(x1, 2, dim=2, keepdim=True)
    x2_norm = torch.norm(x2, 2, dim=2, keepdim=True)
    print(x1_norm.shape, x2_norm.shape)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(x1, x2.transpose(1, 2))
    print(scores0.shape)
    norm0 = torch.bmm(x1_norm, x2_norm.transpose(1,2))
    print(norm0.shape)
    scores0 = scores0 / norm0.clamp(min=eps)

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    return scores0

def caption_loss(cap_output, captions):
    criterion = nn.CrossEntropyLoss()
    caption_loss = criterion(cap_output, captions)
    return caption_loss

def sent_loss(cnn_code, rnn_code, labels, gamma, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * gamma

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    cosine_sim_fn = nn.CosineSimilarity(dim=-1)
    for i in range(batch_size):

        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features

        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # print(word.shape, weiContext.shape)
        # -->batch_size*words_num
        # row_sim = cosine_similarity(word, weiContext)
        row_sim = cosine_sim_fn(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps
