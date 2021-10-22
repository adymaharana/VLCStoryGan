import torchvision.transforms as transforms
import argparse
import os

import flintstones_data as data
from vfid.fid_score import fid_score

def main(args):


    image_transforms = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # def video_transform(video, image_transform):
    #     vid = []
    #     for im in video:
    #         vid.append(image_transform(im))
    #     vid = torch.stack(vid).permute(1, 0, 2, 3)
    #     return vid

    # video_transforms = functools.partial(video_transform,
    #                                      image_transform=image_transforms)  # Only need to feed video later
    #
    # ref_dataset = data.StoryImageDataset(args.img_ref_dir,
    #                                         args.imsize,
    #                                         mode=args.mode,
    #                                         transform=video_transforms)
    # gen_dataset = data.StoryImageDataset(args.img_ref_dir,
    #                                         args.imsize,
    #                                         mode=args.mode,
    #                                         out_img_folder=args.img_gen_dir,
    #                                         transform=video_transforms)
    # vfid = vfid_score(ref_dataset, gen_dataset, cuda=True, normalize=True, r_cache=None)
    # print('Frechet Story Distance: ', vfid)

    # ref_dataset = data.ImageClfDataset(args.img_ref_dir,
    #                                 args.imsize,
    #                                 mode=args.mode,
    #                                 transform=image_transforms)
    # gen_dataset = data.ImageClfDataset(args.img_ref_dir,
    #                                 args.imsize,
    #                                 mode=args.mode,
    #                                 out_img_folder=args.img_gen_dir,
    #                                 transform=image_transforms)

    ref_dataset = data.StoryImageDataset(args.img_ref_dir,
                                    args.imsize,
                                    mode=args.mode,
                                    transform=image_transforms)
    gen_dataset = data.StoryImageDataset(args.img_ref_dir,
                                    args.imsize,
                                    mode=args.mode,
                                    out_img_folder=args.img_gen_dir,
                                    transform=image_transforms)

    fid = fid_score(ref_dataset, gen_dataset, cuda=True, normalize=True, r_cache=os.path.join(args.img_ref_dir, 'fid_cache_%s.npz' % args.mode), batch_size=1)
    print('Frechet Image Distance: ', fid)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
    parser.add_argument('--img_ref_dir', type=str, required=True)
    parser.add_argument('--img_gen_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--imsize', type=int, default=64)
    args = parser.parse_args()

    print(args)
    main(args)
