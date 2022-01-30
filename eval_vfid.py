import torchvision.transforms as transforms
import argparse
import os
from vfid.fid_score import fid_score

def main(args):


    image_transforms = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.task == 'pororo':
        import vlcgan.pororo_data as data
    elif args.task == 'flintstones':
        import vlcgan.flintstones_data as data
    else:
        raise ValueError


    ref_dataset = data.StoryImageDataset(args.img_ref_dir,
                                    args.imsize,
                                    out_img_folder=None,
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
    parser.add_argument('--task', type=str, default='pororo')
    parser.add_argument('--imsize', type=int, default=64)
    args = parser.parse_args()

    print(args)
    main(args)
