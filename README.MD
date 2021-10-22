## Integrating Visuospatial, Linguistic and Commonsense Structure into Story Visualization

PyTorch code for the EMNLP 2021 paper "Integrating Visuospatial, Linguistic and Commonsense Structure into Story Visualization". See the arxiv paper [here](https://arxiv.org/abs/2110.10834).

#### Requirements:
This code has been tested on torch==1.11.0.dev20211014 (nightly) and torchvision==0.12.0.dev20211014 (nightly)

#### Prepare Repository:
 Download the PororoSV dataset and associated files from [here](https://drive.google.com/file/d/1BqKizOZn4o4dbwNGK7ThCDnNEOVAolnf/view?usp=sharing) and save it as ```./data```.
 Download GloVe embeddings (glove.840B.300D) from [here](https://nlp.stanford.edu/projects/glove/). The default location of the embeddings is ```./data/``` (see ```./dcsgan/miscc/config.py```).

#### Extract Constituency Parses:
To install the [Berkeley Neural Parser](https://spacy.io/universe/project/self-attentive-parser) with SpaCy:

```pip install benepar```

To extract parses for PororoSV:

```python parse.py --dataset pororo --data_dir <path-to-data-directory>```

#### Extract Dense Captions:

We use the Dense Captioning Model implementation available [here](https://github.com/soloist97/densecap-pytorch
). Download the pretrained model as outlined in their repository. To extract dense captions for PororoSV:\
```python describe_pororosv.py --config_json <path-to-config> --lut_path <path-to-VG-regions-dict-lite.pkl> --model_checkpoint <path-to-model-checkpoint> --img_path <path-to-data-directory> --box_per_img 10 --batch_size 1```

#### Training VLC-StoryGAN:

To train VLC-StoryGAN for PororoSV:\
```python train_gan.py --cfg ./cfg/pororo_s1_vlc.yml --data_dir <path-to-data-directory> --dataset pororo```\
   
Unless specified, the default output root directory for all model checkpoints is ```./out/```

#### Evaluation Models:
Please see [here](https://github.com/adymaharana/StoryViz) for evaluation models for character classification-based scores, BLEU2/3 and R-Precision.

To evaluate Frechet Inception Distance (FID):\
```python eval_vfid --img_ref_dir <path-to-image-directory-original images> --img_gen_dir <path-to-image-directory-generated-images> --mode <mode>```

## Citation:
```
@inproceedings{maharana2021integrating,
  title={Integrating Visuospatial, Linguistic and Commonsense Structure into Story Visualization},
  author={Maharana, Adyasha and Bansal, Mohit},
  booktitle={EMNLP},
  year={2021}
}
```