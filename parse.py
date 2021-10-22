import os
import csv, json
import spacy, benepar
from tqdm import tqdm
import argparse
from nltk.tree import Tree

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--dataset', type=str, help='dataset_name', default='pororo')
    parser.add_argument('--data_dir', type=str, help='path to data folder', required=True)
    args = parser.parse_args()
    return args


def tree2dict(tree):
    return {tree.label(): {'leaves': tree.leaves(), 'children': [tree2dict(t) if isinstance(t, Tree) else t
                                                                 for t in tree]}}

def get_parse_tree(caption):
    doc = nlp(caption)
    sent = list(doc.sents)[0]
    tokens = [t.text for t in sent]
    t = Tree.fromstring(sent._.parse_string)
    d = tree2dict(t)
    return d, tokens


if __name__ == "__main__":

    print("Loading parser")
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))

    args = parse_args()
    if args.dataset == 'pororo':

        annotations_file = os.path.join(args.data_dir, 'descriptions.csv')
        data_dict = {}
        with open(annotations_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            count = 0
            for i, row in tqdm(enumerate(csv_reader)):
                ep, idx, caption = row
                if ep not in data_dict:
                    data_dict[ep] = {}
                if idx not in data_dict[ep]:
                    data_dict[ep][idx] = []

                d, tokens = get_parse_tree(caption)
                data_dict[ep][idx].append({'tokens': tokens, 'tree': d})

        n_samples = sum([sum([len(v) for k, v in val.items()]) for key, val in data_dict.items()])
        with open(os.path.join(args.data_dir, 'parses.json'), 'w') as f:
            json.dump(data_dict, f, indent=2)

    else:
        raise ValueError