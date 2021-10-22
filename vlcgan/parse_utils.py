import numpy as np
from collections import Counter

def indexsublist(sub, lst):
    """
    Find a sublist in a list
    """
    if type(sub) == list:
        sublen = len(sub)
        first = sub[0] if sub else []
    else:
        raise ValueError
    indx = -1
    while True:
        try:
            indx = lst.index(first, indx + 1)
        except ValueError:
            break
        if sub == lst[indx: indx + sublen]:
            return indx
    raise ValueError


def get_const_tags(tag, tree):

    if tree["leaves"] == tree["children"] and len(tree["leaves"]) == 1:
        # print("Preparing mask of size (1,1,1) for ", tag, tree["leaves"])
        return tree["leaves"], [[tag]], np.ones((1, 1, 1))
    else:
        leaves = tree["leaves"]
        # the mask at current height is all ones because all children nodes are rooted here
        curr_mask = np.ones((len(leaves), len(leaves), 1))
        # print("Created first layer mask of size", curr_mask.shape, "for sub-tree", tree)
        curr_tags = []
        for _ in range(len(leaves)):
            curr_tags.append([tag[:]])
        # print(curr_tags)

        # build sub tree mask like lego. For each child node, get the corresponding 3D mask and iteratively join with
        # masks from other children nodes. First declare an empty mask and increase height based on maximum height
        # encountered while iterating over the child sub-tree
        sub_tree_mask, max_height = None, None
        for t in tree["children"]:
            c_tag = list(t.keys())[0]
            c_tree = t[c_tag]

            if not c_tree['leaves']:
                continue

            start_idx = indexsublist(c_tree["leaves"], leaves)
            end_idx = start_idx + len(c_tree["leaves"])
            c_tokens, c_tags, c_mask = get_const_tags(c_tag, c_tree)
            _, _, height = c_mask.shape
            if sub_tree_mask is None:
                sub_tree_mask = np.zeros((len(leaves), len(leaves), height))
                # print("Created placeholder mask of size", sub_tree_mask.shape, "for", tree)
                max_height = height
            else:
                if height < max_height:
                    # to grow the mask from a child node upto max_height, replicate the last layer of mask
                    c_mask = np.concatenate((np.tile(np.expand_dims(c_mask[:, :, 0], -1), (1,1,max_height-height)), c_mask), axis=-1)
                elif height > max_height:
                    # similarly, to grow the mask upto max_height, replicate the last layer of sub tree mask
                    sub_tree_mask = np.concatenate((np.tile(np.expand_dims(sub_tree_mask[: ,:, 0], -1), (1,1,height-max_height)), sub_tree_mask), axis=-1)
                    max_height = height
                else:
                    # if height is same, do nothing
                    pass
            # print("Received sub-tree mask of dims", c_mask.shape, "for", c_tree)
            sub_tree_mask[start_idx:end_idx, start_idx:end_idx, :] = c_mask

            # update tags
            # print("Received sub-tree tags", c_tags)
            # print(start_idx, end_idx)
            for i in range(start_idx, end_idx):
                assert c_tokens[i-start_idx] == leaves[i]
                curr_tags[i].extend(c_tags[i-start_idx])
            pass

        curr_mask = np.concatenate((curr_mask, sub_tree_mask), axis=-1)
        return leaves, curr_tags, curr_mask

def get_tag_weights(tags, tag_to_idx):

    weight_vector = np.zeros(len(tag_to_idx))
    tag_counts = Counter(tags)
    for tag, count in tag_counts.items():
        weight_vector[tag_to_idx[tag]] = count
    return weight_vector