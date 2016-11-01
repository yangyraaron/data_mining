"""
decision tree module
"""
import copy
from collections import Counter
from math import log


def calc_entropy(data_set):
    """
    calculate entropy, the higher the entorpy, the more mixed up the data is
    -sum(p(xi)*log2p(xi))
    """
    size = len(data_set)
    label_counts = {}
    for feat_vector in data_set:
        label = feat_vector[-1]
        label_counts.setdefault(label, 0)
        label_counts[label] += 1

    entropy = 0.0
    for key, count in label_counts.iteritems():
        prob = float(count) / size
        entropy -= prob * log(prob, 2)

    return entropy


def splite_dataset(data_set, axis, value):
    """
    split data set where axis is equal to value
    """
    result = []
    for feat_vector in data_set:
        if feat_vector[axis] == value:
            reduced_feat_vector = feat_vector[:axis]
            reduced_feat_vector.extend(feat_vector[axis+1:])
            result.append(reduced_feat_vector)

    return result


def choose_best_feature(data_set):
    """
    choose best feature to split dataset
    """
    feature_size = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0; best_feature = -1
    for i in xrange(feature_size):
        feat_list = [eg[i] for eg in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_ds = splite_dataset(data_set, i, value)
            prob = len(sub_ds) / float(len(data_set))
            new_entropy += prob * calc_entropy(sub_ds)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    """
    find the class has greatest frequency
    """
    counter = Counter(class_list)
    cls, _ = counter.most_common(1)

    return cls


def create_tree(data_set, labels):
    """
    crate a decision tree
    """
    labels = copy.copy(labels)
    class_list = [ eg[-1] for eg in data_set]
    # if all classes are same
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # only have class feature
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature(data_set)
    best_feat_cls = labels[best_feat]
    node = {best_feat_cls: {}}
    del(labels[best_feat])
    feat_values = [eg[best_feat] for eg in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_cls = labels[:]
        sub_ds = splite_dataset(data_set, best_feat, value)
        node[best_feat_cls][value] = create_tree(sub_ds, sub_cls)

    return node


def classify(tree, feat_labels, vector):
    """
    classify decision tree
    """
    root_cls = tree.keys()[0]
    node = tree[root_cls]
    feat_index = feat_labels.index(root_cls)
    cls_label = ''
    for cls, ch_node in node.iteritems():
        if vector[feat_index] == cls:
            if isinstance(ch_node, dict):
                cls_label = classify(ch_node, feat_labels, vector)
            else:
                cls_label = node[cls]
    return cls_label


def test():
    """
    decision tree test
    """
    fr = open('data/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = create_tree(lenses, labels)

    return tree


fish_data = [[1, 1, 'yes'],
                 [1, 1, 'yes'],
                 [1, 0, 'no'],
                 [0, 1, 'no'],
                 [0, 1, 'no']]
fish_labels = ['no surfacing', 'flippers']
