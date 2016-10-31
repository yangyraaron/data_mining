"""
decision tree module
"""
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


fish_data = [[1, 1, 'yes'],
                 [1, 1, 'yes'],
                 [1, 0, 'no'],
                 [0, 1, 'no'],
                 [0, 1, 'no']]
fish_labels = ['no surfacing','flippers']
