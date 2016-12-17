"""
adaboost implementation
"""
import pandas
from math import log
from matplotlib import pyplot as plt
from numpy import ones, matrix, mat, shape, inf, zeros, multiply, exp, sign

def load_simple_data():
    datMat = matrix([[ 1. , 2.1],
        [ 2. , 1.1],
        [ 1.3, 1. ],
        [ 1. , 1. ],
        [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def visualize_simple_data():
    """
    visualize simple data by matplotlib
    """
    data, lbs = load_simple_data()
    table = zeros((data.shape[0], 3))
    table[:, :2] = data
    table[:, 2:3] = mat(lbs).T

    df = pandas.DataFrame(table, columns=['x', 'y', 'label'])
    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend()

    plt.show()


def stump_classify(data_matrix, dimen, threshold, thresh_ineq):
    """
    stump classify
    """
    ret_array = ones((data_matrix.shape[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= threshold] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > threshold] = -1.0

    return ret_array


def build_stump(data_arr, cls_labels, d):
    """
    build stump
    """
    data_matrix = mat(data_arr)
    cls_mat = mat(cls_labels).T
    row_size, col_size = shape(data_matrix)
    steps = 10.0; best_stump = {}; best_cls_est = mat(zeros((row_size, 1)))
    min_error = inf

    for i in xrange(col_size):
        min_rag = data_matrix[:, i].min()
        max_rag = data_matrix[:, i].max()
        step_size = (max_rag - min_rag) / steps
        for j in xrange(-1, int(steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (min_rag + float(j) * step_size)
                pred_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = mat(ones((row_size, 1)))
                err_arr[pred_vals == cls_mat] = 0
                weig_error = d.T * err_arr
                print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, thresh_val, inequal, weig_error)
                if weig_error < min_error:
                    min_error = weig_error
                    best_cls_est = pred_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump, min_error, best_cls_est


def train_ds(data_arr, cls_lbs, num_it=40):
    """
    train data set
    """
    weak_cls_arr = []
    row_size = shape(data_arr)[0]
    # intialize equal weight matrix
    D = mat(ones((row_size, 1))/row_size)
    agg_cls_est = mat(zeros((row_size, 1)))
    for i in xrange(num_it):
        best_stump, err, cls_est = build_stump(data_arr, cls_lbs, D)
        print "D:", D.T
        alpha = float(0.5*log((1.0-err)/max(err, 1e-16)))
        best_stump['alhpa'] = alpha
        weak_cls_arr.append(best_stump)
        print "classEst: ", cls_est.T
        expon = mulitply(-1*alpha*mat(cls_lbs).T, cls_est)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        agg_cls_est += alpha * cls_est
        print "agg class est: ", agg_cls_est.T
        agg_errs = multiply(sign(agg_cls_est) != mat(cls_lbs).T, ones((row_size, 1)))
        err_rate = agg_errs.sum() / row_size
        print "total error:", err_rate, "\n"
        if err_rate == 0.0: break

    return weak_cls_arr
