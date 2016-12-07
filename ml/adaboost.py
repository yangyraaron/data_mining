"""
adaboost implementation
"""
from numpy import ones, matrix, mat, shape, inf, zeros

def load_simple_data():
    datMat = matrix([[ 1. , 2.1],
        [ 2. , 1.1],
        [ 1.3, 1. ],
        [ 1. , 1. ],
        [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


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
