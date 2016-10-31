"""
K-nearest neighbars
"""
import os
from numpy import array, tile, zeros
import operator


def create_data_set():
    """
    create data set
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels


def classify(in_x, data_set, labels, k):
    """
    classify
    """
    # # of rows
    size = data_set.shape[0]
    # caculate Euclidian distance between input x and items in data set
    # extend input item to same size of data set
    diff_mat = tile(in_x, (size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # sum a^2 + b^2 + ....
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # end caculate Euclidian distance
    # get sorted index array
    sorted_indicies = distances.argsort()
    class_count = {}
    # find nearest k items from dataset
    for i in xrange(k):
        vote_label = labels[sorted_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # sort the result from largest to smallest
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter,
                                reverse=True)

    return sorted_class_count[0][0]


def file_to_matrix(file_name):
    """
    parse a file and store in a matrix
    """
    file_path = os.path.join('data', file_name)
    if not os.path.exists(file_path):
        print("the file path {} isn't existed".format(file_path))
        return None, None

    fr = open(file_path)
    length = len(fr.readlines())
    mat = zeros((length, 3))
    class_vector = []
    fr = open(file_path)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        cols = line.split('\t')
        # only store first 3 columns
        mat[index, :] = cols[0:3]
        # the last column is the class label
        class_vector.append(int(cols[-1]))
        index += 1

    return mat, class_vector

def auto_norm(data_set):
    """
    auto normalize data set
    scale formula new_value = (old_value - min)/(max-min)
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    # max - min
    ranges = max_vals - min_vals
    norm_ds = zeros(data_set.shape)
    size = data_set.shape[0]
    # old value - min
    norm_ds = data_set - tile(min_vals, (size, 1))
    # final divide
    norm_ds = norm_ds / tile(ranges, (size, 1))

    return norm_ds, ranges, min_vals

def img_to_vector(file_name):
    """
    convert a 32X32 image to 1X1024 vectir
    """
    file_path = os.path.join('data', file_name)
    if not os.path.exists(file_path):
        print 'the file {} is not found'.format(file_path)

    vector = zeros((1, 1024))
    fr = open(file_path)
    for i in xrange(32):
        line = fr.readline()
        for j in xrange(32):
            vector[0, 32*i + j] = int(line[j])

    return vector


def handwriting_test():
    """
    test hand written digits
    """
    labels = []
    training_list = os.listdir('data/trainingDigits')
    training_size = len(training_list)
    training_mat = zeros((training_size, 1024))
    for i in xrange(training_size):
        file_name = training_list[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        labels.append(class_num)
        training_mat[i, :] = img_to_vector('trainingDigits/%s' % file_name)
    test_list = os.listdir('data/testDigits')
    error_count = 0.0
    test_size = len(test_list)
    for i in xrange(test_size):
        file_name = test_list[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        test_vector = img_to_vector('testDigits/%s' % file_name)
        classifed_class = classify(test_vector, training_mat, labels, 3)

        print "the classified class is {0}, the real class is {1}"\
        .format(classifed_class, class_num)

        if classifed_class != class_num:
            error_count += 1.0

    print 'the total number of errors is: %d' % error_count
    print 'the total error rate is: %f' % (error_count/float(test_size))


def dating_test():
    """
    test dating for knn
    """
    ho_ration = 0.10
    mat, labels = file_to_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_values = auto_norm(mat)
    size = norm_mat.shape[0]
    testing_count = int(size * ho_ration)
    error_count = 0.0
    # 10% data used for testing, 90% data used for training
    for i in xrange(testing_count):
        category = classify(norm_mat[i, :],
                          norm_mat[testing_count:size, :],
                          labels[testing_count:size], 3)
        print "classifier get class {0}, actual class is {1}".format(category, labels[i])
        if category != labels[i]:
            error_count += 1.0

    print "the total error rate is: %f" % (error_count / float(testing_count))
