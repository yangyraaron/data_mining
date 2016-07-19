"""
classifiers
"""
import random
import os
import heapq

from algorithm import get_median, get_abs_standard_deviation, manhattan


class SingleClassifier(object):
    """
    the classifier only for one single file data
    """
    DEFAULT_SEPARATOR = '\t'

    def __init__(self, training_files_path, test_file_path, data_format):
        """
        initialization
        :@param training_file_path : the path of training file
        :@param test_file_path : the path of testing file
        :@param data_format: the format of the data
        """
        self._median_deviation = []
        self._training_file_path = training_files_path
        self._test_file_path = test_file_path
        self._format = data_format
        self._training_data = None
        self._test_data = None

        self.init_data()
        for i in xrange(len(self._training_data[0][1])):
            self.normalize_col(i)

    def init_data(self):
        """
        read data from file and init variables
        """
        training_data = []
        for training_file in self._training_file_path:
            single_training_data = self.read_data(training_file)
            training_data.extend(single_training_data)

        self._training_data = training_data

        test_data = self.read_data(self._test_file_path)
        self._test_data = test_data

    def read_data(self, file_path):
        """
        read data from file and init variables
        """
        file_handler = open(file_path)
        lines = file_handler.readlines()
        file_handler.close()

        data = []
        for line in lines:
            fields = line.strip().split(self.DEFAULT_SEPARATOR)
            ignore = []
            vector = []
            is_drop = False
            for i in xrange(len(fields)):
                field_format = self._format[i]
                if field_format == 'num':
                    try:
                        vector.append(float(fields[i]))
                    except Exception:
                        print "can't convert value {} to float, this line will be droped".format(fields[i])
                        is_drop = True
                elif field_format == 'comment':
                    ignore.append(fields[i])
                elif field_format == 'class':
                    classification = fields[i]
            if not is_drop:
                data.append((classification, vector, ignore))

        return data

    def normalize_col(self, col_index):
        """
        normalize columns
        :@param col_index: index of vector element needs to be noramlized
        """
        # the column element from vector
        col = [v[1][col_index] for v in self._training_data]
        median = get_median(col)
        asd = get_abs_standard_deviation(col, median)

        self._median_deviation.append((median, asd))
        for v in self._training_data:
            v[1][col_index] = (v[1][col_index] - median) / asd

    def normalize_vector(self, vector):
        """
        normalize vector
        """
        vector_list = list(vector)
        for i in xrange(len(vector_list)):
            median, asd = self._median_deviation[i]
            vector_list[i] = (vector_list[i] - median) / asd

        return vector_list

    def nearest_neighbor(self, item_vector):
        """
        get nearest neighbor
        :@param : item vector
        """
        distances = [(manhattan(item_vector, item[1]), item)
                     for item in self._training_data]

        return min(distances)[1][0]

    def classify(self, item_vector):
        """
        classify item
        :@param : item vector
        """
        vector = self.normalize_vector(item_vector)
        nearest_class = self.nearest_neighbor(vector)

        return nearest_class

    def test(self):
        """
        test classifier with testing set
        """
        suc_count = 0
        failed_count = 0
        result = {}

        for cls, vector, _ in self._test_data:
            d_cls = self.classify(vector)
            if cls == d_cls:
                suc_count += 1
            else:
                failed_count += 1

            cls_result = result.setdefault(cls, {})
            cls_result.setdefault(d_cls, 0)
            cls_result[d_cls] += 1

        total = suc_count + failed_count

        return round(float(suc_count) / total, 4), result


class KNNSingleClassifier(SingleClassifier):
    """
    single classifier classify by knn algorithem
    """
    def __init__(self, training_files_path, test_file_path, data_format, k):
        """
        initialization
        """
        super(KNNSingleClassifier, self).__init__(training_files_path, test_file_path, data_format)

        self._k = k

    def knn(self, item_vector):
        """
        knn implementation
        """
        items = [(manhattan(item_vector, item[1]), item) for item in self._training_data]
        #get k nearest neighbors by priority queue algorithm
        neighbors = heapq.nsmallest(self._k, items)

        # get a vote for each neighbor
        results = {}
        for neighbor in neighbors:
            the_class = neighbor[1][0]
            results.setdefault(the_class, 0)
            results[the_class] += 1

        result_list = sorted([(result[1], result[0]) for result in results.items()], reverse=True)
        max_votes = result_list[0][0]
        possible_answers = [i[1] for i in result_list if i[0] == max_votes]

        answer = random.choice(possible_answers)

        return answer

    def nearest_neighbor(self, item_vector):
        """
        overried nearest neighbor
        """
        return self.knn(item_vector)


class NFolderCrossValidationClassifier(object):
    """
    Classifier build upon ten folder cross validation
    The input file will be divided into 10 sub files(10 buckets)
    """
    DEFAULT_SEPARATOR = '\t'

    def __init__(self, file_path, class_index, data_format, bucket_name=None, separator='\t', bucket_size=10):
        """
        initialization
        :@param file_name: input file path
        :@param bucket_name: bukect name 
        :@param class_col_index: the class index of vector, as every line of file will be converted a vector
        :@param data_format: the format of data
        :@param separator: separator of line of input file path
        :@param folder_size: how many sub files (buckets) will files divided into
        """
        self._file_path = file_path
        self._bucket_size = bucket_size
        self._class_index = class_index
        self._data_format = data_format.split(separator)
        self._separator = separator
        if not bucket_name:
            self._bucket_name = os.path.splitext(os.path.basename(file_path))[0]
        else:
            self._bucket_name = bucket_name

        self._categorized_data = {}
        self._group_category()
        # initialize the buckets with bucket size of empty vectors
        self._buckets = []
        for _ in xrange(self._bucket_size):
            self._buckets.append([])

        self._results = {}

    def _group_category(self):
        """
        build buckets
        """
        if not self._file_path:
            return
            
        with open(self._file_path) as file_handler:
            lines = file_handler.readlines()

        for line in lines:
            if self._separator != self.DEFAULT_SEPARATOR:
                line = line.replace(self._separator, self.DEFAULT_SEPARATOR)

            vector = line.split()
            category = vector[self._class_index]
            category_lines = self._categorized_data.setdefault(category, [])
            category_lines.append(line)

    def build_buckets(self):
        """
        build buckets based on the category data
        """
        for _, lines in self._categorized_data.iteritems():
            # randomize order of instances for each class
            random.shuffle(lines)
            num = 0
            for line in lines:
                # put the item into bucket horizontally, so that can get average distribution
                # make the training data make much sense.
                self._buckets[num].append(line)
                num = (num + 1) % self._bucket_size
        # write each bucket to sub file
        for bucket_index in xrange(self._bucket_size):
            sub_file = open("data/%s-%02i" %
                            (self._bucket_name, bucket_index + 1), 'w')
            for item in self._buckets[bucket_index]:
                sub_file.write(item)
            sub_file.close()

    def classify(self):
        """
        classify
        """
        index_list = list(xrange(self._bucket_size))
        for test_index in xrange(self._bucket_size):
            training_files = []
            test_file = None
            test_index = test_index + 1
            for index in index_list:
                index = index + 1
                file_path = "data/%s-%02i" % (self._bucket_name, index)
                if test_index == index:
                    test_file = file_path
                else:
                    training_files.append(file_path)

            classifier = KNNSingleClassifier(
                training_files, test_file, self._data_format, 3)
            _, distribution = classifier.test()

            for category, value in distribution.iteritems():
                self._results.setdefault(category, {})
                for sub_category, sub_value in value.iteritems():
                    self._results[category].setdefault(sub_category, 0)
                    self._results[category][sub_category] += sub_value

    def print_result(self):
        """
        print result
        """
        categories = list(self._results.keys())
        categories.sort()

        header = "        "
        sub_header = "    +"
        for category in categories:
            header += category + "  "
            sub_header += "----+"
        print header
        print sub_header

        total = 0.0
        correct = 0.0

        for category in categories:
            row = category + "   |"
            for c2 in categories:
                if c2 in self._results[category]:
                    count = self._results[category][c2]
                else:
                    count = 0
                row += " %2i |" % count
                total += count
                if c2 == category:
                    correct += count
            print row
        print sub_header
        print "\n%5.3f percent correct" % ((correct * 100) / total)
        print "total of %i instances" % total


def test_nfold_classifier():
    """
    main entrence
    """
    # print 'create classifier and build buckets'

    classifier = NFolderCrossValidationClassifier(
        'data/auto-mpg.data', 0, 'class	num	num	num	num	num	comment')
    # classifier.build_buckets()

    # print 'build buckets successfully'

    print "start training"
    classifier.classify()
    classifier.print_result()


def test_pima_classifier():
    """
    test classifier with pima data
    """
    classifier = NFolderCrossValidationClassifier('', 0, 'num	num	num	num	num	num	num	num	class', bucket_name='pima_small/pimaSmall')
    print 'start training'
    classifier.classify()
    classifier.print_result()

def test_single_classifier():
    """
    test single classifier
    """
    data_format = 'comment	class	num	num'.split('\t')
    training_files = ['data/athletes_training_set.txt']
    classifier = SingleClassifier(
        training_files, 'data/athletes_test_set.txt', data_format)
    print classifier.test()

    data_format = 'num	num	num	num	class'.split('\t')
    training_files = ['data/iris_training_set.data']
    classifier = SingleClassifier(
        training_files, 'data/iris_test_set.data', data_format)
    print classifier.test()

    data_format = 'class	num	num	num	num	num	comment'.split('\t')
    training_files = ['data/mpg_training_set.txt']
    classifier = SingleClassifier(
        training_files, 'data/mpg_test_set.txt', data_format)
    print classifier.test()


if __name__ == '__main__':
    #test_single_classifier()
    test_pima_classifier()
