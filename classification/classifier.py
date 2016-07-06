"""
classifiers
"""
import random
import os

from algorithm import get_median, get_abs_standard_deviation, manhattan


class SingleClassifier(object):
    """
    the classifier only for one single file data
    """
    DEFAULT_SEPARATOR = '\t'

    def __init__(self, training_file_path, test_file_path):
        """
        initialization
        :@param training_file_path : the path of training file
        :@param test_file_path : the path of testing file
        """
        self._median_deviation = []
        self._training_file_path = training_file_path
        self._test_file_path = test_file_path
        self._format = None
        self._training_data = None
        self._test_data = None

        self.init_data()
        for i in xrange(len(self._training_data[0][1])):
            self.normalize_col(i)

    def init_data(self):
        """
        read data from file and init variables
        """
        data_format, training_data = self.read_data(self._training_file_path)
        self._format = data_format
        self._training_data = training_data

        _, test_data = self.read_data(self._test_file_path)
        self._test_data = test_data
        
    def read_data(self, file_path):
        """
        read data from file and init variables
        """
        file_handler = open(file_path)
        lines = file_handler.readlines()
        file_handler.close()

        data = []
        data_format = lines[0].strip().split(self.DEFAULT_SEPARATOR)
        for line in lines[1:]:
            fields = line.strip().split(self.DEFAULT_SEPARATOR)
            ignore = []
            vector = []
            for i in xrange(len(fields)):
                field_format = data_format[i]
                if field_format == 'num':
                    vector.append(float(fields[i]))
                elif field_format == 'comment':
                    ignore.append(fields[i])
                elif field_format == 'class':
                    classification = fields[i]

            data.append((classification, vector, ignore))
        
        return data_format, data

    
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
            v[1][col_index] = (v[1][col_index]-median) / asd

    def normalize_vector(self, vector):
        """
        normalize vector
        """
        vector_list = list(vector)
        for i in xrange(len(vector_list)):
            median, asd = self._median_deviation[i]
            vector_list[i] = (vector_list[i] - median)/asd
        
        return vector_list

    def nearest_neighbor(self, item_vector):
        """
        get nearest neighbor 
        :@param : item vector
        """
        distances = [(manhattan(item_vector, item[1]), item) for item in self._training_data]

        return min(distances)

    def classify(self, item_vector):
        """
        classify item
        :@param : item vector
        """
        vector = self.normalize_vector(item_vector)
        nearest = self.nearest_neighbor(vector)

        return nearest[1][0]

    def test(self):
        """
        test classifier with testing set
        """
        suc_count = 0
        failed_count = 0

        for cls, vector, _ in self._test_data:
            d_cls = self.classify(vector)
            if cls == d_cls:
                suc_count += 1
            else:
                failed_count += 1
        
        total = suc_count + failed_count

        return float(suc_count) / total

class NFolderCrossValidationClassifier(object):
    """
    Classifier build upon ten folder cross validation
    The input file will be divided into 10 sub files(10 buckets)
    """
    DEFAULT_SEPARATOR = '\t'

    def __init__(self, file_path, class_index, separator='\t', bucket_size=10):
        """
        initialization
        :@param file_name: input file path
        :@param class_col_index: the class index of vector, as every line of file will be converted a vector
        :@param separator: separator of line of input file path
        :@param folder_size: how many sub files (buckets) will files divided into
        """
        self._file_path = file_path
        self._bucket_size = bucket_size
        self._class_index = class_index
        self._separator = separator
        self._bucket_name = os.path.splitext(os.path.basename(file_path))[0]

        self._categorized_data = {}
        self._group_category()
        # initialize the buckets with bucket size of empty vectors
        self._buckets = []
        for _ in xrange(self._bucket_size):
            self._buckets.append([])

    def _group_category(self):
        """
        build buckets
        """
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
            sub_file = open("data/%s-%02i" % (self._bucket_name, bucket_index+1), 'w')
            for item in self._buckets[bucket_index]:
                sub_file.write(item)
            sub_file.close()


def main():
    """
    main entrence
    """
    print 'create classifier and build buckets'

    classifier = NFolderCrossValidationClassifier('data/auto-mpg.data', 0)
    classifier.build_buckets()

    print 'build buckets successfully'


def test_single_classifier():
    """
    test single classifier
    """
    classifier = SingleClassifier('data/athletes_training_set.txt', 'data/athletes_test_set.txt')
    print classifier.test()
    classifier = SingleClassifier('data/iris_training_set.data', 'data/iris_test_set.data')
    print classifier.test()
    classifier = SingleClassifier('data/mpg_training_set.txt', 'data/mpg_test_set.txt')
    print classifier.test()


if __name__ == '__main__':
    test_single_classifier()
