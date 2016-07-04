"""
classifiers
"""
import random
import os


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
        self._buckets = [[]] * self._bucket_size

    def _group_category(self):
        """
        build buckets
        """
        with open(self._file_path) as file_handler:
            lines = file_handler.readlines()

        for line in lines:
            if self._separator != self.DEFAULT_SEPARATOR:
                line = line.replace(self._separator, self.DEFAULT_SEPARATOR)

            vector = line.split(self.DEFAULT_SEPARATOR)
            category = vector[self._class_index]
            category_vector = self._categorized_data.setdefault(category, [])
            category_vector.append(line)

    def build_buckets(self):
        """
        build buckets based on the category data
        """
        for _, vector in self._categorized_data:
            # randomize order of instances for each class
            random.shuffle(vector)
            num = 0
            for item in vector:
                # put the item into bucket horizontally, so that can get average distribution
                # make the training data make much sense.
                self._buckets[num].append(item)
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
