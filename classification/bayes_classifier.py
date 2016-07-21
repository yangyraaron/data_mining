"""
bayes classification module
"""


class BayesClassifier(object):
    """
    classifier classifies only one file
    """
    def __init__(self, training_file_paths, test_file_path, data_format):
        """
        initialization
        """
        self._training_file_paths = training_file_paths
        self._test_file_path = test_file_path
        self._data_format = data_format
        self._total = 0
        self._classes = {}
        self._counts = {}
        self._prior = {}
        self._conditional = {}

    def training(self):
        """
        start training
        """
        for training_file in self._training_file_paths:
            total, classes, counts = self.read_data(training_file)
            self._total += total
            for category, count in classes.iteritems():
                self._classes.setdefault(category, 0)
                self._classes[category] += count
                self._counts.setdefault(category, {})
                for col, col_value in counts[category].iteritems():
                    self._counts[category].setdefault(col, {})
                    for attr, value in col_value.iteritems():
                        self._counts[category][col].setdefault(attr, 0)
                        self._counts[category][col][attr] += value
        
        # compute prior probabilities
        for category, count in self._classes.iteritems():
            # compute p(h)
            self._prior[category] = float(count) / self._total   
        # compute conditional probabilities p(h|d)
        for category, cols in self._counts.iteritems():
            self._conditional.setdefault(category, {})
            for col, value_counts in cols.iteritems():
                self._conditional[category].setdefault(col, {})
                for attr, count in value_counts.iteritems():
                    # compute p(D|h)
                    self._conditional[category][col][attr] = round(float(count) / self._classes[category], 6)

    def classify(self, item_vector):
        """
        classify
        """
        results = []
        for category, prior in self._prior.iteritems():
            prob = prior
            col = 1
            for attr in item_vector:
                if attr not in self._conditional[category][col]:
                    # didn't find ansy category instance of this attribute occurring with this category
                    # prob should be 0
                    prob = 0
                else:
                    prob = prob * self._conditional[category][col][attr]
                    col += 1

            results.append((round(prob, 6), category))
        print results
        return max(results)[1]

    def read_data(self, file_path):
        """
        read data from file
        """
        file_handler = open(file_path)
        lines = file_handler.readlines()
        file_handler.close()

        total = 0
        classes = {}
        counts = {}
        for line in lines:
            fields = line.strip().split('\t')
            ignore = []
            vector = []
            for i in xrange(len(fields)):
                col_format = self._data_format[i]
                if col_format == 'num':
                    vector.append(float(fields[i]))
                elif col_format == 'attr':
                    vector.append(fields[i])
                elif col_format == 'comment':
                    ignore.append(fields[i])
                elif col_format == 'class':
                    category = fields[i]
            # process instance data
            total += 1
            classes.setdefault(category, 0)
            counts.setdefault(category, {})
            classes[category] += 1
            # process each attribute data of instance
            col = 0
            for col_value in vector:
                col += 1
                counts[category].setdefault(col, {})
                counts[category][col].setdefault(col_value, 0)
                counts[category][col][col_value] += 1
        
        return total, classes, counts


def test_bayes_classifier():
    """
    test bayes classifier
    """
    training_file_paths = []
    for i in xrange(10):
        index = i + 1
        file_path = '%s-%02i' % ('data/iHealth/i', index)
        
        training_file_paths.append(file_path)

    classifier = BayesClassifier(training_file_paths, '', 'attr attr attr attr class'.split(' '))
    print 'classifier start training'
    classifier.training()
    print 'training complete'
    print classifier.classify(['health', 'moderate', 'moderate', 'yes'])


if __name__ == '__main__':
    test_bayes_classifier()
