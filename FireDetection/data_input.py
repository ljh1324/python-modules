import numpy as np

class LabelDataSet:
    def __init__(self, filename):
        file = open(filename, 'r')
        data = [int(x) for x in file.read().split(' ')]

        self.num_examples = data[0]
        self.num_classes = data[1]

        data = data[2:]
        self.data = np.reshape(data, (self.num_examples, self.num_classes))
        file.close()

    def get_num_excamples(self):
        return self.num_examples
    def get_class_n(self):
        return self.num_classes
    def get_data(self):
        return self.data

class ImageDataSet:
    def __init__(self, filename):
        file = open(filename, 'r')
        data = [float(x) for x in file.read().split(' ')]
        self.num_examples = int(data[0])
        self.row = int(data[1])
        self.col = int(data[2])
        self.num_filters = int(data[3])
        self.data = np.reshape(data[4:], (self.num_examples, self.row, self.col, self.num_filters))
        file.close()

    def get_num_examples(self):
        return self.num_examples
    def get_row(self):
        return self.row
    def get_col(self):
        return self.col
    def get_num_filters(self):
        return self.num_filters

