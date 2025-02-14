## Loads .choirdat file's information
# returns a tuple: ((samples, labels), features, classes); where features and
# classes is a number and samples and labels is a list of vectors
import struct
import torch

def load_choirdat(dataset_path, train_data=None, train_label=None):
    with open(dataset_path, 'rb') as f:
        # reads meta information
        features = struct.unpack('i', f.read(4))[0]
        classes = struct.unpack('i', f.read(4))[0]

        # lists containing all samples and labels to be returned
        samples = list()
        labels = list()

        while True:
            # load a new sample
            sample = list()

            # load sample's features
            for i in range(features):
                val = f.read(4)
                if val is None or not len(val):
                    return (samples, labels), features, classes
                sample.append(struct.unpack('f', val)[0])

            # add the new sample and its label
            label = struct.unpack('i', f.read(4))[0]
            if train_data==None:
                samples.append(sample)
                labels.append(label)
            else:
                train_data.append(sample)
                train_label.append(label)
    return (samples, labels), features, classes

dataset_name = 'UCIHAR'
train_path = dataset_name + '/' + dataset_name + '_train.choir_dat'
test_path = dataset_name + '/' + dataset_name + '_test.choir_dat'

(train_x, train_y), features, classes = load_choirdat(train_path)
(test_x, test_y), features, classes = load_choirdat(train_path)

train_x = torch.FloatTensor(train_x)
train_y = torch.IntTensor(train_y)
test_x = torch.FloatTensor(test_x)
test_y = torch.IntTensor(test_y)

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
