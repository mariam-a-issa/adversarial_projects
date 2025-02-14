import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
from dataloader import load_data, save_pickle_data
from model import CNN


def run(args):
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value, num_classes = load_data(args.dataset, args.data)
    train_labels = np.argmax(y_train, axis=1)
    test_labels = np.argmax(y_test, axis=1)

    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    model = CNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    classifier = PyTorchClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), loss=criterion, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=num_classes, )

    classifier.fit(x_train, y_train, batch_size=args.batch_size, nb_epochs=args.epochs)
    torch.save(classifier, os.path.join(args.results, 'model.pth'))

    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    attack = SaliencyMapMethod(classifier=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)

    predictions = classifier.predict(x_train_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
    print("Accuracy on adversarial train examples: {}%".format(accuracy * 100))

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    x_train_adv = x_train_adv.reshape(-1, 28, 28)
    x_test_adv = x_test_adv.reshape(-1, 28, 28)
    save_pickle_data(x_train_adv, train_labels, x_test_adv, test_labels, os.path.join(args.results, '{}_JSMA.pickle'.format(args.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--data', default='./data', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--results', default='./results', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)

    # Set dataset
    d_name = args.dataset.lower()
    if 'fmnist' in d_name or 'fashion' in d_name:
        args.dataset = 'FMNIST'
    elif 'emnist' in d_name:
        args.dataset = 'EMNIST'
    elif 'mnist' in d_name:
        args.dataset = 'MNIST'
    else:
        print('Unknown dataset: {}'.format(args.dataset))
        raise AssertionError

    args.results = os.path.join(args.results, args.dataset)
    os.makedirs(args.results, exist_ok=True)

    # Save args
    with open(os.path.join(args.results, 'args.txt'), 'w') as wf:
        wf.write(str(args))

    run(args)
