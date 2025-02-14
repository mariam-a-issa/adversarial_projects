import attack_kernelbased
import attack_fgsm
import attack_df
import attack_jsma
import attack_gen
import dataloader
import spatial

# import attack_kernelbased
import attack_fgsm_dnn
import attack_df_dnn
# import attack_jsma
import attack_gen_dnn

import torch
import torch.nn as nn
import numpy as np
import time
import onlinehd
import DNNmodels
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm
import copy

import json


def load_dataset(dataset_name):
    X_train, X_test, y_train, y_test = dataloader.load(dataset_name=dataset_name)

    classes = y_train.unique().size(0)
    features = X_train.size(1)

    X_test = X_test[:1000]
    y_test = y_test[:1000]

    return classes, features, X_train, y_train, X_test, y_test

def train_onlinehd(D, classes, features, X_train, y_train, X_test, y_test):
    
    acc_history, test_acc_history = [], []
    model = onlinehd.OnlineHD(classes, features, dim=D)
    best_model = None
    for _ in tqdm.tqdm(range(30)):
        model = model.fit(X_train, y_train, bootstrap=1.0, lr=0.035, epochs=1, one_pass_fit=False)
        
        yhat_train = model(X_train)
        yhat_test = model(X_test)

        acc = (y_train == yhat_train).float().mean().item()
        acc_test = (y_test == yhat_test).float().mean().item()

        if best_model is None or max(test_acc_history) < acc_test:
            best_model = copy.deepcopy(model)

        acc_history.append(acc)
        test_acc_history.append(acc_test)
    model = best_model

    # plt.plot(acc_history, label='train')
    # plt.plot(test_acc_history, label='test')
    # plt.show()
    # print(max(acc_history), max(test_acc_history))
    return model, acc_history, test_acc_history

def kernel_based_attack(onlinehd_model, dnn_model, features, X_test, y_test):
    ret = []
    # with complex encoding (FHRR)
    for e in np.linspace(0.01, 0.1, 10):#[0.01, 0.03, 0.07, 0.1]:
        st = time.perf_counter()
        X_test_noised = attack_kernelbased.genAdversarialNoise(onlinehd_model, X_test, y_test, e)
        running_time = time.perf_counter() - st
        
        yhat_test = onlinehd_model(X_test_noised)
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })

        # print(e, distance, onlinehd_attacked_acc, dnn_attacked_acc)
    return ret

def FGSM(onlinehd_model, dnn_model, features, classes, X_train, X_test, y_test):
    ret = []

    epsilons = np.linspace(0.01, 0.1, 10) # torch.Tensor([0.01, 0.03, 0.07, 0.1])
    verbose = False
    #criterion = nn.CosineEmbeddingLoss()
    criterion = lambda output, model, label: (spatial.cos_cdist(output, model) - label).mean()
    #criterion = lambda output, model, label: output.mean()
    N_VAL_SAMPLES = X_test.data.shape[0]

    attack_samples, acc_results, running_times = attack_fgsm.attack(
        onlinehd_model, X_train, X_test, y_test, epsilons, classes, criterion, N_VAL_SAMPLES, 'cpu', 
        input_dim=features, model_type='dnn', display=False)
  
    for e, X_test_noised, running_time in zip(epsilons, attack_samples, running_times):
        yhat_test = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(e, distance, onlinehd_attacked_acc, dnn_attacked_acc)
    return ret

def FGSM_from_dnn(onlinehd_model, dnn_model, features, classes, X_train, X_test, y_test):
    ret = []

    epsilons = np.linspace(0.01, 0.1, 10) # torch.Tensor([0.01, 0.03, 0.07, 0.1])
    verbose = False
    criterion = nn.CrossEntropyLoss()
    #criterion = lambda output, model, label: (spatial.cos_cdist(output, model) - label).mean()
    #criterion = lambda output, model, label: output.mean()
    N_VAL_SAMPLES = X_test.data.shape[0]

    attack_samples, acc_results, running_times = attack_fgsm_dnn.attack(
        dnn_model, X_train.reshape(len(X_train), features), X_test.reshape(len(X_test), features), y_test, epsilons, classes, criterion, N_VAL_SAMPLES, 'cpu',
        input_dim=features, model_type='dnn', display=False)

    for e, X_test_noised, running_time in zip(epsilons, attack_samples, running_times):
        yhat_test_onlinehd = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test_onlinehd).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(e, distance, onlinehd_attacked_acc, dnn_attacked_acc)

    return ret

def deepfull(onlinehd_model, dnn_model, features, classes, X_test, y_test):
    ret = []

    for max_update in np.linspace(0.01, 0.2, 20):
        X_te_cln = X_test.clone().reshape(-1, 1, features)
        Y_te_cln = y_test.clone()

        x_test_adv = torch.zeros(X_te_cln.shape)
        x_test_pert = torch.zeros(X_te_cln.shape[0], features)

        st = time.perf_counter()
        for i in tqdm.tqdm(range(X_test.shape[0]), disable=True):
            x_test_adv[i], x_test_pert[i] = attack_df.deepfool(X_te_cln[i], onlinehd_model, epsilon=max_update, num_classes=classes, input_dim=features, model_type='dnn')
        running_time = time.perf_counter() - st
        
        X_test_noised = x_test_adv.reshape(-1, features)

        yhat_test = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(max_update, distance, onlinehd_attacked_acc, dnn_attacked_acc)
    
    # unlimited deep full attack
    X_te_cln = X_test.clone().reshape(-1, 1, features)
    Y_te_cln = y_test.clone()

    x_test_adv = torch.zeros(X_te_cln.shape)
    x_test_pert = torch.zeros(X_te_cln.shape[0], features)

    st = time.perf_counter()
    for i in tqdm.tqdm(range(X_test.shape[0]), disable=True):
        x_test_adv[i], x_test_pert[i] = attack_df.deepfool(X_te_cln[i], onlinehd_model, epsilon=None, num_classes=classes, input_dim=features, model_type='dnn')
    running_time = time.perf_counter() - st

    X_test_noised = x_test_adv.reshape(-1, features)

    yhat_test = onlinehd_model(X_test_noised.reshape(-1, features))
    yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

    onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
    dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
    distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

    ret.append({
        'running_time': running_time,
        'distance': distance,
        'onlinehd_attacked_acc': onlinehd_attacked_acc,
        'dnn_attacked_acc': dnn_attacked_acc,
    })
    # print(None, distance, onlinehd_attacked_acc, dnn_attacked_acc)

    return ret

def deepfull_from_dnn(onlinehd_model, dnn_model, features, classes, X_test, y_test):
    ret = []

    for max_update in np.linspace(0.01, 0.2, 20):
        X_te_cln = X_test.clone().reshape(-1, 1, features)
        Y_te_cln = y_test.clone()

        x_test_adv = torch.zeros(X_te_cln.shape)
        x_test_pert = torch.zeros(X_te_cln.shape[0], features)

        st = time.perf_counter()
        for i in tqdm.tqdm(range(X_test.shape[0]), disable=True):
            x_test_adv[i], x_test_pert[i] = attack_df_dnn.deepfool(X_te_cln[i].reshape(1, features), dnn_model, epsilon=max_update, num_classes=classes,
                                                                  input_dim=features, model_type='dnn')
        running_time = time.perf_counter() - st
        
        X_test_noised = x_test_adv.reshape(-1, features)

        yhat_test_onlinehd = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test_onlinehd).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(max_update, distance, onlinehd_attacked_acc, dnn_attacked_acc)
    
    # unlimited deep full attack
    X_te_cln = X_test.clone().reshape(-1, 1, features)
    Y_te_cln = y_test.clone()

    x_test_adv = torch.zeros(X_te_cln.shape)
    x_test_pert = torch.zeros(X_te_cln.shape[0], features)

    st = time.perf_counter()
    for i in tqdm.tqdm(range(X_test.shape[0]), disable=True):
        x_test_adv[i], x_test_pert[i] = attack_df_dnn.deepfool(X_te_cln[i].reshape(1, features), dnn_model, epsilon=None, num_classes=classes,
                                                              input_dim=features, model_type='dnn')
    running_time = time.perf_counter() - st

    X_test_noised = x_test_adv.reshape(-1, features)

    yhat_test_onlinehd = onlinehd_model(X_test_noised.reshape(-1, features))
    yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

    onlinehd_attacked_acc = (y_test == yhat_test_onlinehd).float().mean().item()
    dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
    distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

    ret.append({
        'running_time': running_time,
        'distance': distance,
        'onlinehd_attacked_acc': onlinehd_attacked_acc,
        'dnn_attacked_acc': dnn_attacked_acc,
    })
    # print(None, distance, onlinehd_attacked_acc, dnn_attacked_acc)

    return ret

def genetic_attack(onlinehd_model, dnn_model, features, X_test, y_test):
    ret = []

    alpha_lis = np.linspace(0.0001, 0.0015, 20) #[0.01, 0.03, 0.07, 0.1]
    delta = 0.5
    nData = len(alpha_lis)
    model_accuracy_GA = np.zeros(nData)

    x_test_samp = X_test.cpu()
    y_test_samp = y_test.cpu()

    for n in range(nData):
        alpha = alpha_lis[n]
        # print(f"Alpha : {alpha}")

        st = time.perf_counter()
        x_test_GA, _ = attack_gen.make_GA(onlinehd_model, delta, alpha, x_test_samp, y_test_samp, display=False)
        running_time = time.perf_counter() - st
        
        X_test_noised = x_test_GA.reshape(-1, features)

        yhat_test = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'alpha': alpha,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(alpha, distance, onlinehd_attacked_acc, dnn_attacked_acc)

    #model_accuracy_GA
    return ret

def genetic_attack_from_dnn(onlinehd_model, dnn_model, features, X_test, y_test):
    ret = []

    alpha_lis = np.linspace(0.0004, 0.0037, 20) #[0.01, 0.03, 0.07, 0.1]
    delta = 0.5
    nData = len(alpha_lis)
    model_accuracy_GA = np.zeros(nData)

    x_test_samp = X_test.cpu()
    y_test_samp = y_test.cpu()

    for n in range(nData):
        alpha = alpha_lis[n]
        # print(f"Alpha : {alpha}")

        st = time.perf_counter()
        x_test_GA, _ = attack_gen_dnn.make_GA(dnn_model, delta, alpha, x_test_samp.reshape(-1, features), y_test_samp,
                                                input_dim=features, model_type='dnn', display=False)
        running_time = time.perf_counter() - st
        
        X_test_noised = x_test_GA.reshape(-1, features)

        yhat_test = onlinehd_model(X_test_noised.reshape(-1, features))
        yhat_test_dnn = dnn_model(X_test_noised.reshape(-1, features)).argmax(1)

        onlinehd_attacked_acc = (y_test == yhat_test).float().mean().item()
        dnn_attacked_acc = (y_test == yhat_test_dnn).float().mean().item()
        distance = (X_test_noised.reshape(-1, features) - X_test).norm(dim=-1).mean().item()

        ret.append({
            'running_time': running_time,
            'alpha': alpha,
            'distance': distance,
            'onlinehd_attacked_acc': onlinehd_attacked_acc,
            'dnn_attacked_acc': dnn_attacked_acc,
        })
        # print(alpha, distance, onlinehd_attacked_acc, dnn_attacked_acc)

    #model_accuracy_GA
    return ret

def run_(D, dataset_name, dnn_model_file):
    
    classes, features, X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    model, acc_history, test_acc_history = train_onlinehd(D, classes, features, X_train, y_train, X_test, y_test)

    model_dnn = DNNmodels.NeuralNetwork(num_classes=classes, flatten_layer=False, input_dim=features)
    model_dnn.load_state_dict(torch.load(dnn_model_file))
    model_dnn.eval()

    kernel_based_attack_results = kernel_based_attack(model, model_dnn, features, X_test, y_test)

    FGSM_results = FGSM(model, model_dnn, features, classes, X_train, X_test, y_test)
    FGSM_from_dnn_results = FGSM_from_dnn(model, model_dnn, features, classes, X_train, X_test, y_test)

    deepfull_results = deepfull(model, model_dnn, features, classes, X_test, y_test)
    deepfull_from_dnn_results = deepfull_from_dnn(model, model_dnn, features, classes, X_test, y_test)

    genetic_attack_results = genetic_attack(model, model_dnn, features, X_test, y_test)
    genetic_attack_from_dnn_results = genetic_attack_from_dnn(model, model_dnn, features, X_test, y_test)
    
    return {
        'dim': D,
        'dataset_name': dataset_name,
        'acc_history': acc_history,
        'test_acc_history': test_acc_history,

        'kernel_based_attack': kernel_based_attack_results,

        'FGSM': FGSM_results,
        'FGSM_from_dnn': FGSM_from_dnn_results,

        'deepfull': deepfull_results,
        'deepfull_from_dnn': deepfull_from_dnn_results,

        'genetic_attack': genetic_attack_results,
        'genetic_attack_from_dnn': genetic_attack_from_dnn_results,
    }


DIMs = [512, 1024, 2048, 5120, 10240]
datasets = [('isolet', 'model_dnn_isolet.pt'), ('UCIHAR', 'model_dnn_ucihar.pt'), ('face_full', 'model_dnn_facefull.pt')]

with tqdm.tqdm(total=len(DIMs)*len(datasets)) as pbar:
    for dim in DIMs:
        print('## DIM =', dim)
        for dataset_name, dnn_file_path in datasets:
            print('## dataset_name =', dataset_name)
            results = run_(dim, dataset_name, dnn_file_path)

            with open(f'./results/{dim}_{dataset_name}.json', 'w', encoding='utf8') as f:
                json.dump(results, f)

            pbar.update(1)
