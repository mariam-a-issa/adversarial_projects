import torch
from torch.autograd import Variable

import collections
import numpy as np
import matplotlib.pyplot as plt

import time
import copy
from tqdm import tqdm

is_cuda = torch.cuda.is_available()


def fgsm_attack(x_original, epsilon, gradient):
    # Get Gradient sign
    grad_sign = gradient.sign()
    # Add epsilon*grad_sign perturbation to the original input
    perturbation = epsilon*grad_sign / grad_sign.norm()
    x_perturbed = x_original + perturbation
    return x_perturbed, perturbation


def attack(model, x, x_test, y_test, epsilons, classes, criterion, N_VAL_SAMPLES, device, input_dim=(28, 28), model_type='cnn', display=True):
    acc_results_non = dict()

    # attack_sample = {'0.01':torch.zeros(x_test.shape[0], 28, 28),
    #                 '0.03':torch.zeros(x_test.shape[0], 28, 28),
    #                 '0.07':torch.zeros(x_test.shape[0], 28, 28),
    #                 '0.1':torch.zeros(x_test.shape[0], 28, 28)}
    attack_samples = []
    running_times = []

    for eps in epsilons:
        #eps = x.max() * eps
        correct_unperturbed = 0
        correct_perturbed = 0
        t0 = time.perf_counter()

        if model_type == 'cnn':
            attack = torch.zeros(x_test.shape[0], *input_dim)
            flatten_input_dim = 1
            for d in input_dim: flatten_input_dim *= d
        else:
            attack = torch.zeros(x_test.shape[0], input_dim)
            flatten_input_dim = input_dim
        labels = torch.zeros(x_test.shape[0])

        diffs = torch.zeros(len(x_test))
        for j in tqdm(range(len(x_test)), disable=(not display)):
        ### NOTE: IT WOULD BE MORE EFFICIENT TO ITERATE ONLY ONCE THROUGH THE DATA AND PERFORM ALL THE ATTACKS
            x_origin, y_target = x_test[j], y_test[j]
            x_origin, y_target = x_origin.to(device), y_target.to(device)
            x_origin.requires_grad = True

            output = model(x_origin.reshape(1, 1, *input_dim) if model_type == 'cnn' else x_origin.reshape(1, input_dim))
            y_pred = torch.argmax(output)
            
            if y_pred == y_target:
                correct_unperturbed += 1
            # Calculate loss and gradient
            loss = criterion(output, y_target.unsqueeze(0))
            grad = torch.autograd.grad(outputs=loss, inputs=x_origin)[0]
            model.zero_grad()
            perturbed_x, _ = fgsm_attack(x_origin, epsilon=eps, gradient=grad)
            perturbed_output = model(perturbed_x.reshape(1, 1, *input_dim) if model_type == 'cnn' else perturbed_x.reshape(1, input_dim))
            y_pred_perturbed = torch.argmax(perturbed_output)
            # loss_perturbed = criterion(perturbed_output, y_target)
            diffs[j] = (x_origin - perturbed_x).norm()
            if y_pred_perturbed == y_target:
                correct_perturbed += 1

            attack[j] = perturbed_x
            labels[j] = y_target
            
        attack_samples.append(attack)
        running_time = time.perf_counter() - t0
        running_times.append(running_time)
        # if eps == 0.1:
        #     attack_sample['0.1'] = attack
        # elif eps == 0.07:
        #     attack_sample['0.07'] = attack
        # elif eps == 0.03:
        #     attack_sample['0.03'] = attack
        # else:
        #     attack_sample['0.01'] = attack                
            
        acc_before_attack = correct_unperturbed / N_VAL_SAMPLES
        acc_after_attack = correct_perturbed / N_VAL_SAMPLES
        acc_results_non[eps.item()] = acc_after_attack

        if display:
            print(f'\nFGSM Attack with epsilon = {eps:.5f} | Elapsed time: {time.perf_counter() - t0} seconds.')
            print(f'Accuracy: Before the attack -> {100 * acc_before_attack:.2f}%\t|\tAfter the attack -> {100 * acc_after_attack:.2f}%')
            print(f'mean of norm={diffs.mean()}')          
        
        if model_type == 'cnn' and display:
            plt.imshow(perturbed_x.detach().cpu().numpy().reshape(28, 28), cmap='gray')
            plt.colorbar()
            plt.show()
    acc_results_non[0] = acc_before_attack

    if display:
        return attack_samples, acc_results_non
    return attack_samples, acc_results_non, running_times
