import torch
from torch.autograd import Variable

import collections
import numpy as np
import matplotlib.pyplot as plt

import time
import copy
from tqdm import tqdm

is_cuda = torch.cuda.is_available()

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def deepfool(image, net, epsilon=None, num_classes=10, overshoot=0.02, max_iter=10, input_dim=(28, 28), model_type='cnn'):  #num_classes; mnist:10/ fmnist:10 / emnist:26

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
   
    image = image
    model = net

    I = torch.argsort(-model.scores(image)).tolist()[0]  # sort image score (descending)
    label = I[0]  #label

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = model.scores(x)
    k_i = label

    while k_i == label and loop_i < max_iter:
        pert = image.max().item()
#         pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.cpu().numpy().copy()

            # set new w_k and new f_k

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
            
            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / (np.linalg.norm(w) + 0.001)
        r_tot = np.float32(r_tot + r_i)*3

        final_pert = (1+overshoot)*torch.from_numpy(r_tot).cpu()
        if epsilon is None:
            pert_image = image.cpu() + final_pert
        else:
            pert_image = image.cpu() + epsilon * final_pert / max(overshoot, final_pert.norm())
        
        x = Variable(pert_image, requires_grad=True)
        fs = model.scores(x)
        k_i = torch.argsort(-fs).tolist()[0][0]  # labels for x

        loop_i += 1

    #r_tot = (1+overshoot)*r_tot

    if model_type == 'cnn':
        return pert_image, torch.from_numpy(r_tot.reshape(*input_dim)) #loop_i, label, k_i,
    return pert_image, torch.from_numpy(r_tot.reshape(input_dim)) #loop_i, label, k_i,




