import torch
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import time

class GenAttack(object):
    def __init__(self, model, num_labels, threshold = 0.95, device = 'cpu', input_dim=(28, 28), model_type='cnn'):
        self.model = model.to(device)
        self.device = device
        self.num_labels = num_labels
        self.tlab = None
        self.threshold = threshold

        self.input_dim = input_dim
        self.model_type = model_type

    def set_tlab(self, target):
        self.tlab = torch.zeros((1, self.num_labels)).to(self.device)
        self.tlab[:,target] = 1

    def get_mutation(self, shape, alpha, delta, bernoulli):
        N, features = shape
        U = torch.FloatTensor(N * features).uniform_().to(self.device) \
            * 2 * alpha * delta - alpha * delta
        mask = bernoulli.sample((N*features, )).squeeze()
        mutation = mask * U
        mutation = mutation.view(N, features)
        return mutation
    
    def crossover(self, parents, fitness, population):
        _, features = population.shape
        fitness_pairs = fitness[parents.long()].view(-1, 2)
        prob = fitness_pairs[:, -0] / fitness_pairs.sum()
        parental_bernoulli = torch.distributions.Bernoulli(prob)
        inherit_mask = parental_bernoulli.sample((features,))
        inherit_mask = inherit_mask.view(-1, features)
        parent_features = population[parents.long()]
        children = torch.FloatTensor(inherit_mask.shape).to(device=self.device)
        
        children = self.where(inherit_mask, parent_features[::2], parent_features[1::2])
        return children
        
    def where(self, cond, x_1, x_2):
        return (cond.float() * x_1) + ((1-cond).float() * x_2)

    def get_fitness(self, population, target):
        # population = population.reshape(-1, 784).cpu()
        # population = torch.from_numpy(
            # self.scaler.transform(population)).cuda()
        pop_preds = self.model(population.reshape(population.size(0), 1, *self.input_dim) if self.model_type == 'cnn' else population.reshape(population.size(0), self.input_dim))
        pop_preds = torch.nn.functional.softmax(pop_preds, dim=1)
        all_preds = torch.argmax(pop_preds, dim = 1)
        # print('pop_preds', pop_preds)
        # print('all_preds', all_preds)

        success_pop = (all_preds == target).clone().detach()
        success_pop = success_pop.to(self.device).int()
        success = torch.max(success_pop, dim = 0)

        target_scores = torch.sum(self.tlab * pop_preds, dim = 1)
        sum_others = torch.sum((1 - self.tlab) * pop_preds, dim = 1)
        max_others = torch.max((1 - self.tlab) * pop_preds, dim = 1)

        # the goal is to maximize this loss
        fitness = torch.log(sum_others + 1e-30) - torch.log(target_scores + 1e-30)

        return fitness

    def attack(self, x, target, delta, alpha, p, N, G):
        self.set_tlab(target)
        x = x.to(self.device)
        target = target.to(self.device)
        delta = delta.to(self.device)
        alpha = alpha.to(self.device)
        p = p.to(self.device)
        
        bernoulli = torch.torch.distributions.Bernoulli(p)
        softmax = torch.nn.Softmax(0).to(device=self.device)

        # generate starting population
        features = x.shape[0]
        mutation = self.get_mutation([N, features], alpha, delta, bernoulli)

        # init current population
        Pcurrent = x.expand(N, -1) + mutation
        Pnext = torch.zeros_like(Pcurrent)

        # init previous population with original example
        Pprev = x.expand(N, -1)
        # compute constraints to ensure permissible distance from the original example
        lo = x.min() - alpha[0]*delta[0]
        hi = x.max() + alpha[0]*delta[0]
        
        # start evolution
        for g in range(G):
            # measure fitness with MSE between descriptors
            fitness = self.get_fitness(Pcurrent, target)  # [N]
            # check SSIM
            ssimm = np.zeros(N)
            for i in range(N):
                ssimm[i] = ssim(x.squeeze(0).cpu().numpy(),
                                Pcurrent[i].cpu().numpy(), data_range=1.)  # [N]
            #survivors = ssimm >= 0.95  # [N]
            survivors = ssimm >= self.threshold

            if survivors.sum() == 0:
                # print('All candidates died at generation', g)
                # print('Target = ', target)
                return Pprev.cpu(), False
                
            # temp_Pcurrent = Pcurrent.clone().reshape(-1, 784).cpu()
            # temp_Pcurrent = self.scaler.transform(temp_Pcurrent)
            # temp_Pcurrent = torch.from_numpy(temp_Pcurrent).cuda()
            if target in self.model(Pcurrent.reshape(Pcurrent.shape[0], -1, *self.input_dim) if self.model_type == 'cnn' else Pcurrent.reshape(Pcurrent.shape[0], self.input_dim)).argmax(1):
                # print('Attack Success at generation', g)
                return Pcurrent.cpu(), True

            # choose the best fit candidate among population
            _, best = torch.min(fitness, 0)  # get idx of the best fitted candidate
            # ensure the best candidate gets a place in the next population
            Pnext[0] = Pcurrent[best]

            # generate next population
        #print(pop_preds)
            # ('fitness', fitness)
            probs = softmax(Variable(
                torch.FloatTensor(survivors).to(self.device)) \
                    * Variable(fitness)).data
            # print('probs', probs)
            cat = torch.distributions.Categorical(
                probs[None, :].expand(2 * (N-1), -1))
            parents = cat.sample()  # sample 2 parents per child, total number of children is N-1
            children = self.crossover(parents, fitness, Pcurrent)  # [(N-1) x nchannels x h x w]
            mutation = self.get_mutation([N-1, features], alpha, delta, bernoulli)
            children = children + mutation
            Pnext[1:] = children
            Pprev = Pcurrent  # update previous generation
            Pcurrent = Pnext  # update current generation
            # clip to ensure the distance constraints
            Pcurrent = torch.clamp(Pcurrent, lo, hi)

        # print('All', 5000, 'generations failed.')
        return Pcurrent.cpu(), False
    
    def to(self, device):
        self.device = device


def make_GA(model, delta, alpha, x_test, y_test, input_dim=(28, 28), model_type='cnn', display=True):
    if alpha==0:
        return x_test, y_test
    classes = y_test.unique().size(0)
    attacker = GenAttack(model, classes, input_dim=input_dim, model_type=model_type)

    N = 8
    G = 500
    p = torch.FloatTensor([5e-2])
    alpha = torch.FloatTensor([alpha])
    delta = torch.FloatTensor([delta])

    targets = torch.randint(0, classes, y_test.shape)
    for i in tqdm(range(len(y_test)), disable=(not display)):
        while targets[i] == y_test[i]:
            targets[i] = torch.randint(0,classes, (1,)).item()

    # unif = torch.ones(targets.shape[0])
    # indices = unif.multinomial(1000)

    # x_test = x_test[indices]
    # y_test = y_test[indices]

    success = []
    test_label = []
    test_data = []
    for i in tqdm(range(x_test.size(0)), disable=(not display)):
        temp = attacker.attack(x_test[i], targets[i], delta, alpha, p, N, G)
        if temp[1]:
            # temp_input = scaler.transform(temp[0].clone().reshape(-1, 784))
            # temp_input = torch.from_numpy(temp_input).float().cuda()
            test_data.append(
                torch.from_numpy(temp[0][(model(temp[0].reshape(-1, 1, *input_dim) if model_type == 'cnn' else temp[0].reshape(-1, input_dim)).argmax(1) == targets[i]).nonzero()[0].item()].numpy()))
            success.append(
                torch.from_numpy(temp[0][(model(temp[0].reshape(-1, 1, *input_dim) if model_type == 'cnn' else temp[0].reshape(-1, input_dim)).argmax(1) == targets[i]).nonzero()[0].item()].numpy()))
        else:
            test_data.append(temp[0][0])
        test_label.append(y_test[i])

    temp = torch.stack(test_data)
    temp_label = torch.stack(test_label)

    # temp_input = scaler.transform(temp.reshape(-1, 784))
    # temp_input = torch.from_numpy(temp_input).float().cuda()
    return temp, temp_label


def validate(model,x,y):
    a=time.time()
    print('Validating...')
    yhat = model(x.reshape(x.shape[0], 1, 28, 28)).argmax(1)

    acc = (y == yhat).float().mean()
    print(acc)
    b=time.time()
    print(b-a)
    
    return acc