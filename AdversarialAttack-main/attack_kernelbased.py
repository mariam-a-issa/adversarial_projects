import time

import torch
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import tqdm

import numpy as np
import struct

import matplotlib.pyplot as plt
import scipy.stats as st

import onlinehd
import spatial


#Helper 
def listTensor(lis):
    newlis=[t.numpy() for t in lis]
    return torch.complex(torch.Tensor(np.array(newlis).real),torch.Tensor(np.array(newlis).imag))
def listTensorFloat(lis):
    newlis=[t.numpy() for t in lis]
    return torch.Tensor(np.array(newlis))
def plotMNIST(x):
    plt.imshow(np.reshape(x,(28,28)))


def validate(model,x,y):
    a=time.time()
    print('Validating...')
    yhat = model(x)

    acc = (y == yhat).float().mean()
    print(acc)
    b=time.time()
    print(b-a)
    
    return acc
    

def genAdversarialNoise(model,x,y,alpha=5e-2, display=True):
    
    x_ret=x.clone().detach()
    
    D=model.dim
    
    nData=x.shape[0]
    nFeatures=x.shape[1]

    encoded=model.encode(x)
    scores=model.scores(x)
    
    for n in tqdm.tqdm(range(nData), disable=(not display)):

        ordering=[i[0] for i in sorted(enumerate(scores[n]), key=lambda x:x[1])]
        if ordering[-1]==y[n]:
            
            top1=ordering[-1]
            top2=ordering[-2]

            model_diff=listTensor([model.model[top1],model.model[top2]])
            
            gradientVectorPreprocess=torch.zeros((nFeatures,D),dtype=torch.cfloat)
            gradientVector=torch.zeros(nFeatures)

            for nf in range(nFeatures):
                gradientVectorPreprocess[nf,:]=1j*model.encoder.basis[:,nf]*encoded[n]
    
            temp=spatial.cos_cdist(gradientVectorPreprocess, model_diff)

            gradientVector=temp[:,1]-temp[:,0]
            gradientVector=gradientVector/gradientVector.norm()
            
            x_ret[n]=x_ret[n]+alpha*gradientVector
            
    return x_ret

def genAdversarialNoiseBrokenGrad(model,x,y,alpha=5e-2, dAlpha=5e-2):
    
    x_ret=x.clone().detach()
    
    D=model.dim
    
    nData=x.shape[0]
    nFeatures=x.shape[1]
    
    for k in range(int(alpha/dAlpha)+1):
        encoded=model.encode(x_ret)
        scores=model.scores(x_ret)
    
        for n in range(nData):
            if (n+1)%100==0:
                print(k,n)
            ordering=[i[0] for i in sorted(enumerate(scores[n]), key=lambda x:x[1])]
            if ordering[-1]==y[n]:

                top1=ordering[-1]
                top2=ordering[-2]

                model_diff=listTensor([model.model[top1],model.model[top2]])

                gradientVectorPreprocess=torch.zeros((nFeatures,D),dtype=torch.cfloat)
                gradientVector=torch.zeros(nFeatures)

                for nf in range(nFeatures):
                    gradientVectorPreprocess[nf,:]=1j*model.encoder.basis[:,nf]*encoded[n]

                temp=spatial.cos_cdist(gradientVectorPreprocess, model_diff)

                gradientVector=temp[:,1]-temp[:,0]
                gradientVector=gradientVector/gradientVector.norm()

                x_ret[n]=x_ret[n]+dAlpha*gradientVector
            
        for n in range(nData):
            diff=x_ret[n]-x[n]
            if diff.norm()>alpha:
                x_ret[n]=x[n]+diff*alpha/diff.norm()
                
    return x_ret


def genAdversarialEfficiency(model,x,y):
    
    x_ret=x.clone().detach()
    
    D=model.dim
    
    nData=x.shape[0]
    Lold=np.zeros(nData)+100
    Lnew=np.zeros(nData)+100
    eps=np.zeros(nData)
    nFeatures=x.shape[1]

    encoded=model.encode(x)
    scores=model.scores(x)
    
    for n in range(nData):
        if n%100==0:
            print(n)
        ordering=[i[0] for i in sorted(enumerate(scores[n]), key=lambda x:x[1])]
        if ordering[-1]==y[n]:
            
            top1=ordering[-1]
            top2=ordering[-2]
            Lold[n]=scores[n,top2]-scores[n,top1]
            
            model_diff=listTensor([model.model[top1],model.model[top2]])
            
            gradientVectorPreprocess=torch.zeros((nFeatures,D),dtype=torch.cfloat)
            gradientVector=torch.zeros(nFeatures)

            for nf in range(nFeatures):
                gradientVectorPreprocess[nf,:]=1j*model.encoder.basis[:,nf]*encoded[n]
    
            temp=spatial.cos_cdist(gradientVectorPreprocess, model_diff)

            gradientVector=temp[:,1]-temp[:,0]
            
            eps[n]=-Lold[n]/gradientVector.norm()
            gradientVector=gradientVector/gradientVector.norm()
            
            x_ret[n]=x_ret[n]+eps[n]*gradientVector
            
    scores_new=model.scores(x_ret)
    for n in range(nData):
        if n%100==0:
            print("Scoreds ",n)
        
        if Lold[n]<50:
            ordering=[i[0] for i in sorted(enumerate(scores[n]), key=lambda x:x[1])]
            top1=ordering[-1]
            top2=ordering[-2]
            Lnew[n]=scores_new[n,top2]-scores_new[n,top1]
    
    return Lold,Lnew,eps


def genRandomNoise(model,x,y,alpha=5e-2):
    
    x_ret=x.clone().detach()
    
    x_rand=torch.rand(x_ret.shape)
    eps=1e-8
    eps = torch.tensor(eps, device=x_rand.device)
    norms1 = x_rand.norm(dim=1).unsqueeze_(1).max(eps)
    x_rand.div_(norms1)
    
    x_ret=x_ret+alpha*x_rand
            
    return x_ret

def getGradientTheory(f,nA,nB,model,avg_feature, noise_proj):
    D=model.dim
    nFeatures=f.shape[1]
    
    encodedF=model.encode(listTensorFloat([f]))[0]
    encodedAvg=model.encode(listTensorFloat([avg_feature]))[0]
    
    model_diff=listTensor([model.model[nA],model.model[nB]])
            
    gradientVectorPreprocess=torch.zeros((nFeatures,D),dtype=torch.cfloat)
    gradientVector=torch.zeros(nFeatures)

    for nf in range(nFeatures):
        gradientVectorPreprocess[nf,:]=1j*model.encoder.basis[:,nf]*encodedF

    temp=spatial.cos_cdist(gradientVectorPreprocess, model_diff)

    gradientVector=temp[:,1]-temp[:,0]
    gradientVector=gradientVector/gradientVector.norm()
    
    gradientVectorA=torch.zeros(nFeatures)
    gradientVectorB=torch.zeros(nFeatures)
    
    normA=model.model[nA].norm()
    normB=model.model[nB].norm()
    
    kernFA=spatial.cos_cdist(listTensor([encodedF]),listTensor([model.model[nA]]))[0,0]
    kernFB=spatial.cos_cdist(listTensor([encodedF]),listTensor([model.model[nB]]))[0,0]
    kernFAavg=spatial.cos_cdist(listTensor([encodedF]),model.encode(listTensorFloat([avg_feature[nA]])))[0,0]
    kernFBavg=spatial.cos_cdist(listTensor([encodedF]),model.encode(listTensorFloat([avg_feature[nB]])))[0,0]
    
    gradientVectorA=gradientVectorA+((kernFA-(D/normA)*kernFAavg*noise_proj[nA])@(avg_feature[nA]-f).T).T[0,:]
    gradientVectorB=gradientVectorB+((kernFB-(D/normB)*kernFBavg*noise_proj[nB])@(avg_feature[nB]-f).T).T[0,:]
    
    gradientVectorTheory=gradientVectorA-gradientVectorB
    gradientVectorTheory=gradientVectorTheory/gradientVectorTheory.norm()
    
    return gradientVector,gradientVectorTheory

