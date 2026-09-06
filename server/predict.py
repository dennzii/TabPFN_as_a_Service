import numpy as np
import torch
import torch.nn.functional as F
from TabPFNTransformer import Transformer_TabPFN

def predict(model,device,X_train,y_train,X_test,MAX_COLUMNS = 100):


    n_train = np.array(X_train.shape[0], dtype=np.int64)
    num_classes = len(np.unique(y_train))
    #Z-score normalization


    mean_train = np.mean(X_train,axis= 0)
    std_train = np.std(X_train,axis = 0)

    X_train = (X_train - mean_train) / (std_train + 1e-8)
    
    

    X_test = (X_test - mean_train) / (std_train + 1e-8) # B,sample_size,columns
    k = X_test.shape[1]



    #zero padding
    padded_train = np.zeros(shape=(X_train.shape[0],MAX_COLUMNS),dtype=np.float32)
    padded_test = np.zeros(shape=(X_test.shape[0],MAX_COLUMNS),dtype=np.float32)


    padded_train[:,:k] = X_train * MAX_COLUMNS / k
    padded_test[:,:k] = X_test * MAX_COLUMNS / k

    padded_train = np.concatenate((padded_train,y_train.reshape(-1,1)),axis=1)
    padded_test = np.concatenate((padded_test,np.zeros((X_test.shape[0],1),dtype=np.float32)),axis=1)
    
    full_input = np.concatenate((padded_train,padded_test),axis=0)

    full_input = np.expand_dims(full_input,axis=0)
    #assute input is float32
    full_input = torch.tensor(full_input, dtype=torch.float32, device=device)


    #inference

    logits = model(full_input,n_train)
    
    logits = torch.squeeze(logits,dim=0)# n_test,10
    #clipping non-existing clas indices
    logits = logits[:,:num_classes]

    probs = F.softmax(logits,dim=-1)

    out = torch.argmax(probs,dim=1)
    return out,probs

