from fastapi import FastAPI,HTTPException
import numpy as np
import onnxruntime as ort
import torch
from TabPFNTransformer import Transformer_TabPFN
from predict import predict
from TabPFN_Request import TabPFN_Request

MAX_TRAIN_SAMPLES = 800
app = FastAPI(title='TabPFN v1 Inference Server')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer_TabPFN().to(device)
model.eval()

checkpoint = torch.load("tabpfn_step_18000.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

MAX_CONTEXT_WINDOW = 1024

@app.post('/predict')
def predict_(request : TabPFN_Request):
    X_train = np.array(request.X_train,dtype=np.float32)
    y_train =  np.array(request.y_train,dtype=np.float32)
    X_test =  np.array(request.X_test,dtype=np.float32)

    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        raise HTTPException(400,f'Train sample size cannot exceed {MAX_TRAIN_SAMPLES}. Be Sure your train subsample stratified accoording to target labels')
    elif len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0:
        raise HTTPException(400,f'any of X_train, y_train, X_test cannot be empty!')
    elif len(X_train) != len(y_train):
        raise HTTPException(400,f'sample size of X_train, y_train should be same!')
    elif len(X_train) + len(X_test) > MAX_CONTEXT_WINDOW:
        raise HTTPException(400,f'sumnation lengths of x_train and x_test cannot exceed {MAX_CONTEXT_WINDOW}')
    elif len(np.unique(y_train)) < 2 or len(np.unique(y_train)) > 10:
        raise HTTPException(400,f'2 < class count < 10, got {len(np.unique(y_train))}')
    elif X_train.shape[1] != X_test.shape[1]:
        raise HTTPException(400,f'Feature count mismatch: x_train has {X_train.shape[1]} features, but x_test has { X_test.shape[1]}')
    
    out,probs = predict(model= model,device= device,X_train=X_train,y_train=y_train,X_test=X_test)

    
    return {'out':out.tolist(),
            'probs':probs.tolist()}
