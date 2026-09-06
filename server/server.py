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

@app.post('/predict')
def predict_(request : TabPFN_Request):
    X_train = np.array(request.X_train,dtype=np.float32)
    y_train =  np.array(request.y_train,dtype=np.float32)
    X_test =  np.array(request.X_test,dtype=np.float32)

    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        raise HTTPException(400,f'Train sample size cannot exceed {MAX_TRAIN_SAMPLES}. Be Sure your train subsample stratified accoording to target labels')
    elif len(X_train) == 0 or len(y_train) or len(X_test):
        raise HTTPException(400,f'any of X_train, y_train, X_test cannot be empty!')
    
    out,probs = predict(model= model,device= device,X_train=X_train,y_train=y_train,X_test=X_test)


    return {'out':out.tolist(),
            'probs':probs.tolist()}
