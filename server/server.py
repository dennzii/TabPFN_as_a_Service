from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
import torch
from TabPFNTransformer import Transformer_TabPFN
from predict import predict
from TabPFN_Request import TabPFN_Request


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


    out,probs = predict(model= model,device= device,X_train=X_train,y_train=y_train,X_test=X_test)


    return {'out':out.tolist(),
            'probs':probs.tolist()}
