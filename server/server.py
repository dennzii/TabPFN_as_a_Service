from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel


class input_template(BaseModel):
    X_train = list[list[float]]
    y_train = list[int]
    X_test = list[list[float]]

app = FastAPI(title='TabPFN v1 Inference Server')

session = ort.InferenceSession("tabpfnv1.onnx")
