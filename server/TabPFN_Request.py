from pydantic import BaseModel

class TabPFN_Request(BaseModel):
    X_train : list[list[float]]
    y_train : list[int]
    X_test : list[list[float]]