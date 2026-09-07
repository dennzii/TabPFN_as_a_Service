import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm


def request_inference(df,feature_columns,target_column,target_null_val):
    """
    API call function for TabPFN inference.

    Args:
    train_df (DataFrame): if 'test_df' is None, test data is to be seperated according to 'test_data_target_value' e.g. if 'test_data_target_value' == -1, samples with target values of '-1' is to be test samples.
    feature_columns (list): Feature columns that to form x_test.
    target_column (string): Target value to be predicted
    primary_key (list): primary key columns to be dropped
    target_null_val (int):
    """
    TRAIN_CHUNK_SIZE = 800
    MAX_CONTEXT_LEN = 1024
    test_chunks = []
    server_url = "http://127.0.0.1:8000/predict"
    
    test_df = df[df[target_column] == target_null_val]
    train_df = df[df[target_column] != target_null_val]

    

    #our aim is to sample a stratified subsample with size of 800.
    if len(train_df) > TRAIN_CHUNK_SIZE:
        train_df, _ = train_test_split(
            train_df, train_size=TRAIN_CHUNK_SIZE, stratify=train_df[target_column]
        )
    
    #iki ihtimal var biri kesilmiş train = 800
    # diğeri train=450 e.g.
    target_df = train_df[target_column]
    test_df = test_df[feature_columns]
    train_df = train_df[feature_columns]
    


    TEST_CHUNK_SIZE = MAX_CONTEXT_LEN - len(train_df)

    

    if len(test_df) > TEST_CHUNK_SIZE:
        for i in range(0, len(test_df), TEST_CHUNK_SIZE):
            test_chunk = test_df.iloc[i : i + TEST_CHUNK_SIZE]
            test_chunks.append(test_chunk)
    else:
        test_chunks.append(test_df)

    X_train_list = train_df.values.tolist()
    y_train_list = target_df.values.tolist()

    tcp_session = requests.Session()

    all_outs = []
    all_probs = []

    for chunk in tqdm(test_chunks):

        payload = {"X_train": X_train_list, "y_train": y_train_list, "X_test": chunk.values.tolist()}
        response = tcp_session.post(server_url,json=payload)

        response.raise_for_status()

        data = response.json()

        outs = data['out']
        probs = data['probs']

        all_outs.extend(outs)
        all_probs.extend(probs)
    
    test_df['predicted_label'] = all_outs
    test_df['confidence'] = [max(p) for p in all_probs]

    return test_df


