# TabPFN as a Service

A high-throughput inference server and client adapter for TabPFN (Prior-Data Fitted Networks for Tabular Data), built with FastAPI, PyTorch, and Docker.

Serving TabPFN in production requires managing several structural constraints: sequence context limits (`N_train + N_test <= 1024`), $O(L^2)$ attention memory scaling, and event loop blocking during dense tensor operations. This implementation provides an end-to-end service designed to maximize GPU concurrency while eliminating CUDA out-of-memory (OOM) failures.

---

## Architecture and Concurrency Model

```text
[ Incoming Requests ] 
         │
         ▼
[ FastAPI Event Loop ]  ──► Request validation & JSON parsing (Single Thread / async def)
         │
         ▼
[ asyncio.Semaphore ]   ──► VRAM Gatekeeper (Auto-calibrated concurrecy limit according to available VRAM)
         │
         ▼
[ run_in_threadpool ]   ──► Dispatches PyTorch forward pass to OS worker threads by run_in_threadpool()
         │
         ▼
[ NVIDIA GPU (CUDA) ]   ──► Parallel tensor operations
```

1. **Non-blocking Event Loop:** PyTorch inference is compute-heavy. Executing it directly inside an `async def` route blocks the event loop and stops the server from accepting other traffic. Using `run_in_threadpool` offloads the forward pass to background OS threads. Because PyTorch releases the Python GIL during CUDA execution, multiple inference passes run concurrently on the GPU without freezing FastAPI.
2. **Auto-Calibrated VRAM Semaphore:** At boot time, the server runs a benchmark forward pass with a full context window (`1024` tokens), measures peak activation memory using `torch.cuda.max_memory_allocated`, and calculates available VRAM:
   $$\text{max\_concurrent} = \left\lfloor \frac{\text{VRAM}_{\text{free}}}{\text{Peak}_{\text{activation}} + \text{128MB}} \right\rfloor$$
   An `asyncio.Semaphore` enforces this boundary, queuing extra requests in memory rather than letting the GPU run out of memory.

---

## Quick Start

### 1. Run the Server (Docker + GPU)

Requires the NVIDIA Container Toolkit.

```bash
# Build the image (uses PyTorch cu128 for Blackwell / RTX 50-series support)
docker build -t tabpfn-service .

# Run container with GPU access
docker run --gpus all -p 8000:8000 -e PYTHONUNBUFFERED=1 --name tabpfn-server tabpfn-service
```

During startup, the server inspects the GPU, logs the calibrated concurrency limit, and listens on port `8000`.

---

### 2. Client Usage and API Integration

`client_adapter.py` maps a Pandas DataFrame to the server API, handling train/test splits, stratified subsampling, and dynamic context chunking.

```python
import pandas as pd
import numpy as np
from client.client_adapter import request_inference

# Prepare DataFrame: target column contains labels for train, -1 for test rows
df = pd.DataFrame({
    "feature_1": np.random.randn(1500),
    "feature_2": np.random.randn(1500),
    "target": [0, 1] * 500 + [-1] * 500  # 1000 train rows, 500 test rows
})

# Run inference
results_df = request_inference(
    df=df,
    feature_columns=["feature_1", "feature_2"],
    target_column="target",
    target_null_val=-1,
    server_url="http://localhost:8000/predict"
)

# Inspect predictions
print(results_df[["target", "predicted_label", "confidence"]].head())
```

#### Request Flow and Constraints

```text
[ Raw DataFrame (Train: 1000, Test: 500) ]
                    │
                    ▼
[ Stratified Subsampling ]  ──► Train sample capped at 800 (Server constraint)
                    │
                    ▼
[ Context Window Slicing ]  ──► Test chunk size = 1024 - len(train) = 224
                    │       ──► Slices 500 test rows into: [224, 224, 52] sample sizes
                    ▼
[ HTTP POST /predict ]      ──► Transmits payload via requests.Session:
                                {
                                  "X_train": [[...], ...],  # <= 800 rows, list[list[float]] (Checked by pydantic)
                                  "y_train": [0, 1, ...],   # 2 to 10 classes, list[int]
                                  "X_test":  [[...], ...]   # <= (1024 - N_train), list[list[float]]
                                }
                    │
                    ▼
[ Server Guardrails ]       ──► Validates tensor shape, context boundary (<= 1024), feature match
                    │
                    ▼
[ Output Assembly ]         ──► Merges chunks into DataFrame with 'predicted_label' & 'confidence'
```

---

## Project Structure

```text
├── server/
│   ├── server.py              # FastAPI application with dynamic semaphore
│   ├── predict.py             # Preprocessing, token stacking, and forward pass
│   ├── TabPFNTransformer.py   # 12-layer causal Transformer architecture
│   ├── TabPFN_Request.py      # Pydantic input validation model
│   └── tabpfn_step_18000.pt   # PyTorch model checkpoint
├── client/
│   ├── client_adapter.py      # DataFrame chunker and HTTP client
│   └── run_test.py            # End-to-end evaluation and benchmark script
├── Dockerfile                 # CUDA 12.8 runtime container definition
├── locustfile.py              # Locust stress test suite
└── README.md
```
