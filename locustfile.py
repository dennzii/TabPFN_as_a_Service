from locust import HttpUser, task, between
import numpy as np


N_TRAIN = 200
N_TEST = 50
N_FEATURES = 15
N_CLASSES = 3


class TabPFNUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        rng = np.random.default_rng()

        X_train = rng.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float32)
        y_train = rng.integers(0, N_CLASSES, size=N_TRAIN).astype(np.float32)
        X_test  = rng.standard_normal((N_TEST,  N_FEATURES)).astype(np.float32)

        self.payload = {
            'X_train': X_train.tolist(),
            'y_train': y_train.tolist(),
            'X_test':  X_test.tolist(),
        }

    @task
    def predict(self):
        with self.client.post(
            '/predict',
            json=self.payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f'HTTP {response.status_code}: {response.text[:200]}'
                )
