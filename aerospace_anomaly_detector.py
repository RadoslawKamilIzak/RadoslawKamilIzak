"""
Aerospace Anomaly Detector — time-series anomaly detection for sensor data.
Adapted from aerospace telemetry work; applicable to IoT and industrial ML.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    timestamps: list[int]
    values: list[float]
    scores: list[float]
    anomaly_flags: list[bool]
    threshold: float


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.decoder(out)


class AerospaceAnomalyDetector:
    """
    LSTM autoencoder-based anomaly detector for multivariate sensor time-series.
    Trained on normal operating windows; flags reconstructions exceeding threshold.
    """

    def __init__(self, window_size: int = 30, hidden_size: int = 64, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.threshold_sigma = threshold_sigma
        self.model = LSTMAnomalyDetector(hidden_size=hidden_size)
        self.threshold = 0.0
        self._loss_mean = 0.0
        self._loss_std = 1.0

    def _make_windows(self, series: np.ndarray) -> torch.Tensor:
        windows = [
            series[i:i + self.window_size]
            for i in range(len(series) - self.window_size + 1)
        ]
        return torch.FloatTensor(np.array(windows)).unsqueeze(-1)

    def fit(self, series: np.ndarray, epochs: int = 40, lr: float = 1e-3):
        X = self._make_windows(series)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            pred = self.model(X)
            loss = loss_fn(pred, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X)
        losses = ((recon - X) ** 2).mean(dim=(1, 2)).numpy()
        self._loss_mean = float(losses.mean())
        self._loss_std = float(losses.std())
        self.threshold = self._loss_mean + self.threshold_sigma * self._loss_std

    def predict(self, series: np.ndarray) -> AnomalyResult:
        X = self._make_windows(series)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X)
        scores = ((recon - X) ** 2).mean(dim=(1, 2)).numpy().tolist()
        pad = [0.0] * (self.window_size - 1)
        all_scores = pad + scores
        flags = [s > self.threshold for s in all_scores]
        return AnomalyResult(
            timestamps=list(range(len(series))),
            values=series.tolist(),
            scores=[round(s, 5) for s in all_scores],
            anomaly_flags=flags,
            threshold=round(self.threshold, 5),
        )
