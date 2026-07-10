"""Deep-learning one-step-ahead forecasters: ANN (feedforward) & LSTM.

Both networks are trained ONCE on the standardized training sequences and
then walked forward across the test set, feeding REALIZED (standardized)
test returns into the input window at each step (no retraining). sigma is
the constant std of the training residuals (in return space) and
std_resid is the standardized training residual pool, matching the shared
ForecastResult / BHS-VaR interface used by every other model.
"""

import numpy as np
import torch
import torch.nn as nn

from tsvar import SEED
from tsvar.data import make_sequences
from tsvar.var import ForecastResult


class _ANN(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _LSTM(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(o[:, -1, :]).squeeze(-1)


def _train(model, X, y, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        l = loss(model(Xt), yt)
        l.backward()
        opt.step()
    return model


def walk_forward_dl(train, test, kind, window=10, epochs=30) -> ForecastResult:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mu_, sd_ = train.mean(), train.std()
    z = (train.values - mu_) / sd_
    X, y = make_sequences(z, window)

    if kind == "ann":
        model = _ANN(window)
    elif kind == "lstm":
        model = _LSTM(window)
    else:
        raise ValueError(kind)
    model = _train(model, X, y, epochs)
    model.eval()

    with torch.no_grad():
        resid = y - model(torch.tensor(X, dtype=torch.float32)).numpy()
    sigma = float(np.std(resid) * sd_)
    std_resid = (resid - resid.mean()) / np.std(resid)

    hist = list(z[-window:])
    mu = np.empty(len(test))
    for t in range(len(test)):
        x = torch.tensor(np.array(hist[-window:]), dtype=torch.float32).reshape(1, window)
        with torch.no_grad():
            pred_z = float(model(x).item())
        mu[t] = pred_z * sd_ + mu_
        hist.append((test.values[t] - mu_) / sd_)

    return ForecastResult(mu=mu, sigma=np.full(len(test), sigma), std_resid=std_resid,
                          y_true=test.values, dates=test.index,
                          name=getattr(test, "name", None) or "series")
