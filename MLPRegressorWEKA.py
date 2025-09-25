import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class MLPRegressorWEKA(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_units=10, ridge=1e-4, max_iter=100, tol=1e-5,
                 use_conjugate_gradient=False, activation='tanh', verbose=False):
        self.hidden_units = hidden_units
        self.ridge = ridge
        self.max_iter = max_iter
        self.tol = tol
        self.use_conjugate_gradient = use_conjugate_gradient
        self.activation = activation
        self.verbose = verbose
        
        # To store scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def _get_activation(self):
        if self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError("Unsupported activation. Use 'sigmoid', 'tanh' or 'softplus'.")

    def _build_model(self, input_dim):
        act_fn = self._get_activation()
        model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_units),
            act_fn,
            nn.Linear(self.hidden_units, 1)  # Identity function for regression
        )
        return model

    def fit(self, X, y):
        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.model = self._build_model(X.shape[1])
        
        # Define L2 penalty (ridge)
        def l2_penalty():
            return sum(torch.norm(param)**2 for param in self.model.parameters())

        # Loss + regularization
        def loss_fn(y_pred, y_true):
            mse = nn.MSELoss()(y_pred, y_true)
            ridge_loss = self.ridge * l2_penalty()
            return mse + ridge_loss

        # Choose optimizer
        if self.use_conjugate_gradient:
            optimizer = optim.LBFGS(self.model.parameters(), max_iter=self.max_iter, tolerance_grad=self.tol)
        else:
            optimizer = optim.LBFGS(self.model.parameters(), max_iter=self.max_iter, tolerance_grad=self.tol)

        def closure():
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            if self.verbose:
                print(f'Loss: {loss.item()}')
            return loss

        optimizer.step(closure)
        return self

    def predict(self, X):
        X = self.x_scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()
        preds = self.y_scaler.inverse_transform(preds)
        return preds.flatten()
