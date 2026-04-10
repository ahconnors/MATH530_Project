import numpy as np

class LinearKalmanFilter:
    """
    A simple implementation of a linear Kalman Filter for state estimation in a linear dynamic system.
    """

    def __init__(self):
        self.x_prior = None         # Prior state estimate
        self.x_posterior = None     # Posterior state estimate
        self.P_prior = None         # Prior estimate covariance
        self.P_posterior = None     # Posterior estimate covariance
        self.F = None               # State transition matrix
        self.H = None               # Measurement matrix
        self.R = None               # Measurement noise covariance
        self.Q = None               # Process noise covariance

    def predict(self):
        # Predict the next state
        self.x_prior = np.dot(self.F, self.x_posterior)
        # Predict the next estimate covariance
        self.P_prior = np.dot(self.F, np.dot(self.P_posterior, self.F.T)) + self.Q

    def update(self, z):
        # Compute the Kalman Gain
        y = z - np.dot(self.H, self.x_prior)            # Measurement residual
        S = self.H @ self.P_prior @ self.H.T + self.R   # Residual covariance
        K = self.P_prior @ self.H.T @ np.linalg.inv(S)  # Kalman Gain

        # Update the state estimate
        self.x_posterior = self.x_prior + (K @ y)
        # Update the estimate covariance
        I = np.eye(len(self.P_posterior))
        self.P_posterior = (I - (K @ self.H)) @ self.P_prior

    def skip_update(self):
        """Call this instead of update() when measurement is missing"""
        self.x_posterior = self.x_prior.copy()
        self.P_posterior = self.P_prior.copy()

    def set_matrices(self, F, H, R, Q):
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q

    def set_initial_state(self, x0, P0):
        self.x_posterior = x0
        self.P_posterior = P0

    def get_updated_state(self):
        return self.x_posterior
    
    def get_updated_covariance(self):
        return self.P_posterior
    
    def get_prior_state(self):
        return self.x_prior
    
    def get_prior_covariance(self):
        return self.P_prior