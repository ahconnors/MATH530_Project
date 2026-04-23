import numpy as np
from numpy import linalg
from scipy.linalg import solve_triangular

class SqrtKalmanFilter:
    """
    An implementation of a square root Kalman Filter for state estimation in a linear dynamic syste,
    adapted from A square-root Kalman filter using only QR decompositions (Tracy, 2022).
    """

    def __init__(self):
        self.x_prior = None         # Prior state estimate
        self.x_posterior = None     # Posterior state estimate
        # self.P_prior = None         # Prior estimate covariance
        # self.P_posterior = None     # Posterior estimate covariance
        self.F = None               # State transition matrix
        self.H = None               # Measurement matrix
        self.R = None               # Measurement noise covariance
        self.Q = None               # Process noise covariance
        self.M_posterior = None               # Square root of covariance matrix P
        self.M_prior = None
        self.R_cholesky = None      # Square root of measurement noise covariance
        self.Q_cholesky = None      # Square root of process noise covariance
        self.residual = None        # Most recent residual / Kalman innovation

    #Predict step
    def predict(self):
        # Predict the next state
        self.x_prior = np.dot(self.F, self.x_posterior)
        # Predict the next estimate covariance root
        self.M_prior = self.qr(self.M_posterior @ self.F.T, self.Q_cholesky)
    

    def update(self, z):
        # Compute the innovation and Kalman Gain
        y = z - np.dot(self.H, self.x_prior)            # Measurement residual / innovation
        self.residual = y
        
        # Residual/ innovation covariance, using QR decomposition as describe in Algorithm 1 of Tracy (2022)
        G = self.qr(self.M_prior @ self.H.T, self.R_cholesky)

        # Kalman gain (line 10 in Algorithm 1)
        # Since Python does not have backslash linear solve function, compute Kalman gain
        # with scipy.linalg.solve_triangular
        #  solve G.T K1 = H (G.T is lower triangular)
        K1 = solve_triangular(G.T, self.H, lower = True)

        K2 = K1 @ self.M_prior.T @ self.M_prior

        # Solve G K3 = K2
        K3 = solve_triangular(G, K2, lower = False)
        K = K3.T

        # Update the state estimate
        #state update
        self.x_posterior = self.x_prior + (K @ y)
        
        # Update the estimate covariance
        I = np.eye(len(self.P_posterior))
        self.M_posterior = self.qr(self.M_prior @ (I - K @ self.H).T, self.R_cholesky @ K.T)

    # Helper function to compute R component of QR composition of vertical
    # concatenation of matrices A, B
    def qr(self, A, B):
        concatenated = np.vstack((A,B))
        r = linalg.qr(concatenated, mode = 'r')
        return r

    def skip_update(self):
        """Call this instead of update() when measurement is missing"""
        self.x_posterior = self.x_prior.copy()
        self.M_posterior = self.M_prior.copy()

    def set_matrices(self, F, H, R, Q):
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q
        self.Q_cholesky = linalg.cholesky(Q)
        self.R_cholesky = linalg.cholesky(R)


    def set_initial_state(self, x0, P0):
        self.x_posterior = x0
        self.P_posterior = P0
        self.M_posterior = linalg.cholesky(P0)

    def get_updated_state(self):
        return self.x_posterior
    
    def get_updated_covariance(self):
        return self.M_posterior.T @ self.M_posterior
    
    def get_prior_state(self):
        return self.x_prior
    
    def get_prior_covariance(self):
        return self.M_prior.T @ self.M_prior
    
    def get_last_residual(self):
        return self.residual
    
    