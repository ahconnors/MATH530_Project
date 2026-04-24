'''
    Implementation of a LKF with SVD based on paper by 
    Ordonez et al., doi:10.3390/app10155168
    This KF follows "Algorithm 1": Singular value decomposition-based Kalman filtering
    This does not include the 'Adaptive' KF model described in the paper, 
    so we assume measurement covariance matrix R is time invariant. 
    adaptive_kalman_filter_svd.py implements the adaptive version of this SVD-based KF.
    '''


import numpy as np
from scipy.linalg import sqrtm


class LinearKalmanFilter_with_SVD:
    

    def __init__(self):
        self.x_prior = None         #Prior state estimate
        self.x_posterior = None     #Posterior state estimate
        self.P_prior = None         #Prior estimate covariance
        self.P_posterior = None     #Posterior estimate covariance
        self.F = None               #State transition matrix
        self.H = None               #Measurement matrix
        self.R = None               #Measurement noise covariance
        self.Q = None               #Process noise covariance
        self.D_prior = None         #Singular value matrix of P
        self.D_posterior = None     #Posterior singular value matrix of P
        self.U_prior = None         #Left orthogonal factor of SVD of P
        self.U_posterior = None     #Posterior left orthogonal factor of SVD of P
        self.residual = None        # Most recetn residual /Kalman innovation

    #Initialize matrices U and D
    def initialize(self):
        print(self.P_posterior)
        print(self.x_posterior)
        #Perform initial SVD
        self.U_prior, D_squared, _ = np.linalg.svd(self.P_posterior)
        self.D_prior = np.diag(np.sqrt(D_squared))

        self.U_posterior = self.U_prior.copy()
        self.D_posterior = self.D_prior.copy()
        #Cholesky matrix calculation pre-computed since measurement covariance assumed time-invariant
        self.L = np.linalg.inv((np.linalg.cholesky(self.R)).T)
        #Initialize x_prior to x_posterior for the first iteration
        self.x_prior = self.x_posterior.copy()
        print(self.x_posterior)

    #Step 1 in Algorithm 1: Update SVD using pre-array (eq. 32)
    def update_SVD(self):
        self._check_state()
        #Constrcut pre-array: (m+n) x n matrix
        A = np.concatenate((self.L.T @ self.H @ self.U_prior, np.linalg.inv(self.D_prior)), axis = 0)
        #Obtain pre-array SVD factor V
        #Consider computing this manually and skipping U calculation for efficicency
        _, D_temp, V_transpose = np.linalg.svd(A)    
        V = V_transpose.T
        #Update SVD factors U, D
        self.U_posterior = self.U_prior @ V
        self.D_posterior = np.linalg.inv(np.diag(D_temp))

    #Steps 2 and 3 in Algorithm 1
    def update(self, z):
        self.update_SVD()
        #Compute Kalman Gain (eq 43) (Step 2)
        B = self.H.T @ self.L
        K = self.U_posterior @ (self.D_posterior)**2 @ self.U_posterior.T @ B @ self.L.T
    
        #Update state estimate (Step 3) (eq 44)
        y = z - (self.H @ self.x_prior)
        self.x_posterior = self.x_prior + (K @ y)
        self.residual = y

    def skip_update(self):
        """Call this instead of update() when measurement is missing"""
        self.x_posterior = self.x_prior.copy()
        self.U_posterior = self.U_prior.copy()   # carry forward SVD factors unchanged
        self.D_posterior = self.D_prior.copy()

    #Step 4 in Algorithm 1
    def predict(self):
        #self._check_state()
        #Predict the next state
        self.x_prior = self.F @ self.x_posterior

        #Calculate U, D  SVD factors using eq (21).
        #Construct pre-array: (s+n) x n
        # A = np.concatenate((sqrtm(self.D_posterior) @ self.U_posterior @ self.F.T,
        #                     sqrtm(self.Q).T), axis = 0)
        A = np.concatenate((self.D_posterior @ self.U_posterior.T @ self.F.T,
                            (np.linalg.cholesky(self.Q)).T), axis = 0)
        #Consider computing SVD manually to avoid computing 
        # left orthogonal orthgonal component (high dimension)
        _, D_temp, V_temp = np.linalg.svd(A, full_matrices = False)
        self.U_prior = V_temp.T
        self.D_prior = np.diag(D_temp)
    
    #In this KF we assume R is time-invariant (does not depend on time-step), so we skip Step 5

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
    
    def get_updated_cholesky_matrix(self):
        return self.L
    
    def get_updated_covariance(self):
        return self.P_posterior
    
    def get_prior_state(self):
        return self.x_prior
    
    def get_prior_covariance(self):
        return self.P_prior
    
    def get_last_residual(self):
        return self.residual

    def _check_state(self, label=""):
        problems = []
        for name, val in [("U_prior", self.U_prior), 
                        ("D_prior", self.D_prior),
                        ("U_posterior", self.U_posterior),
                        ("D_posterior", self.D_posterior),
                        ("x_prior", self.x_prior),
                        ("x_posterior", self.x_posterior)]:
            if val is None:
                problems.append(f"{name} is None")
            elif np.isnan(val).any():
                problems.append(f"{name} contains NaN")
            elif np.isinf(val).any():
                problems.append(f"{name} contains Inf")
        if problems:
            raise ValueError(f"[{label}] State corruption: {', '.join(problems)}")