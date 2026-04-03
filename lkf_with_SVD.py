'''
    Implementation of a LKF with SVD based on paper by 
    Ordonez et al., doi:10.3390/app10155168
    This KF follows "Algorithm 1": Singular value decomposition-based Kalman filtering
    This does not include the 'Adaptive' KF model described in the paper, 
    so we assume measurement covariance matrix R is time invariant. 
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
        self.D = None               #Singular value matrix of P
        self.D_posterior = None     #Posterior singular value matrix of P
        self.U = None               #Left orthogonal factor of SVD of P
        self.U_posterior = None     #Posterior left orthogonal factor of SVD of P

    #Initialize matrices U and D
    def initialize(self):
        #Perform initial SVD
        self.U, D_squared, _ = np.linalg.svd(self.P_Posterior)
        self.D = np.sqrt(D_squared)
        #Cholesky matrix calculation pre-computed since measurement covariance assumed time-invariant
        self.L = np.linalg.inv((sqrtm(self.R)).T)

    #Step 1 in Algorithm 1: Update SVD using pre-array (eq. 32)
    def update_SVD(self):
        #Constrcut pre-array: (m+n) x n matrix
        A = np.concatenate((self.L.T @ self.H @ self.U, np.linalg.inv(self.D)), axis = 0)
        #Obtain pre-array SVD factor V
        #Consider computing this manually and skipping U calculation for efficicency
        _, D_temp, V_transpose = np.linalg.svd(A)    
        V = V_transpose.T
        #Update SVD factors U, D
        self.U_posterior = self.U @ V
        self.D_posterior = np.linalg.inv(D_temp)

    #Steps 2 and 3 in Algorithm 1
    def update(self, z):
        #Compute Kalman Gain (eq 43) (Step 2)
        B = self.H.T @ self.L
        K = self.U_posterior @ (self.D_posterior)**2 @ self.U_posterior.T @ B @ self.L.T
    
        #Update state estimate (Step 3) (eq 44)
        y = z - (self.H @ self.x_prior)
        self.x_posterior = self.x_prior + (K @ y)

    #Step 4 in Algorithm 1
    def predict(self):
        #Predict the next state
        self.x_prior = self.F @ self.x_posterior

        #Calculate U, D  SVD factors using eq (21).
        #Construct pre-array: (s+n) x n
        A = np.concatenate((sqrtm(self.D_posterior) @ self.U_posterior @ self.F.T,
                            (sqrtm(self.Q)).T), axis = 0)
        #Consider computing SVD manually to avoid computing 
        # left orthogonal orthgonal component (high dimension)
        _, D_temp, V_temp = np.linalg.svd(A, full_matrices = False)
        self.D = D_temp
        self.U = V_temp
    
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


