import numpy as np
from lkf_with_SVD import LinearKalmanFilter_with_SVD
from collections import deque

class AdaptiveSVDKalmanFilter(LinearKalmanFilter_with_SVD):
    '''Adaptive Kalman Filter implemented with Singular Value Decomposition (SVD)
    
    Implements Algorithm 1 in Ordonez et al. (2020) with adaptive measurement noise
    estimation described in Equations 29-30.

    Addition to base LinearKalmanFilter_with_SVD is matrix R is not fixed
    but updates at every step with a rolling window of raw signal samples used
    to re-estimate noise standard deviation.

    Parameters
    -----------
    window_size : int - number os samples used for noise estimation (>=3)
    min_variance    : float - floor applied to each diagonal of R_k
    '''


    def __init__(self):
        super().__init__()  #Initialize lkf_with_SVD
        self.window_size = 30       # default window size, rollowing window length
        self.min_variance = 1e-10   # floor on estimated variance
        self.signal_buffer = None   #list[deque], containes one deque per measurement channel
        self.R_adaptive = None      #most recently estimated R_k


        # Setup for adaptive Kalman Filter
        def set_adaptive_params(self, window_size = 5):
            # window_size - rolling window length (must be >= 3 so that diff has
            #at least 2 elements)
            if window_size < 3:
                raise ValueError("window_size must be >= 3 (need at least 2 differences.")
            self.window_size = int(window_size)
            
            #Re-intialize buffers if the measurement dimension is already known
            if self.H is not None:
                self._init_buffers()
            
        def set_matrices(self, F, H, R, Q):
            # Delegates to parent LinearKalmanFilter_with_SVD, then initializes
            # one deque per measurement channel
            super().set_matrices(F, H, R, Q)
            self.init_buffers()

        def _init_buffers(self):
            # create separate rolling buffer for each measurement channel:
            # m deques with max length of window_size, where m is number of channels
            m = self.H.shape[0]
            self.signal_bufer = [deque(maxlen = self.window_size) for _ in range(m)]
        

        def push_signal_sample(self, measurement):
            # Append one raw measurement vector to rolling noise-estimation buffers

            # This is called even during signal outage, using last valid reading

            measurement = np.asarray(measurement, dtype = float).ravel()
            if self.signal_buffer is None:
                raise RuntimeError("Call set_matrices() before push_signal_sample().")
            if len(measurement) != len(self.signal_buffer):
                raise ValueError(
                    f"sample length {len(measurement)} does not match"
                    f"number of measurement channels {len(self.signal_buffer)}"
                )
            
            for ch, buf in enumerate(self.signal_buffer):
                buf.append(measurement[ch])

        # Adaptive R Estimation (Equations 29-30)

        def _estimate_R(self):
            #Compute the time-variant measurement noise covariance R_k

            # If a channel's buffer is too short (<3 samples), the corresponding
            # diagonal is taken from the nominal R from initialization
            # Returns R_k: (m,m) diagional ndarray

            m = self.H.shape[0]
            variances = np.empty(m)

            for ch, buf in enumerate(self.signal_buffer):
                if len(buf) <3:
                    #Fall back to nominal R
                    variances[ch] = self.R[ch,ch]
                else:
                    signal_segment  = np.array(buf, dtype = float)  
                    differences     = np.diff(signal_segment)                       # Eq 30 numerator
                    sigma_noise     = np.std(differences, ddof = 1) / np.sqrt(2.0)  # Eq 30 
                    variances[ch]   = max(sigma_noise **2, self.min_variance)       # Eq 29

            return np.diag(variances)
            

    # Override parent update function for adaptive R_k
        def update(self, z):
            # Adaptive measurement-update step.

            #Step 5 of Algorithm 1
            # Estimate R_k from current signal window
            R_nominal = self.R  # Save nominal R
            self.R  = self._estimate_R() #Use adaptive R_k
            self.R_adaptive  = self.R.copy()    # Save R_adaptive


            # Run the SVD update using adapted R_k
            super().update(z)

            # Restore nominal R so repeated set_matrices() calls are consistent
            self.R = R_nominal


        # Accessor function
        def get_adaptive_R(self):
            # Returns most recently estimated R_k, or None before first update
            return self.R_adaptive