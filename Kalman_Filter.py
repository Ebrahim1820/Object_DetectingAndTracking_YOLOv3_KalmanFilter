'''
    This class is a implementation of Kalman Filter Algorithm
    to track object in real time.

    File Name        : Kalman_Filter
    File Description : High performance tracking by the use of
                       Kalman Filter Algorithm
    Author           : Ebrahim Najafi
    Date created     : 08/21/2019
    Python Version   : 3.4
'''
import numpy as np

class KalmanFilter:
    """
    The Kalman Filter is a unsupervised algorithm for tracking a single object
    in a continuous state space. Given a sequence of noisy measurements,
    the Kalman Filter is able to recover the “true state” of
    the underling object being tracked.Common uses for the Kalman Filter
    include radar and sonar tracking and state estimation in robotics.
    Referances: https://pykalman.github.io
    Formulated as follow:
            Prediction:
                u’k|k-1 = Fu’k-1|k-1
                Pk|k-1 = FPk-1|k-1 FT + Q
            Update:
                C = APk|k-1 AT + R
                Kk = Pk|k-1 AT C-1
                u’k|k = u’k|k-1 + Kk (bk – Au’k|k-1)
                Pk|k = Pk|k-1 – Kk CKT
    """
    def __init__(self):

        self.delta_Time = .005

        self.A = np.array([[1,0],[0,1]])      # Matrix in Observation equations
        self.u = np.array([[0],[0.]])      # state vecotr position, velocity
      

        # b is center coordinates of each object
        self.b = np.array([[0],[255]])      # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.,self.delta_Time],[0.,1.]])  # state transition matrix
        self.R = np.array([[1.,0.],[0.,1.]]) # observation noise matrix
        self.Q = np.array([[1.,0.],[0.,1.]])          # process noise matrix
        self.previousResult = np.array([[0],[255]])


    def prediction(self):

        self.u = np.round(np.dot(self.F , self.u))
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.previousResult = self.u

        return self.u

    def update(self, b , flag):
        # b: is a vector of observations
        # ckeck: if "true" prediction
        # result will be updated else detection

        # update using prediction
        if not flag:
            self.b =  self.previousResult
        # update using detection
        else:
            self.b = b

        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A, self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))

        self.previousResult = self.u

        return self.u
check = True
vector_of_observations = 14.5
kf = KalmanFilter()
print(kf.prediction())
print(kf.update(vector_of_observations,check))



