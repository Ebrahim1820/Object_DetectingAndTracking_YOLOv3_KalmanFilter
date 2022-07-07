import numpy as np
from Kalman_Filter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track(object):

    """
    This class will track every object which detected
    """

    def __init__(self, prediction, trackIdCount):
        """
        Prediction: this value will predict centroids of object to track
        trackIdCount: Assigne ID number for each track object
        """

        self.track_id = trackIdCount
        self.kf = KalmanFilter()
        self.prediction = np.asarray(prediction)  # predicted centroids of (X and Y) and convert as array
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace the object path


class Tracker(object):

    """
    This class keep updates track vectors of object tracked
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def update(self, detections):

        # Create tracks if no tracks vector found

        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids

        n = len(self.tracks)
        m = len(detections)

        cost = np.zeros(shape=(n, m))
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:

                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] + diff[1][0] * diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # avrage squared Error
        cost = (0.5) * cost

        assignment = []
        for _ in range(n):
            assignment.append(-1)  # initial values -1

        # Using Hungarian Algorithm assign the correct
        # detected measurements to predicted tracks
        row_indx, col_indx = linear_sum_assignment(cost)
        for i in range(len(row_indx)):
            assignment[row_indx[i]] = col_indx[i]

        # identify tracks with no assignment
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1
        # if tracks are not detected for a long time , remove them

        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)

        # only when skipped frame exceeds max
        if len(del_tracks) > 0:
            for idx in del_tracks:
                if idx < len(self.tracks):
                    del self.tracks[idx]
                    del assignment[idx]
                else:
                    print("Error: Id is greater than length of tracks")

        # now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # start new tracking object
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update kalmanFilter state, previousResult and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].kf.prediction()

            if (assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].kf.update(detections[assignment[i]], 1)

            else:
                self.tracks[i].prediction = self.tracks[i].kf.update(np.array([[0], [0]]), 0)

            if (len(self.tracks[i].trace) > self.max_trace_length):
                #print('tracks[i].trace', tracks[i].trace)
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].kf.previousResult = self.tracks[i].prediction
