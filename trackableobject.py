

class TrackableObject:
    def __init__(self, objectId, centroid):
        # store the object Id, then initialize a list of centroids
        # using the current centroid

        self.objectId = objectId
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.Counted = False
