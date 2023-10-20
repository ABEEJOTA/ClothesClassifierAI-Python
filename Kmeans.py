__authors__ = ['1526641']
__group__ = 'DJ.15'

import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictÂºionary with options
            """
        self.num_iter = 10000
        self.K = K
        self._init_X(X)
        self._init_options(options)  






    def _init_X(self, X: np.ndarray):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        if type(X) is not np.ndarray:
            Arr = np.array(X)
        else:
            Arr = X.copy()

        shape = (np.prod(Arr.shape[:-1]), Arr.shape[-1])
        self.X = Arr.reshape(shape)
        self.X = self.X.astype(float)



    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################







    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.options['km_init'].lower() == 'first':
            c = [list(self.X[0])]
            for i in self.X:
                if len(c) == self.K:
                    break
                i = list(i)
                if i not in c:
                    c.append(i)

            self.centroids = np.array(c)
            self.old_centroids = np.array(c)
        elif self.options['km_init'].lower() == 'linear':

            unique = np.unique(self.X, axis=0)
            offset = unique.shape[0] // self.K
            idx = np.array(range(self.K))
            centroids = unique[offset * idx]
            self.centroids = centroids
            self.old_centroids = centroids.copy()

        elif self.options['km_init'].lower() == 'rgb':
            rgb = np.array([
                [255, 0, 0],  # red
                [0, 255, 0],  # green
                [0, 0, 255],  # blue
                [0, 0, 0],  # black
                [255, 248, 220],  # brown
                [255, 192, 203],  # pink
                [128, 128, 128],  # grey
                [255, 255, 0],  # Yellow
                [255, 165, 0],  # Orange
                [128, 0, 128],  # Purple
                [255, 255, 255],  # white
            ])
            idx = np.array([i % rgb.shape[0] for i in range(0, self.K + 1)])
            self.centroids = rgb[idx]
            self.old_centroids = self.centroids.copy()
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
            
            





    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        
        """
        
        distancia = distance(self.X, self.centroids)
        self.labels = np.argmin(distancia, axis = 1)
        """
        d = distance(self.X, self.centroids)
        self.labels = d.argmin(axis=1)
        
        
        
        

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        
        self.old_centroids = self.centroids.copy()
        for j, centroid in enumerate(self.centroids):
            self.centroids[j] = np.mean(self.X[j == self.labels, :], axis=0)
        
        
        
        
        
        

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        return (np.isclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])).all()

        
        
        
    
    
    
    
    
    
    

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        
        self._init_centroids()
        i = 0
        while i < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            if self.converges():
                break
            i += 1
        return i
        
    
    
    
    
    
    

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
    
        z=0
        for j, centroid in enumerate(self.centroids):
            distancia = np.linalg.norm(self.X[j == self.labels] - centroid, axis=1)
            distancia = distancia*2
            z += np.sum(distancia)
        return z / self.X.shape[0]
        
      




     

    
    
    
    

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        old = None
        for K in range(self.K, max_K):
            self.K = K
            self.fit()
            new = self.whitinClassDistance()
            dec = 100 * new / (old if old is not None else float("inf"))
            old = new
            if 100 - dec < 20:
                break
        self.K -= 1
        self.fit()
        
        

        
        
        
        
        
        


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """


    e = np.empty((X.shape[0], C.shape[0]))
    for i, centroid in enumerate(C):
        e[:, i] = np.linalg.norm(X - centroid, axis=1)
    return e







def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    e = utils.get_color_prob(centroids)
    e = np.argmax(e, axis=1)
    c = utils.colors[e]
    c, count = np.unique(c, return_counts=True)
    if count[c == 'White'] == 1:  # remove white color if there's just one
        c = c[c != 'White']
    c = np.hstack((c, np.array([''] * (11 - len(c)))))
    return c