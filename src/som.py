# Generic imports.
import time
import numpy as np

# Specific imports.
from datetime import datetime
from scipy.spatial import distance
from numpy.linalg import norm as la_norm

# Public interface.
__all__ = ['SOM']

# Current version.
__version__ = '0.1.0'

# Author.
__author__ = "Michalis Vrettas, PhD"

# Email.
__email__ = "michail.vrettas@gmail.com"


# Self Organizing Map class.
class SOM(object):
    """
    Description:
    This class implements a Self-Organizing-Map algorithm. Steps in brief:

    1) Initialize the weight vectors in the map (network nodes) using uniformly
    random values between a predefined range. The default range: (-1, 1) can be
    set only during the construction of the object.

    2) Randomly pick an input (training) vector. The order of the input vectors
    changes periodically every 'N' epochs.

    3) Traverse each node in the map:
       3.1) Use the Euclidean distance formula (i.e. norm) to find similarities
            between the input vector and the map's weight vectors
       3.2) Track the node that produces the smallest distance (this node is the
            best matching unit -- BMU)

    4) Update the weight vectors of the nodes in the neighborhood of the BMU
    (including the BMU itself) by pulling them closer to the input vector.

    5) Repeat from step [2] until convergence (or a maximum number of iteration
    has been reached). Here convergence is measured with the mean absolute error,
    but we can also change it to something more suitable if necessary.
    """

    def __init__(self, m=11, d=1, u_range=(-1, 1), metric='euclidean'):
        """
        Description:
        Constructor for a self-organizing-map object. The network is assumed
        square (i.e. grid shape is MxM).  The depth of the network should be
        always the same as the size of the input vectors 'd'.

        Args:
        - m: grid size (m x m).
        - d: neuron size (always same as input vector size).
        - u_range: the range of the uniform numbers in the initialization
        (default=(-1, 1)).
        - metric: string determining the way to compute the distance between
        the nodes in the update stage.

        Supported metrics are: all the scipy.spatial.distances with default
        parameters.

        Note:
        The constructor can also accept a string which can be converted to a
        numerical value, e.g. '10'. But if the conversion fails e.g. '1.398',
        then a ValueError exception will be raised.

        Raises:
        - ValueError: If any of the inputs is non-positive, or the range limits
        (low, high) are not ordered correctly. Also, if the chosen metric is not
        on the list with the supported values.
        """

        # Create a random number generator.
        self._rng = np.random.default_rng()

        # Make sure the inputs are integer values.
        # NOTE: This will raise an error if the conversion fails.
        self._m = int(m)
        self._d = int(d)

        # Negative values for grid size are not permitted.
        if self._m < 1:
            raise ValueError(f" SOM: Grid size input 'M' can't be negative: {self._m}")
        # _end_if_

        if self._d < 1:
            raise ValueError(f" SOM: Input vector size 'd' can't be negative: {self._d}")
        # _end_if_

        # Get the [low, high] limits of the uniform numbers.
        low_lim, high_lim = u_range

        # Sanity check.
        if low_lim >= high_lim:
            raise ValueError(" SOM: Input limits are invalid. NOTE: u_range = (low_lim, high_lim).")
        # _end_check_

        # Uniformly distributed random numbers: U(low, high).
        self._neurons = low_lim + (high_lim - low_lim) * self._rng.random((m, m, d))

        # Add the limits in the object.
        self._limits = (low_lim, high_lim)

        # Add the metric.
        if metric in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                      'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']:
            # Set the metric as object variable.
            self._metric = metric
        else:
            raise ValueError(f" SOM: Unknown metric selected: {metric}.")
        # _end_if_

        # Placeholder for U-Matrix.
        self._uMat = None

    # _end_def_

    @property
    def grid_size(self):
        """
        Description:
        Accessor (get) for the grid size of the SOM network.
        It is assumed that the grid is square (e.g. MxM).
        """
        return self._m

    # _end_def_

    @property
    def neuron_size(self):
        """
        Description:
        Accessor (get) of the neuron size (depth of network).
        """
        return self._d

    # _end_def_

    @property
    def shape(self):
        """
        Description:
        Returns the full shape of the SOM network (e.g. MxMxd).
        """
        return self._neurons.shape

    # _end_def_

    @property
    def distance_metric(self):
        """
        Description:
        Accessor (get) for the distance metric of the SOM network.
        """
        return self._metric

    # _end_def_

    def reset_network(self, r_seed=None):
        """
        Description:
        Re-initializes the neurons with random values. It uses the same
        range limits: (low, high) as the ones given in the constructor.
        Args:
        - r_seed: random seed for the reset of the network. If no value
        is given then the seed is not reset.
        NOTE:
        The old values of the _neurons will be overwritten. No warnings
        or extra care is taken.
        """
        # If the user has given a seed USE IT!
        if r_seed:
            self._rng = np.random.default_rng(int(r_seed))
        # _end_seed_

        # Get the limits of the object.
        low_lim, high_lim = self._limits

        # Uniformly distributed random numbers.
        self._neurons = low_lim + (high_lim - low_lim) * self._rng.random((self._m,
                                                                           self._m,
                                                                           self._d))
    # _end_def_

    def find_nearest_node(self, vec):
        """
        Description:
        Finds the closest node of the grid (i.e. shortest distance) to the input vector.

        Args:
            - vec: input vector with the training sample (dx1).

        Returns:
            - a tuple with the coordinates, on the map, of the winner node (best matching unit).

        Note:
        We use vectorizations to speed up performance.  This is optimized for the default
        'Euclidean' distances, by using np.linalg.norm() function. Other distance metrics
        can also be exploited using scipy.spatial.distance package.
        """

        # For the default metric use linalg.norm instead.
        if self._metric == 'euclidean':
            # Find the Euclidean distances (vectorized).
            ri = la_norm(self._neurons - vec, ord=2, axis=2)
        else:
            # Get the distances with the selected metric.
            ri = distance.cdist(self._neurons.reshape(self._m * self._m, self._d),
                                vec.reshape(1, self._d), metric=self._metric)

            # Go back to MxM.
            ri = ri.reshape(self._m, self._m)
        # _end_if_

        # Return the coordinates of the Best Matching Unit (node).
        return np.unravel_index(np.argmin(ri, axis=None), ri.shape)

    # _end_def_

    def _update_nodes(self, t_node, vec, eta, tk):
        """
        Description:
        Updates the nodes (weights of the network) by centering the peak at the t_node.
        All the updates happen with the same learning rate 'eta',  although this value
        can change (adaptive learning rate) during the fitting course.

        Args:
            - t_node: a tuple (row, col) with the coordinates of the BMU node.
            - vec: input (training) vector.
            - eta: learning rate (how fast/slow the weights are updated).
            - tk: is the current iteration, this affects the window of the neighborhood.

        Note(1):
        The nodes are updated according to:
        n_{i} = n_{i} + u_{i}*eta*(x_{j} - n_{i}), where:
            - n_{i} is the i-th node (vector)
            - u_{i} is a function (Gaussian in this case) that defines how the neighbours
              around the center node are going to be affected.
            - eta is the learning rate.
            - x_{j} is the current input vector.

        Note(2):
        The localization radius (window of influence) is starting by including the whole
        network, and gradually, in time, it reduces  to a single node.  This approach of
        localization could be revisited.
        """

        # Local RBF kernel function.
        def rbf(u, sig):
            return np.exp(-0.5 * u.dot(u) / sig)

        # Local bounds check function.
        # NOTE: This works because the network is assumed m*m.
        def out_of_bounds(lx):
            return lx < 0 or lx >= self._m

        # Extract the (row, col) of the node.
        row, col = t_node

        # Centre (coordinates) vector.
        mu = np.array(t_node)

        # Characteristic length scale. Controls how fast the weights
        # will decrease as they move away from the center vector mu.
        sigma = 1.0

        # Max radius size.
        r_max = int(0.5 * self._m - 1)

        # Estimate the local radius, around the center: [0, r_max].
        r_loc = int(r_max * 0.98 ** tk)

        # Update all the neighbouring neurons.
        # NB: The '+1' is added because the upper limit is exclusive.
        for i in range(row - r_loc, row + r_loc + 1):

            # Avoid going out of bounds.
            if out_of_bounds(i):
                continue
            # _end_check_

            for j in range(col - r_loc, col + r_loc + 1):

                # Avoid going out of bounds.
                if out_of_bounds(j):
                    continue
                # _end_check_

                # Get the difference of the current node from the centroid.
                d_ij = mu - [i, j]

                # Update the weights.
                self._neurons[i, j] += eta * rbf(d_ij, sigma) * (vec - self._neurons[i, j])
            # _end_columns_

        # _end_rows_

    # _end_def_

    def _compute_u_matrix(self):
        """
        Description:
        Computes the U-Matrix of the SOM network. This visualization squashes
        the 3D network map into a 2D image.
        """
        # Add the matrix on the first call.
        if self._uMat is None:
            # Estimate the dimensions of the U-Matrix.
            m = 2 * self._m - 1

            # Set the matrix in the object.
            self._uMat = np.zeros((m, m))
        else:
            # By default, U-Matrix is cleared before computed.
            self._uMat *= 0
        # _end_check_

        # Get the dimensions of the U-Matrix
        _m, _n = self._uMat.shape

        # Step -1-
        # NOTE: Horizontal distances.
        for i in range(self._m):
            for j in range(self._m - 1):
                self._uMat[2 * i, 2 * j + 1] = la_norm(self._neurons[i, j] - self._neurons[i, j + 1])
        # _end_Step_1_

        # Step -2-
        # NOTE: Vertical distances.
        for i in range(self._m - 1):
            for j in range(self._m):
                self._uMat[2 * i + 1, 2 * j] = la_norm(self._neurons[i, j] - self._neurons[i + 1, j])
        # _end_Step_2_

        # Step -3-
        # NOTE: Diagonal distances.
        for i in range(1, _m - 1, 2):
            # Map the index to the network range.
            ip = int(i / 2)

            for j in range(1, _n - 1, 2):
                # Map the index to the network range.
                jp = int(j / 2)

                # First diagonal distance.
                # [i-1, j-1] ---> [i+1, j+1]
                dist_1 = la_norm(self._neurons[ip, jp] - self._neurons[ip + 1, jp + 1])

                # Second diagonal distance.
                # [i-1, j+1] ---> [i+1, j-1]
                dist_2 = la_norm(self._neurons[ip, jp + 1] - self._neurons[ip + 1, jp])

                # Average the two distances.
                self._uMat[i, j] = 0.5 * (dist_1 + dist_2)
        # _end_Step_3_

        # Step -4-
        # NOTE: Averaged nodes.
        for i in range(0, _m, 2):
            for j in range(0, _n, 2):
                # Collects the neighboring distances.
                dist = []

                # Collect only the valid ones.
                # NB: '+2' is for the exclusive upper limit.
                for r_n in range(i - 1, i + 2):
                    # Check bounds.
                    if r_n < 0 or r_n >= _m:
                        continue

                    for c_n in range(j - 1, j + 2):
                        # Check bounds.
                        if c_n < 0 or c_n >= _n:
                            continue

                        # Put the distance in the list.
                        dist.append(self._uMat[r_n, c_n])
                    # _end_for_
                # _end_for_

                # Check if for some reason the list is empty.
                if not dist:
                    continue
                # _end_empty_

                # Average all the distances.
                self._uMat[i, j] = np.mean(dist)
        # _end_Step_4_

    # _end_def_

    def train(self, x, epochs=100, tol=1.0e-5, l_rate=None, n_update=10):
        """
        Description:
        Train the network for 'epochs' iterations. This is the main fitting
        process.

        Args:
        - x: training dataset (NxD)
        - epochs: maximum number of iterations
        - tol: tolerance to terminate the fit process
        - l_rate: learning rate (e.g. 0.01). If no value is given it will
        use a slowly decaying exponential function and will reduce in runtime.
        - n_update: number of iterations to display progress
        """
        # Local learning rate function.
        if l_rate:
            if not isinstance(l_rate, float):
                raise TypeError(" SOM.train: Learning rate 'l_rate' should be float in [0, 1]. ")
            # _end_if_

            # Make sure the value is within bounds.
            l_rate = max(1.0e-5, min(l_rate, 1.0))

            # Constant function (ignore the input).
            def eta(_):
                return l_rate
        else:
            # Slowly decaying exponential functions.
            def eta(ik):
                return max(1.0e-5, 0.98 ** ik)
        # _end_if_

        # Get the dimensions of the training data 'X'.
        # row: is the number of training samples
        # col: is the dimensionality of each sample
        row, col = x.shape

        # Check the dimensionality of training data with that
        # of the neurons in the SOM network. They should match.
        if col != self._d:
            raise ValueError(" SOM.train: Dimension mismatch.")
        # _end_if_

        # Make sure 'epochs' is integer.
        nit = int(epochs)

        # Check for positive input.
        if nit < 0:
            raise ValueError(" SOM.train: Epochs is negative.")
        # _end_if_

        # Index range of the input (training) data.
        x_range = np.arange(row)

        # Perturb the order of the input samples (in-place).
        self._rng.shuffle(x_range)

        # Track the error (for further analysis).
        epoch_error = np.zeros(nit)

        # Converged tuple: (flag, iteration).
        has_converged = (False, 0)

        # Display message.
        print(" SOM training started ...")

        # Start timing.
        start_t = time.time()

        # Run maximum 'epoch' iterations.
        for i in range(nit):
            # Initial copy of the network.
            grid_i = self._neurons.copy()

            # Run through all the input vectors.
            for j in x_range:
                # Find the winner node (best matching unit).
                bmu_node = self.find_nearest_node(x[j])

                # Update the neuron values. All the nodes are
                # updated with the same learning rate: eta(i).
                self._update_nodes(bmu_node, x[j], eta(i), i)
            # _end_samples_

            # NOTE: Here we use the mean absolute error (MAE).
            epoch_error[i] = np.sum(np.abs(grid_i - self._neurons)) / self._neurons.size

            # Check for convergence. Let it run for a few iterations.
            if i > 50:
                # The first condition will check the error on the current epoch.
                # The second will  identify if the  algorithm  has stuck to some
                # local minimum. In this case the difference  between the errors
                # will fluctuate around a value higher than the TOL but if 'n=5'
                # of them are similar, their difference are expected to be small
                # enough to detect it here.
                if epoch_error[i] <= tol or \
                        np.all(np.abs(np.diff(epoch_error[i - 5:i])) <= 1.0e-4):
                    # Change the flag and record the final iteration.
                    has_converged = (True, i)

                    # Exit the training loop.
                    break
                # _end_if_
            # _end_convergence_check_

            # Every N_UPDATES epochs.
            if (i + 1) % n_update == 0:
                # Display progress so far.
                print(" [{0}]: Epoch {1}: Error {2:.6f}".
                      format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i + 1, epoch_error[i]))

                # Perturb the order of input vectors.
                self._rng.shuffle(x_range)
            # _end_update_
        # _end_epochs_

        # Stop timing.
        stop_t = time.time()

        # Display time info.
        print(" Training process ended in: {0:.3f} sec.".format(stop_t - start_t))

        # Check if it has converged prematurely.
        if has_converged[0]:
            # This is the iteration it finished.
            k = has_converged[1]

            # Display an informative message.
            print(" SOM training converged at epoch {0} with error {1:.6f}.".
                  format(k + 1, epoch_error[k]))

            # Store the errors up to k-th iteration.
            setattr(self, "errors", epoch_error[0:k])
        else:
            print(" SOM training finished with error {0:.6f}.".
                  format(epoch_error[-1]))

            # Include the errors in the object.
            setattr(self, "errors", epoch_error)
        # _end_if_

        # Compute the uMatrix (for consistency).
        self._compute_u_matrix()

    # _end_def_

    @property
    def u_matrix(self):
        """
        Description:
        Accessor (get) of the U-Matrix.

        Note: If the matrix does not exist it will create it first.
        """

        # Check if the matrix exists.
        if not self._uMat:
            self._compute_u_matrix()
        # _end_if_

        return self._uMat

    # _end_def_

    @property
    def get_map(self):
        """
        Description:
        Returns the neurons [(M x M x d) weights of the SOM].
        """
        # Return the neurons.
        return self._neurons

    # _end_def_

    def save_som_to_hdf(self, filename, overwrite=False):
        """
        Description:
        Saves the current network as HDF5 file. If the network is "untrained"
        the field 'errors' will not exist. Hence, a warning message will be
        prompted. The UMatrix will be created on the current map (if it isn't
        created yet).

        Args:
            - filename: to save the map.
            - overwrite: if True it will overwrite the file, if it exists.
        """

        # Import the HDF5 File locally.
        from h5py import File as hdf5_File

        try:
            # This will prevent overwriting the output file by accident.
            fmode = "w-" if not overwrite else "w"

            # Open the hdf5 file for write.
            with hdf5_File(filename, fmode) as f5:
                # These fields should always exist in the object.
                f5.create_dataset("init_limits", data=self._limits)
                f5.create_dataset("dist_metric", data=self._metric)
                f5.create_dataset("network_map", data=self._neurons)

                # If the uMat does not exist, it will be created.
                f5.create_dataset("uMatrix", data=self.u_matrix)

                # Check if the errors exists as attribute.
                if hasattr(self, "errors"):
                    f5.create_dataset("epoch_error", data=self.errors)
                else:
                    print(" SOM.saveNetToHDF: Network does not seem to be trained.")
                # _end_if_
            # _end_with_file_
        except Exception as e0:
            raise RuntimeError(" SOM.saveMapToHDF: {0}".format(e0))
        # _end_try_

    # _end_def_

    def __str__(self):
        """
        Description:
        Defines the way to print a SOM object.
        """
        return str(" -- Self Organizing Map --\n"
                   " Network dimensions: {0}\n"
                   " Distance metric: {1}\n"
                   " Initialization limits: {2}\n".format(self._neurons.shape,
                                                          self._metric,
                                                          self._limits))
    # _end_def_

# _end_class_
