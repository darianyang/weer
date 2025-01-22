import numpy as np
from sklearn.cluster import KMeans
from numpy.typing import ArrayLike
import tqdm.auto as tqdm
from copy import deepcopy


class StratifiedClusters:
    """Class for performing stratified k-means clustering."""

    def __init__(self, n_clusters: int, bin_bounds: ArrayLike, is_2d: bool = False):
        """
        Parameters
        ----------
        n_clusters: int, Number of clusters in each stratum

        bin_bounds: array-like, boundaries of stratified bins. Should not include -inf, +inf

        is_2d: bool, Whether the bin bounds are for 2D stratification
        """

        self.n_clusters = n_clusters
        self.is_2d = is_2d

        if is_2d:
            self.bin_boundaries = [np.concatenate([[-np.inf], bounds, [np.inf]]) for bounds in bin_bounds]
        else:
            self.bin_boundaries = np.concatenate([[-np.inf], bin_bounds, [np.inf]])

        self.kmeans_models = {}

        self.kmeans_seed = 1337
        self.max_iter = 1000

        self.coords_to_stratify = None

        self.disable_progress = False

    def fit(self, data: ArrayLike, coords_to_stratify: tuple = (0,)):
        """
        Fits the stratified clusterer model.

        Parameters
        ----------
        data: Input points. Should be 2 dimensions, (frame, coordinates).

        coords_to_stratify: tuple, Coordinates to stratify on (i.e. trajectories).
        """

        if self.coords_to_stratify is not None and not self.coords_to_stratify == coords_to_stratify:
            print(f"Warning: Changing the coordinates to stratify from {self.coords_to_stratify} to {coords_to_stratify}")
        self.coords_to_stratify = coords_to_stratify

        assert len(np.array(data).shape) <= 2, "Dimensionality not correct, expected ndim<=2"

        if self.is_2d:
            bin_pairs = zip(self.bin_boundaries[0][:-1], self.bin_boundaries[0][1:], self.bin_boundaries[1][:-1], self.bin_boundaries[1][1:])
        else:
            bin_pairs = zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])

        for i, bin_pair in tqdm.tqdm(enumerate(bin_pairs), 
                                     total=len(self.bin_boundaries[0]) - 1 if self.is_2d else len(self.bin_boundaries) - 1):

            kmeans_estimator = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                n_init='auto'
            )

            # Get the points in this bin
            if self.is_2d:
                bin_lower_x, bin_upper_x, bin_lower_y, bin_upper_y = bin_pair
                points_in_bin = np.where(
                    (data[..., coords_to_stratify[0]] >= bin_lower_x) &
                    (data[..., coords_to_stratify[0]] < bin_upper_x) &
                    (data[..., coords_to_stratify[1]] >= bin_lower_y) &
                    (data[..., coords_to_stratify[1]] < bin_upper_y)
                )
            else:
                bin_lower, bin_upper = bin_pair
                points_in_bin = np.where(
                    (data[..., coords_to_stratify[0]] >= bin_lower) &
                    (data[..., coords_to_stratify[0]] < bin_upper)
                )

            # clustering requires at least as many points as clusters
            if points_in_bin[0].shape[0] < self.n_clusters:
                print(f"Warning: Not enough points {points_in_bin} in bin {i}")
                #self.kmeans_models[i] = deepcopy(kmeans_estimator)
                self.kmeans_models[i] = None
                continue

            try:
                kmeans_estimator.fit(data[points_in_bin])
            except ValueError as e:
                print("i, bin_pair:", i, bin_pair)
                print("points_in_bin:", points_in_bin)
                raise e

            self.kmeans_models[i] = deepcopy(kmeans_estimator)

    def predict(self, data: ArrayLike):
        """
        Assigns stratified clusters to a set of input data.

        Parameters
        ----------
        data: Array-like, The set of samples to assign to clusters

        Returns
        -------
        Integer cluster assignments
        """

        discretized = np.full((data.shape[0]), fill_value=-1, dtype=int)

        cluster_offset = 0

        if self.is_2d:
            bin_pairs = zip(self.bin_boundaries[0][:-1], self.bin_boundaries[0][1:], self.bin_boundaries[1][:-1], self.bin_boundaries[1][1:])
        else:
            bin_pairs = zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])

        for i, bin_pair in tqdm.tqdm(enumerate(bin_pairs), 
                                     total=len(self.bin_boundaries[0]) - 1 if self.is_2d else len(self.bin_boundaries) - 1, 
                                     disable=self.disable_progress):

            # Get the points in this bin
            if self.is_2d:
                bin_lower_x, bin_upper_x, bin_lower_y, bin_upper_y = bin_pair
                points_in_bin = np.where(
                    (data[:, self.coords_to_stratify[0]] >= bin_lower_x) &
                    (data[:, self.coords_to_stratify[0]] < bin_upper_x) &
                    (data[:, self.coords_to_stratify[1]] >= bin_lower_y) &
                    (data[:, self.coords_to_stratify[1]] < bin_upper_y)
                )
            else:
                bin_lower, bin_upper = bin_pair
                points_in_bin = np.where(
                    (data[:, self.coords_to_stratify[0]] >= bin_lower) &
                    (data[:, self.coords_to_stratify[0]] < bin_upper)
                )

            _clustering = self.kmeans_models[i]

            # If no matches, skip (duh)
            if not points_in_bin[0].shape == (0,) and _clustering is not None:

                discretization = _clustering.predict(data[points_in_bin])
                discretized[points_in_bin] = discretization
                discretized[points_in_bin] = discretized[points_in_bin] + cluster_offset
                cluster_offset += len(_clustering.cluster_centers_)

            elif _clustering is None:
                # Assign to the closest bin clustering model
                closest_bin = self._find_closest_bin(data[points_in_bin])
                if closest_bin is not None:
                    _clustering = self.kmeans_models[closest_bin]
                else:
                    # If no matches, or empty bin fit, assign to bin 0
                    discretized[points_in_bin] = 0
                cluster_offset += 1

        assert not -1 in discretized, "Something didn't get correctly discretized"
        return discretized

    def _find_closest_bin(self, points):
        """
        Finds the closest bin with a valid clustering model.

        Parameters
        ----------
        points: Array-like, The set of points to find the closest bin for

        Returns
        -------
        The index of the closest bin with a valid clustering model
        """
        min_distance = float('inf')
        closest_bin = None

        for i, model in self.kmeans_models.items():
            if model is not None:
                distances = np.linalg.norm(model.cluster_centers_ - points[:, None], axis=2)
                min_dist = np.min(distances)
                if min_dist < min_distance:
                    min_distance = min_dist
                    closest_bin = i

        return closest_bin

    def remove_state(self, state_to_remove: int):
        """
        Removes a cluster by index, and re-indexes the remaining clusters to be consecutive.

        Parameters
        ----------
        state_to_remove: int, The index of the state to remove

        Returns
        -------
        The index of the removed state, in the space of the ORIGINAL clustering the model was built with.
        """

        cluster_offset = 0

        for i, bin_bounds in enumerate(zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])):

            _clustering = self.kmeans_models[i]

            # Check if any of the states to be removed are in this bin
            if state_to_remove in range(cluster_offset, cluster_offset + len(_clustering.cluster_centers_)):

                index_within_stratum = state_to_remove - cluster_offset
                _clustering.cluster_centers_ = np.delete(_clustering.cluster_centers_, index_within_stratum, axis=0)

                # Get the original index, before any cleaning was done
                original_index = index_within_stratum + i * self.n_clusters
                return original_index

            cluster_offset += len(_clustering.cluster_centers_)

    @property
    def cluster_centers(self):

        cluster_centers = []

        for model in self.kmeans_models.values():
            cluster_centers.append(model.cluster_centers_)

        return np.concatenate(cluster_centers)