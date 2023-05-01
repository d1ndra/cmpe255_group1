#Note: This contains the PCA classes from the 255 final project.
# PCA  Alex

# #1 - Custom PCA class which has methods for transforming, inverse transforming, and getting loadings (essentially components)
# Note: some normalization issues could be addressed but it all works

from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MPCA:
    # Note: can change n_components = None for a selection based on variable per cent of variance explained
    def __init__(self, n_components=1):
        # Initialize class variables that will be used later
        self.n_components = n_components
        self.projection_matrix = None
        self.evr_ = None

    # Perform PCA on the input data and returns the transformed data
    def fit_transform(self, X):
        # Convert to mean = 0
        X_cent = X - X.mean(axis=0)

        # Compute the svd as described in the report
        u, s, vh = np.linalg.svd(X_cent, full_matrices=False)

        # Compute the projection matrix which we will use to transform the data matrix
        self.projection_matrix = vh.T

        # Project the data onto the principal components
        X_pca = X_cent.dot(self.projection_matrix)

        # NOTE: this is explained in great detail the report.
        # Ax X_pca = us, we can find the explained variance ratios by using
        # The normalized square norms of the data points along the principal components
        # This is one of the benefits of using the SVD method as it doesn't require finding the covariance matrix
        # Alternatively, we can think about it in the context of eigenvalues -- as the
        # singular values are related to these via square rooting, it makes sense that the distances should be squared
        # as X_pca = us is something like the square root of the eigendecomposition of X^TX
        # As I explain in the paper, this is a bit more efficient than the covariance one.
        evr = np.sum(X_pca ** 2, axis=0) / np.sum(X_cent ** 2)

        # Assign the value of evr_ attribute
        self.evr_ = evr

        # Compute the cumulative explained variance ratio [note: no longer used as class var*, 4/29]
        cum_evr = np.cumsum(evr)

        # NOTE: it currently defaults to 1 but could be useful in a future imnplementation
        if self.n_components is None:
            self.n_components = np.sum(cum_evr < 0.75) + 1

        return X_pca[:, :self.n_components]

    # function to project data onto the principal components after the model has been fit
    def transform(self, X):
        # Center the data
        X_cent = X - X.mean(axis=0)
        # Project the data onto the principal components
        X_pca = X_cent.dot(self.projection_matrix)
        return X_pca[:, :self.n_components]

        # Note components typically refer to slightly different concepts,

    # with components being the principal components themselves, and loadings
    # being the normalized versions of their coefficients,
    #  but here they're all normalized so I only left the loadings.
    # There may be some instability with non-normalized data
    def get_loadings(self):
        loadings = self.projection_matrix[:, :self.n_components]
        return loadings

    # Inverse transform function as the projection matrix is orthogonal
    # we can simply multiply by its transpose here.
    def inverse_transform(self, X_transformed):
        # Project the transformed data back onto the original feature space
        X_original = X_transformed.dot(self.projection_matrix.T)

        return X_original


# Basic Linear regression class that uses gradient descent. It is
# then used by the PCA regression class we define later.
# This is basically following exactly what we had on homework #2
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None

    def fit(self, X, y):
        # Initialize the model parameters
        self.theta = np.zeros(X.shape[1]).reshape(-1, 1)

        # Perform gradient descent
        for i in range(self.num_iterations):
            # Calculate the predicted values of y
            y_pred = np.dot(X, self.theta)
            # Calculate the error between the predicted values and the actual values of y
            error = y_pred - y
            # Calculate the gradient of the cost function with respect to theta
            gradient = np.dot(X.T, error) / len(X)
            # Update the values of theta
            self.theta -= self.learning_rate * np.dot(X.T, error) / len(X)

    def predict(self, X):
        # Calculate the predicted values of y using the learned parameters
        y_pred = np.dot(X, self.theta)

        return y_pred


# #2 - PCA  Regression
# Note: I had some trouble with the normalizations but this seems to work well.
# The problem is that the data has to be normalized for the PCA to run but then
# denormalized to get predictions and this was giving me some trouble.
# As it is the normalization is done and undone in the fit and predict methods.

class PCALinearRegression:
    # initialize variables to use
    def __init__(self, n_components, learning_rate=0.01, num_iterations=1000):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.pca = MPCA(n_components)
        self.regression = LinearRegressionGD(learning_rate, num_iterations)
        self.feature_mean = None
        self.feature_std = None

    # Note this will store the normalization features for use in predicting. This
    # denormalization is a source of a few difficulties.
    def fit(self, X, feature_indices):
        # Normalizations done separately for a more robust predictive algorithm
        # Normalize the selected features
        self.feature_mean = X[:, feature_indices].mean(axis=0)
        self.feature_std = X[:, feature_indices].std(axis=0)
        X_norm = (X[:, feature_indices] - self.feature_mean) / self.feature_std

        # Normalize the remaining features
        X_r = np.delete(X, feature_indices, axis=1)
        X_r_mean = X_r.mean(axis=0)
        X_r_std = X_r.std(axis=0)
        X_r_norm = (X_r - X_r_mean) / X_r_std

        # Perform PCA on the remaining features
        X_pca = self.pca.fit_transform(X_r_norm)

        # Fit the linear regression model using the principal components
        self.regression.fit(X_pca, X_norm)

    # Note: issues with normalization seem to be resolved but this is all complicated
    # by the fact that you need to transform to PCA space to make predictions.
    # Note: this uses the features of the training set on predictions, may be an issue?
    def predict(self, X, feature_indices):
        # Normalize the aset
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_norm = (X - X_mean) / X_std

        # Remove the selected features to perform PCA on the remaining features
        X_r = np.delete(X_norm, feature_indices, axis=1)

        # Use the transform method mentioned earlier to transform the new data onto the principal components
        X_pca = self.pca.transform(X_r)

        # Predict the normalized target features using the linear regression model inside
        y_pred_norm = self.regression.predict(X_pca)

        # Denormalize the predicted values using the normaliation features of the training set [ISSUE?]
        y_pred = y_pred_norm * self.feature_std + self.feature_mean

        return y_pred

    # Note: [fixed]
    def mse(self, X, feature_indices):
        y_pred = self.predict(X, feature_indices)
        mse = np.mean((X[:, feature_indices] - y_pred) ** 2)
        return mse

    # R-squared [issues with PCA transform?]
    # Standard r squared calculation.
    def r_squared(self, X, feature_indices):
        y_pred = self.predict(X, feature_indices)
        ss_res = np.sum((X[:, feature_indices] - y_pred) ** 2)
        ss_tot = np.sum((X[:, feature_indices] - X[:, feature_indices].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared


# #3 - PCA Outliers
# Finding Outliers
# This is a class that extends MPCA that provides a handy framework for outlier detection using PCA
# as it performs the principal component analysis and then returns information about the outliers
# while also providing a graphical interface for visualizing the outliers projected onto the plane
# of the first 2 principal components. Of course it also inherits the PCA features of the superclass.

# Note: I didn't just put this in the PCA class since I was afraid of tinkering with it after getting
# it to work. However, this still works fin as again it has all the MPCA methods and can be used that way.

class OutPCA(MPCA):
    def __init__(self, n_components):
        # Initialize from superclass
        super().__init__(n_components)

    # main method which will take the outlers and plot them
    def plot_outliers(self, X):
        # Use the transform method from MPCA class
        principal_components = self.transform(X)

        # Calculate distances from the origin
        distance = np.sqrt(np.sum(np.square(principal_components), axis=1))
        # Calculate the median distance from the origin
        median_distance = np.median(distance)
        # Calculate the median of those distances
        mad_distance = np.median(np.abs(distance - median_distance))

        # Identify outliers based on MADs
        outliers_combined = self.get_outliers_combined(distance, median_distance, mad_distance)

        # We're going to plot the points using the top 2 components as the axes, although there may be more dimensions:
        # the data is essentially projected onto the plane of the first components
        sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=outliers_combined,
                        palette='Set2')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Outlier Detection with PCA')
        self.create_legend()
        plt.show()

    # Note we use median absolute deviation instead of mean or sd to get a more robust algorithm as values are not squared
    def get_outliers_combined(self, distance, median_distance, mad_distance):
        # 2 mads, 4 mads, 8 mads, 16 mads outliers
        outliers1 = (distance > (median_distance + 2 * mad_distance))
        outliers2 = (distance > (median_distance + 4 * mad_distance))
        outliers3 = (distance > (median_distance + 8 * mad_distance))
        outliers4 = (distance > (median_distance + 16 * mad_distance))
        return 4 * outliers4 + 3 * outliers3 + 2 * outliers2 + 1 * outliers1

    # Create the framework for efficiently visualizing the graph that will describe outliers
    def create_legend(self):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='0-2 MADs', markerfacecolor=sns.color_palette('Set2')[0],
                   markersize=8),
            Line2D([0], [0], marker='o', color='w', label='2-4 MADs', markerfacecolor=sns.color_palette('Set2')[1],
                   markersize=8),
            Line2D([0], [0], marker='o', color='w', label='4-8 MADs', markerfacecolor=sns.color_palette('Set2')[2],
                   markersize=8),
            Line2D([0], [0], marker='o', color='w', label='8-16 MADs', markerfacecolor=sns.color_palette('Set2')[3],
                   markersize=8),
            Line2D([0], [0], marker='o', color='w', label='16+ MADs', markerfacecolor=sns.color_palette('Set2')[4],
                   markersize=8)]

        plt.legend(handles=legend_elements, title='Outlier')

    # Method to return the indeces of the n highest outlers of the dataset
    def get_n_highest_outliers(self, X, n=5):
        # Note: there's some redundancy with the plot method but it was easier to just repeat the calculation rather than do it sep.
        principal_components = self.transform(X)
        distance = np.sqrt(np.sum(np.square(principal_components), axis=1))
        median_distance = np.median(distance)
        mad_distance = np.median(np.abs(distance - median_distance))

        # Identify outliers based on MADs
        outliers_combined = self.get_outliers_combined(distance, median_distance, mad_distance)

        # Get the indices of the N highest outliers
        sorted_indices = np.argsort(outliers_combined)[-n:]
        return sorted_indices


# Create an OutPCA object
myoutliers = OutPCA(n_components=5)
myoutliers.fit_transform(X_normalized)
myoutliers.plot_outliers(X_normalized)

n_highest_outliers = 5

# Get outlier indices
outlier_indices = myoutliers.get_n_highest_outliers(X_normalized, n=n_highest_outliers)
print(outlier_indices)
merged_df.iloc[outlier_indices]
# set max rows and columns to None
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

outlier_points = merged_df.iloc[outlier_indices]
print(outlier_points)
