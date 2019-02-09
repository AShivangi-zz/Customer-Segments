import numpy as np
import pandas as pd
from IPython.display import display
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
import visuals as vs

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")

display(data.describe())
indices = [100, 200, 300]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print("Chosen samples of wholesale customers dataset:")
display(samples)

percentiles_data = 100*data.rank(pct=True)
percentiles_samples = percentiles_data.iloc[indices]
seaborn.heatmap(percentiles_samples, annot=True)


# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(labels='Frozen', axis=1)
# Split the data into training and testing sets(0.25) using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, data['Frozen'], test_size=0.25, random_state=42)
# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print ("Score:{:.4f}".format(score))

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')


correlations = data.corr(method='spearman')
seaborn.heatmap(correlations,
            cmap='RdBu',
            alpha=0.8,
            linewidth=0.6,
            vmin=-1.0,
            vmax=1.0, cbar=True, square=True, mask=correlations==1.0, annot=True)

# Scale the data using the natural logarithm
log_data = np.log(data)
# Scale the sample data using the natural logarithm
log_samples = np.log(samples)
# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Display the log-transformed sample data
display(log_samples)

list_outliers = np.array([], dtype='int')
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(a=log_data[feature], q=25)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(a=log_data[feature], q=75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    print("Data points considered outliers for the feature '{}':".format(feature))
    outlier = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(outlier)
    list_outliers = np.append(list_outliers, outlier.index)
    
# Select the indices for data points you wish to remove
unique, counts = np.unique(list_outliers, return_counts=True)
outliers = unique[counts > 1]
print ("\nData points that are considered outliers for more than 1 feature:\n")
print (outliers)

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
pca.fit(good_data)
# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)
#explained variance ratios
ex_variance = pca.explained_variance_ratio_

print('First two components: {:.4f}'.format(ex_variance[:2].sum()))
print('First four components: {:.4f}'.format(ex_variance[:4].sum()))

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)
# Transform the good data using the PCA fit above
reduced_data = pca.fit_transform(good_data)
# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
# Create a biplot
vs.biplot(good_data, reduced_data, pca)

# Apply your clustering algorithm of choice to the reduced data
clusterer =GMM(n_components=2, n_init=10)
clusterer.fit(reduced_data)
# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)
# Find the cluster centers
centers = clusterer.means_
# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)
# Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds)
print ("The silhouette score is: {:.3f}".format(score))

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

# Inverse transform the centers
log_centers = pca.inverse_transform(centers)
# Exponentiate the centers
true_centers = np.exp(log_centers)
# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)
display(samples)

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)
