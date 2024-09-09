import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#  Data Aggregation
def aggregate_data(df):
    # Replace missing values with mean for TCP DL and UL retransmission volumes
    df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
    df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
    
    # Replace missing values for average RTT columns with the mean
    df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
    df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)

    # Aggregate data per customer (assuming MSISDN/Number is the customer identifier)
    aggregated = df.groupby('MSISDN/Number').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': 'first'  # bythe first occurrence
    }).reset_index()

    return aggregated

# aggregated_data = aggregate_data(data)

# Compute Top, Bottom, and Most Frequent Values
def compute_values(df):
    # Get top, bottom, and most frequent values for TCP DL
    tcp_dl_top = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
    tcp_dl_bottom = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
    tcp_dl_most_frequent = df['TCP DL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

    # Ensure that tcp_dl_most_frequent has 10 entries
    if len(tcp_dl_most_frequent) < 10:
        tcp_dl_most_frequent = tcp_dl_most_frequent.tolist() + [tcp_dl_most_frequent[0]] * (10 - len(tcp_dl_most_frequent))

    # Get top, bottom, and most frequent values for TCP UL
    tcp_ul_top = df['TCP UL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
    tcp_ul_bottom = df['TCP UL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
    tcp_ul_most_frequent = df['TCP UL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

    # Ensure that tcp_ul_most_frequent has 10 entries
    if len(tcp_ul_most_frequent) < 10:
        tcp_ul_most_frequent = tcp_ul_most_frequent.tolist() + [tcp_ul_most_frequent[0]] * (10 - len(tcp_ul_most_frequent))

    # Create a summary DataFrame for better readability
    tcp_dl_summary = pd.DataFrame({
        'Top 10': tcp_dl_top,
        'Bottom 10': tcp_dl_bottom,
        'Most Frequent': tcp_dl_most_frequent
    })

    tcp_ul_summary = pd.DataFrame({
        'Top 10': tcp_ul_top,
        'Bottom 10': tcp_ul_bottom,
        'Most Frequent': tcp_ul_most_frequent
    })

    # Print summaries for better visibility
    print("TCP DL Summary:")
    print(tcp_dl_summary)
    print("\nTCP UL Summary:")
    print(tcp_ul_summary)

    return {
        'TCP DL Summary': tcp_dl_summary,
        'TCP UL Summary': tcp_ul_summary
    }
     

# values = compute_values(aggregated_data)

# Distribution of Average TCP Retransmission per Handset Type
def distribution_per_handset(df, top_n=10):
    tcp_dl_distribution = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index()
    tcp_ul_distribution = df.groupby('Handset Type')['TCP UL Retrans. Vol (Bytes)'].mean().reset_index()

    # Limit to top N handset types based on average TCP DL
    tcp_dl_distribution = tcp_dl_distribution.nlargest(top_n, 'TCP DL Retrans. Vol (Bytes)')
    tcp_ul_distribution = tcp_ul_distribution.nlargest(top_n, 'TCP UL Retrans. Vol (Bytes)')

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=tcp_dl_distribution)
    plt.title('Average TCP DL Retransmission per Handset Type')
    plt.xticks(rotation=90)

    plt.subplot(1, 2, 2)
    sns.barplot(x='Handset Type', y='TCP UL Retrans. Vol (Bytes)', data=tcp_ul_distribution)
    plt.title('Average TCP UL Retransmission per Handset Type')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

# distribution_per_handset(aggregated_data)


# K-Means Clustering
def perform_clustering(df):
    # Selecting features for clustering
    features = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']]
    
    # Ensure the features are numeric
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    
    # Get centroids of the clusters
    cluster_centroids = kmeans.cluster_centers_
    
     # Select only numeric columns for describing each cluster
    numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']
    
    # Describe each cluster by calculating the mean of numeric columns
    cluster_description = df.groupby('Cluster')[numeric_columns].mean().reset_index()
    
    return cluster_description, cluster_centroids, df

def calculate_scores(clustered_df, cluster_centroids):
    # Identify the least engaged and worst experience clusters
    least_engaged_cluster = 0
    worst_experience_cluster = 0
    
    # Features used in clustering
    features = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']
    
    # Calculate engagement and experience scores
    clustered_df['Engagement Score'] = clustered_df[features].apply(lambda x: euclidean(x, cluster_centroids[least_engaged_cluster]), axis=1)
    clustered_df['Experience Score'] = clustered_df[features].apply(lambda x: euclidean(x, cluster_centroids[worst_experience_cluster]), axis=1)
    
    return clustered_df

# Calculate Satisfaction Score
def calculate_satisfaction_scores(clustered_df_with_scores):
    clustered_df_with_scores['Satisfaction Score'] = clustered_df_with_scores[['Engagement Score', 'Experience Score']].mean(axis=1)
    return clustered_df_with_scores

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def build_regression_model(df):
    # Features and target variable
    X = df[['Engagement Score', 'Experience Score']]
    y = df['Satisfaction Score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

# Build the model and get evaluation metric
regression_model, mse = build_regression_model(clustered_df_with_scores)
print(f"Mean Squared Error of the Regression Model: {mse}")

from sklearn.cluster import KMeans

def perform_kmeans_on_scores(df, n_clusters=2):
    # Features for clustering
    features = df[['Engagement Score', 'Experience Score']]
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Score Cluster'] = kmeans.fit_predict(features)
    
    # Get centroids of the clusters
    score_cluster_centroids = kmeans.cluster_centers_
    
    return df, score_cluster_centroids

# Run K-Means clustering on scores
clustered_df_with_scores, score_cluster_centroids = perform_kmeans_on_scores(clustered_df_with_scores)
print("Score Clusters:")
print(clustered_df_with_scores[['Score Cluster']].head())


# task 4.5
def aggregate_scores_per_cluster(df):
    aggregation = df.groupby('Cluster')[['Satisfaction Score', 'Experience Score']].mean().reset_index()
    return aggregation

# Aggregate scores per cluster
aggregated_scores = aggregate_scores_per_cluster(clustered_df_with_scores)
print("Aggregated Scores per Cluster:")
print(aggregated_scores)



# # Calculate Satisfaction Score
# def calculate_satisfaction_scores(clustered_df_with_scores):
#     clustered_df_with_scores['Satisfaction Score'] = clustered_df_with_scores[['Engagement Score', 'Experience Score']].mean(axis=1)
#     return clustered_df_with_scores

# # Calculate the satisfaction scores
# clustered_df_with_scores = experience_aggreagte.calculate_satisfaction_scores(clustered_df_with_scores)

# # Get the top 10 satisfied customers
# top_10_satisfied_customers = clustered_df_with_scores.sort_values(by='Satisfaction Score', ascending=False).head(10)

# # Print the top 10 satisfied customers
# print("Top 10 Satisfied Customers:")
# print(top_10_satisfied_customers[['User Id', 'Satisfaction Score']])


# # Calculate Engagement and Experience Scores
# def calculate_scores(clustered_df, cluster_centroids):
#     # Identify the least engaged and worst experience clusters
#     # For simplicity, let's assume cluster 0 is the least engaged and cluster 0 is the worst experience.
#     least_engaged_cluster = 0
#     worst_experience_cluster = 0
    
#     # Features used in clustering
#     features = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']
    
#     # Calculate engagement and experience scores
#     clustered_df['Engagement Score'] = clustered_df[features].apply(lambda x: euclidean(x, cluster_centroids[least_engaged_cluster]), axis=1)
#     clustered_df['Experience Score'] = clustered_df[features].apply(lambda x: euclidean(x, cluster_centroids[worst_experience_cluster]), axis=1)
    
#     return clustered_df

# # Combine Both Steps
# def clustering_and_scoring(df):
#     # Step 1: Perform Clustering
#     cluster_description, cluster_centroids, clustered_df = perform_clustering(df)
    
#     # Step 2: Calculate Engagement and Experience Scores
#     clustered_df_with_scores = calculate_scores(clustered_df, cluster_centroids)
    
#     return cluster_description, cluster_centroids, clustered_df_with_scores

# # Assuming `df` is your DataFrame
# # Running the combined function
# cluster_description, cluster_centroids, clustered_df_with_scores = clustering_and_scoring(df)

# # Print results
# print("Cluster Description:")
# print(cluster_description)

# print("\nCluster Centroids:")
# print(cluster_centroids)

# print("\nClustered DataFrame with Scores:")
# print(clustered_df_with_scores[['Cluster', 'Engagement Score', 'Experience Score']].head())


# # Assign Engagement and Experience scores 
    
# def assign_scores(df, engagement_cluster_centers, experience_cluster_centers):
#     # Calculate Euclidean distance to the least engaged cluster center
#     df['Engagement Score'] = euclidean_distances(
#         df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']],
#         [engagement_cluster_centers]
#     ).flatten()

#     # Calculate Euclidean distance to the worst experience cluster center
#     df['Experience Score'] = euclidean_distances(
#         df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']],
#         [experience_cluster_centers]
#     ).flatten()

#     return df




    # # Describe each cluster
    # cluster_description = df.groupby('Cluster').mean().reset_index()
    # return cluster_description

# cluster_info = perform_clustering(aggregated_data)

# Print cluster descriptions
# print(cluster_info)

# Task 4.4: Dashboard Development
# Note: Implementation would typically involve a framework like Dash or Streamlit.
# Here, we will just outline the components required.

def create_dashboard():
    # KPIs to visualize
    # 1. Average TCP DL Retransmission
    # 2. Average TCP UL Retransmission
    # 3. Average RTT DL and UL
    
    # Dashboard Usability: Ensure all visualizations are clear and labeled.
    # Interactive Elements: Use dropdowns for filtering by Handset Type.
    # Visual Appeal: Maintain a clean layout with consistent color schemes.
    
    # Deployment: Use a framework like Dash or Streamlit to deploy the dashboard online.
    pass  # Replace this with actual dashboard code using a visualization library.

# Uncomment the line below to create the dashboard
# create_dashboard()