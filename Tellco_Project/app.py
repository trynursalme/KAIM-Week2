import sys
import os
os.chdir('..')
sys.path.append(os.path.abspath('../TellCo_Project/scripts/experience_aggreagte.py'))
import sys
sys.path.append('../TellCo_Project/scripts/load_data.py')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data_from_postgres


# import function from the script
#from scripts import experience_aggreagte
# Load your data from the PostgreSQL database (this will use your existing function)
data = load_data_from_postgres("SELECT * FROM aggregate_data")
df = load_data_from_postgres(data)

# import function from the script

from scripts import experience_aggreagte 
aggregate_data =experience_aggreagte.aggregate_data(df)
distribution_per_handset =experience_aggreagte.distribution_per_handset(df)
perform_clustering =experience_aggreagte.perform_clustering(df)
calculate_scores =experience_aggreagte.calculate_scores(df)
calculate_satisfaction_scores =experience_aggreagte.calculate_satisfaction_scores(df)
build_regression_model =experience_aggreagte.build_regression_model(df)
perform_kmeans_on_scores =experience_aggreagte.perform_kmeans_on_scores(df)
aggregate_scores_per_cluster =experience_aggreagte.aggregate_scores_per_cluster(df)
#     aggregate_data, 
#     compute_values, 
#     distribution_per_handset, 
#     perform_clustering, 
#     calculate_scores, 
#     calculate_satisfaction_scores, 
#     build_regression_model, 
#     perform_kmeans_on_scores, 
#     aggregate_scores_per_cluster
# )
# Sidebar for navigation

#Title of the dashboard
st.title("TellCo Dataset Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ('User Overview Analysis', 'User Engagement Analysis', 'Experience Analysis', 'Satisfaction Analysis'))

# User Overview Analysis Page
if option == 'User Overview Analysis':
    st.header("User Overview Analysis")
    
    # Display a plot (Replace this with any plot for user overview analysis)
    values = experience_aggreagte.compute_values(aggregate_data)
    st.subheader("Top/Bottom/Most Frequent TCP DL Values")
    st.write(values['TCP DL Summary'])
    
    # Visualization for TCP retransmission distribution
    st.subheader("Distribution of TCP Retransmission per Handset Type")
    plt.figure(figsize=(10,6))
    distribution_per_handset(aggregate_data, top_n=10)
    st.pyplot(plt)

# User Engagement Analysis Page
if option == 'User Engagement Analysis':
    st.header("User Engagement Analysis")
    
    # Perform clustering
    cluster_description, cluster_centroids, clustered_df = perform_clustering(aggregate_data)
    
    st.subheader("Cluster Descriptions")
    st.write(cluster_description)
    
    # Add a visualization for engagement (replace with appropriate one)
    st.subheader("TCP Retransmission Volume by Cluster")
    sns.barplot(data=cluster_description, x='Cluster', y='TCP DL Retrans. Vol (Bytes)')
    plt.title("Average TCP DL Retransmission Volume per Cluster")
    st.pyplot(plt)

# Experience Analysis Page
if option == 'Experience Analysis':
    st.header("Experience Analysis")
    
    # Calculate scores
    clustered_df_with_scores = calculate_scores(clustered_df, cluster_centroids)
    
    st.subheader("Engagement & Experience Scores")
    st.write(clustered_df_with_scores[['MSISDN/Number', 'Engagement Score', 'Experience Score']].head(10))

    # Visualization of scores
    st.subheader("Distribution of Engagement and Experience Scores")
    plt.figure(figsize=(10,6))
    sns.histplot(clustered_df_with_scores['Engagement Score'], kde=True, label='Engagement Score')
    sns.histplot(clustered_df_with_scores['Experience Score'], kde=True, color='orange', label='Experience Score')
    plt.legend()
    st.pyplot(plt)

# Satisfaction Analysis Page
if option == 'Satisfaction Analysis':
    st.header("Satisfaction Analysis")

    # Calculate satisfaction score and run regression model
    clustered_df_with_scores = calculate_satisfaction_scores(clustered_df_with_scores)
    regression_model, mse = build_regression_model(clustered_df_with_scores)
    
    st.subheader("Satisfaction Scores")
    st.write(clustered_df_with_scores[['MSISDN/Number', 'Satisfaction Score']].head(10))
    
    st.subheader("Regression Model - Satisfaction Score Prediction")
    st.write(f"Mean Squared Error: {mse}")
    
    # Visualize satisfaction scores clustering
    clustered_df_with_scores, score_cluster_centroids = perform_kmeans_on_scores(clustered_df_with_scores)
    
    st.subheader("Satisfaction Score Clusters")
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=clustered_df_with_scores, x='Engagement Score', y='Experience Score', hue='Score Cluster', palette='coolwarm')
    plt.title("Engagement vs Experience Scores by Cluster")
    st.pyplot(plt)

# Final deployment button
st.sidebar.markdown("---")
st.sidebar.write("**Deploy the Dashboard**")
if st.sidebar.button('Deploy'):
    st.sidebar.success("The dashboard has been successfully deployed!")

st.sidebar.title("TellCo Dashboard")
option = st.sidebar.selectbox("Select Analysis", ("User Overview Analysis", "User Engagement Analysis", "Experience Analysis", "Satisfaction Analysis"))

# # Page: User Overview Analysis
# if option == "User Overview Analysis":
#     st.title("User Overview Analysis")

#     # Aggregate the data using your custom function
#     aggregated_data = experience_aggreagte.aggregate_data(data)

#     # Display the aggregated data
#     st.write("### Aggregated Data Overview")
#     st.dataframe(aggregated_data.head())

#     # Compute top/bottom values and frequent values
#     st.write("### Top, Bottom, and Most Frequent TCP Retransmission Volumes")
#     summaries = experience_aggreagte.compute_values(aggregated_data)
#     st.write("#### TCP DL Summary")
#     st.dataframe(summaries['TCP DL Summary'])
#     st.write("#### TCP UL Summary")
#     st.dataframe(summaries['TCP UL Summary'])

#     # Distribution of average TCP retransmission per handset type
#     st.write("### TCP Retransmission Distribution per Handset Type")
#     distribution_per_handset(aggregated_data)

# # Page: User Engagement Analysis
# elif option == "User Engagement Analysis":
#     st.title("User Engagement Analysis")

#     # Perform clustering using your custom function
#     cluster_description, cluster_centroids, clustered_data = perform_clustering(aggregate_data)
#     # Calculate engagement and experience scores based on cluster centroids
#     clustered_df_with_scores = calculate_scores(clustered_data, cluster_centroids)

#     # Calculate satisfaction score by averaging engagement and experience scores
#     clustered_df_with_scores = calculate_satisfaction_scores(clustered_df_with_scores)
#     # Display cluster descriptions
#     st.write("### Cluster Descriptions")
#     st.dataframe(cluster_description)

#     # Scatter plot for Engagement vs Experience by Cluster
#     st.write("### Engagement vs Experience")
#     fig, ax = plt.subplots()
#     sns.scatterplot(x='TCP DL Retrans. Vol (Bytes)', y='TCP UL Retrans. Vol (Bytes)', hue='Cluster', data=clustered_data, ax=ax)
#     st.pyplot(fig)

# # Page: Experience Analysis
# elif option == "Experience Analysis":
#     st.title("Experience Analysis")

#     # Reuse clustering results
#     st.write("### Cluster Experience Metrics")
#     st.dataframe(experience_aggreagte.cluster_description)

#     # Display additional network metrics (RTT)
#     st.write("### Average RTT per Cluster")
#     fig, ax = plt.subplots()
#     sns.barplot(x='Cluster', y='Avg RTT DL (ms)', data=clustered_data, ax=ax)
#     sns.barplot(x='Cluster', y='Avg RTT UL (ms)', data=clustered_data, ax=ax, color="red")
#     st.pyplot(fig)

# # Page: Satisfaction Analysis
# elif option == "Satisfaction Analysis":
#     st.title("Satisfaction Analysis")

#     # Calculate engagement and experience scores
#     clustered_with_scores = calculate_scores(clustered_data, cluster_centroids)

#     # Calculate satisfaction scores
#     clustered_with_scores = calculate_satisfaction_scores(clustered_with_scores)

#     # Display satisfaction scores
#     st.write("### Top 10 Satisfied Customers")
#     top_customers = clustered_with_scores[['MSISDN/Number', 'Satisfaction Score']].nlargest(10, 'Satisfaction Score')
#     st.dataframe(top_customers)

#     # Build regression model
#     st.write("### Satisfaction Prediction Model")
#     model, mse = build_regression_model(clustered_with_scores)
#     st.write(f"Mean Squared Error of the Model: {mse}")

#     # Run k-means clustering on scores
#     st.write("### K-Means Clustering on Satisfaction Scores")
#     clustered_with_scores, score_centroids = perform_kmeans_on_scores(clustered_with_scores)
#     st.write("Score Clusters")
#     st.dataframe(clustered_with_scores[['Score Cluster', 'Engagement Score', 'Experience Score']].head())

#     # Aggregate satisfaction and experience scores per cluster
#     st.write("### Aggregated Scores per Cluster")
#     aggregated_scores = aggregate_scores_per_cluster(clustered_with_scores)
#     st.dataframe(aggregated_scores)

# # Load your data from the PostgreSQL database (this will use your existing function)
# data = load_data_from_postgres("SELECT * FROM aggregate_data")

# from scripts import experience_aggreagte

# # Sidebar for navigation
# st.sidebar.title("TellCo Dashboard")
# option = st.sidebar.selectbox("Select Analysis", ("User Overview Analysis", "User Engagement Analysis", "Experience Analysis", "Satisfaction Analysis"))

# # Placeholder for different pages
# if option == "User Overview Analysis":
#     st.title("User Overview Analysis")
    
#     # Assuming you have a function to perform aggregation and transformations for the overview
#     st.write("## Aggregated Overview Metrics")
#     aggregated_data = aggregate_data(data)  # Replace this with the correct aggregation function
    
#     # Display the summary of top/bottom values and frequencies
#     st.write("### Top and Bottom TCP Retransmission Volumes")
#     st.dataframe(aggregated_data[['MSISDN/Number', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].sort_values(by='TCP DL Retrans. Vol (Bytes)', ascending=False).head(10))
    
#     st.write("### Most Frequent Handsets by Users")
#     most_frequent_handsets = data['Handset Type'].value_counts().head(10)
#     st.bar_chart(most_frequent_handsets)

# elif option == "User Engagement Analysis":
#     st.title("User Engagement Analysis")
    
#     # Assuming you already have a function to compute engagement metrics and perform clustering
#     st.write("## Clustering Results")
#     clustered_data = compute_engagement_clusters(data)  # Replace with your clustering function
    
#     # Display cluster descriptions
#     st.write("### Cluster Descriptions")
#     st.dataframe(clustered_data[['Cluster', 'Engagement Metric 1', 'Engagement Metric 2']])
    
#     # Plot engagement vs experience for each cluster
#     st.write("### Engagement vs Experience by Cluster")
#     fig, ax = plt.subplots()
#     sns.scatterplot(x='Engagement Score', y='Experience Score', hue='Cluster', data=clustered_data, ax=ax)
#     st.pyplot(fig)

# elif option == "Experience Analysis":
#     st.title("Experience Analysis")
    
#     # Assuming you're analyzing network metrics like RTT and retransmission
#     st.write("## Experience Metrics per Cluster")
#     experience_data = compute_experience_metrics(data)  # Replace with your experience metrics function
    
#     st.write("### RTT and Retransmission per Cluster")
#     fig, ax = plt.subplots()
#     sns.barplot(x='Cluster', y='Avg RTT DL (ms)', data=experience_data, ax=ax, label="RTT DL")
#     sns.barplot(x='Cluster', y='Avg RTT UL (ms)', data=experience_data, ax=ax, label="RTT UL", color="salmon")
#     st.pyplot(fig)
    
#     # Display retransmission metrics
#     st.write("### TCP Retransmission Volumes by Cluster")
#     retransmission_data = experience_data.groupby('Cluster').mean()[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']]
#     st.bar_chart(retransmission_data)

# elif option == "Satisfaction Analysis":
#     st.title("Satisfaction Analysis")
    
#     # Assuming you have functions to compute engagement and experience scores
#     st.write("## Satisfaction Scores")
#     scores = compute_satisfaction_scores(data)  # Replace with the function that calculates satisfaction
    
#     st.write("### Top 10 Satisfied Customers")
#     top_customers = scores[['MSISDN/Number', 'Satisfaction Score']].nlargest(10, 'Satisfaction Score')
#     st.dataframe(top_customers)
    
#     # Build and evaluate your regression model
#     st.write("## Satisfaction Prediction Model")
#     model, mse = build_satisfaction_regression_model(data)  # Replace with your regression model function
#     st.write(f"Mean Squared Error of the model: {mse}")
    
#     # Optionally visualize the regression results
#     st.write("### Regression Plot")
#     fig, ax = plt.subplots()
#     sns.regplot(x='Actual Satisfaction', y='Predicted Satisfaction', data=data, ax=ax)
#     st.pyplot(fig)
