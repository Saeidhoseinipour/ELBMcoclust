# Load necessary libraries
library(biclustpl)
library(R.matlab)
library(clusterSim)
library(fpc)
library(cluster)
library(MASS)
library(flexclust)
library(mclust)

# Set the file path
file_name <- "D:/My papers/Application/ELBMcoclust/WebACE.mat"

# Number of iterations
num_iterations <- 100

# Initialize vectors to store accuracy and ARI values
accuracy_values <- numeric(num_iterations)
ari_values <- numeric(num_iterations)

# Loop over iterations
for (i in 1:num_iterations) {
  # Read the .mat file
  mydata <- readMat(file_name)
  
  # Extract data matrix and true labels
  data_matrix <- as.matrix(mydata[["fea"]])
  true_labels <- mydata[["gnd"]]
  
  # Specify the desired number of row and column clusters
  n_row_clusters <- 20
  n_col_clusters <- 5
  
  # Run the biclust_dense function
  result <- biclust_dense(data_matrix, row_clusters = n_row_clusters, col_clusters = n_col_clusters,
                          family = "poisson", nstart = 1, loglik_seq = FALSE,
                          epsilon = 1e-6, maxit = 100L, trace = FALSE)
  
  # Extract row cluster assignments
  row_cluster_labels <- result$row_clusters
  sorted_row_cluster_labels <- sort(row_cluster_labels)
  
  # Calculate accuracy
  correct_assignments <- sum(true_labels == sorted_row_cluster_labels)
  total_assignments <- length(true_labels)
  accuracy_score <- correct_assignments / total_assignments
  
  # Calculate Adjusted Rand Index
  adjusted_rand_index <- adjustedRandIndex(true_labels, sorted_row_cluster_labels)
  
  # Store accuracy and ARI values
  accuracy_values[i] <- accuracy_score
  ari_values[i] <- adjusted_rand_index
}

# Print the accuracy and ARI values for each iteration
cat("Accuracy values:", accuracy_values, "\n")
cat("ARI values:", ari_values, "\n")

install.packages("writexl")
library(writexl)

# ... Your previous code ...

# Initialize vectors to store accuracy and ARI values
accuracy_values <- numeric(num_iterations)
ari_values <- numeric(num_iterations)

# Loop over iterations
for (i in 1:num_iterations) {
  # ... Your previous code ...
  
  # Store accuracy and ARI values
  accuracy_values[i] <- accuracy_score
  ari_values[i] <- adjusted_rand_index
}

# Create a data frame to store accuracy and ARI values
results_df <- data.frame(Accuracy = accuracy_values, ARI = ari_values)

# Write the data frame to an Excel file
write_xlsx(results_df, "biclustering_results.xlsx")
