#install.packages("biclustpl")
install.packages("clusterSim")
install.packages("fpc")
install.packages("cluster")
install.packages("MASS")
install.packages("flexclust")
install.packages("mclust")
install.packages("writexl")

ls(pos = "package:biclustpl")

# Load the necessary packages
library(biclustpl)

# Your matrix data (replace this with your actual data)
library(R.matlab)
# Set the file path



# Load necessary libraries
library(biclustpl)
library(R.matlab)
library(clusterSim)
library(fpc)
library(cluster)
library(MASS)
library(flexclust)
library(mclust)
# Load necessary libraries
library(Matrix)
library(writexl)

# Set the file path
file_name <- "D:/My papers/Application/ELBMcoclust/Classic3.mat"

# Read the .mat file
mydata <- readMat(file_name)

# Data matrix
X_Classic3 <- as.matrix(mydata$A)
#print(X_Classic3)
#X_Classic3_sum_1 <- X_Classic3 / sum(X_Classic3)

#E_m1 <- matrix(1, nrow = 4303, ncol = 1)
#X_Classic3_normalization <- X_Classic3 %*% diag(sqrt(colSums(X_Classic3 %*% t(X_Classic3) %*% E_m1)^-0.5))
#print(X_Classic3_normalization)
#print(colSums(X_Classic3)[, drop = FALSE])
#freq <- colSums(X_Classic3)

# True labels list [0,0,0,..,1,1,1,..,2,2,2] n_row_cluster = 3
true_labels <- as.vector(mydata$labels)
true_labels <- true_labels + 1

# Extract data matrix and true labels
#data_matrix <- as.matrix(mydata[["fea"]])
#true_labels <- mydata[["gnd"]]







# Number of iterations
num_iterations <- 100

# Initialize vectors to store accuracy and ARI values
accuracy_values <- numeric(num_iterations)
ari_values <- numeric(num_iterations)

# Loop over iterations
for (i in 1:num_iterations) {
  # Specify the desired number of row and column clusters
  n_row_clusters <- 3
  n_col_clusters <- 3
  
  # Run the biclust_dense function
  result <- biclust_dense(X_Classic3, row_clusters = n_row_clusters, col_clusters = n_col_clusters,
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
#cat("Accuracy values:", accuracy_values, "\n")
#cat("ARI values:", ari_values, "\n")



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
write_xlsx(results_df, "PLB_Classic3_R1.xlsx")

