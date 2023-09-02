#install.packages("biclustpl")
install.packages("clusterSim")
install.packages("fpc")
install.packages("cluster")
install.packages("MASS")
install.packages("flexclust")
install.packages("mclust")
install.packages("writexl")
library(writexl)

ls(pos = "package:biclustpl")

# Load the necessary packages
library(biclustpl)

# Your matrix data (replace this with your actual data)
library(R.matlab)
# Set the file path
file_name <- "D:/My papers/Application/ELBMcoclust/cstr.mat"

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
  n_row_clusters <- 4
  n_col_clusters <- 4
  
  # Run the biclust_dense function
  result <- biclust_dense(data_matrix, row_clusters = n_row_clusters, col_clusters = n_col_clusters,
                          family = "poisson", nstart = 1, loglik_seq = FALSE,
                          epsilon = 1e-5, maxit = 100L, trace = FALSE)
  
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
write_xlsx(results_df, "PLB_CSTR_R1.xlsx")

