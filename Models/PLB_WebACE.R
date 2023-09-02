#install.packages("biclustpl")
install.packages("clusterSim")
install.packages("fpc")
install.packages("cluster")
install.packages("MASS")
install.packages("flexclust")
install.packages("mclust")
# Load the 'mclust' package if it's not already loaded
if (!requireNamespace("mclust", quietly = TRUE)) {
  install.packages("mclust")
  library(mclust)
}



library("mclust")
ls(pos = "package:biclustpl")
# Load the necessary packages
library(biclustpl)
# Your matrix data (replace this with your actual data)
library(R.matlab)


# Set the file path
file_name <- "D:/My papers/Application/ELBMcoclust/WebACE.mat"


# Read the .mat file
mydata <- readMat(file_name)


true_labels <- mydata[["gnd"]]
mydata[["fea"]]
data_matrix <- as.matrix(mydata[["fea"]])  # Replace ... with your data values
#print(data_matrix)


# Confusion matrix
confusion_matrix <- table(true_labels, true_labels)
#print(confusion_matrix)



#true_labels <- mydata[["gnd"]]
#mydata[["fea"]]
#data_matrix <- as.matrix(mydata[["fea"]])  # Replace ... with your data values
#print(data_matrix)
# Specify the desired number of row and column clusters


n_row_clusters <- 20
n_col_clusters <- 5

# Run the biclust_dense function
result <- biclust_dense(data_matrix, row_clusters = n_row_clusters, col_clusters = n_col_clusters,  family = "poisson",
                        nstart = 1, loglik_seq = FALSE,
                        epsilon = 1e-6, maxit = 10000L, trace = FALSE)

# Extract row and column cluster assignments
row_cluster_labels <- result$row_clusters
col_cluster_labels <- result$col_clusters

sorted_row_cluster_labels <- sort(row_cluster_labels)
sorted_col_cluster_labels <- sort(col_cluster_labels)

# Print the extracted cluster labels
#cat("Row Cluster Labels:", row_cluster_labels, "\n")
#cat("Column Cluster Labels:", col_cluster_labels, "\n")


# Calculate accuracy
correct_assignments <- sum(true_labels == sorted_row_cluster_labels)
total_assignments <- length(true_labels)
accuracy_score <- correct_assignments / total_assignments

# Print the calculated accuracy
cat("Accuracy:", accuracy_score, "\n")


# Calculate Adjusted Rand Index
adjusted_rand_index <- adjustedRandIndex(true_labels, sorted_row_cluster_labels)

# Print the calculated Adjusted Rand Index
cat("Adjusted Rand Index:", adjusted_rand_index, "\n")
