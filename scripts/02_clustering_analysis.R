# ==============================================================================
# Script:  02_clustering_analysis.R
# Purpose: Perform K-Means clustering on the IBM Telco Customer Churn dataset
#          to identify actionable customer segments for the Churn DSS.
# Input:   data/raw/Telco_customer_churn.csv
# Outputs: outputs/figures/03_kmeans_optimal_k.png
#          outputs/figures/04_kmeans_clusters.png
#          outputs/tables/01_cluster_profiles.csv
# ==============================================================================

# load packages
library(tidyverse)
library(here)
library(janitor)
library(cluster)
library(factoextra)
library(patchwork)

# set global seed for reproducibility
set.seed(42)

# Ensure output directories exist
dir.create(here("outputs","figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("outputs","tables"), recursive = TRUE, showWarnings = FALSE)

# load and clean data
raw_data_path <- here("data","raw","Telco_customer_churn.csv")

telco_clean <- read_csv(raw_data_path, show_col_types = FALSE) |>
  clean_names() |>
  mutate(
    total_charges = replace_na(as.numeric(total_charges), 0),
    churn_flag = if_else(churn_label == "Yes", 1L, 0L)
  )


# Feature Selection & Scaling
cluster_features <- telco_clean |>
  select(tenure_months, monthly_charges, total_charges, cltv)

# standardize features (mean = 0, sd = 1) so Euclidean distance is unbiased
scaled_matrix <- scale(cluster_features)

# Determine optimal number of clusters (k) via Elbow (WSS) and Avg Silhouette Width across k = 2..8
p_elbow <- fviz_nbclust(scaled_matrix, kmeans, method = "wss", k.max = 8, nstart = 25) +
  labs(title = "Elbow Method (Within-Cluster SS)", x = "Number of Clusters (k)") + 
  theme_minimal()

p_sil <- fviz_nbclust(scaled_matrix, kmeans, method = "silhouette", k.max = 8, nstart = 25) +
  labs(title = "Average Silhouette Method", x = "Number of Clusters (k)") +
  theme_minimal()

p_optimal_k <- p_elbow + p_sil

ggsave(
  filename = here("outputs", "figures", "03_kmeans_optima_k.png"),
  plot = p_optimal_k,
  width = 11,
  height = 5,
  dpi = 300
)


# Fit Final K-Means MOdel
OPTIMAL_K <- 4
kmeans_model <- kmeans(
  x = scaled_matrix,
  centers = OPTIMAL_K,
  nstart = 25,
  iter.max = 100
)

p_clusters <- fviz_cluster(
  object = kmeans_model,
  data = scaled_matrix,
  geom = "point",
  ellipse.type = "convex",
  pointsize = 1.2,
  alpha = 0.4,
  ggtheme = theme_minimal()
) +
  labs(
    title = paste0("K-Means Customer Segments (k = ", OPTIMAL_K, ")"),
    subtitle = "Projected onto the first two Principal Components"
  )

ggsave(
  filename = here("outputs", "figures", "04_kmeans_clusters.png"),
  plot = p_clusters,
  width = 8,
  height = 6,
  dpi = 300
)

# Profile Clusters Against Business & Churn Metrics
telco_clustered <- telco_clean |>
  mutate(cluster = factor(kmeans_model$cluster))

cluster_profiles <- telco_clustered |>
  group_by(cluster) |>
  summarise(
    customer_count = n(),
    pct_of_customers = round(n() / nrow(telco_clustered) * 100, 1),
    avg_tenure_months = round(mean(tenure_months), 1),
    avg_monthly_charge = round(mean(monthly_charges), 2),
    avg_total_charge = round(mean(total_charges), 2),
    avg_cltv = round(mean(cltv), 0),
    churn_rate_pct = round(mean(churn_flag) * 100, 1),
    pct_month_to_month = round(mean(contract == "Month-to-month") * 100, 1),
    .groups = "drop"
  ) |>
  arrange(desc(churn_rate_pct))

print(cluster_profiles)

write_csv(
  cluster_profiles,
  here("outputs", "tables", "01_cluster_profiles.csv")
)
