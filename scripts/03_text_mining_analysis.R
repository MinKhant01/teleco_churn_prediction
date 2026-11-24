# ==============================================================================
# Script: 03_text_mining_analysis.R
# Purpose: Apply Zhao (2012) Chapter 10 text mining techniques to the
#          IBM Telco Customer Churn dataset (Churn Reason text field).
# ==============================================================================

# Load Libraries
library(tidyverse)
library(janitor)
library(here)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(cluster)

set.seed(42)

# Set up and ensure output directories
dir.create(here("outputs", "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("outputs", "tables"), recursive = TRUE, showWarnings = FALSE)

# Load & Prepare Text Data
raw_telco <- read_csv("data/raw/Telco_customer_churn.csv", show_col_types = FALSE) |>
  clean_names()

# Filter to churned customers with a recorded churn reason
churn_text_df <- raw_telco |>
  filter(!is.na(churn_reason), str_trim(churn_reason) != "") |>
  select(customer_id, churn_reason)

cat("Total churn reason documents: ", nrow(churn_text_df), "\n")

# Build a Volatile Corpus
corpus_raw <- VCorpus(VectorSource(churn_text_df$churn_reason))

# Text Transformation
corpus_clean <- corpus_raw |>
  tm_map(content_transformer(tolower)) |>
  tm_map(removePunctuation) |>
  tm_map(removeNumbers) |>
  tm_map(removeWords, stopwords("english")) |>
  tm_map(stripWhitespace)

# Keep a pre-stemmed copy for reference
corpus_pre_stem <- corpus_clean

# Stemming Words
corpus_stemmed <- tm_map(corpus_clean, stemDocument)

# Build Term-Document Matrix
tdm <- TermDocumentMatrix(corpus_stemmed, control = list(wordLengths = c(2, Inf)))
inspect(tdm[1:10, 1:10])

tdm_matrix <- as.matrix(tdm)
term_freq <- sort(rowSums(tdm_matrix), decreasing = TRUE)

term_freq_df <- tibble(
  term = names(term_freq),
  frequency = as.integer(term_freq)
)

# Export frequent terms table
write_csv(term_freq_df, here("outputs", "tables", "02_tm_frequent_terms.csv"))

# Frequent Terms & Associations
freq_terms_50 <- findFreqTerms(tdm, lowfreq = 50)
print(freq_terms_50)

# plot top 15 frequent terms
p_freq <- term_freq_df |>
  slice_head(n = 15) |>
  ggplot(aes(x = reorder(term, frequency), y = frequency)) +
  geom_col(fill = "#2c7fb8") +
  geom_text(aes(label = frequency), hjust = -0.15, size = 3.5) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Top 15 Most Frequent Terms in Customer Churn Reasons",
    subtitle = "Stemmed terms from IBM Telco Customer Churn Dataset (n = 1,869)",
    x = "Stemmed Term",
    y = "Document Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = here("outputs", "figures", "05_tm_frequent_terms.png"),
  plot = p_freq,
  width = 8,
  height = 5,
  dpi = 300
)

# Term associations
target_terms <- intersect(c("competitor", "support", "price", "servic"), rownames(tdm_matrix))
assoc_list <- findAssocs(tdm, terms = target_terms, corlimit = 0.20)

assoc_df <- imap_dfr(assoc_list, \(vals, term_name) {
  if (length(vals) == 0) return(tibble())
  tibble(
    target_term = term_name,
    associated_term = names(vals),
    correlation = as.numeric(vals)
  )
})

write_csv(assoc_df, here("outputs", "tables", "03_tm_term_associations.csv"))

# Word Cloud
png("outputs/figures/06_tm_wordcloud.png", width = 1800, height = 1400, res = 300)
wordcloud(
  words = term_freq_df$term,
  freq = term_freq_df$frequency,
  min.freq = 20,
  max.words = 100,
  random.order = FALSE,
  rot.per = 0.20,
  colors = brewer.pal(8, "Dark2")
)
dev.off()

# Clustering Words via Hierarchical Clustering
tdm_sparse <- removeSparseTerms(tdm, sparse = 0.96)
m_words <- as.matrix(tdm_sparse)

dist_matrix <- dist(scale(m_words), method = "euclidean")
hc_words <- hclust(dist_matrix, method = "ward.D2")

png("outputs/figures/07_tm_word_dendrogram.png", width = 2000, height = 1200, res = 300)
plot(
  hc_words,
  main = "Hierarchical Clustering of Frequent Churn Reason Terms",
  xlab = "Stemmed Terms",
  sub = "Ward.DS Linkage on Standardized Term-Document Matrix"
)
rect.hclust(hc_words, k = 4, border = "steelblue")
dev.off()

# Clustering Documents (Churn Reasons) with k-Means
dtm_matrix <- t(as.matrix(tdm_sparse))

non_empty_rows <- rowSums(dtm_matrix) > 0
dtm_filtered <- dtm_matrix[non_empty_rows, ]

k_clusters <- 4
km_docs <- kmeans(dtm_filtered, centers = k_clusters, nstart = 25, iter.max = 100)

cluster_summary <- map_dfr(1:k_clusters, \(cl) {
  cluster_docs <- dtm_filtered[km_docs$cluster == cl, , drop = FALSE]
  top_terms <- sort(colMeans(cluster_docs), decreasing = TRUE)[1:5]
  tibble(
    cluster = cl,
    doc_count = nrow(cluster_docs),
    pct_of_clustered = round(100 * nrow(cluster_docs) / nrow(dtm_filtered), 1),
    top_terms = paste(names(top_terms), collapse = ", ")
  )
})

print(cluster_summary)
write_csv(cluster_summary, "outputs/tables/04_tm_doc_clusters.csv")

cat("\nText mining pipeline completed successfully.\n")


