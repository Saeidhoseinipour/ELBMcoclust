# **Classic3** 
 Classic3 dataset containing 3891 documents by merging the popular MEDLINE, CISI, and CRANFIELD sets. MEDLINE consists of 1033 abstracts from “Medical” papers, CISI consists of 1460 abstracts from “Information Retrieval” papers, and CRANFIELD consists of 1398 abstracts from “Aeronautical Systems” papers.


```python
from ELBMcoclust.Models.coclust_SELBMcem import CoclustSELBMcem

SELBM = CoclustSELBMcem(n_row_clusters = 3, n_col_clusters = 3, model = "Poisson")
SELBM.fit(X_Classic3)

freq_cluster = X_Classic3_sum_1.sum(axis = 0)                                     # For wordcloud plot
freq_block = SELBM.R.T@X_Classic3_sum_1                                           # n_row_cluster x number of column
freq_block_2 = SELBM.R.T@X_Classic3_sum_1@SELBM.C                                 # n_row_cluster x number of column

w = mydata_2['term_labels']                                                       # 4303  Words for WC plot 


dic_word_block = {'Words': w.tolist(),
                  'Freq_cluster': freq_cluster.tolist(),
                  'Freq_row_cluster_1': freq_block[0,:],
                  'Freq_row_cluster_2': freq_block[1,:],
                  'Freq_row_cluster_3': freq_block[2,:], 
                  'Column labels': np.sort(SELBM.column_labels_)}
print(dic_word_block, dic_word_block)

df_1 = df[df['Column labels'] == 1]
df_2 = df[df['Column labels'] == 2]
df_3 = df[df['Column labels'] == 3]

print(df_1['Freq_row_cluster_1'].sum(),df_2['Freq_row_cluster_1'].sum(),df_3['Freq_row_cluster_1'].sum())
print(df_1['Freq_row_cluster_2'].sum(),df_2['Freq_row_cluster_2'].sum(),df_3['Freq_row_cluster_2'].sum())
print(df_1['Freq_row_cluster_3'].sum(),df_2['Freq_row_cluster_3'].sum(),df_3['Freq_row_cluster_3'].sum())
```


|                  | Word Cluster 1 (low frequency)                 | Word Cluster 2 ( medium frequency)                 | Word Cluster 3 (hight frequency)                 |
|------------------|--------------------------|--------------------------|--------------------------|
| Document Cluster 1 (Medical)           | 0.06122926646589793      | 0.013789848175136924     | 0.19179006662817732      |
| Document Cluster 2  (Information Retrieval)          | 0.042118526378204624     | 0.013142290948242232     | 0.19034671618268917      |
| Document Cluster 3  (Aeronautical Systems)          | 0.13270632109476188      | 0.017987267308502503     | 0.33688969681838765      |



|                  | Word Cluster 1 (low frequency)                 | Word Cluster 2 ( medium frequency)                 | Word Cluster 3 (hight frequency)                 |
|------------------|--------------------------|--------------------------|--------------------------|
| Document Cluster 1 (Medical)           | 60249      | 3430     | 5384      |
| Document Cluster 2  (Information Retrieval)          | 52954     | 3534     | 5754      |
| Document Cluster 3  (Aeronautical Systems)          |  110727      | 4505     | 9811      |

**Normalized Version**:

|                     | Freq_cluster | Freqency Medical cluster | Freqency Information Retrieval cluster | Freqency Aeronautical Systems cluster | Column labels |
|---------------------|--------------|--------------------|--------------------|--------------------|---------------|
| count               | 4303 | 4303        | 4303        | 4303        | 4303   |
| mean                | 0.000232     | 0.000062           | 0.000057           | 0.000113           | 2.579131      |
| std                 | 0.000393     | 0.000121           | 0.000140           | 0.000297           | 0.791459      |
| min                 | 0.000031     | 0.000000           | 0.000000           | 0.000000           | 1.000000      |
| 25%                 | 0.000055     | 0.000008           | 0.000000           | 0.000004           | 3.000000      |
| 50%                 | 0.000101     | 0.000023           | 0.000020           | 0.000027           | 3.000000      |
| 75%                 | 0.000230     | 0.000066           | 0.000055           | 0.000090           | 3.000000      |
| max                 | 0.005715     | 0.002520           | 0.003636           | 0.005356           | 3.000000      |
|                     |              |                    |                    |                    |               |
| Words               | Freq_cluster | Freq_row_cluster_1 | Freq_row_cluster_2 | Freq_row_cluster_3 | Column labels |
| contribution        | 0.000144     | 0.000035           | 0.000059           | 0.000051           | 1             |
| catheterization     | 0.000074     | 0.000074           | 0.000000           | 0.000000           | 1             |
| distribution        | 0.002173     | 0.000191           | 0.000328           | 0.001654           | 1             |
| dr                  | 0.000133     | 0.000078           | 0.000055           | 0.000000           | 1             |
| cp                  | 0.000074     | 0.000004           | 0.000000           | 0.000070           | 1             |
| ...                 | ...          | ...                | ...                | ...                | ...           |
| equilibrium         | 0.000714     | 0.000051           | 0.000004           | 0.000659           | 3             |
| librarianship       | 0.000343     | 0.000012           | 0.000304           | 0.000027           | 3             |
| extremely           | 0.000133     | 0.000035           | 0.000027           | 0.000070           | 3             |
| disturbance         | 0.000222     | 0.000055           | 0.000000           | 0.000168           | 3             |
| ribonucleic         | 0.000039     | 0.000039           | 0.000000           | 0.000000           | 3             |


**Non-normalized version**:

|                   | Freq_cluster | Freq_row_cluster_1 | Freq_row_cluster_2 | Freq_row_cluster_3 | Column labels |
|-------------------|--------------|--------------------|--------------------|--------------------|---------------|
| count             | 4303.000000  | 4303.000000        | 4303.000000        | 4303.000000        | 4303.000000   |
| mean              | 59.574251    | 16.049965          | 14.464792          | 29.059493          | 1.253079      |
| std               | 100.797875   | 31.003533          | 35.402298          | 76.031959          | 0.622938      |
| min               | 8.000000     | 0.000000           | 0.000000           | 0.000000           | 1.000000      |
| 25%               | 14.000000    | 2.000000           | 0.000000           | 1.000000           | 1.000000      |
| 50%               | 26.000000    | 7.000000           | 5.000000           | 7.000000           | 1.000000      |
| 75%               | 59.000000    | 17.000000          | 13.000000          | 23.000000          | 1.000000      |
| max               | 1465.000000  | 646.000000         | 929.000000         | 1373.000000        | 3.000000      |
|			|	|			|			|			|		|
| **Words**             | **Freq_cluster** | **Freq_row_cluster_1** | **Freq_row_cluster_2** | **Freq_row_cluster_3** | **Column labels** |
| contribution      | 37.0         | 10.0               | 14.0               | 13.0               | 1             |
| catheterization   | 19.0         | 19.0               | 0.0                | 0.0                | 1             |
| distribution      | 557.0        | 55.0               | 78.0               | 424.0              | 1             |
| dr                | 34.0         | 20.0               | 14.0               | 0.0                | 1             |
| cp                | 19.0         | 1.0                | 0.0                | 18.0               | 1             |
| equilibrium       | 183.0        | 13.0               | 1.0                | 169.0              | 3             |
| librarianship     | 88.0         | 3.0                | 78.0               | 7.0                | 3             |
| extremely         | 34.0         | 9.0                | 7.0                | 18.0               | 3             |
| disturbance       | 57.0         | 14.0               | 0.0                | 43.0               | 3             |
| ribonucleic       | 10.0         | 10.0               | 0.0                | 0.0                | 3             |

```python
df_1 = df[df['Column labels'] == 1]

#print(dict(df_1))
df_1.sort_values(by="Freq_cluster", ascending=False).head(20)
```

|  index in data matrix  | Words        |   Freq_cluster |   Freq_row_cluster_1 |   Freq_row_cluster_2 |   Freq_row_cluster_3 |   Column labels |
|---:|:-------------|---------------:|---------------------:|---------------------:|---------------------:|---------------:|
|  2 | distribution |            557 |                   63 |                   90 |                  404 |              1  |
| 29 | ae           |            344 |                    0 |                    8 |                  336 |              1  |
| 74 | due          |            317 |                  119 |                   26 |                  172 |              1  |
| 26 | tn           |            251 |                    0 |                   11 |                  240 |              1  |
|111 | dna          |            218 |                  216 |                    2 |                    0 |              1  |
|109 | age          |            165 |                  132 |                   32 |                    1 |              1  |
|115 | day          |            136 |                  106 |                   25 |                    5 |              1  |
| 47 | ii           |            131 |                   57 |                   55 |                   19 |              1  |
|118 | description  |            126 |                   31 |                   66 |                   29 |              1  |
| 71 | man          |            115 |                   70 |                   45 |                    0 |              1  |
| 93 | end          |            115 |                   35 |                   41 |                   39 |              1  |
| 20 | mg           |             89 |                   89 |                    0 |                    0 |              1  |
|114 | app          |             89 |                    0 |                    9 |                   80 |              1  |
| 95 | rna          |             88 |                   88 |                    0 |                    0 |              1  |
| 19 | ca           |             86 |                   51 |                   35 |                    0 |              1  |
| 61 | arc          |             82 |                    1 |                    1 |                   80 |              1  |
| 77 | sdi          |             74 |                    2 |                   72 |                    0 |              1  |
| 85 | rae          |             72 |                    0 |                    0 |                   72 |              1  |
|  7 | hr           |             70 |                   70 |                    0 |                    0 |              1  |
| 84 | aid          |             68 |                   16 |                   34 |                   18 |              1  |



<img alt="Bar charts top 1000 words in classic3 dataset obtined by PoissonSELBM for co-clustering, Saeid Hoseinipour, Co-clustering, Text mining, Latent block model, word cloud" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/bar_chart_all_words_classic3_top_1000.svg">



<img alt="Word clouds top 60 words in classic3 dataset obtined by PoissonSELBM for co-clustering, Saeid Hoseinipour, Co-clustering, Text mining, Latent block model, word cloud" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/WC_classic3_three_color_3_3.svg">



# Refrences
[1] [Dhillon, I.S.et al., Information-theoretic co-clustering, Proceedings of the ninth ACM SIGKDD International
		Conference on Knowledge Discovery and Data Mining, 89-98, 2003](https://dl.acm.org/doi/abs/10.1145/2487575).
  
[2] [Saeid, Hoseinipour et al, Sparse Expoential Family Latent Block Model for Co-clustering (2023), Advances in Data Analysis and Classification (preprint).]()
