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





<img alt="Sample SVG Image" src="https://github.com/Saeidhoseinipour/ELBMcoclust/blob/main/Images/bar_chart_all_words_classic3_top_1000.svg">



# Refrences
[1] [Dhillon, I.S.et al., Information-theoretic co-clustering, Proceedings of the ninth ACM SIGKDD International
		Conference on Knowledge Discovery and Data Mining, 89-98, 2003](https://dl.acm.org/doi/abs/10.1145/2487575).
