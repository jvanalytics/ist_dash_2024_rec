# notas para aulas de non supervised

# aula 1


## analise de correlaçao

![alt text](aula_1_correlacao.jpg)

- anova, chi quadrado
- dataset com carateristicas demograficas e muitas carateristicas. fazem se testes para avaliar redundancia e avaliaçao de poder preditivo sobre as mesmas variaveis
- variaveis entrada vs variaveis de saida


## exploracao: input vs output

![alt text](aula_1_input_output.jpg)

- dados tabulares: series, imagem, texto, eventos, relacional
- exemplo imagem. input: imagem. cada pixel será uma variavel. output: label (por exemplo: é um cão?)


## data Exploration and preprocessing
![alt text](aula_1_data_exploration.jpg)

- exercicio: https://web.ist.utl.pt/rmch/dash/guides/DataExploration.html
- usou-se dataset 'virus' em vez de 'iris'
- referencia a metodos de pre processamento (MVs, outliers, scaling, balancing, discretizaçao, enconding)

## clustering part 1
https://e.tecnicomais.pt/pluginfile.php/350464/mod_resource/content/4/03a%20Clustering%20Part1.pdf

![alt text](aula_1_clustering.jpg)

### 1. Descriçao com explicabilidade dos preditores, clustering
### 2. aplicaçao (cenarios, educaçao, ecommerce (catalogo, comportamental))

### 3. clustering 
- simples vs hierarquicas
- exclusivas vs nao exclusivas
- suaves vs estritas (hard)


# aula 2

## clustering part 2
https://e.tecnicomais.pt/pluginfile.php/350467/mod_resource/content/3/03b%20Clustering%20Part2.pdf

![alt text](aula_2_clustering.jpg)
![alt text](aula_2_clustering_categorical.jpg)

- observaçao de dados em matriz, exploraçao, amostragem e eficiencia. Referencia a distancia das observaçoes (ex: knn)


### Clustering features:
#### 1. distance
 ![alt text](aula_2_clustering_distance.jpg)
- numeric, nominal, ordinal, non iid (multivariate, time series, image, geo, events)
    - nominal usa a distancia Hamming
![alt text](aula_2_clustering_distance_hamming.jpg)
    - distancia euclidiana, manhattan, chebyshev, coseno
        - tbm se pode calcular correlaçao pearson, spearman
    - escolha de semelhança vs distancia depende do conhecimento de dominio

### 2. approach (abordagens)
![alt text](aula_2_clustering_approach.jpg)

![alt text](aula_2_clustering_approach_2.jpg)

    - atençao a questao de dummyfication. é so para itens com cardinalidade maior que 2 sem ordem.
    - partitioning, hierarchical, density-based, model-based

### Density Based
Density-based clustering is a method that identifies clusters in data by looking for regions of high density separated by regions of low density. One of the most popular algorithms for density-based clustering is DBSCAN (Density-Based Spatial Clustering of Applications with Noise). DBSCAN groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It is particularly effective for discovering clusters of arbitrary shape and for handling noise in the data. For more details, refer to the [DBSCAN documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.DBSCAN.html).

### partitioning clustering
Partitioning clustering is a method that divides the data into distinct clusters, where each data point belongs to exactly one cluster. The goal is to optimize the partitioning by minimizing the within-cluster variance. One of the most common algorithms for partitioning clustering is K-means, which iteratively assigns data points to clusters based on the nearest mean value. Another popular algorithm is K-medoids, which is more robust to noise and outliers. For more details, refer to the [K-means documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

### hierarchical clustering
![alt text](aula_2_clustering_hierarchical.jpg)
Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. There are two main types of hierarchical clustering algorithms: agglomerative and divisive. Agglomerative clustering is a "bottom-up" approach where each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. Divisive clustering is a "top-down" approach where all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy. This method is particularly useful for data that has a nested structure. For more details, refer to the [Hierarchical Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering).


### model-based clustering
Model-based clustering is a method that assumes the data is generated by a mixture of underlying probability distributions, each representing a different cluster. This approach uses statistical models to estimate the parameters of these distributions and assign data points to clusters based on the likelihood of belonging to each distribution. One of the most common algorithms for model-based clustering is the Gaussian Mixture Model (GMM), which assumes that the data is generated from a mixture of Gaussian distributions. For more details, refer to the [Gaussian Mixture Model documentation](https://scikit-learn.org/stable/modules/mixture.html).


### Exercicio Clustering

- notebook https://web.ist.utl.pt/rmch/dash/guides/Clustering%20in%20Python.html
- exercicio pratico https://web.ist.utl.pt/rmch/dash/exercises/01%20Clustering.pdf
- soluçao exercicio https://web.ist.utl.pt/rmch/dash/exercises/01%20Clustering%20Solutions.pdf 

##### para agglomerative clustering pode-se usar sklearn ou formula especifica 
```python
from sklearn.metrics import pairwise_distances

def mydistance(x1, x2):
    res = 0.0001
    for j, weight in enumerate([1,2,3,1]):
        res += weight*abs(x1[j]-x2[j])
    return res

def sim_affinity(X):
    return pairwise_distances(X, metric=mydistance)
```

### Avaliaçao

#### Cohesion vs Separation

In clustering, cohesion and separation are two important measures used to evaluate the quality of the clusters formed.

**Cohesion** (also known as intra-cluster distance) measures how closely related the items in a cluster are. It is typically quantified by the Sum of Squared Errors (SSE), which calculates the total squared distance between each data point and the centroid of its assigned cluster. A lower cohesion value indicates that the data points within a cluster are closer to each other, suggesting a more compact and well-defined cluster.

**Separation** (also known as inter-cluster distance) measures how distinct or well-separated a cluster is from other clusters. It is often quantified by metrics such as the Silhouette Score, which considers both the cohesion within clusters and the separation between clusters. A higher separation value indicates that the clusters are more distinct from each other, suggesting better-defined boundaries between clusters.

For more details, refer to the [Silhouette Score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

```python
from sklearn.metrics import silhouette_score

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Calculate Silhouette Score
sil_score = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
```

#### SSE (Cohesion)
![alt text](aula_2_clustering_SSE_evaluation.jpg)
Sum of Squared Errors (SSE) is a metric used to evaluate the performance of clustering algorithms. It measures the total squared distance between each data point and the centroid of its assigned cluster. A lower SSE indicates that the data points are closer to their respective centroids, suggesting better clustering performance. For more details, refer to the [SSE documentation](https://en.wikipedia.org/wiki/Residual_sum_of_squares).

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Calculate SSE
sse = kmeans.inertia_
print(f'Sum of Squared Errors (SSE): {sse}')
```