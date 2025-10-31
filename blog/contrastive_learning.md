---
layout: post.html
title: Contrastive Learning
tags: post
date: 2025-09-19T16:11
---
In order to understand more about the task for [Clustering BioTrove Data Competition by Kaggle](https://www.kaggle.com/competitions/biotrove-clustering), the set goals must be explored in order to work on a proper solution.

The goals are stated as below:
1. Learn image embeddings using contrastive learning with family priors, autoencoders, or other deep representation methods.
2. Cluster images using algorithms such as k-means, hierarchical clustering, DBSCAN, or graph-based methods.
3. Submit cluster assignments that will be compared against hidden genus- and species-level labels using **Normalized Mutual Information (NMI)**.

In this document, the technique called “contrastive learning” is explored in several categories.

# 01.What is Contrastive Learning?

Contrastive learning trains models to map similar inputs (positive pairs) close together in an embedding space while pushing dissimilar inputs (negative pairs) far apart. This yields powerful representations for downstream tasks such as classification, retrieval, and clustering.


Contrastive learning is a **self-supervised** approach in which the main objective is to learn an embedding function.

$$
f_\theta : \mathcal{X} \rightarrow \mathbb{R}^{d}
$$

that captures semantic similarity without requiring manual labels. By constructing pairs of inputs and defining a contrastive loss, the model discovers structure from data alone.

# 02:Foundations

## 02.1:Embedding Space and Similarity
- Embedding function maps the input $ x$ to a dimensional vector.

$$
f_\theta(x) : \mathcal{X} \rightarrow \mathbb{R}^{d}
$$

- Similarity measure is used to quantify the similarity between embedding pairs.
$$
sim(\bar{u}^{}, \bar{v}^{}) = \frac{\bar{u}^\top \bar{v}^{}}{\| \bar{u}^{} \|  \| \bar{v}^{} \|}
$$

## 02.2:Data Augmentation
Data Augmentation is done for each sample $ x$. From a single sample, two correlated views are generated ($ x_i$ and $ x_j$) using list of random transformations (e.g. `transforms.HorizontalFlip`, `transforms.RandomResizedCrop` etc.)

$$
\tilde{x}_i = t(x_k) \rightarrow t \sim \mathcal{T}
$$

$$
\tilde{x}_j = t'(x_k) \rightarrow t' \sim \mathcal{T}
$$

where:
- $x_k$: Sampled minibatch $ \{ x_k \}_{k=1}^{N}$
- $t$ and $ t'$: Transformation functions (derived from the same $ \mathcal{T}$ transformation set)
- $\tilde{x}_i$ and $ \tilde{x}_j$: Two views of $ x_k$, described as **positive pairs**

These generated views form **positive pair** and the rest of the data in the batch are considered as **negatives.**

## 02.3:Loss
Given a batch of $ N$ samples, with each with two views ($ \tilde{x}_i$ and $ \tilde{x}_j$), $ 2N$ embeddings will be in the end.

For a positive pair of $ (i,\ j)$, `NT-Xent` loss can be defined as:

$$
\mathscr{l}\_{i, j} = -\log{ \frac{ \frac{\exp(\text{sim}(z\_i, z\_j))} {\tau} } { \sum\_{k \ = \ 1}^{2N} \mathbb{1}\_{ \big[ k \ \neq \ i \big] } \frac{\exp(\text{sim}(z\_i, z\_k))} {\tau}} }
$$

where:
- $z_i = f_\theta(x_i)$: Normalized embedding
- $\tau > 0$: **Temperature**
- $\mathbb{1}_{ \lceil k \neq i \rceil }$: Indicator Function

$$
\mathbb{1}_{ \big[ k \ \neq \ i \big] } = \begin{cases} 1, & \text{if $k \neq i$} \\\ 0, & \text{if $k = i$} \end{cases}
$$

The total loss over the batches averages both directions:

$$
\mathscr{L} = \frac{1} {2N} \sum\_{i = 1}^{N} \big( \mathscr{l}\_{i,\ j(i)} + \mathscr{l}\_{j(i),\ i} \big)
$$

- Temperature $\tau$ controls the **sharpness of the distribution**.
	- Smaller $\tau$ emphasizes **hard negatives.**
    
- Embeddings $z$ are **typically unit-norm** to ensure cosine similarity equivalence to dot product.
