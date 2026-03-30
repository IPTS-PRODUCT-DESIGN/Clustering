# Recursive Semantic Decomposition for Requirements Clustering

A recursive, tree-structured clustering pipeline for software and systems requirements. The pipeline embeds requirements with a sentence transformer (BGE), reduces dimensionality with UMAP, clusters with HDBSCAN, and recursively decomposes large clusters until a minimum size threshold is reached. Cluster labels are generated via c-TF-IDF (statistical) or optionally via an LLM (Claude). Clustering quality is validated at every level with DBCV.


## Overview

The core idea is hierarchical topic decomposition: rather than producing a single flat partition, the pipeline builds a **tree of requirement groups** where each node is a semantically coherent cluster. Large clusters are recursively split into finer sub-topics until the groups are small enough to be meaningful, while noise points are reassigned to their nearest cluster at each level.

The repository also includes an evaluation module with keyword-based and LLM-based ground truth generation, plus standard clustering metrics (ARI, NMI, V-measure, Fowlkes-Mallows, homogeneity, completeness) and a confusion matrix visualization.


## Pipeline

```
Raw requirements (JSON list of strings)
  BGE embedding  (BAAI/bge-large-en-v1.5)
  UMAP reduction  (n_neighbors, n_components, min_dist)
  HDBSCAN clustering  (min_cluster_size, leaf selection)
       noise points, reassigned to nearest cluster (cosine similarity)
  c-TF-IDF or LLM labeling per cluster
  Recurse into clusters ≥ min_size_for_recursion
  Output: TreeNode hierarchy + flat DataFrame export
```


## Requirements

Python 3.9+
An Anthropic API key (only if `use_llm_labels=True`)


### Python packages

```
sentence-transformers
umap-learn
hdbscan
scikit-learn
pandas
numpy
matplotlib
anthropic
graphviz        
```

You also need the `graphviz` system package:

```bash
# Ubuntu / Debian
sudo apt-get install -y graphviz
# conda
conda install -y graphviz
```


### 1. Install dependencies
```bash
pip install sentence-transformers umap-learn hdbscan scikit-learn pandas numpy matplotlib anthropic graphviz
```


### 2. Prepare your data

Create a JSON file containing a flat list of requirement strings:
```json
[
  "The system shall provide real-time telemetry updates.",
  "Users shall be able to register new UAVs via the dashboard.",
  "The middleware shall forward MAVLink messages to the GCS."
]
```


### 3. Run the pipeline

```python
from code_cluster import RecursiveSemanticDecomposition, load_requirements_json
texts = load_requirements_json("path/to/requirements.json")

pipeline = RecursiveSemanticDecomposition(
    embedding_model        = "BAAI/bge-large-en-v1.5",
    min_cluster_size       = 4,
    min_size_for_recursion = 8,
    umap_n_neighbors       = 10,
    umap_n_components      = 5,
    umap_min_dist          = 0.0,
    cluster_selection_method = "leaf",
    ctfidf_top_n           = 5,
    random_state           = 42,
    use_llm_labels         = True,           
    anthropic_api_key      = "...",       
)
tree = pipeline.fit(texts, root_label="Product Requirements")
pipeline.print_tree()
```


### 4. Export results

```python
df = pipeline.to_dataframe()
df.to_excel("clustering_results.xlsx", index=False)
diag = pipeline.get_diagnostics()
print(diag)
```


### 5. Evaluate against ground truth

```python
from code_cluster import generate_ground_truth_llm, evaluate_clustering
y_true = generate_ground_truth_llm(texts, api_key="sk-...")
metrics = evaluate_clustering(df, y_true, save_prefix="my_eval")
```


## Configuration

| Parameter | Default | Description |
| `embedding_model` | `BAAI/bge-large-en-v1.5` | Sentence transformer model for encoding requirements |
| `min_cluster_size` | `4` | HDBSCAN minimum cluster size |
| `min_size_for_recursion` | `8` | Clusters smaller than this are kept as leaf nodes |
| `umap_n_neighbors` | `10` | UMAP locality parameter |
| `umap_n_components` | `5` | UMAP target dimensionality |
| `umap_min_dist` | `0.0` | UMAP minimum distance |
| `cluster_selection_method` | `leaf` | HDBSCAN cluster selection (`leaf` or `eom`) |
| `ctfidf_top_n` | `5` | Number of top c-TF-IDF terms per cluster label |
| `use_llm_labels` | `False` | Use Claude to generate human-readable cluster labels |
| `anthropic_api_key` | `None` | Anthropic API key (required when `use_llm_labels=True`) |


## Project structure

```
code_cluster.ipynb
README.md
requirements.txt
data/               
example.json
```


## Evaluation metrics

The evaluation module computes:
- Adjusted Rand Index (ARI) agreement between predicted and true partitions, corrected for chance
- Normalized Mutual Information (NMI) shared information between clusterings
- V-measure, Homogeneity, Completeness conditional entropy-based measures
- Fowlkes-Mallows Index geometric mean of precision and recall at pair level
- Confusion matrix with Hungarian algorithm alignment for best label matching

Ground truth can be generated via keyword heuristics (domain-specific regex patterns included for Dronology) or via LLM-based classification.


## Citation
If you use this pipeline or datasets in academic work, please cite accordingly.