# GNN-CL: GNN Fraud Detection System

A production-ready Graph Neural Network (GNN) based fraud detection system with interactive visualization, real-time monitoring, and analytics.

-----

## Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [Architecture](https://www.google.com/search?q=%23architecture)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Quick Start](https://www.google.com/search?q=%23quick-start)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Model Details](https://www.google.com/search?q=%23model-details)
  - [API Reference](https://www.google.com/search?q=%23api-reference)
  - [License](https://www.google.com/search?q=%23license)

-----

## Features

  * **Heterogeneous GNN Model**: R-GCN hybrid architecture for user-merchant transaction graphs.
  * **Real-time Detection**: Edge-level fraud prediction.
  * **Interactive Visualization**: D3.js force-directed graph with filtering capabilities.
  * **Analytics**: Network metrics calculation and community detection.
  * **Scalability**: Architecture supports caching and batch processing.
  * **Explainable AI**: Attention-based edge scoring and feature importance.

-----

## Architecture

```text
┌───────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│   Flask Web Server    │      │      Data Layer      │      │    GNN Model Layer    │
│                       │      │                      │      │                       │
│  • REST API           │ ───▶ │  • nodes.json (Cache)│ ───▶ │  • NodeEncoder        │
│  • Templates (HTML)   │      │  • edges.json (Cache)│      │  • RelationSummarizer │
│  • Static (JS/CSS)    │      │  • transactions (CSV)│      │  • EdgeClassifier     │
└───────────────────────┘      └──────────────────────┘      └───────────────────────┘
```

-----

## Installation

### Prerequisites

  * Python 3.8+
  * CUDA-capable GPU (optional)
  * 8GB+ RAM

### Setup

1.  **Clone and Environment**

    ```bash
    git clone https://github.com/yourusername/gnn-fraud-detection.git
    cd gnn-fraud-detection
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies**

    ```bash
    # PyTorch with CUDA 11.8 support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

    # Application requirements
    pip install flask pandas numpy scikit-learn geopy tqdm
    ```

-----

## Quick Start

1.  **Generate Data**
    Creates 500k synthetic transactions with fraud patterns.

    ```bash
    python data_gen.py
    ```

2.  **Preprocess**
    Converts CSV to PyTorch Geometric HeteroData.

    ```bash
    python preprocess_pyg.py
    ```

3.  **Train Model**
    Trains the R-GCN model (default: 100 epochs).

    ```bash
    python train_pyg.py
    ```

4.  **Cache Data**
    Generates inference results for visualization.

    ```bash
    python generate_cache.py
    ```

5.  **Run Application**

    ```bash
    python app.py
    ```

    Access the dashboard at `http://127.0.0.1:5000`.

-----

## Project Structure

```text
gnn-fraud-detection/
├── data/
│   ├── raw/                  # Generated CSV transactions
│   └── processed/            # PyG HeteroData and JSON caches
├── models/
│   ├── fraud_gnn_pyg.py      # GNN architecture definition
│   ├── predict_pyg.py        # Inference logic
│   └── dataset_pyg.py        # Data loading utilities
├── static/                   # JS (D3.js) and CSS
├── templates/                # HTML templates
├── data_gen.py               # Synthetic data generator
├── train_pyg.py              # Training script
├── app.py                    # Flask entry point
└── requirements.txt          # Dependencies
```

-----

## Model Details

### Architecture: FraudGNNHybrid

The model utilizes a Relational Graph Convolutional Network (R-GCN) designed for heterogeneous graphs.

1.  **Node Encoders**: Linear projections with ReLU activation and Dropout ($p=0.3$).
2.  **Relationship Summarizer**: Aggregates multi-relation neighborhoods using SAGEConv layers.
3.  **Core Node Intensifier**: Attention mechanism to weigh node importance.
4.  **Edge Classifier**: MLP classifier taking concatenated node embeddings and temporal edge encodings.

### Training Parameters

  * **Optimizer**: Adam
  * **Loss Function**: Focal Loss ($\alpha=0.25$, $\gamma=2.0$)
  * **Batch Size**: 16,384 edges
  * **Hidden Dimensions**: 128

-----

## API Reference

**Base URL**: `http://localhost:5000`

### Endpoints

  * `GET /api/graph`

      * **Params**: `nodes` (int), `edges` (int)
      * **Returns**: Sampled graph structure with risk scores.

  * `GET /api/node/<node_id>`

      * **Returns**: Detailed analytics and fraud patterns for a specific node.

  * `GET /api/metrics`

      * **Returns**: System-wide performance metrics and graph statistics.

  * `GET /api/search`

      * **Params**: `q` (string)
      * **Returns**: List of nodes matching the query.

-----

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
