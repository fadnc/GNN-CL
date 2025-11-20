import json
import os
import torch
from torch_geometric.data import HeteroData
from models.fraud_gnn_pyg import FraudGNNHybrid
from tqdm import tqdm
from collections import defaultdict

GRAPH_PATH = "data/processed/fraud_graph_pyg.pt"
MODEL_PATH = "best_fraudgnn.pth"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

# -------------------------------------------------------------------
# LOAD GRAPH
# -------------------------------------------------------------------
print("\nLoading graph...")
graph_ck = torch.load(GRAPH_PATH, map_location="cpu")
hetero = graph_ck.get("data", graph_ck)

user_x = hetero['user'].x
merch_x = hetero['merchant'].x
edge_index = hetero['user','transacts','merchant'].edge_index
edge_attr = hetero['user','transacts','merchant'].edge_attr

num_edges = edge_index.size(1)
num_users = user_x.size(0)
num_merchants = merch_x.size(0)

print(f"Users: {num_users}, Merchants: {num_merchants}, Edges: {num_edges}")

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
print("\nLoading model...")
model = FraudGNNHybrid(
    user_in_dim=user_x.size(1),
    merchant_in_dim=merch_x.size(1),
    edge_attr_dim=edge_attr.size(1),
    hidden_dim=128,
    relation_names=[('user','transacts','merchant'), ('merchant','receives','user')]
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------------------------------------------
# COMPUTE NODE EMBEDDINGS
# -------------------------------------------------------------------
print("\nComputing node embeddings...")
with torch.no_grad():
    hetero['user'].x = user_x.to(DEVICE)
    hetero['merchant'].x = merch_x.to(DEVICE)
    node_embs = model.forward_nodes(hetero)

# Initialize node risk accumulators
user_risks = defaultdict(list)
merchant_risks = defaultdict(list)

# -------------------------------------------------------------------
# COMPUTE EDGE PREDICTIONS (in batches)
# -------------------------------------------------------------------
edges = []
batch_size = 4096

print("\nComputing fraud predictions for edges...")
for start in tqdm(range(0, num_edges, batch_size)):
    end = min(start + batch_size, num_edges)

    src = edge_index[0, start:end]
    dst = edge_index[1, start:end]
    attr = edge_attr[start:end]

    src_h = node_embs["user"][src.to(DEVICE)]
    dst_h = node_embs["merchant"][dst.to(DEVICE)]
    attr = attr.to(DEVICE)

    with torch.no_grad():
        logits = model.edge_classifier(src_h, dst_h, attr)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    # Store edges and accumulate node risks
    for i, p in zip(range(start, end), probs):
        u = int(edge_index[0, i])
        m = int(edge_index[1, i])
        
        edge = {
            "edge_id": i,
            "source": f"user_{u}",
            "target": f"merch_{m}",
            "edge_attr": edge_attr[i].tolist(),
            "amount": float(edge_attr[i][0].item()),
            "pred_prob": float(p),
            "is_suspicious": p > 0.5
        }
        edges.append(edge)

        # Accumulate risks for averaging
        user_risks[u].append(p)
        merchant_risks[m].append(p)

# -------------------------------------------------------------------
# CALCULATE AVERAGE NODE RISKS
# -------------------------------------------------------------------
print("\nCalculating node risk scores...")

nodes = []

# Process users
for i in range(num_users):
    if i in user_risks:
        # Average risk across all transactions
        avg_risk = sum(user_risks[i]) / len(user_risks[i])
        risk_pct = avg_risk * 100
        
        # More conservative threshold for flagging
        is_suspicious = avg_risk > 0.7  # Changed from 0.5 to 0.7
    else:
        risk_pct = 0.0
        is_suspicious = False
    
    nodes.append({
        "id": f"user_{i}",
        "type": "user",
        "risk_score": round(risk_pct, 2),
        "is_suspicious": is_suspicious,
        "name": f"user_{i}",
        "orig_id": i
    })

# Process merchants
for j in range(num_merchants):
    if j in merchant_risks:
        # Average risk across all transactions
        avg_risk = sum(merchant_risks[j]) / len(merchant_risks[j])
        risk_pct = avg_risk * 100
        
        # More conservative threshold
        is_suspicious = avg_risk > 0.7
    else:
        risk_pct = 0.0
        is_suspicious = False
    
    nodes.append({
        "id": f"merch_{j}",
        "type": "merchant",
        "risk_score": round(risk_pct, 2),
        "is_suspicious": is_suspicious,
        "name": f"merch_{j}",
        "orig_id": j
    })

# -------------------------------------------------------------------
# CALCULATE AND DISPLAY STATISTICS
# -------------------------------------------------------------------
print("\n" + "="*60)
print("STATISTICS")
print("="*60)

# Overall stats
total_nodes = len(nodes)
suspicious_nodes = sum(1 for n in nodes if n['is_suspicious'])
fraud_rate = (suspicious_nodes / total_nodes * 100) if total_nodes > 0 else 0

print(f"\nNodes:")
print(f"  Total: {total_nodes:,}")
print(f"  Suspicious: {suspicious_nodes:,}")
print(f"  Fraud Rate: {fraud_rate:.2f}%")

# User stats
user_nodes = [n for n in nodes if n['type'] == 'user']
suspicious_users = sum(1 for n in user_nodes if n['is_suspicious'])
user_fraud_rate = (suspicious_users / len(user_nodes) * 100) if user_nodes else 0

print(f"\nUsers:")
print(f"  Total: {len(user_nodes):,}")
print(f"  Suspicious: {suspicious_users:,}")
print(f"  Fraud Rate: {user_fraud_rate:.2f}%")

# Merchant stats
merchant_nodes = [n for n in nodes if n['type'] == 'merchant']
suspicious_merchants = sum(1 for n in merchant_nodes if n['is_suspicious'])
merchant_fraud_rate = (suspicious_merchants / len(merchant_nodes) * 100) if merchant_nodes else 0

print(f"\nMerchants:")
print(f"  Total: {len(merchant_nodes):,}")
print(f"  Suspicious: {suspicious_merchants:,}")
print(f"  Fraud Rate: {merchant_fraud_rate:.2f}%")

# Edge stats
total_edges = len(edges)
suspicious_edges = sum(1 for e in edges if e['is_suspicious'])
edge_fraud_rate = (suspicious_edges / total_edges * 100) if total_edges > 0 else 0

print(f"\nEdges:")
print(f"  Total: {total_edges:,}")
print(f"  Suspicious: {suspicious_edges:,}")
print(f"  Fraud Rate: {edge_fraud_rate:.2f}%")

# Risk distribution
risk_scores = [n['risk_score'] for n in nodes]
print(f"\nRisk Score Distribution:")
print(f"  Min: {min(risk_scores):.2f}%")
print(f"  Max: {max(risk_scores):.2f}%")
print(f"  Mean: {sum(risk_scores)/len(risk_scores):.2f}%")
print(f"  Median: {sorted(risk_scores)[len(risk_scores)//2]:.2f}%")

# -------------------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------------------
print("\n" + "="*60)
print("SAVING FILES")
print("="*60)

nodes_path = os.path.join(OUT_DIR, "nodes.json")
edges_path = os.path.join(OUT_DIR, "edges.json")

with open(nodes_path, "w") as f:
    json.dump(nodes, f, indent=2)

with open(edges_path, "w") as f:
    json.dump(edges, f, indent=2)

print(f"\n✓ Saved: {nodes_path}")
print(f"✓ Saved: {edges_path}")
print(f"\n✓ DONE - Cache ready for visualization")
print(f"\nExpected fraud rate in visualization: {fraud_rate:.2f}%")