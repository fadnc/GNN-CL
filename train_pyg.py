# train_pyg.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import os
from models.fraud_gnn_pyg import FraudGNNHybrid
from models.dataset_pyg import load_processed_graph, make_edge_index_splits

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'], metrics['recall'], metrics['f1'] = p, r, f
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    return metrics

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    logits: (N, C)
    targets: (N,)
    """
    probs = F.softmax(logits, dim=1)
    pt = probs[range(len(targets)), targets]
    logp = torch.log(pt + 1e-12)
    loss = -alpha * ((1 - pt) ** gamma) * logp
    return loss.mean()

def train_epoch(model, data, optimizer, train_idx, batch_size=8192,
                use_focal=True, class_weights=None):

    model.train()
    total_loss = 0.0
    n_batches = 0

    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    # move node features to device once
    data['user'].x = data['user'].x.to(DEVICE)
    data['merchant'].x = data['merchant'].x.to(DEVICE)

    perm = torch.randperm(train_idx.size(0), device=train_idx.device)
    train_idx = train_idx[perm]

    for i in range(0, train_idx.size(0), batch_size):

        batch_edges = train_idx[i:i+batch_size]

        # 1️⃣ recompute node embeddings EVERY BATCH
        node_embs = model.forward_nodes(data)

        src = edge_index[0, batch_edges].to(DEVICE)
        dst = edge_index[1, batch_edges].to(DEVICE)
        e_attr = edge_attr[batch_edges].to(DEVICE)
        y = labels[batch_edges].to(DEVICE)

        logits = model.forward_edges(node_embs, src, dst, e_attr)

        if use_focal:
            loss = focal_loss(logits, y, alpha=0.25, gamma=2.0)
        else:
            loss = F.cross_entropy(logits, y, weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)



@torch.no_grad()
def evaluate(model, data, split_idx):
    model.eval()

    data['user'].x = data['user'].x.to(DEVICE)
    data['merchant'].x = data['merchant'].x.to(DEVICE)

    node_embs = model.forward_nodes(data)

    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    batch_size = 32768
    preds, probs, y_true = [], [], []

    for i in range(0, split_idx.size(0), batch_size):
        batch_edges = split_idx[i:i+batch_size]

        src = edge_index[0, batch_edges].to(DEVICE)
        dst = edge_index[1, batch_edges].to(DEVICE)
        e_attr = edge_attr[batch_edges].to(DEVICE)
        y = labels[batch_edges].to(DEVICE)

        logits = model.forward_edges(node_embs, src, dst, e_attr)
        p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
        pr = logits.argmax(dim=1).cpu().numpy()

        preds.append(pr)
        probs.append(p)
        y_true.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    y_true = np.concatenate(y_true)

    return compute_metrics(y_true, preds, probs)


def main():
    # Load graph
    data, u_map, m_map = load_processed_graph("data/processed/fraud_graph_pyg.pt", device='cpu')
    train_idx, val_idx, test_idx = make_edge_index_splits(data)
    print("Train/Val/Test sizes:", train_idx.size(0), val_idx.size(0), test_idx.size(0))

    # dims
    user_in = data['user'].x.size(1)
    merch_in = data['merchant'].x.size(1)
    edge_attr_dim = data['user','transacts','merchant'].edge_attr.size(1)
    hidden = 128

    model = FraudGNNHybrid(user_in, merch_in, edge_attr_dim, hidden_dim=hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_f1 = 0.0
    for epoch in range(1, 101):
        loss = train_epoch(model, data, optimizer, train_idx, batch_size=16384)
        val_metrics = evaluate(model, data, val_idx)
        test_metrics = evaluate(model, data, test_idx) if epoch % 10 == 0 else None
        print(f"Epoch {epoch} | Train loss {loss:.4f} | Val F1 {val_metrics['f1']:.4f} | Val AUC {val_metrics.get('auc',0):.4f}")
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "best_fraudgnn.pth")
            print("Saved best model.")

    print("Training finished. Best val F1:", best_val_f1)

if __name__ == "__main__":
    main()
