import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils import subgraph
from model import WrappedModel
from config import EXPLANATION_DIR


def init_explainer(model):
    wrapped = WrappedModel(model)
    return Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type=None,
        model_config=ModelConfig(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        )
    )


def explain_graphs(model, graphs, top_pct=0.2):
    os.makedirs(EXPLANATION_DIR, exist_ok=True)
    explainer = init_explainer(model)
    results = []
    for idx, data in enumerate(graphs):
        data.num_nodes = data.x.size(0)
        data = data.to(model.base.device)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        explanation = explainer(
            x=data.x.float(),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr.view(-1,1).float(),
            batch=batch,
            graph_features=data.graph_features.float(),
        )
        mask = explanation.node_mask.detach().cpu()
        k    = max(1, int(top_pct * mask.numel()))
        topn = mask.topk(k).indices
        ei_s, ea_s = subgraph(
            topn, data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
            edge_attr=data.edge_attr.view(-1,1).cpu(),
        )
        G = nx.DiGraph()
        for (u,v),w in zip(ei_s.t().cpu().numpy(), ea_s.cpu().numpy()):
            G.add_edge(int(u), int(v), weight=float(w))
            results.append({'U':int(u),'V':int(v),'W':float(w),'graph_idx':idx})
        pos = nx.spring_layout(G)
        plt.figure(figsize=(5,5))
        widths = [d['weight']*5 for _,_,d in G.edges(data=True)]
        colors = ['red' if n in topn.cpu().numpy() else 'lightgray' for n in G.nodes()]
        nx.draw(G, pos,
                with_labels=True,
                node_size=[300 if c=='red' else 100 for c in colors],
                node_color=colors,
                width=widths)
        plt.title(f"Graph {idx} — Top {int(top_pct*100)}% Nodes")
        plt.axis('off')
        plt.savefig(EXPLANATION_DIR / f"graph_{idx}.png")
        plt.close()
    return results