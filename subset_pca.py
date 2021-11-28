from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
from pyvis.network import Network


class SubsetPCA:
    def __init__(self, cutoff=0.5):
        self.cutoff = cutoff

    def component(self, node, node_set, edge_set):
        node_set.remove(node)
        comp = set([node])
        if not node in edge_set:
            return set()
        neighbors = edge_set[node]
        for neighbor in neighbors:
            if not neighbor in node_set:
                continue
            comp.update(self.component(neighbor, node_set, edge_set))
        return comp

    def find_components(self, node_set, edge_set):
        components = []
        while len(node_set) > 0:
            comp = self.component(list(node_set)[0], node_set, edge_set)
            components.append(comp)
        return components

    def transform(self, df):
        out = pd.DataFrame()
        for i in range(len(self.components)):
            comp = self.components[i]
            pca = self.comp_pcas[i]
            name = ' '.join(comp)
            out[name] = pca.transform(df[comp]).flatten()
        return out

    def fit(self, df):
        original_features = df.columns.values
        correlations = df.corr()
        high_corr = {}
        edge_set = {}
        for node in original_features:
            edge_set[node] = []
        for col_name, col in correlations.items():
            for row_name, entry in col.items():
                if np.abs(entry) > self.cutoff and col_name != row_name:
                    name = col_name+' '+row_name if col_name < row_name else row_name+' '+col_name
                    high_corr[name] = entry
                    edge_set[col_name].append(row_name)
                    edge_set[row_name].append(col_name)
        components = self.find_components(set(original_features), edge_set)
        comp_pcas = []
        for comp in components:
            df_comp = df[comp]
            if len(df_comp) > 5000:
                df_comp = df_comp[:5000]
            pca = KernelPCA(n_components=1, kernel='rbf').fit(df_comp)
            comp_pcas.append(pca)
        self.components = components
        self.comp_pcas = comp_pcas
        self.high_corr = high_corr
        self.edge_set = edge_set
        self.train = df
        return components, comp_pcas

    def visualize_components(self):
        net = Network()
        node_set = self.train.columns.values
        for node in node_set:
            net.add_node(node, label=str(node))
        for name, corr in self.high_corr.items():
            n1, n2 = name.split()
            net.add_edge(n1, n2)
        net.show('network'+'.html')
