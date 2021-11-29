from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from pyvis.network import Network


class SubsetPCA:
    def __init__(self, value=0.5, type='cutoff'):
        self.type = type
        if type == 'cutoff':
            self.cutoff = value
            self.out_dim = None
        elif type == 'dim':
            self.out_dim = value
            self.cutoff = None

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

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

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

        entries = []
        for col_name, col in correlations.items():
            for row_name, entry in col.items():
                if col_name == row_name:
                    continue
                name = col_name+'@@'+row_name if col_name < row_name else row_name+'@@'+col_name
                entries.append((np.abs(entry), name))
        entries.sort(key=lambda x: -x[0])
        if self.type == 'cutoff':
            for entry, name in [(e, n) for e, n in entries if e > self.cutoff]:
                if np.abs(entry) > self.cutoff:
                    n1, n2 = name.split('@@')
                    high_corr[name] = entry
                    edge_set[n1].append(n2)
                    edge_set[n2].append(n1)
        elif self.type == 'dim':
            components = []
            visited = set()
            lost_dims = 0
            orig_dims = len(original_features)
            for entry, name in entries:
                n1, n2 = name.split('@@')
                if (not n1 in visited) and (not n2 in visited):
                    # print('neither in')
                    visited.add(n1)
                    visited.add(n2)
                    components.append(set([n1, n2]))

                    high_corr[name] = entry
                    edge_set[n1].append(n2)
                    edge_set[n2].append(n1)
                    lost_dims += 1
                elif (n1 in visited) and (not n2 in visited):
                    # print('first in')
                    visited.add(n2)
                    index = [n1 in c for c in components].index(True)
                    comp = components[index]
                    comp.add(n2)
                    components[index] = comp

                    high_corr[name] = entry
                    edge_set[n1].append(n2)
                    edge_set[n2].append(n1)
                    lost_dims += 1
                elif (not n1 in visited) and (n2 in visited):
                    # print('second in')
                    visited.add(n1)
                    index = [n2 in c for c in components].index(True)
                    comp = components[index]
                    comp.add(n1)
                    components[index] = comp

                    high_corr[name] = entry
                    edge_set[n1].append(n2)
                    edge_set[n2].append(n1)
                    lost_dims += 1
                else:
                    i1 = [n1 in c for c in components].index(True)
                    i2 = [n2 in c for c in components].index(True)
                    if i1 == i2:
                        # print('same clust')
                        high_corr[name] = entry
                        edge_set[n1].append(n2)
                        edge_set[n2].append(n1)
                    else:
                        # print('diff clust')
                        c1 = components[i1]
                        c2 = components[i2]
                        del components[i2]
                        i1 = [n1 in c for c in components].index(True)
                        components[i1] = c1 | c2

                        high_corr[name] = entry
                        edge_set[n1].append(n2)
                        edge_set[n2].append(n1)
                        lost_dims += 1
                # print(entry, name)
                # print(components)
                # print(orig_dims, lost_dims)
                if orig_dims - lost_dims <= self.out_dim:
                    break

        components = self.find_components(set(original_features), edge_set)
        comp_pcas = []
        for comp in components:
            df_comp = df[comp]
            if len(df_comp) > 5000:
                df_comp = df_comp[:5000]
            pca = PCA(n_components=1).fit(df_comp)
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
            n1, n2 = name.split('@@')
            net.add_edge(n1, n2)
        net.show('network'+'.html')


class SubsetKernelPCA:
    def __init__(self, cutoff=0.5, kernel='rbf'):
        self.cutoff = cutoff
        self.kernel = kernel

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
            pca = KernelPCA(n_components=1, kernel=self.kernel).fit(df_comp)
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
