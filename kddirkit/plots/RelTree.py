import dgl
import networkx as nx
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class RelTree(object):
    def __init__(self, nodes_data, edges_data):
        super(RelTree, self).__init__()

        self.nodes_data = nodes_data
        self.edges_data = edges_data
        self.src = nyt_data.edges_data['src'].to_numpy()
        self.dst = nyt_data.edges_data['des'].to_numpy()
        self.g = dgl.graph((self.dst, self.src))

        self.savePath = "./data_analysis/tree.png"
        self.rel2id_path = "D:/PycharmProjects/KGNS/raw_data/relation2id.csv"
        self.rel2id_pd = pd.read_csv(self.rel2id_path)
        self.virtual_weight = 570088
        self.weight_rel2id_pd = pd.DataFrame([self.virtual_weight], columns=['Counts']).append(self.rel2id_pd[['Counts']]).fillna(0)
        self.weight_rel2id_np = self.weight_rel2id_pd.to_numpy()
        # weight_rel2id_np = weight_rel2id_np[np.isnan(weight_rel2id_np)] = 0
        self.weight_rel2id_Tensor = torch.LongTensor(self.weight_rel2id_np)

    def out_degree2color(self, out_degree):
        if out_degree==1 and out_degree<5 :
            return 'g'
        if out_degree>1 and out_degree<5 :
            return 'y'
        elif out_degree>=5  :
            return 'r'
        else:
            return 'b'

    def longTail2color(self, instanceCounts):
        if instanceCounts>=1 and instanceCounts<=200 :
            return 'g'
        if instanceCounts>200 and instanceCounts<=10000 :
            return 'y'
        elif instanceCounts>10000:
            return 'r'
        else:
            return 'b'

    @property
    def OutDegreeTree(self, savePath = None):
        tree_g = self.g.to_networkx().to_directed()
        fig = plt.figure(figsize=(16, 9))

        pos = nx.drawing.nx_agraph.graphviz_layout(tree_g, prog='dot')
        values = [self.out_degree2color(tree_g.out_degree(i)) for i in range(len(self.nodes_data))]

        values[0] = 'purple'
        pos[0] = [1800, 306]
        nx.draw(tree_g, pos, with_labels=True, node_color=values, arrows=True, alpha=0.5)

        color = ['purple', 'red', 'yellow', 'green', 'blue']  # 指定bar的颜色
        labels = ['virtual relation', 'counts of sub chidr en $> 5$', 'counts of sub chidren $\geq 1$',
                  'counts of sub chidren $=1$', 'bottom relation']  # legend标签列表，上面的color即是颜色列表
        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
        plt.legend(handles=patches)
        if savePath != None:
            fig.savefig(savePath)  # 保存图片 ./data_analysis/tree.png]
        return fig

    @property
    def InstanceTree(self, savePath = None):
        tree_g = self.g.to_networkx().to_directed()
        # tree_g.add_node("root")
        fig = plt.figure(figsize=(16, 9))

        pos = nx.drawing.nx_agraph.graphviz_layout(tree_g, prog='dot')
        values = [self.longTail2color(self.weight_rel2id_np[i]) for i in range(len(self.nodes_data))]
        # tree_g.remove_node(1)
        values[0] = 'purple'
        pos[0] = [1800, 306]
        nx.draw(tree_g, pos, with_labels=True, node_color=values, arrows=True, alpha=0.5)

        color = ['purple', 'red', 'yellow', 'green', 'blue']  # 指定bar的颜色
        labels = ['virtual relation', 'counts of instances $> 10000$', 'counts of instances $> 200$',
                  'counts of instances $\geq=1$', 'counts of instances $=0$']  # legend标签列表，上面的color即是颜色列表
        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
        plt.legend(handles=patches)
        if savePath != None:
            fig.savefig('./data_analysis/tree.png')  # 保存图片
        return fig

if __name__ == '__main__':
    from kddirkit.dataloaders.LoadNYT import *
    nyt_data = NYTDataLoader()
    RelTree = RelTree(nyt_data.nodes_data, nyt_data.edges_data)
    fig = RelTree.OutDegreeTree
    plt.show()