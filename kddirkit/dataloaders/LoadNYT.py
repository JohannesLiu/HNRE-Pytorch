import numpy as np
import pandas as pd
import os

import torch

class NYTDataLoader(object):
    def __init__(self):
        super(NYTDataLoader, self).__init__()
        self.relationVectorFile_NYT = "./raw_data/TransEEmbeddings/NYT-10/relationVector.txt"
        self.relationHierFile_NYT = "./data_analysis/hier_relation_data/hier-rel-NYT-Clusters.csv"
        self.relationHierFile_NYT_PS = "./data_analysis/hier_relation_data/hier-rel-NYT-PS-Clusters.csv"
        self.data_path = "./data/"

class NYTGraphDataLoader(NYTDataLoader):
    def __init__(self):
        super(NYTGraphDataLoader, self).__init__()
        self._data, self._nodes_data,  self._edges_data = self.reset_parameter()

    def reset_parameter(self):
        # relationVectorFile_NYT = "./raw_data/TransEEmbeddings/NYT-10/relationVector.txt"
        # relationHierFile_NYT = "./data_analysis/hier_relation_data/hier-rel-NYT-Clusters.csv"
        # relationHierFile_NYT_PS = "./data_analysis/hier_relation_data/hier-rel-NYT-PS-Clusters.csv"

        relationVectorFile_NYT = pd.read_table(self.relationVectorFile_NYT, names=["rel", "vec"])
        rel_NYT_list = relationVectorFile_NYT["rel"].tolist()
        rel_NYT_ndarray = np.array(rel_NYT_list)
        vec_NYT_list = relationVectorFile_NYT["vec"].tolist()
        vec_NYT_ndarray = np.ndarray(shape=[52, 50])
        for i in range(len(vec_NYT_list)):
            vec_tmp_string = vec_NYT_list[i].replace('[', "").replace(']', "").split(',')
            vec_tmp_ndarray = np.array(vec_tmp_string).astype(np.float)
            vec_NYT_ndarray[i] = (vec_tmp_ndarray)
        relationVector_NYT_np = np.column_stack((rel_NYT_list, vec_NYT_ndarray))
        relationVector_NYT_pd = pd.DataFrame(relationVector_NYT_np,
                                             columns=(["rel"] + ["transe_dim_" + str(i) for i in range(50)]))

        rel_NYT_hier2_PS_pd = pd.read_csv(self.relationHierFile_NYT_PS).drop(0).reset_index(drop=True)

        order = ['rel', 'src', 'des', 'hier', 'cluster'] + ["transe_dim_" + str(i) for i in range(50)]
        # 先筛选出hier0, hier1, hier2
        rel_NYT_hier0_pd = pd.read_csv(self.relationHierFile_NYT)
        rel_NYT_hier0_pd = rel_NYT_hier0_pd[rel_NYT_hier0_pd['hier'] == 0]
        rel_NYT_hier1_pd = pd.read_csv(self.relationHierFile_NYT)
        rel_NYT_hier1_pd = rel_NYT_hier1_pd[rel_NYT_hier1_pd['hier'] == 1]
        rel_NYT_hier2_pd = pd.read_csv(self.relationHierFile_NYT)
        rel_NYT_hier2_pd = rel_NYT_hier2_pd[rel_NYT_hier2_pd['hier'] == 2]
        rel_NYT_hier3_pd = pd.read_csv(self.relationHierFile_NYT)
        rel_NYT_hier3_pd = rel_NYT_hier3_pd[rel_NYT_hier3_pd['hier'] == 3].reset_index(drop=True)
        # 将embeddings与各个分表连接
        rel_NYT_hier0_pd = pd.merge(relationVector_NYT_pd, rel_NYT_hier0_pd, how='inner')[order]
        rel_NYT_hier1_pd = np.column_stack(
            (rel_NYT_hier1_pd, np.ndarray(shape=[len(rel_NYT_hier1_pd), 50], dtype=float)))
        rel_NYT_hier1_pd = pd.DataFrame(rel_NYT_hier1_pd, columns=(
                    ["rel", "src", "des", "hier", 'cluster'] + ["transe_dim_" + str(i) for i in range(50)]))
        rel_NYT_hier2_pd = np.column_stack(
            (rel_NYT_hier2_pd, np.ndarray(shape=[len(rel_NYT_hier2_pd), 50], dtype=float)))
        rel_NYT_hier2_pd = pd.DataFrame(rel_NYT_hier2_pd, columns=(
                    ["rel", "src", "des", "hier", 'cluster'] + ["transe_dim_" + str(i) for i in range(50)]))
        for i in range(50):
            rel_NYT_hier0_pd[['transe_dim_' + str(i)]] = rel_NYT_hier0_pd[['transe_dim_' + str(i)]].astype('float')
            rel_NYT_hier1_pd[['transe_dim_' + str(i)]] = rel_NYT_hier1_pd[['transe_dim_' + str(i)]].astype('float')
            rel_NYT_hier2_pd[['transe_dim_' + str(i)]] = rel_NYT_hier2_pd[['transe_dim_' + str(i)]].astype('float')
        rel_NYT_hier0_pd[['src']] = rel_NYT_hier0_pd[['src']].astype('int')
        rel_NYT_hier1_pd[['src']] = rel_NYT_hier1_pd[['src']].astype('int')
        rel_NYT_hier2_pd[['src']] = rel_NYT_hier2_pd[['src']].astype('int')
        rel_NYT_hier0_pd[['des']] = rel_NYT_hier0_pd[['des']].astype('int')
        rel_NYT_hier1_pd[['des']] = rel_NYT_hier1_pd[['des']].astype('int')
        rel_NYT_hier2_pd[['des']] = rel_NYT_hier2_pd[['des']].astype('int')
        rel_NYT_hier0_pd[['hier']] = rel_NYT_hier0_pd[['hier']].astype('int')
        rel_NYT_hier1_pd[['hier']] = rel_NYT_hier1_pd[['hier']].astype('int')
        rel_NYT_hier2_pd[['hier']] = rel_NYT_hier2_pd[['hier']].astype('int')
        rel_NYT_hier0_pd[['cluster']] = rel_NYT_hier0_pd[['cluster']].astype('int')
        rel_NYT_hier1_pd[['cluster']] = rel_NYT_hier1_pd[['cluster']].astype('int')
        rel_NYT_hier2_pd[['cluster']] = rel_NYT_hier2_pd[['cluster']].astype('int')

        for k in range(50):
            for i in range(len(rel_NYT_hier1_pd)):
                dimSum_k = 0
                sameCount = 0
                for j in range(len(rel_NYT_hier0_pd)):
                    if (rel_NYT_hier0_pd.loc[j, 'des'] == rel_NYT_hier1_pd.loc[i, 'src']):
                        dimSum_k += rel_NYT_hier0_pd.loc[j, 'transe_dim_' + str(k)]
                        sameCount += 1
                rel_NYT_hier1_pd.loc[i, 'transe_dim_' + str(k)] = dimSum_k / sameCount
        for k in range(50):
            for i in range(len(rel_NYT_hier2_pd)):
                dimSum_k = 0
                sameCount = 0
                for j in range(len(rel_NYT_hier1_pd)):
                    if (rel_NYT_hier1_pd.loc[j, 'des'] == rel_NYT_hier2_pd.loc[i, 'src']):
                        dimSum_k += rel_NYT_hier1_pd.loc[j, 'transe_dim_' + str(k)]
                        sameCount += 1
                rel_NYT_hier2_pd.loc[i, 'transe_dim_' + str(k)] = dimSum_k / sameCount
        for k in range(50):
            for i in range(len(rel_NYT_hier3_pd)):
                dimSum_k = 0
                sameCount = 0
                for j in range(len(rel_NYT_hier2_pd)):
                    if (rel_NYT_hier2_pd.loc[j, 'des'] == rel_NYT_hier3_pd.loc[i, 'src']):
                        dimSum_k += rel_NYT_hier2_pd.loc[j, 'transe_dim_' + str(k)]
                        sameCount += 1
                rel_NYT_hier3_pd.loc[i, 'transe_dim_' + str(k)] = dimSum_k / sameCount

        rel_NYT_hier = rel_NYT_hier0_pd.append(rel_NYT_hier1_pd).append(rel_NYT_hier2_pd).append(
            rel_NYT_hier3_pd).reset_index(drop=True)
        rel_NYT_hier.loc[:, 'hier'] = rel_NYT_hier2_PS_pd.loc[:, 'hier']
        rel_NYT_hier.src = rel_NYT_hier.src + 1
        rel_NYT_hier.des = rel_NYT_hier.des + 1

        NANode_List = [['NA_node', 1, 0, 1, -1] + 50 * [0]]
        NANode_pd = pd.DataFrame(NANode_List, columns=order)
        rel_NYT_hier = NANode_pd.append(rel_NYT_hier).reset_index(drop=True)

        virtualNode_List = [['virtual_node', 0, -1, -1, -2] + 50 * [0]]
        virtualNode_pd = pd.DataFrame(virtualNode_List, columns=order)
        data = rel_NYT_hier.copy()
        data.hier = data.hier
        data.loc[data['des'] == -1, 'des'] = 0
        data = virtualNode_pd.append(data).reset_index(drop=True)
        # data.des = data.des-1

        for k in range(50):
            dimSum_k = 0
            sameCount = 0
            for j in range(len(rel_NYT_hier2_pd)):
                dimSum_k += rel_NYT_hier2_pd.loc[j, 'transe_dim_' + str(k)]
                sameCount += 1
            dimSum_k += rel_NYT_hier3_pd.loc[0, 'transe_dim_' + str(k)]
            sameCount += 1
            data.loc[0, 'transe_dim_' + str(k)] = dimSum_k / sameCount

        top_nodes = data[data['des'] == 0]
        top_nodes = top_nodes.reset_index().src

        nodes_data = data[['src', 'rel', 'hier']]
        nodes_data.loc[:, 'src'] = nodes_data.loc[:, 'src']

        nodes_data = nodes_data
        edges_data = data[['src', 'des']][1:]

        return data, nodes_data, edges_data

    @property
    def data(self):
        return self._data

    @property
    def nodes_data(self):
        return self._nodes_data

    @property
    def edges_data(self):
        return self._edges_data

class NYTTrainDataLoader(NYTDataLoader):
    def __init__(self, device= 'cpu'):
        super(NYTTrainDataLoader, self).__init__()
        self.num_classes = 53
        self.instance_triple = np.load(self.data_path + 'train_instance_triple.npy')
        self.instance_scope = np.load(self.data_path + 'train_instance_scope.npy')
        self.len = np.load(self.data_path + 'train_len.npy')
        self.label = np.load(self.data_path + 'train_label.npy')
        self.word = np.load(self.data_path + 'train_word.npy')
        self.pos1 = np.load(self.data_path + 'train_pos1.npy')
        self.pos2 = np.load(self.data_path + 'train_pos2.npy')
        self.mask = np.load(self.data_path + 'train_mask.npy')
        self.instance_scope_Tensor = torch.LongTensor(self.instance_scope).to(device)
        self.len_Tensor = torch.LongTensor(self.len).to(device)
        self.label_Tensor = torch.LongTensor(self.label).to(device)
        self.word_Tensor = torch.LongTensor(self.word).to(device)
        self.pos1_Tensor = torch.LongTensor(self.pos1).to(device)
        self.pos2_Tensor = torch.LongTensor(self.pos2).to(device)
        self.mask_Tensor = torch.LongTensor(self.mask).to(device)




class NYTTestDataLoader(NYTDataLoader):
    def __init__(self, mode, device= 'cpu'):
        super(NYTTestDataLoader, self).__init__()
        self._mode = mode
        if self._mode == 'pr' or self._mode == 'hit_k_100' or self._mode == 'hit_k_200':
            self.instance_triple = np.load(self.data_path + 'test_entity_pair.npy')
            self.instance_scope = np.load(self.data_path + 'test_entity_scope.npy')
            self.len = np.load(self.data_path + 'test_len.npy')
            self.label = np.load(self.data_path + 'test_label.npy')
            self.word = np.load(self.data_path + 'test_word.npy')
            self.pos1 = np.load(self.data_path + 'test_pos1.npy')
            self.pos2 = np.load(self.data_path + 'test_pos2.npy')
            self.mask = np.load(self.data_path + 'test_mask.npy')
            self.exclude_na_flatten_label = np.load(self.data_path + 'all_true_label.npy')
        else:
            self.instance_triple = np.load(self.data_path + 'pn/test_entity_pair_pn.npy')
            self.instance_scope = np.load(self.data_path + 'pn/test_entity_scope_' + mode + '.npy')
            self.len = np.load(self.data_path + 'pn/test_len_' + mode + '.npy')
            self.label = np.load(self.data_path + 'pn/test_label_' + mode + '.npy')
            self.word = np.load(self.data_path + 'pn/test_word_' + mode + '.npy')
            self.pos1 = np.load(self.data_path + 'pn/test_pos1_' + mode + '.npy')
            self.pos2 = np.load(self.data_path + 'pn/test_pos2_' + mode + '.npy')
            self.mask = np.load(self.data_path + 'pn/test_mask_' + mode + '.npy')
            self.exclude_na_flatten_label = np.load(self.data_path + 'pn/true_label.npy')
        self.instance_scope_Tensor = torch.LongTensor(self.instance_scope).to(device)
        self.len_Tensor = torch.LongTensor(self.len).to(device)
        self.label_Tensor = torch.LongTensor(self.label).to(device)
        self.word_Tensor = torch.LongTensor(self.word).to(device)
        self.pos1_Tensor = torch.LongTensor(self.pos1).to(device)
        self.pos2_Tensor = torch.LongTensor(self.pos2).to(device)
        self.mask_Tensor = torch.LongTensor(self.mask).to(device)
        self.exclude_na_flatten_label_Tensor = torch.LongTensor(self.exclude_na_flatten_label).to(device)

        f = open("raw_data/relation2id.txt", "r")
        content = f.readlines()[1:]
        self.id2rel = {}
        for i in content:
            rel, rid = i.strip().split()
            self.id2rel[(int)(rid)] = rel
        f.close()

        self.fewrel_100 = {}
        f = open("data/rel100.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_100[i.strip()] = 1
        f.close()

        self.fewrel_200 = {}
        f = open("data/rel200.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_200[i.strip()] = 1
        f.close()


    @property
    def mode(self):
        return self._mode

if __name__ == '__main__':
    nyt_data = NYTGraphDataLoader()
    print(nyt_data.data)
    print(nyt_data.nodes_data)
    print(nyt_data.edges_data)