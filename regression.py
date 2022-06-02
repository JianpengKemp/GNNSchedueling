import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
from data import SyntheticDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import AvgPooling, GNNExplainer,AvgPooling
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Regress(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Regress, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,allow_zero_in_degree = True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim,allow_zero_in_degree = True)
        self.pool = AvgPooling()
        self.MLP_layer = MLPReadout(hidden_dim, 1)   # 1 out dim since regression problem 

    def forward(self, graph, feat, eweight=None):
        feat = F.relu(self.conv1(graph, feat))
        feat = F.relu(self.conv2(graph, feat))
        
        #feat = F.relu(self.conv2(graph, feat))
        with graph.local_scope():
            #graph.ndata['h'] = feat
            #hg = dgl.mean_nodes(graph, 'h')
            hg = self.pool(graph, feat)
            return self.MLP_layer(hg)

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=3): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

# prepare dataset 
dataset = SyntheticDataset()
num_examples = len(dataset)
num_train = int(num_examples * 0.8)
train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False)


# train loop
device = torch.device('cpu')
model = Regress(19, 150).to(device)
print(model)
num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print("Number of Trainable Parameters:",num_params)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(400):
    for batched_graph, labels in train_dataloader:
        feats = batched_graph.ndata['type']
        #feat1 = batched_graph.ndata['type']
        #feat2 = batched_graph.ndata['prop'].unsqueeze(1)
        #feats = torch.cat((feat1,feat2),dim=1)
        pred = model(batched_graph, feats)
        pred = torch.squeeze(pred,1)
        loss = F.l1_loss(pred, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

# test 
loss = 0
for batched_graph, labels in test_dataloader:
    feats = batched_graph.ndata['type']
    #feat1 = batched_graph.ndata['type']
    #feat2 = batched_graph.ndata['prop'].unsqueeze(1)
    #feats = torch.cat((feat1,feat2),dim=1)
    pred = model(batched_graph, feats)
    pred = torch.squeeze(pred,1)
    loss = F.l1_loss(pred, labels)

print('Test accuracy:', loss.item())

 # Explain the prediction for graph 0
explainer = GNNExplainer(model, num_hops=1)
g, target = dataset[0]
nodes = g.nodes()
features = g.ndata['type']
#features = torch.cat((g.ndata['type'],g.ndata['prop'].unsqueeze(1)),dim=1)
output = model(g,features)
print("test regression value:", output)
print("real value:", target)

feat_mask, edge_mask = explainer.explain_graph(g, features) 


# visualize
space = {'OUTSIDE': 0, 'BASEMENT':1, 'STAIRS':2, 'CORRIDOR': 3, 'TECHNICAL ROOM':4, 'BEDROOM':5, 
    'BATHROOM':6, 'DINING/LIVING ROOM':7, 'KITCHEN':8, 'HALL':9, 'WC':10, 'WALK-IN CLOSET':11, 
    'GALLERY':12, 'SHOWER/WC':13, 'PANTRY':14, 'STORAGE':15, 'STUDY':16,'COAT RACK':17, 'DIELE  ':18}
space = {y: x for x, y in space.items()} 
print(space)
G = dgl.to_networkx(g)
widths = [5 * i for i in edge_mask.tolist()]
nodelist = G.nodes()
labellist = [space[i] for i in np.argmax(g.ndata['type'],axis=1).tolist()]
edgelist = G.edges()
plt.figure(figsize=(12,8))

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,
                       nodelist=nodelist,
                       node_size=1800,
                       node_color='black')
nx.draw_networkx_edges(G,pos,
                       edgelist = edgelist,
                       width=widths,
                       edge_color='blue')
nx.draw_networkx_labels(G, pos=pos,
                        labels=dict(zip(nodelist,labellist)),
                        font_color='white',
                        font_size=5)
plt.box(False)
plt.show()


