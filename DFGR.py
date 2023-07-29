# @Time    : 2022/12/14 0014 20:07
# @Author  : TAO XU
# @File    : CGNN.py
import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm

from resnet_fcn import resnet18, resnet34, resnet50,resnet101, resnet152
from vgg_fcn import vgg11, vgg11_bn,vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from densenet_fcn import densenet121, densenet169, densenet201, densenet161
from VIT import vit_base_patch16_224, vit_base_patch32_224
from  inception_fcn import inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool , GIN, GlobalAttention, EdgeConv
from torch_geometric.nn.norm import BatchNorm, GraphNorm, LayerNorm
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

class attention(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gate_nn = nn.Linear(hidden_channels, 1)
        self.nn = nn.ReLU()
    def forward(self, x, batch):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)
        size = batch[-1].item()+1
        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        attention = softmax(gate, batch, num_nodes=size)
        out = scatter_add(attention*x, batch, dim=0, dim_size=size)
        return out, attention

class CGNN(torch.nn.Module):
    def __init__(self,embedding_size,hidden_size, backbone, adj, args, ):
        super(CGNN,self).__init__()

        self.backbone = backbone
        self.conv_down = nn.Conv2d(embedding_size, hidden_size, kernel_size=1)
        self.bn_down = nn.BatchNorm2d(hidden_size)
        self.rule_down = nn.ReLU(True)
        self.args = args
        if args.gnn_layer in ['gat', 'GAT', 'Gat']:
            self.GNN_Layer = GATLayer_mutihead(hidden_size, hidden_size, args.adj_model, adj, args.batch_size, args=args)
            self.fc = torch.nn.Linear(hidden_size*args.heads, args.num_classes)
        elif args.gnn_layer in ['gcn', 'GCN', 'Gcn']:
            self.GNN_Layer = GCNLayer(hidden_size, hidden_size, args.adj_model, adj, args.batch_size, args=args)
            self.fc = torch.nn.Linear(hidden_size,args.num_classes)
        else:
            pass
    def forward(self,x):
        x = self.backbone(x)
        if self.args.model_name in ['VITB16GNN', 'VITB32GNN']:
            b, wh, c = x.shape
            x = x.transpose(1, 2)
            x = x.reshape(b, c, int(wh**0.5), int(wh**0.5))
        x = self.conv_down(x)
        x = self.rule_down(self.bn_down(x))

        x, A = self.GNN_Layer(x)
        if self.args.heads > 4 and self.args.gnn_layer in ['gat', 'GAT', 'Gat']:
            x = F.dropout(x, p=0.0, training=True)
        return self.fc(x), A, x

class GATLayer_mutihead(nn.Module):
    def __init__(self, in_features, out_features, adj_mode, adj_all, batch_size, args):
        super(GATLayer_mutihead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_mode = adj_mode
        self.adj_all = adj_all
        self.batch_size = batch_size

        self.args = args
        self.conv1 = GATConv(in_channels=in_features, out_channels=out_features, heads=self.args.heads, dropout=0.0)
        self.BatchNorm1 = BatchNorm(in_channels=out_features * self.args.heads,)
        self.relu1 = nn.ReLU(inplace=True)


        self.attention_mean = attention(in_features * self.args.heads)

    def forward(self, x):
        b, c, w, h = x.shape
        if b == self.args.train_number%self.batch_size: # train的tail 7165
            adj, _ = self.adj_all[self.adj_mode]['tail_train']
        elif b== self.args.test_number%self.batch_size:# val的tail 1788
            adj, _ = self.adj_all[self.adj_mode]['tail_val']
        elif b==self.batch_size:
            adj, _ = self.adj_all[self.adj_mode]['batch']
        else:
            print(b+""+self.batch_size)
            adj = None
        batch = torch.reshape(torch.LongTensor([[i] * w * h for i in range(b)]), [-1, ]).cuda()
        adj = adj.cuda()

        x = x.reshape([b, c, -1])
        x = x.transpose(1, 2)
        x = x.reshape([-1, c])
        # block 1
        x = self.conv1(x, adj)
        x = self.BatchNorm1(x)
        x = self.relu1(x)
        # readout
        #x = global_mean_pool(x, batch)
        x, attention = self.attention_mean(x, batch)
        return x, attention

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_mode, adj_all, batch_size, args):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.adj_mode = adj_mode
        self.adj_all = adj_all
        self.batch_size = batch_size

        self.args = args

        self.conv1 = GATConv(in_channels=in_features, out_channels=out_features, heads=1, dropout=0.0)
        self.BatchNorm1 = BatchNorm(in_channels=out_features)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention_mean = attention(out_features)

    def forward(self, x):
        b, c, w, h = x.shape
        if b == self.args.train_number%self.batch_size: # train的tail 7165
            adj, _ = self.adj_all[self.adj_mode]['tail_train']
        elif b== self.args.test_number%self.batch_size:# val的tail 1788
            adj, _ = self.adj_all[self.adj_mode]['tail_val']
        elif b==self.batch_size:
            adj, _ = self.adj_all[self.adj_mode]['batch']
        else:
            print(b+""+self.batch_size)
            adj = None
        batch = torch.reshape(torch.LongTensor([[i] * w * h for i in range(b)]), [-1, ]).cuda()
        adj = adj.cuda()
        if self.args.model_name not in ['VITB16GNN', 'VITB32GNN']:
            x = x.reshape([b, c, -1])
            x = x.transpose(1, 2)
            x = x.reshape([-1, c])
        # block 1
        x = self.conv1(x, adj)
        x = self.BatchNorm1(x)
        x = self.relu1(x)

        # readout
        # x = global_mean_pool(x, batch)
        x, attention = self.attention_mean(x, batch)
        return x, attention

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_mode, adj_all, batch_size, args):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_mode = adj_mode
        self.adj_all = adj_all
        self.batch_size = batch_size
        self.args = args

        self.conv1 = GCNConv(in_channels=in_features, out_channels=out_features)
        self.BatchNorm1 = BatchNorm(in_channels=out_features)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention_mean = attention(out_features)

    def forward(self, x):
        b, c, w, h = x.shape
        if b == self.args.train_number%self.batch_size: # train的tail 7165
            adj, _ = self.adj_all[self.adj_mode]['tail_train']
        elif b== self.args.test_number%self.batch_size:# val的tail 1788
            adj, _ = self.adj_all[self.adj_mode]['tail_val']
        elif b==self.batch_size:
            adj, _ = self.adj_all[self.adj_mode]['batch']
        else:
            print(b+""+self.batch_size)
            adj = None
        batch = torch.reshape(torch.LongTensor([[i] * w * h for i in range(b)]), [-1, ]).cuda()
        adj = adj.cuda()
        x = x.reshape([b, c, -1])
        x = x.transpose(1, 2)
        x = x.reshape([-1, c])

        # block 1
        x = self.conv1(x, adj)
        x = self.BatchNorm1(x)
        x = self.relu1(x)

        # readout
        # x = global_mean_pool(x, batch)
        x, attention = self.attention_mean(x, batch)
        return x, attention

def CGNNRes18(adj, args):
    backbone = resnet18(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def CGNNRes34(adj, args):
    backbone = resnet34(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def CGNNRes50(adj, args):
    backbone = resnet50(pretrained=True)
    return CGNN(embedding_size=2048, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNRes101(adj, args):
    backbone = resnet101(pretrained=True)
    return CGNN(embedding_size=2048, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNRes152(adj, args):
    backbone = resnet152(pretrained=True)
    return CGNN(embedding_size=2048, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNVgg11(adj, args):
    backbone = vgg11(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNVgg13(adj, args):
    backbone = vgg13(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNVgg16(adj, args):
    backbone = vgg16(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNVgg19(adj, args):
    backbone = vgg19(pretrained=True)
    return CGNN(embedding_size=512, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNDense121(adj, args):
    backbone = densenet121(pretrained=True)
    return CGNN(embedding_size=1024, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args, )
def CGNNDense169(adj, args):
    backbone = densenet169(pretrained=True)
    return CGNN(embedding_size=1664, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def CGNNDense201(adj, args):
    backbone = densenet201(pretrained=True)
    return CGNN(embedding_size=1920, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def CGNNInceptionV3(adj, args):
    backbone = inception_v3(pretrained=True)
    return CGNN(embedding_size=2048, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def VITB16GNN(adj, args):
    backbone = vit_base_patch16_224()
    return CGNN(embedding_size=768, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)
def VITB32GNN(adj, args):
    backbone = vit_base_patch32_224()
    return CGNN(embedding_size=768, hidden_size=args.hidden_size, backbone=backbone, adj=adj, args=args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Propert CGNN for OAI in pytorch')
    parser.add_argument('-adj_model', default=4, type=str,
                        help='the adj_model for feature map')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--num_classes',
                        help='the number of classification',
                        type=int, default=5)
    args = parser.parse_args()
    model = CGNN(64,64,adj=None,args=args)
    print(model)
