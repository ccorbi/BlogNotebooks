from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, inDim, outDim, hidDim=400, cuda=True):
        super(VAE, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.cuda = cuda

        self.fc1 = nn.Linear(inDim, hidDim)
        self.fc21 = nn.Linear(hidDim, outDim)
        self.fc22 = nn.Linear(hidDim, outDim)
        self.fc3 = nn.Linear(outDim, hidDim)
        self.fc4 = nn.Linear(hidDim, inDim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.inDim))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

class fusion_VAE(nn.Module):
    def __init__(self, RNA, methylation, miRNA, sub_depth=2, whole_depth=2, rna_decay=5, methy_decay=5, mirna_decay=5, whole_decay=5, cuda=False):
        super(fusion_VAE, self).__init__()

        def layer_construction(in_features, num_layers=2, decay=5):
            layers, sizes = [], []
            for i in range(num_layers):
                tmp_in = in_features / (decay ** i)
                tmp_out = in_features / (decay ** (i + 1))
                layers.append(nn.Linear(tmp_in, tmp_out))
                sizes.append(tmp_out)
            return nn.ModuleList(layers), sizes

        self.isCuda= cuda
        rna_size, methy_size, mirna_size = RNA.shape[-1], methylation.shape[-1], miRNA.shape[-1]
        self.rna_layers, rna_sizes = layer_construction(rna_size, num_layers=sub_depth, decay=rna_decay)
        self.methy_layers, methy_sizes = layer_construction(methy_size, num_layers=sub_depth, decay=methy_decay)
        self.mirna_layers, mirna_sizes = layer_construction(mirna_size, num_layers=sub_depth, decay=mirna_decay)

        self.sizes = [rna_size + methy_size + mirna_size]
        for i in range(sub_depth):
            self.sizes.append(rna_sizes[i] + methy_sizes[i] + mirna_sizes[i])
        
        self.whole_layers, whole_sizes = layer_construction(self.sizes[-1], num_layers=whole_depth, decay=whole_decay)
        self.sizes.extend(whole_sizes)

        self.hidden_features = self.sizes[-1] / whole_decay
        self.mu, self.logvar = nn.Linear(self.sizes[-1], self.hidden_features), nn.Linear(self.sizes[-1], self.hidden_features)
        self.sizes.append(self.hidden_features)

        print("=======================>")
        print( "Sub-network Layers Size")
        command = "Total: %d\tRNA: %d\tMethylation: %d\tmiRNA: %d"
        print( command % (self.sizes[0], rna_size, methy_size, mirna_size))
        for i in range(sub_depth-1):
            print(command % (self.sizes[i+1], rna_sizes[i], methy_sizes[i], mirna_sizes[i]))
        print( "Integrating Layers Size"    )
        for i in range(whole_depth+1):
            print("Total: %d" % self.sizes[sub_depth+i])
        print("Hidden Size: %d" % self.hidden_features)
        print("=======================>")

        self.reconstruction_layers = []
        for i in range(len(self.sizes)-1):
            self.reconstruction_layers.append(nn.Linear(self.sizes[-(i+1)], self.sizes[-(i+2)]))
        self.reconstruction_layers = nn.ModuleList(self.reconstruction_layers)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward_layers(self, data, layers, relu=True):
        h = data
        for i in range(len(layers)):
            if relu:
                h = self.relu(layers[i](h))
            else:
                h = layers[i](h)
        return h

    def encode(self, RNA, methylation, miRNA):
        h_rna = self.forward_layers(RNA, self.rna_layers)
        h_methy = self.forward_layers(methylation, self.methy_layers)
        h_mirna = self.forward_layers(miRNA, self.mirna_layers)
        tmp = torch.cat((h_rna, h_methy, h_mirna), dim=1)
        h = self.forward_layers(tmp, self.whole_layers, relu=True)
        mu, logvar = self.mu(h), self.logvar(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.forward_layers(z, [self.reconstruction_layers[i] for i in range(len(self.reconstruction_layers)-1)], relu=True)
        x = self.sigmoid(self.reconstruction_layers[-1](h))
        return x

    def forward(self, RNA, methylation, miRNA):
        mu, logvar = self.encode(RNA, methylation, miRNA)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
