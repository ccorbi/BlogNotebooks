from __future__ import print_function
import argparse
import os.path as ops
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import fusion_VAE
from sklearn import preprocessing
import numpy as np
from utils import cox, visualize, build_args
import random


def evaluate(epoch):
    model.eval()
    sub_RNA, sub_methy, sub_miRNA = wrapper(RNA), wrapper(methylation), wrapper(miRNA)
    recon_batch,mu, logvar = model(sub_RNA, sub_methy, sub_miRNA)
    if args.cuda:
        hidden = mu.cpu().data.numpy()
    else:
        hidden = mu.data.numpy()

    if survival.shape[-1] == 3:
        p_value, ratio, surv_info = cox(hidden, survival)
        print('Epoch %d ====> Cox P-value: %.7f' % (epoch, p_value))
        surv_info.to_csv('./temp/survie_info_{}.csv'.format(epoch), index=False)
    else:
        # TO-DO: for all data
        types = set(survival[:, -1].tolist())
        for type_ in types:
            indices = np.where(survival[:, -1] == type_)[0]
            p_value, ratio, surv_info= cox(hidden[indices], survival[indices])
            print('Epoch %d ====> Type: %s, Cox P-value: %.7f' % (epoch, type_, p_value))
        visualize(hidden, survival[:, -1], outfp='%d.png' % (epoch))
    return p_value


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD, BCE, KLD


def tcga_load(disease):
    def read(fp):
        return np.loadtxt(fp, delimiter='\t', dtype=str, skiprows=1)[:, 1:-1].T.astype(float)

    RNA = 'data/%s/%s_Gene_Expression.txt' % (disease, disease)
    methylation = 'data/%s/%s_Methy_Expression.txt' % (disease, disease)
    miRNA = 'data/%s/%s_Mirna_Expression.txt' % (disease, disease)
    RNA, methylation, miRNA = read(RNA), read(methylation), read(miRNA)
    survival = np.loadtxt('data/%s/%s_Survival.txt' % (disease, disease), delimiter='\t', skiprows=1, dtype=str)
    return RNA, methylation, miRNA, survival


def preprocess(data):
    return preprocessing.minmax_scale(data)
    #return preprocessing.normalize(data, norm='l2')

def wrapper(data):
    data = torch.from_numpy(data)
    data = Variable(data).type(torch.FloatTensor)
    if args.cuda:
        data = data.cuda()
    return data

def train(epoch):
    model.train()
    train_loss = 0
    indices = np.arange(num_patients)
    for i in range(int(num_patients / args.batch_size)):
        sub_indices = indices[(i * args.batch_size): ((i+1) * args.batch_size)]
        sub_RNA, sub_methy, sub_miRNA = RNA[sub_indices], methylation[sub_indices], miRNA[sub_indices]
        sub_data = Variable(torch.from_numpy(np.concatenate([sub_RNA, sub_methy, sub_miRNA], axis=1)).type(torch.FloatTensor))
        if args.cuda:
            sub_data = sub_data.cuda()
        sub_RNA, sub_methy, sub_miRNA = wrapper(sub_RNA), wrapper(sub_methy), wrapper(sub_miRNA)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(sub_RNA, sub_methy, sub_miRNA)
        loss, BCE, KLD = loss_function(recon_batch, sub_data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tBCELoss: {}'.format(
                epoch, (i + 1) * args.batch_size, num_patients,
                100. * (i + 1)/ int(num_patients / args.batch_size),
                loss.data[0] / num_patients, BCE))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / num_patients))


# init
args = build_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(0)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# load data

RNA, methylation, miRNA, survival = tcga_load(args.disease)
RNA, methylation, miRNA = preprocess(RNA), preprocess(methylation), preprocess(miRNA)
num_patients = len(RNA)

# init model
model = fusion_VAE(RNA, methylation, miRNA, sub_depth=args.sub_depth, \
    whole_depth=args.whole_depth, rna_decay=args.rna_decay, methy_decay=args.methy_decay, mirna_decay=args.mirna_decay, whole_decay=args.whole_decay, cuda=args.cuda)

if args.cuda:
    print("Using CUDA")
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

optimizer = optim.Adam(model.parameters(), lr=1e-3)


lowest_pval = 1
lowest_epoch = 0

# training

for epoch in range(1, args.epochs + 1):
    train(epoch)
    p_value = evaluate(epoch)
    if p_value < lowest_pval:
        lowest_pval = p_value
        lowest_epoch = epoch
    print("lowet_pval:",lowest_pval,"at epoch:",lowest_epoch)

print(lowest_pval)
