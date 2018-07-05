from __future__ import division
from __future__ import print_function
from lifelines.statistics import logrank_test, pairwise_logrank_test,multivariate_logrank_test
from sklearn import mixture, manifold, cluster
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from rpy2.robjects import IntVector, Formula
from rpy2.robjects.packages import importr
surv = importr('survival')
stats = importr('stats')

def build_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--disease', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    # Model parameter
    parser.add_argument('--sub-depth', type=int, default=2,
                        help='how many layers for subnetwork')
    parser.add_argument('--rna-decay', type=int, default=20,
                        help='how decay for subnetwork')
    parser.add_argument('--methy-decay', type=int, default=20,
                        help='how decay for subnetwork')
    parser.add_argument('--mirna-decay', type=int, default=2,
                        help='how decay for subnetwork')

    parser.add_argument('--whole-depth', type=int, default=2,
                        help='how many layers for integrative network')
    parser.add_argument('--whole-decay', type=int, default=2,
                        help='how decay for integrative network')

    return parser.parse_args()

def cox(hidden, survival, epoch, method='MG'):
    def clustering(hidden, method):
        if method == 'KNN':
            clf = cluster.KMeans(n_clusters=3)
            clf.fit(hidden)
            return clf.predict(hidden)

        if method == 'MG':
            clf = mixture.BayesianGaussianMixture(n_components=10,n_init=10,max_iter=500, covariance_type='full')
            clf.fit(hidden)
            return clf.predict(hidden)

    predicts = clustering(hidden, method)
    T1, E1, G1 = [], [], []

    #print(predicts)
    unique, counts = np.unique(predicts, return_counts=True)

    #print(np.asarray((unique, counts/predicts.shape[0])).T)
    # easier to grep
    clusters = np.asarray((unique, counts/predicts.shape[0])).T
    for i in clusters:
        print('Epoch {} # CLUSTER: {} '.format(epoch, i))

    sscore = silhouette_score(hidden, predicts)
    print('Epoch {}  # SILHOUTE:  {}'.format(epoch, sscore) )
    for i in range(len(predicts)):
        T1.append(survival[i,-2])
        E1.append(survival[i,-1])
    #temp = np.array(T1)
    #print(temp)
    #print(temp.astype(int))
    # print(T1)
    # print('\n')
    # print(E1)
    # print('\n')
    # print(predicts)
    info = pd.DataFrame({'status':T1,'survive':E1,'clusters':predicts})
    ratio = 0
    formula = Formula('x~y')
    env = formula.environment
    env['y'] = IntVector(predicts)
    env['x'] = surv.Surv(IntVector(np.array(T1).astype(int)),IntVector(np.array(E1).astype(int)))
    result = surv.survdiff(formula)
    p_value = 1 - np.array(stats.pchisq(result[4],len(set(predicts)) -1))
    # R pvalue

    #result = multivariate_logrank_test(np.array(T1),  np.array(predicts).astype(int), np.array(E1) )
    
    #p_value = result.p_value
    return p_value, ratio, info

def visualize(hidden, labels, outfp='tmp.png'):
    clf = manifold.TSNE(n_components=2)
    X = clf.fit_transform(hidden)
    data = np.concatenate([labels, X], axis=1)
    data = pd.DataFrame(data, columns=['group', 'c1', 'c2'])
    sns.lmplot('c1', 'c2', data=data, fit_reg=False, hue='group', legend=False)
    sns.despine(top=False, right=False)
    plt.legend(loc=1)
    plt.ylabel('Component 1', fontweight='bold')
    plt.xlabel('Component 2', fontweight='bold')
    plt.tight_layout()
    plt.savefig(outfp)
    plt.close()
