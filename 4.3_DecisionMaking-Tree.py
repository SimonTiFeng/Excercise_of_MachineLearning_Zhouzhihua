import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.optimize as opt

data = pd.read_csv('Excercise_of_MachineLearning_Zhouzhihua/table_4.2.csv')

# data mapping
color_mapping = {'青绿': 1, '乌黑': 0, '浅白': 2}
root_mapping = {'蜷缩': 1, '稍蜷': 2, '硬挺': 3}
sound_mapping = {'浊响': 1, '沉闷': 2, '清脆': 3}
texture_mapping = {'清晰': 1, '稍糊': 2, '模糊': 3}
umbilicus_mapping = {'凹陷': 1, '稍凹': 2, '平坦': 3}
touch_mapping = {'硬滑': 1, '软粘': 2}
good_bad_mapping = {'好瓜': 1, '坏瓜': 0}

# data replacement
data['色泽'] = data['色泽'].replace(color_mapping)
data['根蒂'] = data['根蒂'].replace(root_mapping)
data['敲声'] = data['敲声'].replace(sound_mapping)
data['纹理'] = data['纹理'].replace(texture_mapping)
data['脐部'] = data['脐部'].replace(umbilicus_mapping)
data['触感'] = data['触感'].replace(touch_mapping)
data['好坏'] = data['好坏'].replace(good_bad_mapping)

def discretize(column, q):
    return pd.qcut(column, q=q, labels=False) + 1

data['密度'] = discretize(data['密度'], q=3)
data['含糖率'] = discretize(data['含糖率'], q=3)

y = data['好坏']
X = data.drop('编号', axis=1)
X = X.drop('好坏', axis=1)
print(X)


def entropy(y):
    unique_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X, y, feature):
    H_y = entropy(y)
    feature_values, feature_counts = np.unique(X[feature], return_counts=True)
    probabilities = feature_counts / len(X)
    cond_entropy = 0
    for value, prob in zip(feature_values, probabilities):
        subset_y = y[X[feature] == value]
        cond_entropy += prob * entropy(subset_y)
    return H_y - cond_entropy

feature_names = X.columns.tolist()

def best_split(X, y):
    best_gain = 0
    best_feature = None
    for feature in X.columns:
        gain = information_gain(X, y, feature)
        if gain > 0:
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
    return best_gain, best_feature

class TreeNode:
    def __init__(self, feature_name=None, branches=None, *, value=None):
        self.feature_name = feature_name  
        self.branches = branches if branches is not None else []  
        self.value = value  

    def is_leaf_node(self):
        return self.value is not None  

def build_tree(X, y, depth=0, max_depth=3):
    num_samples, num_features = X.shape
    num_labels = len(np.unique(y))
    
    if depth >= max_depth or num_labels == 1:
        leaf_value = np.argmax(np.bincount(y))
        return TreeNode(value=leaf_value)
    
    _, best_feature = best_split(X, y)
    
    if best_feature is None:
        leaf_value = np.argmax(np.bincount(y))
        return TreeNode(value=leaf_value)
    
    branches = []
    unique_values = np.unique(X[best_feature])
    
    for value in unique_values:
        subset_indices = X[best_feature] == value
        branch_subtree = build_tree(X[subset_indices], y[subset_indices], depth + 1, max_depth)
        branches.append((value, branch_subtree))  
    
    return TreeNode(feature_name=best_feature, branches=branches)


def predict(tree, sample):
    if tree.is_leaf_node():
        return tree.value
    branch = next(branch for value, branch in tree.branches if sample[tree.feature_name] == value)
    return predict(branch, sample)

def calculate_accuracy(tree, X, y):
    predictions = [predict(tree, x) for _, x in X.iterrows()]
    return np.mean(predictions == y)

tree = build_tree(X, y, max_depth=3)

accuracy = calculate_accuracy(tree, X, y)
print(f'模型在训练集上的准确率为: {accuracy * 100:.2f}%')

def print_tree(node, depth=0):
    if node.is_leaf_node():
        print("\t" * depth + f"Leaf: {node.value}")
    else:
        for value, branch in node.branches:
            print("\t" * depth + f"[{node.feature_name} == {value}]")
            print_tree(branch, depth + 1)

print_tree(tree)