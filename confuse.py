import os
from collections import defaultdict
import utils
import numpy as np
import matplotlib.pyplot as plt

label2index = {'Social sciences and society': 0, 'Sports and recreation': 1, 'Natural sciences': 2, 'Language and literature': 3, 'Geography and places': 4, 'Music': 5, 'Media and drama': 6, 'Art and architecture': 7, 'Warfare': 8, 'Engineering and technology': 9, 'Video games': 10, 'Philosophy and religion': 11, 'Agriculture, food and drink': 12, 'History': 13, 'Mathematics': 14, 'Miscellaneous': 15}

shortlabel2index = {'Social': 0, 'Sports': 1, 'Sciences': 2, 'Literature': 3, 'Geography': 4, 'Music': 5, 'Media': 6, 'Art': 7, 'Warfare': 8, 'Engineering': 9, 'Video': 10, 'Philosophy': 11, 'Agriculture': 12, 'History': 13, 'Mathematics': 14, 'Misc': 15}

naming = {'lo_fast_ft' : 'lower-case fasttext', 'lo_glove_ft' : 'GloVe', 'lo_rand_ft' : 'lower-case random embed', 'lo_w2v_ft' : 'lower-case word2vec', 'nb' : 'Naive Bayes', 'up_fast_ft' : 'Fasttext', 'up_rand_ft' : 'Random embed', 'up_w2v_ft' : 'Word2vec'}

index2label = {}
for l, i in shortlabel2index.items():
    index2label[i] = l

correct_labels = []
with open("./data/topicclass/topicclass_valid.txt", 'r') as f:
    for line in f:
        lab = line.split("|||")[0].strip()
        correct_labels.append(lab)

all_labels = set(correct_labels)

m = 11

pred_dir = "predict/"
for model in os.listdir(pred_dir):
    print(model)
    fname = os.path.join(pred_dir, model, "valid_preds")
    pred_labels = open(fname, 'r').readlines()
    pred_labels = [x.strip() for x in pred_labels]
    wrongs = np.zeros((16, 16))
    for c, p in zip(correct_labels, pred_labels):
        if c != p:
            wrongs[label2index[c], label2index[p]] += 1
    wrongs /= 11
    wrongs = 1 - wrongs

    
    #  plt.figure(figsize=(7, 7))
    plt.imshow(wrongs, cmap='gray')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_yticks(np.arange(0, 16, 1))
    ax.set_xticklabels([index2label[i] for i in range(16)], rotation='vertical')
    ax.set_yticklabels([index2label[i] for i in range(16)])
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    plt.title(naming[model])
    plt.tight_layout()
    plt.savefig(model + ".png")
