import colorcet as cc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import random 
from sklearn.metrics.pairwise import cosine_similarity


def visualize_action_features():

    actions = {
        1:	'bow',					
        2:	'brushing hair',					
        3:	'brushing teeth',					
        4:	'check time (from watch)',					
        5:	'cheer up',					
        6:	'clapping',					
        7:	'cross hands in front (say stop)',					
        8:	'drink water',					
        9:	'drop',					
        10:	'eat meal/snack',					
        11:	'falling',					
        12:	'giving something to other person',					
        13:	'hand waving',					
        14:	'handshaking',					
        15:	'hopping (one foot jumping)',					
        16:	'hugging other person',					
        17:	'jump up',					
        18:	'kicking other person',					
        19:	'kicking something',					
        20:	'make a phone call/answer phone',					
        21:	'pat on back of other person',					
        22:	'pickup',					
        23:	'playing with phone/tablet',					
        24:	'point finger at the other person',					
        25:	'pointing to something with finger',					
        26:	'punching/slapping other person',					
        27:	'pushing other person',					
        28:	'put on a hat/cap',					
        29:	'put something inside pocket',					
        30:	'reading',					
        31:	'rub two hands together',					
        32:	'salute',					
        33:	'sitting down',					
        34:	'standing up',					
        35:	'take off a hat/cap',					
        36:	'take off glasses',					
        37:	'take off jacket',					
        38:	'take out something from pocket',					
        39:	'taking a selfie',					
        40:	'tear up paper',					
        41:	'throw',					
        42:	'touch back (backache)',					
        43:	'touch chest (stomachache/heart pain)',					
        44:	'touch head (headache)',					
        45:	'touch neck (neckache)',					
        46:	'typing on a keyboard',					
        47:	'use a fan (with hand or paper)/feeling warm',					
        48:	'wear jacket',					
        49:	'wear on glasses',					
        50:	'wipe face',					
        51:	'writing'					
    }
    indicies = sorted([str(x) for x in range(1, 52)])

    features_file = '/home/praveen/Research/MultiView_Actions/features/PKcsv1-act.pkl'

    outputs = pickle.load(open(features_file, 'rb'))
    labels, features = list(outputs.keys()), list(outputs.values())
    features = np.array(features).squeeze()

    action_labels = np.array([label.split('_')[2] for label in labels])
    view_labels = np.array([label.split('_')[-1] for label in labels])

    action_classes, counts = np.unique(action_labels, return_counts=True)

    action_classes = action_classes[np.argsort(counts)[::-1]]

    action_classes = action_classes[:10]

    action_indicies = [i for i in range(len(action_labels)) if action_labels[i] in action_classes]

    _action_labels = [actions[indicies.index(lbl)] for lbl in action_labels[action_indicies]]
    _features = features[action_indicies]

    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(_features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    features_tsne = tsne.fit_transform(features_pca)

    data_frame = pd.DataFrame(data={'tsne-2d-one': features_tsne[:, 0], 'tsne-2d-two': features_tsne[:, 1], 'Class': _action_labels})

    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
        hue='Class',
        palette=sns.color_palette(cc.glasbey, n_colors=10),
        data=data_frame,
        legend='brief',
        alpha=0.9
    )

    h,l = ax.get_legend_handles_labels()
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.legend(h[1:],l[1:], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('tsne_action.png')


def visualize_view_features():

    features_file = '/home/praveen/Research/MultiView_Actions/features/PKcsv1-view.pkl'

    outputs = pickle.load(open(features_file, 'rb'))
    labels, features = list(outputs.keys()), list(outputs.values())
    features = np.array(features).squeeze()

    action_labels = np.array([label.split('_')[2] for label in labels])
    view_labels = np.array([label.split('_')[-1] for label in labels])

    action_classes, counts = np.unique(action_labels, return_counts=True)

    action_classes = action_classes[np.argsort(counts)[::-1]]

    action_classes = action_classes[:10]

    action_indicies = [i for i in range(len(action_labels)) if action_labels[i] in action_classes]

    _action_labels = action_labels[action_indicies]
    _view_labels = view_labels[action_indicies]
    _features = features[action_indicies]

    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(_features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    features_tsne = tsne.fit_transform(features_pca)

    data_frame = pd.DataFrame(data={'tsne-2d-one': features_tsne[:, 0], 'tsne-2d-two': features_tsne[:, 1], 'Class': _view_labels})

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(x='tsne-2d-one', y='tsne-2d-two',
        hue='Class',
        palette=sns.color_palette(cc.glasbey, n_colors=3),
        data=data_frame,
        legend='brief',
        alpha=0.9
    )

    h,l = ax.get_legend_handles_labels()
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.legend(h[1:],l[1:], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('tsne_view.png')


def visualize_distances():
    actions = {
        1:	'bow',					
        2:	'brushing hair',					
        3:	'brushing teeth',					
        4:	'check time (from watch)',					
        5:	'cheer up',					
        6:	'clapping',					
        7:	'cross hands in front (say stop)',					
        8:	'drink water',					
        9:	'drop',					
        10:	'eat meal/snack',					
        11:	'falling',					
        12:	'giving something to other person',					
        13:	'hand waving',					
        14:	'handshaking',					
        15:	'hopping (one foot jumping)',					
        16:	'hugging other person',					
        17:	'jump up',					
        18:	'kicking other person',					
        19:	'kicking something',					
        20:	'make a phone call/answer phone',					
        21:	'pat on back of other person',					
        22:	'pickup',					
        23:	'playing with phone/tablet',					
        24:	'point finger at the other person',					
        25:	'pointing to something with finger',					
        26:	'punching/slapping other person',					
        27:	'pushing other person',					
        28:	'put on a hat/cap',					
        29:	'put something inside pocket',					
        30:	'reading',					
        31:	'rub two hands together',					
        32:	'salute',					
        33:	'sitting down',					
        34:	'standing up',					
        35:	'take off a hat/cap',					
        36:	'take off glasses',					
        37:	'take off jacket',					
        38:	'take out something from pocket',					
        39:	'taking a selfie',					
        40:	'tear up paper',					
        41:	'throw',					
        42:	'touch back (backache)',					
        43:	'touch chest (stomachache/heart pain)',					
        44:	'touch head (headache)',					
        45:	'touch neck (neckache)',					
        46:	'typing on a keyboard',					
        47:	'use a fan (with hand or paper)/feeling warm',					
        48:	'wear jacket',					
        49:	'wear on glasses',					
        50:	'wipe face',					
        51:	'writing'					
    }
    action_indicies = sorted([str(x) for x in range(1, 52)])

    pca = PCA(n_components=32)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    
    action_features_file = '/home/praveen/Research/MultiView_Actions/features/PKcsv1-act.pkl'

    outputs = pickle.load(open(action_features_file, 'rb'))
    action_labels, action_features = list(outputs.keys()), list(outputs.values())
    action_features = np.array(action_features).squeeze()
    action_features = pca.fit_transform(action_features)
    action_features = tsne.fit_transform(action_features)

    view_features_file = '/home/praveen/Research/MultiView_Actions/features/PKcsv1-view.pkl'

    outputs = pickle.load(open(view_features_file, 'rb'))
    view_labels, view_features = list(outputs.keys()), list(outputs.values())
    view_features = np.array(view_features).squeeze()
    view_features = pca.fit_transform(view_features)
    view_features = tsne.fit_transform(view_features)

    action_classes = np.array([label.split('_')[2] for label in action_labels])
    action_classes, counts = np.unique(action_classes, return_counts=True)
    action_classes = action_classes[np.argsort(counts)[::-1]]
    action_classes = action_classes[:10]

    outputs = {}
    assert len(view_labels) == len(action_labels)
    for i in range(len(action_labels)):
        action_label = action_labels[i]
        if not action_label.split('_')[2] in action_classes:
            continue
        assert action_label in view_labels
        view_label_indx = view_labels.index(action_label)
        action_feature = action_features[i]
        view_feature = view_features[view_label_indx]
        outputs[action_label] = (action_feature, view_feature)
    
    action_features = [v[0] for k, v in outputs.items()]
    view_features = [v[1] for k, v in outputs.items()]
    labels = [k for k, v in outputs.items()]
    action_distances = cosine_similarity(action_features)
    view_distances = cosine_similarity(view_features)
    
    for i in range(len(labels)):
        central_node_index = i
        distances = np.stack([action_distances[central_node_index], view_distances[central_node_index]], axis=1)
        _action_labels = [actions[action_indicies.index(lbl)] for lbl in np.array([label.split('_')[2] for label in labels])]
        _view_labels = np.array([label.split('_')[-1] for label in labels])

        data_frame = pd.DataFrame(data={'Action Distance': distances[:, 0], 'View Distance': distances[:, 1], 'Class': _action_labels, 'View': _view_labels})

        plt.figure(figsize=(12, 6))
        markers = {"R": "o", "M": "X", "L": "P"}

        ax = sns.scatterplot(x='Action Distance', y='View Distance',
            hue='Class',
            style='View',
            markers=markers,
            palette=sns.color_palette(cc.glasbey, n_colors=10),
            data=data_frame,
            legend='brief',
            alpha=0.9
        )

        h,l = ax.get_legend_handles_labels()
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.set_ylabel('')
        ax.set_xlabel('')
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        l1 = ax.legend(h[1:-4],l[1:-4], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, prop={'size': 14})
        l2 = ax.legend(h[-4:],l[-4:], loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))
        ax.add_artist(l1)

        #plt.legend(h[1:],l[1:], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4, prop={'size': 14})
        plt.tight_layout()
        plt.savefig('./outputs/distance_visualization_' + str(i) + '.png')
        plt.close()
        


if __name__ == '__main__':
    #visualize_action_features()
    #visualize_view_features()
    visualize_distances()
    