import numpy as np
from collections import defaultdict


def parameter_learn(filtered_sentences, labels):
    """
    training script
    filtered_sentences <List> : filtered sentences for each publication
    labels <List> : ground truth datasets for each publications

    eg :
    filtered_sentences[i] = 'Ta-Feng dataset includes 10000 transactional data'
    labels[i] = ['Ta-Feng_2008', 'Ta-Feng_2010'] (all the datasets in the ith
    publication)
    """
    dataset_word_dict = defaultdict(lambda: defaultdict(lambda: 0))
    word_dataset_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for sent, datasets in zip(filtered_sentences, labels):
        for dataset in datasets:
            for w in sent.split(' '):
                if dataset_word_dict[dataset][w] == 0:
                    dataset_word_dict[dataset]['U_COUNT'] += 1
                if word_dataset_dict[w][dataset] == 0:
                    word_dataset_dict[w]['U_COUNT'] += 1

                dataset_word_dict[dataset][w] += 1
                dataset_word_dict[dataset]['COUNT'] += 1
                word_dataset_dict[w][dataset] += 1
                word_dataset_dict[w]['COUNT'] += 1
    return dataset_word_dict, word_dataset_dict


def predict(filtered_sentence, dataset_word_dict, word_dataset_dict,
            top_k=5):
    """
    prediction step
    filtered_sentence <str> : filtered sentences for a test instance
    """
    predictions = {}
    datasets = len(list(dataset_word_dict.keys()))

    for dataset in dataset_word_dict:
        temp_score = 0
        for w in filtered_sentence.split(' '):
            if word_dataset_dict[w]['U_COUNT']:
                temp_score += np.log(1 + datasets/word_dataset_dict[w]['U_COUNT']) * \
                np.log((dataset_word_dict[dataset][w] + 1)/(word_dataset_dict[w]['COUNT'] + datasets))
        predictions[dataset] = temp_score

    sorted_preds = sorted(predictions.items(),
                          key=lambda kv: kv[1],
                          reverse=True)
    return sorted_preds[:top_k]
