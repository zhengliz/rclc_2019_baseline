```python
import einb
from data_utils import load_rclc_corpus, read_pub_text 
from eval_utils import top5UptoD_err, top5UptoD_prec
from sentence_filtering import SentenceFilterClass
from statistics import mean 
from urllib.parse import urlparse

```

## Load RCLC Corpus


```python
TEXT_PATH = '../data/resource/pubs/text/'
corpus = load_rclc_corpus('../data/corpus.jsonld', 
                          '../data/resource/dataset/html/',
                          '../data/resource/pubs/json/')
datasets = corpus['datasets']
pubs = corpus['pubs']
sent_filter = SentenceFilterClass()
```

## Let's prepare the publications for our experiment. 

We extract abstract, content, and references from the publications.


```python
pub_contexts = []
pub_labels = []
for pub in pubs:
    context = [pub['dct:title']['@value'], pub['dct:publisher']['@value']]
    parsed = pub['parsed_pub']
    metadata = parsed['metadata']
    if metadata['abstractText']:
        context.append(metadata['abstractText'])

    if metadata['sections']:
        s_text = [sec['text'] for sec in metadata['sections'] if len(sec['text']) > 0]
    else:
        _id = urlparse(pub['@id']).fragment.split('-')[1]
        s_text = read_pub_text(TEXT_PATH + _id + '.pdf.txt')
    context.extend(s_text)
    
    if metadata['references']:
        ref_titles = [ref['title'] for ref in metadata['references']]
    context.extend(ref_titles)
    
    filtered_sent = sent_filter.final_approach(' '.join(context))
    pub_contexts.append(' '.join(filtered_sent))
    
    
    citations = pub['cito:citesAsDataSource']
    if isinstance(citations, list):
        pub_label = [c['@id'] for c in citations]
    elif isinstance(citations, dict):
        pub_label = [citations['@id']]
    pub_labels.append(pub_label)
    
print(f'Number of publications: {len(pub_contexts)}')
```

    Number of publications: 480


## First, we want to see whether our model can overfit the corpus

We train the baseline model on all publications in the corpus, then we predict the same set of publications. The baseline model is based on our [rcc submission](https://github.com/LARC-CMU-SMU/coleridge-rich-context-larc). 


```python
# Train a model
dataset_word_dict, word_dataset_dict = einb.parameter_learn(pub_contexts,pub_labels)

# Perform prediction
errs = []
for i, context in enumerate(pub_contexts):
    # print(f'{pub_labels[i]}')
    preds = einb.predict(context, dataset_word_dict, word_dataset_dict, 5)
    # print(f'{preds}')
    errs.append(top5UptoD_err(pub_labels[i], [p[0] for p in preds]))
print(f'top5UptoD error rate: {mean(errs)}')
```

    top5UptoD error rate: 0.11086805555555555


## Now, We conduct 5-fold Cross Validation Evaluation


```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=2019)
cv_errs = []
iter = 0
for train_index, test_index in kf.split(pub_contexts):
    print(f'Fold {iter}')
    X_train = [pub_contexts[i] for i in train_index]
    X_test = [pub_contexts[i] for i in test_index]
    y_train = [pub_labels[i] for i in train_index]
    y_test = [pub_labels[i] for i in test_index]
    
    dataset_word_dict, word_dataset_dict = einb.parameter_learn(X_train,y_train)
    errs = []
    for i, context in enumerate(X_test):
        # print(f'{y_test[i]}')
        preds = einb.predict(context, dataset_word_dict, word_dataset_dict, 5)
        # print(f'{preds}')
        errs.append(top5UptoD_err(y_test[i], [p[0] for p in preds]))
    cv_errs.append(mean(errs))
    print(f'top5UptoD error rate: {mean(errs)}')
    iter += 1

print(f'Average top5UptoD error rate: {mean(cv_errs)}')
```

    Fold 0
    top5UptoD error rate: 0.3072916666666667
    Fold 1
    top5UptoD error rate: 0.23663194444444444
    Fold 2
    top5UptoD error rate: 0.17708333333333334
    Fold 3
    top5UptoD error rate: 0.1876736111111111
    Fold 4
    top5UptoD error rate: 0.17326388888888888
    Average top5UptoD error rate: 0.21638888888888888



```python

```
