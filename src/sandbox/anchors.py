import sklearn
import sklearn.ensemble
from sklearn.feature_extraction.text import CountVectorizer
from anchor import utils, anchor_tabular, anchor_text
from os import path
import numpy as np
import spacy

# https://github.com/marcotcr/anchor
# https://christophm.github.io/interpretable-ml-book/anchors.html

def load_polarity():
    data = []
    labels = []
    f_names = ['rt-polarity.neg', 'rt-polarity.pos']
    for (l, f) in enumerate(f_names):
        for line in open(path.join(path.dirname(__file__), "../../data/sentiment-sentences", f), 'rb'):
            try:
                line.decode('utf8')
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels


def tabular():
    # Get dataset and train explainer
    dataset = utils.load_dataset("adult", balance=True,
                                 dataset_folder=path.join(path.dirname(__file__), "../../data/"))
    explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names,
                                                      dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)

    # Train black box model
    c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
    c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    predict_fn = lambda x: c.predict(explainer.encoder.transform(x))
    print(f"Train: {sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train))}")
    print(f"Test: {sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test))}")

    # Get explanation for sample
    idx = 0
    np.random.seed(1)
    print(f"Prediction: ", explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
    exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)
    print(f"Anchor: {(' AND '.join(exp.names()))}")
    print(f"Precision: {exp.precision()}")
    print(f"Coverage: {exp.coverage()}")

    # Calculate explanation coverage and precision on test set
    fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
    print(f"Anchor test coverage: {fit_anchor.shape[0]/float(dataset.test.shape[0])}")
    print(f"Anchor test precision: {np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1)))}")


def text():
    nlp = spacy.load("en_core_web_lg")

    data, labels = load_polarity()
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2,
                                                                                      random_state=42)
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1,
                                                                                    random_state=42)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)

    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)
    test_vectors = vectorizer.transform(test)
    val_vectors = vectorizer.transform(val)

    c = sklearn.linear_model.LogisticRegression()
    # c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=10)
    c.fit(train_vectors, train_labels)
    preds = c.predict(val_vectors)
    print('Val accuracy', sklearn.metrics.accuracy_score(val_labels, preds))

    def predict_lr(texts):
        return c.predict(vectorizer.transform(texts))

    explainer = anchor_text.AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)

    np.random.seed(1)
    text = 'This is a good book .'
    pred = explainer.class_names[predict_lr([text])[0]]
    alternative = explainer.class_names[1 - predict_lr([text])[0]]
    print('Prediction: %s' % pred)
    exp = explainer.explain_instance(text, predict_lr, threshold=0.95, use_proba=True)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))


if __name__ == '__main__':
    #tabular()
    text()
