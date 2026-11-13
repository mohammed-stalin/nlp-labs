# Lab 3 - Short-Answer Grading and Sentiment Classification

[Back to all labs](../README.md)

## Abstract

This lab studies supervised learning from text through two tasks: predicting numerical short-answer scores and classifying tweet sentiment. It compares sparse lexical features with averaged Word2Vec embeddings. The regression notebook contains saved evaluation results; the classification notebook defines experiments but does not contain completed metric outputs.

## Objectives and data

| File | Role |
| --- | --- |
| [ex1.ipynb](ex1.ipynb) | Short-answer preprocessing, regression, evaluation, and plots |
| [answers.csv](answers.csv) | 2,442 records with `id`, `answer`, `score`, and `correct` fields |
| [ex2.ipynb](ex2.ipynb) | Tweet preprocessing, representation learning, and classification code |
| [twitter_training.csv](twitter_training.csv) | Training tweets with ID, entity/topic, sentiment, and text fields |
| [twitter_validation.csv](twitter_validation.csv) | Separate supplied validation tweets |

The grading experiment predicts `score` from `answer`; it does not incorporate the `correct` field as an answer-reference feature. Both tasks are supervised prediction experiments rather than next-token language modeling.

## Methodology

### Short-answer score regression

Answers are lowercased and tokenized, punctuation and English stop words are removed, and Porter stemming is followed by WordNet lemmatization. The notebook builds four feature representations: bag-of-words, TF-IDF, and mean word vectors from 100-dimensional Word2Vec CBOW and Skip-gram models. The Word2Vec configurations use a context window of five and `min_count=1`.

Each representation is split into 80% training and 20% testing with `random_state=42`. Support vector regression, linear regression, and decision-tree regression are evaluated with MSE and RMSE. Later cells also compute MAE, R^2, and MAPE and plot the comparisons.

### Tweet sentiment classification

Tweets are cleaned by removing URLs, mentions, hashtags, and punctuation, followed by English stop-word removal, lemmatization, and stemming. The training data is split 80/20 with seed 42. Word2Vec is learned on the training split; bag-of-words and TF-IDF are likewise fitted on training text.

The code defines SVM, multinomial Naive Bayes, logistic regression, and AdaBoost classifiers. Evaluation functions return accuracy, weighted F1, and a classification report. The final results dictionary includes logistic-regression and AdaBoost combinations; SVM entries are commented out. The supplied validation CSV is loaded but is not used for the final evaluation in the current code.

## Recorded regression results

The table below reproduces the first MSE/RMSE comparison cell, rounded to four decimals. Lower values indicate smaller prediction errors.

| Representation | Regressor | MSE | RMSE |
| --- | --- | ---: | ---: |
| TF-IDF | SVR | **0.9578** | **0.9787** |
| TF-IDF | Linear regression | 2.6744 | 1.6354 |
| TF-IDF | Decision tree | 1.6899 | 1.3000 |
| CBOW | SVR | 1.7285 | 1.3147 |
| CBOW | Linear regression | 1.2975 | 1.1391 |
| CBOW | Decision tree | 1.8476 | 1.3593 |
| Skip-gram | SVR | 1.4878 | 1.2198 |
| Skip-gram | Linear regression | 1.2330 | 1.1104 |
| Skip-gram | Decision tree | 1.7355 | 1.3174 |
| Bag-of-words | SVR | 1.1196 | 1.0581 |
| Bag-of-words | Linear regression | 5.1984 | 2.2800 |
| Bag-of-words | Decision tree | 1.6574 | 1.2874 |

**SVR with TF-IDF is the strongest combination in this saved comparison.** Its additional outputs show MAE **0.6883** and R^2 **0.2511**. The positive but modest R^2 suggests substantial room for improvement. Later evaluation cells refit the unseeded decision trees and produce somewhat different scores.

The classification notebook contains a saved `KeyboardInterrupt` in its initial imports and no completed accuracy/F1 outputs. A best sentiment classifier cannot be selected from the committed evidence.

## Limitations and improvements

- In the regression notebook, vectorizers and Word2Vec are fitted before the split. This exposes feature learning to test text and limits the strength of the reported comparison. Split raw text first and fit feature extraction only on training data.
- MAPE is extremely large in the saved output and is unsuitable for interpreting errors when target scores include zero or values near zero. Prefer MAE, RMSE, and R^2 here.
- Word2Vec training and decision trees are not fully seeded; a single random split does not quantify result variability.
- Tweet CSVs have no header, but the pandas reads use default header inference. A reproduction should use `header=None` with explicit column names to retain the first record.
- Multinomial Naive Bayes requires nonnegative inputs, while Word2Vec vectors may contain negative values. The embedding-based Naive Bayes calls need revision or exclusion before a complete classification run.

## Reproduction notes

Open Jupyter from this folder. Imports include `pandas`, `numpy`, `nltk`, `spacy`, `gensim`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib`. The regression notebook loads `en_core_web_sm`, and the notebooks download NLTK tokenizer, stop-word, and WordNet resources. No pinned environment is supplied; library versions can affect compatibility and results. Metrics above are historical saved outputs, not a fresh execution.

## Conclusion

The recorded regression experiment favors TF-IDF with SVR over the tested alternatives. A stronger conclusion requires leakage-free feature fitting and repeated evaluation. The sentiment experiment provides a useful starting pipeline, with execution and model-input issues still to resolve before reporting results.
