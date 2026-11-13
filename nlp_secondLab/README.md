# Lab 2 - Rule-Based Extraction and Text Representations

[Back to all labs](../README.md)

**Author:** Mohammed AACHABI - **Course:** MST IASD

## Abstract

This lab explores two ways to turn text into usable information: extracting purchase details with explicit rules, and representing Arabic text numerically for NLP. A supplementary Spark notebook compares sentiment classifiers. The exercises connect interpretable extraction rules with sparse features and learned word vectors.

## Objectives and files

| Notebook | Experiment |
| --- | --- |
| [ex1.ipynb](ex1.ipynb) | Extract quantities, product descriptions, and prices; generate an itemized bill |
| [ex2.ipynb](ex2.ipynb) | Arabic preprocessing, one-hot encoding, bag-of-words, TF-IDF, word embeddings, and t-SNE |
| [models.ipynb](models.ipynb) | Supplementary PySpark sentiment-classification experiments |

## Methodology

### Rule-based bill generation

The first exercise processes an English purchase paragraph with spaCy, removes adjectives and stop words, and strips selected weight units. A regular expression matches quantity expressions, product descriptions, and dollar prices. A custom number-word converter supplies numeric quantities; the code multiplies quantity by unit price and displays the bill with `tabulate`.

### Arabic preprocessing and representation

The second exercise reads the first two documents from the local MongoDB collection `nlp.aljazera`, expecting a `content` field. It cleans URLs, email-like strings, diacritics, repeated characters, digits, and punctuation. It then tokenizes text, applies a custom rule for words beginning with waw, removes Arabic stop words, and compares Qalsadi lemmatization with Tashaphyne light stemming.

The representation experiments include a hand-built one-hot vocabulary, scikit-learn bag-of-words and TF-IDF matrices, a small NumPy skip-gram demonstration, a GloVe lookup, and Gensim FastText. The skip-gram example uses just three words from the first sentence and 50-dimensional embeddings over 5,001 iterations. t-SNE plots project one-hot, bag-of-words, TF-IDF, and FastText representations into two dimensions.

### Supplementary sentiment classification

The Spark notebook reads `twitter_training.csv`, drops missing records, and creates pipelines with tokenization, stop-word removal, count features, label indexing, and classifiers. Its baseline comparisons use an 80/20 random split with seed 42. A further experiment defines bigram TF-IDF features and random-forest cross-validation.

## Results and discussion

The saved bill example produces a total of **$7,948.50**. This confirms extraction on the supplied paragraph, rather than general accuracy across arbitrary purchase descriptions.

The Arabic notebook saves vocabularies, numerical representations, a training-loss plot, nearest-word examples, and a four-panel t-SNE figure. These are qualitative demonstrations: there is no downstream benchmark establishing a best embedding method.

Selected saved Spark outputs are:

| Experiment | Accuracy |
| --- | ---: |
| Decision tree, depth 5 | 36.75% |
| Logistic regression, initial pipeline | 83.73% |
| Logistic regression, later variant dropping ID | 84.95% |

These values describe individual saved runs. Pipeline variants and Spark execution details prevent treating them as a controlled comparison of every design choice.

## Limitations

The number-word converter does not fully implement English number composition, and digit quantities are not handled by its dictionary. Extraction rules also assume particular currency and sentence formats. Arabic normalization and splitting every initial waw can remove meaningful distinctions. The tiny embedding corpus and t-SNE projections do not support broad semantic claims.

The GloVe lookup expects the unbundled file `../glove.6B.50d.txt`; vocabulary coverage must be checked before interpreting vectors. The MongoDB article corpus is not bundled. The Spark CSV is also absent from this folder, although a copy exists in [Lab 3](../nlp_thirdLab/twitter_training.csv).

## Reproduction notes

Run notebooks from this folder. Main imports include `spacy`, `nltk`, `tabulate`, `word2number`, `pymongo`, `pyarabic`, `tashaphyne`, `qalsadi`, `scikit-learn`, `numpy`, `matplotlib`, `gensim`, and TensorFlow/Keras. The bill exercise requires `en_core_web_sm`; Arabic preprocessing requires the MongoDB documents and NLTK resources. The Spark experiment additionally needs PySpark and Java, and its CSV path must point to the bundled Lab 3 data. The reported results are taken from saved outputs; the notebooks were not rerun for this report.

## Conclusion

Explicit rules make extraction easy to inspect but sensitive to wording. Numerical text representations offer a route to learning from text, while this lab's small examples show why meaningful evaluation needs larger corpora and consistent experimental conditions.
