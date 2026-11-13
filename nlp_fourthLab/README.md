# Lab 4 - Arabic Relevance Scoring and Transformer Experiments

[Back to all labs](../README.md)

## Abstract

This lab connects corpus collection, recurrent neural networks, and pretrained transformers. It builds an Arabic news corpus with heuristic relevance scores, compares sequence models for score prediction, fine-tunes GPT-2 on quotes, and sets up BERT sentiment classification on fashion reviews. Saved results support the recurrent-model and GPT-2 experiments; the BERT notebook does not contain completed evaluation metrics.

## Objectives and files

| Notebook or artifact | Purpose |
| --- | --- |
| [scraping.ipynb](scraping.ipynb) | Collect Hespress article links and text; assign heuristic scores |
| [part1.ipynb](part1.ipynb) | Arabic preprocessing and recurrent-model regression |
| [part2.ipynb](part2.ipynb) | GPT-2 quote fine-tuning and generation |
| [part3.ipynb](part3.ipynb) | BERT binary sentiment-classification workflow |
| [data/](data/) | Article exports, quote splits, and Amazon Fashion review data |
| [data_cleaned.csv](data_cleaned.csv) | 220 preprocessed Arabic records with scores |
| [generated_quotes.txt](generated_quotes.txt) | Bundled generated-text artifact |
| [texts_with_scores.csv](texts_with_scores.csv) | Additional saved text-and-score artifact |

## Methodology

### 1. Corpus construction and relevance labels

The scraping notebook uses Selenium to load dynamic Hespress search results and Requests/BeautifulSoup to extract article paragraphs. It searches for Stalin-related articles and a contrasting Eid al-Adha query. Saved collection messages report 195 and 30 links respectively; the displayed scored CSV contains 225 rows before preprocessing.

The implemented score combines text length, contributing up to three points, with keyword occurrences weighted by 1.5. The sum is capped at ten and rounded down. These targets are rule-generated proxies for relevance, not human judgments of article quality or validated readability/engagement scores.

### 2. Arabic relevance regression

`part1.ipynb` cleans Arabic text, removes stop words, and demonstrates tokenization. It then loads `data_cleaned.csv`, converts `clean_text` to padded token sequences, and standardizes `score` with `StandardScaler`. Two splits with seed 42 create 70% training, 15% validation, and 15% testing partitions.

The initial models use 100-dimensional embeddings and 64-unit recurrent layers, trained with Adam and MSE for ten epochs with batch size 64. Although named `RNN` and `Bidirectional RNN` in the notebook, their actual recurrent layers are LSTM and bidirectional LSTM. The separately named `LSTM` baseline duplicates the first architecture. A second experiment introduces stacked layers, 50% dropout, and early stopping.

### 3. GPT-2 quote fine-tuning

`part2.ipynb` loads GPT-2, adds padding support, appends end-of-text markers to quote lines, and resizes token embeddings. Training runs for three epochs with learning rate `5e-5`, 500 warmup steps, sequences limited to 400 tokens, and optimizer updates after two processed sequences. The DataLoader itself uses batch size one.

After validation, the generation code loads epoch index 2 and attempts 1,000 continuations from `QUOTE:`. It samples from the top 20 candidates for the first three generated tokens and the top three afterward, stopping at an end-of-text token or 100 generated tokens. Only completed sequences are written, so the attempt count is not a guaranteed output count.

### 4. BERT review classification

`part3.ipynb` loads 3,176 Amazon Fashion reviews, identifies 16 missing review texts, fills them with empty strings, and maps ratings of four or five to positive, with ratings of one through three mapped to negative. `bert-base-uncased` tokenization uses a maximum length of 128. The code prepares a 90/10 training/validation split with seed 42, batch size two, and a three-epoch training function with learning rate `2e-5`.

## Recorded results

Relevance-regression MSE is measured on **standardized targets**, so it is not error directly expressed on the original 0-10 score scale.

| Notebook model label | Initial MSE | Stacked/dropout MSE |
| --- | ---: | ---: |
| RNN (implemented with LSTM) | **0.6945** | 0.9722 |
| Bidirectional RNN (bidirectional LSTM) | 0.7484 | 0.8611 |
| GRU | 0.8553 | **0.7844** |
| LSTM | 0.7714 | 0.9190 |

The initial LSTM-based `RNN` has the lowest saved MSE overall. GRU performs best within the second experiment, but adding stacking and dropout does not improve every model. Because the first and fourth baseline architectures are identical, their different results also illustrate training variability.

GPT-2 records a mean validation loss of **4.3708** and a successful generation-file message. This is a language-model loss, not a direct measure of quote originality, coherence, or factual accuracy. The repository includes `generated_quotes.txt`, while the current generation cell names its output `generated_quotes_epoch_2.txt`; these should not be assumed to be the same run.

The BERT notebook defines and calls a training function, but the saved cell output contains model-initialization messages rather than completed epoch results. Its evaluation cell has no saved metrics, so this report makes no claim about trained BERT accuracy or F1.

## Limitations

The Arabic tokenizer and score scaler are fitted before splitting; a stricter experiment would fit both on training data only. Heuristic labels measure agreement with the scoring rule, and the small corpus limits generalization. WordNet lemmatization in the preprocessing code does not provide an Arabic-specific morphology model. Multiple architectures, random initialization, and one split are insufficient for a robust ranking.

The GPT-2 experiment lacks a reported test-set result and a systematic assessment of generated text. BERT needs an actual training run and evaluation against a baseline before its performance can be discussed. The notebooks and outputs document exploratory experiments rather than a finished deployed system.

## Reproduction notes

Open Jupyter in this folder and review paths before running. The collection and preprocessing notebooks use `requests`, `beautifulsoup4`, `selenium`, `pandas`, `numpy`, `nltk`, `pymongo`, `pyarabic`, `tashaphyne`, and `qalsadi`; regression uses TensorFlow/Keras and scikit-learn; transformer experiments use PyTorch and Transformers.

- Replace the scraping notebook's original machine-specific absolute paths with paths under this folder. Selenium collection also requires Chrome and a compatible browser driver.
- Point GPT-2's `train2.txt`, `valid2.txt`, and `test2.txt` paths to their bundled locations under `data/`.
- The notebooks contain informal notes stored as code cells; convert those cells to Markdown or skip them during execution.
- Saved model weights and tokenizer files referenced by inference cells are not bundled. Train and save the required artifacts first; preserve the target scaler for inverse transformation of relevance predictions.
- Model downloads require network access. No pinned environment is supplied, and older Transformers/Keras imports may need adaptation to the installed versions.

All numerical findings above come from the committed outputs and data; training was not rerun for this report.

## Conclusion

The lab demonstrates a progression from handcrafted relevance targets to neural regression and transformer adaptation. Its clearest findings are the saved recurrent-model comparison and completed GPT-2 validation run. Better label validation, training-only preprocessing, reproducible environments, and a completed BERT experiment are the next steps toward stronger conclusions.

## License

This lab includes an [MIT license](LICENSE).
