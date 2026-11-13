# Lab 1 - Web Scraping and an Arabic NLP Pipeline

[Back to all labs](../README.md)

**Author:** Mohammed AACHABI - **Course:** MST IASD

## Abstract

This lab follows a small data pipeline from an Arabic web page to structured storage and linguistic analysis. A population table is extracted with Requests and BeautifulSoup, exported to CSV, inserted into MongoDB, and converted into text for preprocessing and named entity recognition. The experiment illustrates both the mechanics of an NLP pipeline and the limits of applying general-purpose language tools to Arabic.

## Objective

Extract structured information from HTML, preserve it in reusable formats, and explore how cleaning, tokenization, stop-word removal, stemming, lemmatization, part-of-speech tagging, and entity recognition affect the resulting text.

## Data and files

| File | Role |
| --- | --- |
| [lab1(nlp).ipynb](lab1%28nlp%29.ipynb) | Complete scraping, storage, and NLP experiment |
| [countries.csv](countries.csv) | Saved population table: 240 records and six columns |
| [Atelier1.pdf](Atelier1.pdf) | Original lab assignment |

The scraped source is an Arabic Wikipedia table of countries and dependencies by population. Its fields cover rank, country or territory, population, census date, share of world population, and source. The CSV is a historical snapshot; its figures should not be treated as current statistics.

## Methodology

1. Request the page and locate its `wikitable sortable` table using BeautifulSoup.
2. Extract header and row text into a pandas DataFrame, then export `countries.csv`.
3. Convert rows to dictionaries and insert them into the local MongoDB collection `test.country_population`.
4. Serialize the records as text. Apply regular-expression cleaning, NLTK tokenization, Arabic stop-word removal, and the Arabic Snowball stemmer.
5. Compare this pipeline with demonstrations using Porter stemming and WordNet lemmatization, then apply NLTK part-of-speech tagging.
6. Run spaCy's `xx_ent_wiki_sm` entity recognizer and inspect the predicted spans and labels.

## Results and discussion

The saved notebook records an HTTP 200 response, a populated table, and successful MongoDB insertion. The bundled CSV contains 240 country or territory records. Intermediate outputs show the conversion from raw records to tokens and stemmed text.

The entity predictions also expose a quality problem: numeric fragments, database identifiers, and census phrases receive entity labels. These examples demonstrate that obtaining predictions does not establish accurate Arabic entity recognition. No annotated reference set or quantitative NER evaluation is included.

## Limitations

- Porter stemming, WordNet lemmatization, and the default NLTK POS tagger are English-oriented choices in this notebook; their output is not a validated Arabic linguistic analysis.
- `xx_ent_wiki_sm` is the multilingual model loaded by the code, despite the notebook comment describing it as Arabic-specific.
- Text serialization includes MongoDB identifiers, and punctuation removal can change numerical meaning. Structured numeric fields should be handled separately from prose.
- The scraper depends on the source page's table structure. Repeating `insert_many` can insert duplicate records.

## Reproduction notes

Open Jupyter in this lab folder. The notebook uses `requests`, `beautifulsoup4`, `pandas`, `pymongo`, `nltk`, and `spacy`, plus the `xx_ent_wiki_sm` spaCy model and the NLTK resources requested in its cells. MongoDB must be available at `localhost:27017` for the storage section. Live scraping and resource downloads need network access; the bundled CSV supports offline inspection of the extracted table.

The observations above come from the committed notebook outputs and CSV, rather than a new execution.

## Conclusion

The lab establishes a complete extraction-to-analysis workflow. Its main lesson is that language-specific preprocessing and careful separation of text from metadata are necessary before interpreting linguistic predictions.
