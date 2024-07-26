# NLP tools demonstrations

In this folder, we will show some NLP tools and how they are used.
The demonstrations are structured according to the course chapters.
This means, we will show only the task described in a given chapter.

## Non-exhaustive list of NLP tools

|Tool|Demo|Language|Tasks|
|---|---|---|---|
| [AllenNLP](https://allennlp.org/)|[Demo](https://demo.allennlp.org/) |Python | |
| [BERT](https://github.com/google-research/bert) | |Python|LM|
| [CogCompNLP](https://github.com/CogComp/cogcomp-nlp) | | Java | |
| [compromise](https://github.com/spencermountain/compromise) | |Javascript| |
| [BlingFire](https://github.com/microsoft/BlingFire) | |Multiple (C++)|Preprocessing|
| [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) | [Demo](https://corenlp.run/) |Java| |
| [ERNIE](https://github.com/thunlp/ERNIE) | |Python|LM|
| [FastText](https://fasttext.cc/) | |Python|Text classification and word representation|
| [Flair](https://github.com/flairNLP/flair) | |Python|NER, PoS|
| [Gensim](https://radimrehurek.com/gensim/) | |Python|Topic modeling|
| [GPT-2](https://github.com/openai/gpt-2) | |Python|LM|
| [IceCaps](https://github.com/microsoft/icecaps) | |Python|Conversation agent|
| [jiant](https://github.com/nyu-mll/jiant) | |Python|Research tasks|
| [JsLingua](https://github.com/kariminf/jslingua) | |Javascript||
| [natural](https://github.com/NaturalNode/natural) | |Nodejs| |
| [NeuralCoref](https://github.com/huggingface/neuralcoref) | |Python(Spacy)|CoRef|
| [NLP Architect](https://github.com/IntelLabs/nlp-architect) | |Python| |
| [NLP.js](https://github.com/axa-group/nlp.js) | |Nodejs| |
| [NLTK](https://www.nltk.org/) | |Python| |
| [OpenNLP](https://opennlp.apache.org/) | |Java| |
| [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) | | Python | |
| [retext](https://github.com/retextjs/retext) | | Nodejs | |
| [spaCy](https://spacy.io/) | |Python| |
| [Spark NLP](https://nlp.johnsnowlabs.com) | | | |
| [Textacy](https://readthedocs.org/projects/textacy/) | | Python | |
| [Texar-pytorch](https://github.com/asyml/texar-pytorch) | |Python| |
| [TextBlob](https://github.com/sloria/TextBlob) | |Python| |
| [Tika](https://tika.apache.org/) | |Java|Text extraction|
| [pattern](https://github.com/clips/pattern) | |Python| |

## Demonstrations

We will use [Jupyter Notebook](https://jupyter.org/) to document these tools.
By default, this plateform supports Python since it is developped on it.
As for Java, we use [Ganymede](https://github.com/allen-ball/ganymede).

### CH03: Basic text processing

- [CoreNLP](CH03/basic_java_CoreNLP.ipynb): Java
  - I. Preprocessing pipeline
  - II. Get tokens
  - III. Get sentences
  - IV. Get lemmas
  - V. Other languages
- [LangPi](CH03/basic_java_LangPi.ipynb): Java
  - I. Text normalization
  - II. Text segmentation
  - III. StopWords Filtering
  - IV. Text stemming
- [NLTK](CH03/basic_python_NLTK.ipynb): Python
  - I. Text tokenization (sentences, words)
  - II. StopWords filtering
  - III. Stemming
  - IV. Lemmatization
  - V. Distance
- [OpenNLP](CH03/basic_java_OpenNLP.ipynb): Java
  - I. Language detection (detection, training)
  - II. Sentence boundary detection (detection, training)
  - III. Word tokenization
  - TODO: complete
- [spaCy](CH03/basic_python_spaCy.ipynb): Python
  - I. Sentence tokenization
  - II. Words tokenization
  - III. StopWords filtering
  - IV. Lemmatization

### CH04: Language models

- [Keras(TensorFlow)](CH04/lm_python_Keras.ipynb): Python
  - I. Simple FeedForward 3-gram model (with, without embedding)
  - II. LSTM model
- [NLTK](CH04/lm_python_NLTK.ipynb): Python
  - I. Preprocessing (padding)
  - II. NGrams
  - III. Vocabulary
  - IV. Language models (MLE, Smoothed models)

### CH05: Part of Speech (PoS) tagging

- [flair](CH05/sequences_python_flair.ipynb) : Python
  - I. Data preparation: Tokenization, Corpus preparation
  - II. Part of Speech (PoS) tagging: Tagging, Training
  - II. Named Entity Recognition (NER): Recognition, Training
- [NLTK](CH05/sequences_python_NLTK.ipynb) : Python
  - I. Part of Speech (PoS) tagging: Default, RegEx, CRF, HMM, PerceptronTagger, Brill
  - II. Named Entity Recognition (NER): default, training
  - III. Chunking

### CH06: Parsing

- [CoreNLP](CH06/parsing_java_CoreNLP.ipynb) : Java
  - I. Constituancy Parsing
  - II. Dependency Parsing %TODO complete
- [NLTK](CH06/parsing_python_NLTK.ipynb) : Python
  - I. Grammars: Context Free grammar (CFG), Loading grammars, Treebanks
  - II. Parsing: Recursive Descent, Shift-Reduce, Chart
  - III. Generation

### CH07: Word semantics

- [Gensim](CH07/encoding_python_gensim.ipynb) : Python
  - I. TF-IDF
  - II. LSA
  - III. LDA
  - IV. Word2Vec
  - V. Fasttext
- [NLTK](CH07/encoding_python_NLTK.ipynb) : Python
  - I. WordNet: Synsets, SynSet properties and relations, Lemma relations
  - II. Operations: Lowest Common Hypernyms, Similarity
  - III. Multilingual WordNet
- [Scikit-learn](CH07/encoding_python_sklearn.ipynb) : Python
  - I. Vectorization: TF, IDF, LSA
  - II. Parameters: reading, preprocessing
  - III. Similarity
- [TensorFlow-BERT](CH07/encoding_python_TF_BERT.ipynb) : Python
  - I. Text preprocessing: Using tf-hub model, Training a preprocessor
  - II. Text encoding: Using tf-hub model, Train a model from scratch
  - III. Fine-Tuning
- [TensorFlow-ELMo](CH07/encoding_python_TF_ELMo.ipynb) : Python
  - I. Text encoding
  - II. Using ELMo

### CH08: Sentence semantics

- [NLTK](CH08/sentsem_python_NLTK.ipynb) : Python
  - I. Thematic relations: FrameNet, PropBank, VerbNet
  - II. Propositional Logic: Expressions and proves, First-Order Logic (FOL)
  - III. Semantic parsing: λ-Calculus, analysis
- [AMRLib-spacy](CH08/sentemn_python_amrlib.ipynb) : Python

### CH09: Coreference detection

- [CoreNLP](CH09/coref_java_CoreNLP.ipynb) : Java

### CH10: Discourse coherence


### CH11: Some applications

- [Keras(TensorFlow)](CH11/MT_python_keras.ipynb) : Python
  - Text reading and dataset creation
  - Text preprocessing
  - The encoder/decoder model
  - Training
  - Translate
  - Evaluation (NLTK-BLEU)
