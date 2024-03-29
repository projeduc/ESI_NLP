# <img src="extra/logo/esi.nlp-logo.png" width=100px> "Natural language processing" course (ESI, Alger)

[![Type](https://img.shields.io/badge/type-Course-0014A8.svg?style=flat)](https://github.com/projeduc/ESI_NLP)
[![License](https://img.shields.io/badge/license-CC--BY_4.0-0014A8.svg?style=flat)](https://creativecommons.org/licenses/by/4.0/deed.en)
[![GitHub release](https://img.shields.io/github/release/projeduc/ESI_NLP.svg)](https://github.com/projeduc/ESI_NLP/releases)
[![Github All Releases](https://img.shields.io/github/downloads/projeduc/ESI_NLP/total.svg)](https://github.com/projeduc/ESI_NLP/releases)
[![Github Release](https://img.shields.io/github/downloads/projeduc/ESI_NLP/latest/total.svg)](https://github.com/projeduc/ESI_NLP/releases/latest)

This repository contains NLP course presentations, demos and labs proposed by Ecole nationale Supéieure d'Informatique (ESI), Alger, Algeria.

[Download the course HERE](https://github.com/projeduc/ESI_NLP/releases/latest)

Natural Language Processing (NLP) is an interdisciplinary field involving linguistics, computer science, and artificial intelligence. It is dedicated to the understanding and generation of human language by computers. With the emergence of language models such as OpenAI's GPT, Google's BERT, and others, NLP has become a rapidly growing field with increasing importance in various industries and domains.

This field is often taught as a course in the "artificial intelligence" and "data science" specializations. It is part of branches of artificial intelligence such as computer vision, robotics, machine reasoning, etc. This does not mean that other specializations do not need this course; it is just less prominent than other courses in those specializations. In software engineering, for example, this field can be exploited to design a system that constructs an entire software architecture from a textual description. Natural language is part of human-machine interaction modes. In information systems, techniques from this field are often used for information retrieval and extraction. Data extraction is a phase of an ETL (Extract, Transform, Load) system where data can be textual (unstructured). In systems and networks, various injection attacks, such as SQL injection, can be treated similarly to natural language.

This course try to cover both research and developpement aspects of this field. A wide range of topics will be presented, which are organized into several levels: lexicon, syntax, semantics, pragmatics, and discourse. It is important to note that this course has a significant similarity to the compilation course. Both courses share almost the same pipeline: lexical parsing, syntactic parsing, and semantic parsing. They have common techniques, but those of the compilation course are not always applicable to natural language. The latter is different from a programming language, which is well-defined and therefore contains less ambiguity.

## Syllabus

1. **Introduction to NLP**
    - History
        - AI birth and golden era
        - AI winter
        - AI spring
    - Language processing levels
        - Phonology
        - Morphology and syntax
        - Semantic
        - Pragmatic et discourse
    - NLP applications
        - Tasks
        - Systems
        - Business
    - NLP challenges
        - Ressources
        - Language understanding
        - EEvaluation
        - Ethics
1. **ML for NLP**
    - Machine learning
        - Traditional ML
        - MLP
        - CNN
        - RNN
    - Text classification
        - Traditional ML and MLP
        - CNN
        - RNN
    - Sequences classification
        - IOB notation
        - HMM
        - MEMM
        - RNN
    - Attention
        - Seq2Seq
        - Seq2Seq with attention
        - Attention types
        - Transformers
1. **Basic text processing**
    - Character processing
        - Regular expressions
        - Edit distance
    - Text segmentation
        - Sentence boundary disambiguation
        - Words tokenization
    - Text normalization
    - Text filtering
    - Morphology
        - Word formation
        - Form reducing
1. **Language models**
    - N-gram language models
        - Formulation
        - Smoothing
    - Neural language models
        - Muli-Layers Perceptron (MLP)
        - Recurrent neural networks (RNN)
    - Evaluation
        - Approches
        - Perplexity
1. **Part of Speech (PoS) tagging**
    - Sequence labeling
    - Ressources
    - Approches
        - Hidden Markov models (HMM)
        - Maximum entropy
        - Neural networks
1. **Parsing**
    - Syntactic structures
        - Constituency formalisms
        - Dependency formalisms
    - Constituency parsing
        - CKY algorithm
        - Probabilistic CKY algorithm
    - Dependency parsing
        - Transition-based
        - Graph-based
1. **Word semantics**
    - Lexical databases
        - Semantic relations
        - Wordnet
        - Other ressources
    - Vector representation of words
        - TF-IDF
        - Word-Word
        - Latent semantic analysis (LSA)
    - Word embedding
        - Word2vec
        - GloVe
        - Contextual embedding
        - Models evaluation
    - Word sense disambiguation
        - Knowledge bases based
        - Machine learning based
1. **Sentence semantics**
    - Semantic roles
        - Roles
        - FrameNet
        - PropBank
    - Semantic role labeling
        - Features based
        - Neural networks based
    - Semantic representation of sentences
        - First-order logic
        - Graphs (AMR)
        - Semantic parsing
1. **Coreference detection**
    - References
        - References forms
        - Referencing manner
        - coreference relations properties
    - Resolution of coreferences
        - Mention detection
        - Coreferences linking
        - Evaluation
    - Related Tasks
        - Entity linking
        - Named-entity recognition (NER)
1. **Discourse coherence**
    - Coherence relations
        - Rhetorical Structure Theory (RST)
        - Penn Discourse TreeBank (PDTB)
    - Discourse structure-based analysis
        - RST analysis
        - PDTB analysis
    - Discourse entity-based analysis
        - Centering theory
        - Entity Grid model
1. **Some applications**
    - Machine translation
        - Direct approach
        - Transfert-based appoach
        - Interlingua appoach
        - Statistical appoach
        - Exemples-based appoach
        - Neural appoach
    - Automatic text summarization
        - Statistical appoach
        - Graph-based appoach
        - Linguistic appoach
        - Machine learning based appoach
    - Questions/Answering
        - Information retrieval based appoach
        - Knowledge based appoach
        - Language models based appoach
    - Dialog systems
        - Task-oriented: Frame-based
        - Task-oriented: Dialogue-State
        - Rules-based chatbots
        - IR-based chatbots
        - Text generation based chatbots
    - Sentiment analysis
        - Knowledge based
        - Machine learning based
        - Hybrid
    - Lisibility
        - Formulas
        - Machine learning
    - Speech recognition
        - Features extraction
        - Recognition
    - Speech synthesis


## License

Copyright (C) 2022-2024  Abdelkrime Aries (english version)

Copyright (C) 2020-2021  Abdelkrime Aries (french version)


[Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.en)

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material
for any purpose, even commercially.

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
