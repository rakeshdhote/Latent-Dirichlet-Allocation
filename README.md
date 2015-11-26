# Latent-Dirichlet-Allocation

The repo consists of the files for the Topic Modeling implemenation using the Latent Dirichlet Allocation (LDA) to uncover hidden thematic structure in the [arXiv(http://www.arxiv.org)] articles corpus.  

The Python code uses the [Gensim](https://radimrehurek.com/gensim/), [NLTK](http://www.nltk.org/), [Wordcloud](https://github.com/amueller/word_cloud), and [MatPlotLib](http://matplotlib.org/) packages. The program executes the following tasks:  

*Pre-processing:*   
*  Cleans the arXiv articles corpus data using Python re package   
*  Tokenize, stemming, stopword removal, and lammetization (WordNet) 

*Processing:*  
*  Converts the corpus into the 'bag-of-words'
*  Converts the corpus to vector space model
*  Implements TF-IDF (term frequency inverse document frequency)
*  Implements Latent Dirichlet Allocation algorithm
*  Conducts similarity index for a test document  

*Post-Processing:*  
*  Topic probabilities processing
*  Generate wordcloud visualisation
*  Advanced data visualization (to be added...)
