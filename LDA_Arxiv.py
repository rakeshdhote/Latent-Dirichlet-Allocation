# -*- coding: utf-8 -*-
"""
Topic Modeling
Latent Dirichlet Allocation 
For Analysing Themes in the arXiv corpus

@author: rakesh dhote
www.rakeshdhote.com
"""
#############################################################
# Install modules
#import logging
import nltk
import gensim
from gensim import corpora, models, similarities
from os import path
from wordcloud import WordCloud
from collections import defaultdict
#from pprint import pprint   
import matplotlib.pyplot as plt
import time
#############################################################
# Word tokenization
def NLTK_tokenize(words):
    return nltk.tokenize.word_tokenize(words)

##############################################################
# Stop word 
def NLTK_stopwords(words):
    stop_words = set(nltk.corpus.stopwords.words("english") + list(string.punctuation) + ['rt', 'via', "i'm", 'en', ":)" ,":-)" ,"=)", ":(" ,":-(" ,"=(", "&", "de", ")", "<3","//"])
    return [w for w in words if not w in stop_words]
    
#############################################################
# Stemming
def NLTK_stemmer(words):
    ps = nltk.stem.PorterStemmer()
    return [ps.stem(w) for w in words]

#############################################################
# Lammetization
def NLTK_lammetization(words):
    return [[nltk.stem.WordNetLemmatizer().lemmatize(l) for l in w] for w in words]
    
#############################################################
# Read data
def read_data(dirn,filen):
	fname = path.join(dirn, filen) 
	with open(fname, "r") as f:
	    documents = []
	    for line in f:
	        documents.append(line)
	    return documents
     
#############################################################
# clean data
def clean_data(documents):
    
	# stopword list
    stoplist = set('all just being over both through yourselves its before herself had should to only under ours has do them his very they not during now him nor did this she each further where few because doing some are our ourselves out what for while does above between t be we who were here hers by on about of against s or own into yourself down your from her their there been whom too themselves was until more himself that but don with than those he me myself these up will below can theirs my and then is am it an as itself at have in any if again no when same how other which yo after most such why a off i yours so the having once using paper also presents present found '.split())

	# remove common words and tokenize 
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    
    # Apply lammetization
    texts = NLTK_lammetization(texts)
    
    frequency = defaultdict(int)

    # remove words that appear only once
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts 

#############################################################
# create dictionary and corpus
def create_dictcorpus(texts):
    
    # create dictionary:
    dictionary = corpora.Dictionary(texts)
    # create corpus from the bag of words using doc2bow()
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary, corpus    
    
#############################################################
# create and save dictionary and corpus
def dictcorpus(dirn, dictn, corpn, texts):
    
    dictn = dictn + '.dict'
    corpn = corpn + '.mm'
    
    dicfile = path.join(dirn, dictn)
    corpusfile = path.join(dirn, corpn)    
    
    dictionary, corpus = create_dictcorpus(texts)
    
    dictionary.save(dicfile)  #save dictionary   
    corpora.MmCorpus.serialize(corpusfile, corpus) #save corpus 
    
    return dictionary, corpus
    
#############################################################
# TF-IDF transformations    
def ifidf_transformation(corpus):
    
    tfidf = models.TfidfModel(corpus) 
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf    

#############################################################
# LDA model
def topics_lda(ntopics, npass, iterations, dictionary, corpus):
    
    lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, \
    num_topics = ntopics, update_every = 0, passes = npass, iterations = iterations) 
    
    return lda                                 

#############################################################
# Write LDA topics to file
def ldatopics_file(dirn, ldatopics, lda, ntopics):
    
    fname = path.join(dirn, ldatopics) 
    
    ldatext_file = open(fname, "w")
    for i in range(0,ntopics):
        ldatext_file.write(lda.print_topics(ntopics)[i][1] + '\n')
    ldatext_file.close()  
    
#############################################################
def topics_lsi(corpus_tfidf, dictionary, ntopics):
    
    lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = ntopics) 
    corpus_lsi = lsi[corpus_tfidf]
    
    return lsi, corpus_lsi

#############################################################
# Similarity 
def similarity_lda(dictionary, corpus, lda, dirn, similarity_docname):
    
    fname = path.join(dirn, similarity_docname) 
    
    index = similarities.MatrixSimilarity(lda[corpus])
    index.save("simIndex.index")
    
    doc = open(fname, 'r').read()
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lda = lda[vec_bow] # convert the query to LDA space
    
    sims = index[vec_lda]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    
    return sims

#############################################################
# Plot wordcloud
def lda_wordcloud(dirn, ldatopics, ntopics, nrow, figname):
    
    fname = path.join(dirn, ldatopics)
    figname = path.join(dirn, figname)
    
    final_topics = open(fname, 'r')

    i = 0 # dummy variable
    
    if (ntopics % nrow != 0):
        ncol = ntopics/nrow + 1
    else:
        ncol = ntopics/nrow
        
    fig = plt.figure(figsize=(8,4))
    
    for line in final_topics:
        i = i + 1
        scores = [float(x.split("*")[0]) for x in line.split(" + ")]
        words = [x.split("*")[1] for x in line.split(" + ")]
        freq = []
        freq = zip(words, scores) # tuple of frequencies 
        
        temp = WordCloud(background_color='white').generate_from_frequencies(freq)
        ax = plt.subplot(nrow, ncol, i)  
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Topic %d' % i)
        plt.tight_layout()
        plt.imshow(temp)
        
    final_topics.close()    
    fig.savefig(figname, dpi = 300)

#############################################################
# Main Program
if __name__ == '__main__':

    start = time.time()
    ############################
    # preamble    
    dirn = "./Arxiv" # directory name
    name = "Arxiv"
    filen = name + ".txt" #file name
    dictn = name + "_dict" # dictionary name
    corpn = name + "_corpus" # corpus name
    ldatopics = name + "_ldatopics.txt"
    similarity_docname = name + "_test.txt"
    ldamodel = name + "_LDA.lda"
    figname = name + "_LDA_Plot.png"
    
    ############################    
	# read data
    documents = read_data(dirn,filen)

    ############################
	# clean data
    texts = clean_data(documents)

    ############################
    # create dicationary and corpus
    dictionary, corpus = dictcorpus(dirn, dictn, corpn, texts)
    
#	print(dictionary)
#    print(dictionary.token2id)
#    print(corpus)  

    ############################
    # TF-IDF transformations
    corpus_tfidf = ifidf_transformation(corpus)
    
#    for doc in corpus_tfidf:
#        print(doc)
    
    ############################ 
    ## Latent Dirichlet Allocation
           
    ntopics = 15 # Number of topics
    npass = 100 # number of passess
    iterations = 100 # number of iterations
    
    lda = topics_lda(ntopics, npass, iterations, dictionary, corpus_tfidf)
    
    # save LDA topics to a text file
    ldatopics_file(dirn, ldatopics, lda, ntopics)
    
    lda.print_topics(ntopics) 
    lda.save(ldamodel)

    ############################
    ## Latent Semantic Indexing

#    lsi, corpus_lsi = topics_lsi(corpus_tfidf, dictionary, ntopics)
#    lsi.print_topics(ntopics)                                  
#                                  
#    for doc in corpus_lsi: 
#        print(doc)      

    ############################
    ## Similarity Index - cosine similarity
    # sims = similarity_lda(dictionary, corpus, lda, dirn, similarity_docname)
    
    # # print first n list items
    # n  = 12
    # print sims[1:n]

    ############################
    ## plot Wordcloud
    nrow = 3 # number of rows in figure
    lda_wordcloud(dirn, ldatopics, ntopics, nrow, figname)
    
    end = time.time()
    
    print 'Time elapsed = ', (end - start)/60, ' mins'












