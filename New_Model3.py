import math
import os
from nltk.stem import PorterStemmer
from NewsItem import NewsItem, Assignment_1
from itertools import islice

def B_Model2(query, stop_words, inputfolder):

    files = [f"./{inputfolder}/{file}" for file in os.listdir(inputfolder)] #Save all file paths within the input folder into array
    stemmer = PorterStemmer() 
    documentColl = {} #Collection of NewsItem objects
    termSet= {} #Set of all terms in document collection
    queryDict = Assignment_1.Q_Parser(query, stop_words, stemmer) #Query document language


    for filename in files:

        item = NewsItem.NewsItem() #Initaialize new NewsItem object
        docId = Assignment_1.getDocId(filename) 
        item.setNewsId(docId)
        text = Assignment_1.getTextSec(filename)
    
        size = 0 #Initaialize size of NewsItem document

        for line in text.splitlines():

            line = Assignment_1.cleanLine(line) #clean the line

            for word in line.split(): #Tokenization

                size += 1
                word = word.lower() #Lower case the word
                word = stemmer.stem(word) #Stem the word

                if (not word in stop_words and len(word) > 2):
                    item.add_term(word)
                    #term into big C
                    try:
                        termSet[word] += 1
                    except:
                        termSet[word] = 1

            item.setSize(size)

        documentColl[item.getNewsId()] = item

    #Calculate Score First Pass 
    results = {}
    termsSize = len(termSet)
    mu = 1000 #Typical values of mu is 1000-2000 (play with value)

    for docId, newsItem in documentColl.items():
        d_terms = newsItem.get_termList()
        d_size = newsItem.getSize()
        score = 0
        for q_term in queryDict.keys():
            try:
                f_qi_d = d_terms[q_term]
            except:
                f_qi_d = 0
            
            numerator = f_qi_d + mu*(termSet[q_term]/termsSize)
            denominator = d_size + mu
            score += math.log10((numerator/denominator))
        
        results[newsItem.getDocId()] = score

    ## Step 5 top -k
    a = 10 #k-value
    Set_C = {k: v for i, (k, v) in enumerate(results.items()) if i < a}

    # step 6
    vocabulary = [] 
    z = 10 #highest-probability words in doc (10-25 typical vaules)

    for newsItem in Set_C.keys():
        terms = newsItem.get_termList()
        first_z_words = list(islice(terms, z))
        vocabulary.append(first_z_words)

    vocabulary.append(list(queryDict.keys())) #query expansion

    #step 7

    relevance_model = {}
    for word in vocabulary:
        score = 0

        for docId, newsItem in Set_C.keys():
            terms = newsItem.get_termList()
            doc_size = newsItem.getSize()

            try:
                score += terms[word]/doc_size + results[docId]
            except:
                score = terms[word]/doc_size + results[docId]

        relevance_model[word] = score

    #step 8
    results = {}
    termsSize = len(Set_C)
    mu = 1000 #Typical values of mu is 1000-2000 (play with value)

    for docId, newsItem in documentColl.items():
        d_terms = newsItem.get_termList()
        d_size = newsItem.getSize()
        score = 0
        for q_term in vocabulary:
            try:
                f_qi_d = d_terms[q_term]
            except:
                f_qi_d = 0
            
            numerator = f_qi_d + mu*(termSet[q_term]/termsSize)
            denominator = d_size + mu
            score +=  math.log10(relevance_model[word]) * math.log10((numerator/denominator))
        
        results[newsItem.getDocId()] = score

    results_sorted = dict(sorted(results.items(), key=lambda item: sum(item[1].values()), reverse=False)) #ascedning order. smaller value = better
    return results_sorted