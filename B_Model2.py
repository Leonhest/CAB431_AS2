import math
import os
from nltk.stem import PorterStemmer
from NewsItem import NewsItem, Assignment_1

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

    ####################################ß
    result = {}
    termsSize = len(termSet)
    la = 0.2

    for docId, newsItem in documentColl.items():
        d_terms = newsItem.get_termList()
        d_size = newsItem.getSize()
        score = 0
        for q_term, freq in queryDict.items():
            try:
                f_qi_d = d_terms[q_term]
            except:
                f_qi_d = 0
            
            score += math.log10((1-la)*(f_qi_d/d_size) + (la*(termSet[q_term]/termsSize)))
        
        result[newsItem.getDocId()] = score

    return result


if __name__ == '__main__':

    #Read and save stop words from text file
    stopwords_f = open('common-english-words.txt', 'r') 
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    inputfolder = "RCV1v2"
    folderSet = [f"./{inputfolder}/{file}" for file in os.listdir(inputfolder)] #Save all file paths within the input folder into array
    query = ""

    for folder in folderSet:
        rankings = B_Model2(query, stop_words, inputfolder)
        #do something...

    "For testing P values."
    # P = []
    # precision_list = []
    # num_of_rel = 0

    # for index, doc_id in enumerate(islice(results_rank.keys(), 10), start=1):
    #     rel = float(results_rel[doc_id])
    #     P.append(rel)

    #     num_of_rel += rel
    #     precision = num_of_rel / index
    #     precision_list.append(precision)

    #     print(f"At position {index} docID: {doc_id}, precision = {precision}")

    # average_precision = sum(precision_list) / len(precision_list)
    # print(f"The average precision = {average_precision}")
