import os
import string
import NewsItem
from nltk.stem import PorterStemmer



def getDocId(filename):
    """
    Find the item ID in a given document

    Parameters:
    filename (string): Path to file

    Returns:
    itemID (string): The item ID of the document
    """

    file = open(rf"{filename}", "r") #Open document file

    for line in file.readlines():
        
        words = line.split() #Split each line into individual words

        for word in words:
            
            if ("itemid" in word):
                
                file.close() #Close document file
                itemId = word.strip('itemid="') #Parse the ID before saving it

                return itemId

def getTextSec(filename):
    """
    Find the text section in a given document

    Parameters:
    filename (string): Path to file

    Returns:
    text (string): Text section of the document
    """
    text = "" #Initailize text as empty string

    file = open(rf"{filename}", "r") #Open document file

    line = file.readline() #Initailize line

    while line:

        #find start of text section
        if line.startswith("<text>"):
            line = file.readline() # skip first line
            #find end of text sectino
            while not line.startswith("</text>"):
                text += line #Save each line to variable text
                line = file.readline()

            file.close() #Close file
            return text.strip() #Parse the text section before returning it
        
        line = file.readline()

def cleanLine(line):
    """
    Parse a given string by removing numbers, punctuation, white space etc.

    Parameters:
    line (string): Sentence from the text section of a document

    Returns:
    line (string): Parsed sentence from the text section of a document
    """
    line = line.replace("<p>", "").replace("</p>", "")
    line = line.translate(str.maketrans('', '', string.digits)).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    line = line.replace("\\s+", " ")
    return line.strip()

def News_Parser(stop_words, inputfolder):
    """
    Builds a dictionary of NewsItem objects for a given dataset.

    Parameters:
    stop_words (list of string): List of common english words
    inputfolder (string): Path to folder containing the news item document dataset

    Returns:
    Rcv1Coll (dict):
        - keys (string): NewsItem ID
        - values (NewsItem): Object
    """
    files = [f"./{inputfolder}/{file}" for file in os.listdir(inputfolder)] #Save all file paths within the input folder into array
    stemmer = PorterStemmer() 
    Rcv1Coll = {} #Collection of NewsItem objects
    
    for filename in files:

        item = NewsItem.NewsItem() #Initaialize new NewsItem object
        docId = getDocId(filename) 
        item.setNewsId(docId)
        text = getTextSec(filename)
    
        size = 0 #Initaialize size of NewsItem document

        for line in text.splitlines():

            line = cleanLine(line) #clean the line

            for word in line.split(): #Tokenization
                size += 1
                word = word.lower() #Lower case the word
                word = stemmer.stem(word) #Stem the word
                if (not word in stop_words and len(word) > 2):
                    item.add_term(word)
            
            item.setSize(size)

        Rcv1Coll[item.getNewsId()] = item

    return Rcv1Coll


def Q_Parser(query, stop_words):
    """
    Tokenizes and Parsers a given query

    Parameters:
    query (string): Sentence used to search for documents
    stop_words (list of string): List of common english words

    Returns:
    index (dict):
        - keys (string): Term
        - values (interger): Frequency
    """
    index = {} #Initailize dictionary
    stemmer = PorterStemmer()
    query = cleanLine(query) #clean the line

    for word in query.split(): #Tokenization
            word = word.lower() #Lower case the word
            word = stemmer.stem(word) #Stem the word
            if (not word in stop_words and len(word) > 2):
                try:
                    index[word] += 1
                except:
                    index[word] = 1

    return index





