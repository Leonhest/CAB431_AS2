import string
import re

class NewsItem:

    def __init__(self):
        self.newsID = "" #NewsItem ID
        self.terms = {} #List of Terms
        self.item_size = 0 #Size of document

    def getNewsId(self):
        """
        Return NewsItem ID
        """
        return self.newsID

    def setNewsId(self, newsID):
        """
        Set NewsItem ID

        Parameters:
        newsID (string): newsID
        """
        self.newsID = newsID
    
    def get_termList(self):
        """
        Return a sortted list of terms
        """
        listSorted = dict(sorted(self.terms.items(), key=lambda item: item[1], reverse=True))
        return listSorted

    def add_term(self,term):
        """
        Add term to list of terms

        Parameters:
        term (string): term
        """
        try:
            self.terms[term] += 1
        except KeyError:
            self.terms[term] = 1
    
    def getSize(self):
        """
        Return the size of the document
        """
        return self.item_size
    
    def setSize(self, size):
        """
        Set the size of the document

        Parameters:
        size (string): size of document
        """
        self.item_size = size

class Assignment_1:
    """
    collection of method from  1 to be used in assignment 2

    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    @staticmethod
    def Q_Parser(query, stop_words, stem):
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
        stemmer = stem
        query = Assignment_1.cleanLine(query) #clean the line

        for word in query.split(): #Tokenization
                word = word.lower() #Lower case the word
                word = stemmer.stem(word) #Stem the word
                if (not word in stop_words and len(word) > 2):
                    try:
                        index[word] += 1
                    except:
                        index[word] = 1

        return index
