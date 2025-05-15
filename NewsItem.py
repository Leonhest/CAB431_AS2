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

