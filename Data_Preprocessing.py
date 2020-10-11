import re
import _pickle as cPickle
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



#stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def data_preprocessing(review):
    review = re.sub(re.compile('<.*?>'), '', review)
    review = re.sub('[^A-Za-z0-9]+',' ', review)
    review = review.lower()
    #tokens = nltk.word_tokenize(review)
    #review = [word for word in tokens if word not in stop_words]
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ''.join(review)
    return review
