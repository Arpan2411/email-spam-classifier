import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure required NLTK resources are available before tokenization.
# NLTK 3.8+ uses punkt_tab for word_tokenize.
for resource in ('stopwords', 'punkt_tab'):
    try:
        if resource == 'stopwords':
            stopwords.words('english')
        else:
            nltk.data.find(f'tokenizers/{resource}/english.pickle')
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
stem_words = PorterStemmer()
def data_preprocess_pipeline(text):
    # converting text to lower case
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    # keeping only alphabets and numerical values
    for i in text:
        if i.isalnum():
            y.append(i)
    text.clear()
    text = y.copy()
    y.clear()
    # removing stop words
    for i in text:
        if i not in stop_words:
            y.append(i)
    text.clear()
    text = y.copy()
    y.clear()
    # applying stemming
    for i in text:
        y.append(stem_words.stem(i))
    text.clear()
    text = y.copy()
    y.clear()
    return " ".join(text) # converting the list to string