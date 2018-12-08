# HOW TO CALL FUNCTION

# import importlib.util
# spec = importlib.util.spec_from_file_location("nltk_pack", "nltk_pack.py")
# pack = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(pack)
# pack.word_tokenize(w)

def get_wordnet_pos(treebank_tag):
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
        # get only non proper noun
#         if treebank_tag.startswith('NNP'):
#             return wordnet.NOUN
#         else:
#             return ''
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
def lemmatizer(word,wtype):
    from nltk.stem.wordnet import WordNetLemmatizer
    w_type = get_wordnet_pos(wtype)
    
    if (w_type != ''):
        return [WordNetLemmatizer().lemmatize(word,w_type),w_type]
    else:
        return [word,wtype]
    
def word_tokenize(review):
    import nltk.tokenize as nt
    import nltk
    import pandas as pd 
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # clean punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sent=tokenizer.tokenize(review.lower())
    
    # clean stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    import numpy as np
    stop_words = np.append(np.array(stop_words),np.array(['s','ll'])) # get stopword
    words = [w for w in tokenized_sent if not w in stop_words]
    pos=pd.DataFrame(nltk.pos_tag(words), columns=['word','type'])
    
#     ps = PorterStemmer()
#     pos['word'] = [ps.stem(w) for w in pos['word']]
    # normalize words by lemmatizing ex. walking -> walk
    newpos = pd.DataFrame([lemmatizer(w[0],w[1]) for w in np.array(pos)], columns=['newword','newtype'])

    return newpos[newpos['newtype'].isin(['n','v','r','a'])] # take only noun, verb, adv, adj 

def word_tokenize2(review):
    import nltk.tokenize as nt
    import nltk
    import pandas as pd 
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # clean punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sent=tokenizer.tokenize(review.lower())
    
    # clean stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    import numpy as np
    stop_words = np.append(np.array(stop_words),np.array(['s','ll'])) # get stopword
    words = [w for w in tokenized_sent if not w in stop_words]
    pos=pd.DataFrame(nltk.pos_tag(words), columns=['word','type'])
#     ps = PorterStemmer()
#     pos['word'] = [ps.stem(w) for w in pos['word']]
    # normalize words
    newpos = pd.DataFrame([lemmatizer(w[0],w[1]) for w in np.array(pos)], columns=['newword','newtype'])
#     print(newpos[newpos['newtype'].isin(['r','a'])])
    return newpos[newpos['newtype'].isin(['r','a'])] # take only noun, verb, adv, adj 

def word_analyze(review):
    # compute sentiment scores (polarity) and labels
    import numpy as np
    from afinn import Afinn
    af = Afinn() 
#     data = [word_tokenize(w)['newword'] for w in review]
#     return [print(w) for w in data for i in w]
    data = word_tokenize(review)['newword']
    return [af.score(w) for w in data]

def getwordlist(data):
    import collections
    import numpy as np
    import pandas as pd
    from collections import Counter
    countdict = Counter([i for w in [word_tokenize(w)['newword'] for w in data] for i in w])
    dict = pd.DataFrame.from_dict(countdict, orient='index').reset_index()
    return dict

def getfeature(data, max = 1000):
    
    # HOW TO USE
    # pack.getfeature(textreviewlist)
    data = [' '.join((word_tokenize(w)['newword'])) for w in data]
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = max)
    X = vectorizer.fit_transform(data)
    print(len(vectorizer.get_feature_names()[:]))
    return pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()[:])

def getfeature2(data, max = 1000, ngram=1):
    
    # HOW TO USE
    # pack.getfeature(textreviewlist)
    # this version 2 filter N and V out
    data2 = [' '.join((word_tokenize2(w)['newword'])) for w in data]
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, ngram), analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = max)
    X = vectorizer.fit_transform(data2)
    return pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()[:])

def getlist(data):
    
    # HOW TO USE
    # pack.getfeature(textreviewlist)
    data = [' '.join((word_tokenize(w)['newword'])) for w in data]
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = None)
    X = vectorizer.fit_transform(data)
    return vectorizer.get_feature_names()[:]



