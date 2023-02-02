import pickle
import string
import nltk

nltk.download('omw-1.4')
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim


nltk.download('stopwords')
# from nltk.corpus import stopwords
en_stop = set(nltk.corpus.stopwords.words('english'))

nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import spacy

spacy.load('en_core_web_sm')
from spacy.lang.en import English

parser = English()
stemmer = SnowballStemmer('english')

NUM_TOPICS = 20
STEMMING_CRYPTO_STOPWORDS = ["exchang", "servic"]
CRYPTO_STOPWORDS = ["exchange", "service"]


# https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        # elif token.like_url:
        #     lda_tokens.append('URL')
        # elif token.orth_.startswith('@'):
        #     lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def clean_token(token):
    return token.strip(string.punctuation)


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    token_prepared = tokenize(text)
    token_prepared = [clean_token(token) for token in token_prepared]
    token_prepared = [token for token in token_prepared if token != ""]
    token_prepared = [token for token in token_prepared if token not in en_stop]

    # remove any of our custom stopwords
    # token_prepared = [token for token in token_prepared if token not in CRYPTO_STOPWORDS]

    # choose if we want stemming, or lemmatizing
    token_prepared = [stemmer.stem(token) for token in token_prepared] # stem word
    # token_prepared = [get_lemma(token) for token in token_prepared] # use lemma
    # token_prepared = [get_lemma2(token) for token in token_prepared] # use lemma2

    # remove any of our custom stopwords (after NLP)
    token_prepared = [token for token in token_prepared if token not in STEMMING_CRYPTO_STOPWORDS]

    return token_prepared


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    text_data = []
    with open('data/amanda_orgs.csv', encoding="utf8") as f:
        for line in f:
            tokens = prepare_text_for_lda(line)
            # if random.random() > .99:
            #     print(tokens)
            text_data.append(tokens)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    # Loop Run
    # start = 5
    # limit = 60
    # step = 5
    # perplexity_list = []
    # coherence_list = []
    # for num_topics in range(start, limit, step):
    #     print("Number of topics: " + str(num_topics))
    #     ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    #     cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence="u_mass")
    #     # print("coherence: " + str(cm.get_coherence()))
    #     # print("perplexity:" + str(ldamodel.log_perplexity(corpus)))
    #
    #     perplexity_list.append(ldamodel.log_perplexity(corpus))
    #     coherence_list.append(cm.get_coherence())
    #
    #     # ldamodel.save('model5.gensim')
    #     # topics = ldamodel.print_topics(num_words=5)
    #     # for topic in topics:
    #     #     print(topic)
    #
    #     # lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    #     # pyLDAvis.save_html(lda_display, 'lda_result.html')

    # print("Perplexity List: " + str(perplexity_list))
    # print("Coherence List: " + str(coherence_list))

    # Solo Run
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence="u_mass")
    print("coherence: " + str(cm.get_coherence()))
    print("perplexity:" + str(ldamodel.log_perplexity(corpus)))
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display, 'lda_result.html')
