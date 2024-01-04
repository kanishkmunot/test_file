import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import nltk
import random
import string
import re, unicodedata
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia as wk
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ignore warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize chatbot
englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(englishBot)
trainer.train("chatterbot.corpus.english")

# Load data
data = open('haus.txt', 'r', errors='ignore')
raw = data.read()

# Converting all data into a list of sentences
sent_tokens = nltk.sent_tokenize(raw)


def Normalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    rmv = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        rmv.append(text)
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list


welcome_input = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
welcome_response = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]


def welcome(user_response):
    for word in user_response.split():
        if word.lower() in welcome_input:
            return random.choice(welcome_response)


def generateResponse(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0) or "tell me about" in user_response:
        print("Checking Wikipedia")
        if user_response:
            robo_response = wikipedia_data(user_response)
            return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


def wikipedia_data(input):
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            wiki = wk.summary(topic, sentences=3)
            return wiki
    except Exception as e:
        print("No content has been found")


def mainloop(user_response):
    flag = True
    while flag:
        user_response = input("You: ").lower()
        if user_response not in ['bye', 'shutdown', 'exit', 'see you', 'quit']:
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print("Chatterbot: You are welcome..")
            else:
                if welcome(user_response) is not None:
                    print("Chatterbot: " + welcome(user_response))
                else:
                    print("Chatterbot: ", end="")
                    print(generateResponse(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print("Chatterbot: Bye!!! ")


if __name__ == '__main__':
    mainloop("")
