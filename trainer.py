import gensim
import string
import spacy
import numpy as np
import joblib
import pandas as pd
from operator import add
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import Word2Vec
from nltk.data import find
from model_trainer import ClassifierTrainer
from settings import WORD2VEC_MODEL_PATH, TRAINING_DATA_PATH, BINARY_MODEL


class Feedback:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.result = []
        word2vec_sample = str(find(WORD2VEC_MODEL_PATH))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        self.binary_model = joblib.load(BINARY_MODEL)
        self.classifier = ClassifierTrainer()

    @staticmethod
    def calculate_text_feature(word_features):
        text_feature = word_features[0]
        for w_feature in word_features[1:]:
            text_feature = list(map(add, text_feature, w_feature))
        return text_feature

    def tokenize_word(self, sentence):
        symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "/"]

        tokens = self.nlp(sentence)
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

        tokens = lemmas

        tokens = [tok for tok in tokens if tok not in STOP_WORDS]
        tokens = [tok for tok in tokens if tok not in symbols]
        tokens = [tok for tok in tokens if tok != ""]

        return tokens

    def extract_feature_sentence(self, text):

        text_features = []
        if len(text) > 1:
            for raw_sentence in text:
                words = self.tokenize_word(sentence=raw_sentence)

                for t_word in words:
                    try:
                        word_feature = self.word2vec_model[t_word.replace(" ", "")]
                        text_features.append(word_feature)
                    except Exception as e:
                        print(e)
        else:
            text = " ".join(map(str, text))
            words = self.tokenize_word(sentence=text)
            for t_word in words:
                try:
                    word_feature = self.word2vec_model[t_word.replace(" ", "")]
                    text_features.append(word_feature)
                except Exception as e:
                    print(e)

        return text_features

    def extract_features(self):
        train_x_feature = []
        train_y_feature = []

        col_list = ['reviews', 'label']

        data = pd.read_csv(TRAINING_DATA_PATH, usecols=col_list)

        for re_row, lbl_row in zip(data['reviews'], data['label']):
            text = re_row.split('.')
            try:
                sentences = []
                for sentence in text:
                    if not sentence == "":
                        sentences.append(sentence.lower())
                text_features = self.extract_feature_sentence(text=sentences)
                text_feature = self.calculate_text_feature(word_features=text_features)
                train_x_feature.append(text_feature)
                train_y_feature.append(lbl_row)
            except Exception as e:
                print(e)
                print(text)

        return train_x_feature, train_y_feature

    def train_binary_classification(self):
        train_x_data, train_y_data = self.extract_features()
        models, accuracies = self.classifier.train_best_model(model_name="RBF SVM", x_data=train_x_data, y_data=train_y_data)
        max_model = models[accuracies.index(max(accuracies))]
        print("INFO:: Best model:", max_model)
        self.classifier.train_best_model(model_name=max_model, x_data=train_x_data, y_data=train_y_data)

        return

    def run(self, text):
        text = text.split('.')
        sentences = []
        for sentence in text:
            if not sentence == "":
                sentences.append(sentence.lower())
        text_feature = self.calculate_text_feature(self.extract_feature_sentence(text=sentences))
        predict_result = self.binary_model.predict(np.array(text_feature).reshape(1, -1))
        if predict_result[0] == 'Pos':
            result = 'Positive'
        else:
            result = 'Negative'
        print(f"Result: {predict_result[0]}")

        return result


if __name__ == '__main__':
    Feedback().train_binary_classification()
