import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

WORD2VEC_MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')
BINARY_MODEL = os.path.join(CUR_DIR, 'utils', 'model', 'binary_model.pkl')
TRAINING_DATA_PATH = os.path.join(CUR_DIR, 'data', '4training.csv')
