import re
from string import punctuation
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

FILENAME = 'spam.csv'
PREPARED = 'prepared.csv'
ENCODING = 'iso-8859-1'
NUMBER_STR = 'aanumbers'
SEED = 43
TRAIN_FRAC = 0.8


def get_data(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, usecols=[0, 1],
                       header=0,
                       dtype='string',
                       names=['Target', 'SMS'],
                       encoding=ENCODING).fillna('')


def prepare_data(data: pd.DataFrame) -> None:
    en_sm_model = spacy.load('en_core_web_sm')
    remove_punct = str.maketrans('', '', punctuation)
    with_numbers = re.compile(r'^.*\d.*$')

    def prepare_line(line: str) -> str:
        word_list = []
        for i in en_sm_model(line.lower()):
            word = i.lemma_.translate(remove_punct)
            word = re.sub(with_numbers, NUMBER_STR, word)
            if len(word) > 1 and word not in STOP_WORDS:
                word_list.append(word)
        return ' '.join(word_list)

    data['SMS'] = data['SMS'].apply(prepare_line)


def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    shuffle = data.sample(frac=1, random_state=SEED)
    last = int(shuffle.shape[0] * TRAIN_FRAC)
    return shuffle[:last], shuffle[last:]


def bag_of_words(data: pd.Series) -> pd.DataFrame:
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    return pd.DataFrame(vectorizer.transform(data).toarray(),
                        columns=vectorizer.get_feature_names_out())


class CustomNB:
    def __init__(self, alpha=1.0):
        self.alpha: float = alpha
        self.p_words: pd.DataFrame = pd.DataFrame()
        self.priors: pd.Series = pd.Series()
        self.vocab_set: set[str] = set()

    def fit(self, data: pd.DataFrame):
        bow = data.reset_index(drop=True).join(bag_of_words(data.SMS))
        n_words = bow.groupby('Target').sum(numeric_only=True).T
        self.p_words = (n_words + self.alpha).divide(n_words.sum() + self.alpha * n_words.shape[0])
        self.priors = data.Target.value_counts() / data.shape[0]
        self.vocab_set = set(self.p_words.index)

    def predict(self, msg: str) -> str:
        words = [w for w in msg.split() if w in self.vocab_set]
        p_msg: pd.Series = self.p_words.loc[words].prod() * self.priors
        result = 'unknown'
        if p_msg.spam > p_msg.ham:
            result = 'spam'
        elif p_msg.spam < p_msg.ham:
            result = 'ham'
        return result


try:
    df = get_data(PREPARED)
except FileNotFoundError:
    df = get_data(FILENAME)
    prepare_data(df)
    df.to_csv(PREPARED, index=False, encoding=ENCODING)

train_set, test_set = split_data(df)

# model = CustomNB()
# model.fit(train_set)
# actual = test_set.Target
# predicted = test_set.SMS.apply(model.predict)

train_bow = bag_of_words(train_set.SMS)
train_target = train_set.Target.reset_index(drop=True) == 'spam'
test_bow = bag_of_words(test_set.SMS)
test_target = test_set.Target.reset_index(drop=True) == 'spam'

missing_features_train = [w for w in train_bow.columns if w not in test_bow.columns]
missing_features_test = [w for w in test_bow.columns if w not in train_bow.columns]
test_bow = (test_bow.drop(columns=missing_features_test).
            join(pd.DataFrame(0, columns=missing_features_train, index=test_bow.index)).
            sort_index(axis='columns'))

model = MultinomialNB()
model.fit(train_bow, train_target)

actual = test_target
predicted = model.predict(test_bow)
tn, fp, fn, tp = confusion_matrix(actual, predicted).flatten()

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 * precision * recall / (precision + recall)

print({'Accuracy': accuracy,
       'Recall': recall,
       'Precision': precision,
       'F1': f1})
