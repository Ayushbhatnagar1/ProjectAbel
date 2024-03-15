import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter

def load_lm_dictionary(lm_path):
    lm_dict = {}
    with open(lm_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lm_dict[row['Word'].lower()] = {'positive': int(row.get('Positive', 0)),
                                             'negative': int(row.get('Negative', 0))}
    return lm_dict

def load_correa_dictionary(correa_path):
    correa_df = pd.read_excel(correa_path)
    correa_dict = {}
    for index, row in correa_df.iterrows():
        word = str(row['Word']).lower().strip()
        positive_score = row['Positive'] if not pd.isna(row['Positive']) else 0
        negative_score = row['Negative'] if not pd.isna(row['Negative']) else 0
        correa_dict[word] = {'positive': int(positive_score),
                             'negative': int(negative_score)}
    return correa_dict

def load_neutral_dictionary(neutral_path):
    neutral_df = pd.read_excel(neutral_path)
    neutral_dict = {row['Word'].lower(): 'neutral' for index, row in neutral_df.iterrows()}
    return neutral_dict

def combine_dictionaries(*dicts):
    combined_dict = {}
    for d in dicts:
        for word, scores in d.items():
            if word not in combined_dict:
                combined_dict[word] = scores
            else:
                combined_dict[word]['positive'] += scores.get('positive', 0)
                combined_dict[word]['negative'] += scores.get('negative', 0)
    return combined_dict

def tokenize(document):
    return word_tokenize(document)

def score_document(document, financial_dictionary, neutral_dictionary):
    words = tokenize(document.lower())
    word_freq = Counter(words)

    positive_words = []
    negative_words = []
    neutral_words = []

    for word in words:
        if word in financial_dictionary:
            if financial_dictionary[word].get('positive', 0) > 0:
                positive_words.append(word)
            if financial_dictionary[word].get('negative', 0) > 0:
                negative_words.append(word)
        if word in neutral_dictionary:
            neutral_words.append(word)

    positive_score = sum(word_freq[word] for word in positive_words)
    negative_score = sum(word_freq[word] for word in negative_words)
    neutral_count = sum(word_freq[word] for word in neutral_words)

    total_words = len(words)
    sentiment_score = (positive_score - negative_score) / total_words if total_words > 0 else 0
    neutral_proportion = neutral_count / total_words if total_words > 0 else 0

    return {
        'sentiment_score': sentiment_score,
        'positive_score': positive_score,
        'negative_score': negative_score,
        'neutral_count': neutral_count,
        'neutral_proportion': neutral_proportion,
        'positive_words': positive_words,
        'negative_words': negative_words,
        'neutral_words': neutral_words
    }

lm_dictionary = load_lm_dictionary('lm.csv')
correa_dictionary = load_correa_dictionary('correa1.xlsx')
neutral_dictionary = load_neutral_dictionary('neutral.xlsx')

combined_dictionary = combine_dictionaries(lm_dictionary, correa_dictionary)

with open('file1.txt', 'r') as file:
    document_text = file.read()

result = score_document(document_text, combined_dictionary, neutral_dictionary)

print(f"Sentiment Score: {result['sentiment_score']}")
print(f"Positive Score: {result['positive_score']}, Words: {set(result['positive_words'])}")
print(f"Negative Score: {result['negative_score']}, Words: {set(result['negative_words'])}")
print(f"Neutral Count: {result['neutral_count']}, Words: {set(result['neutral_words'])}")
print(f"Neutral Proportion: {result['neutral_proportion']}")
