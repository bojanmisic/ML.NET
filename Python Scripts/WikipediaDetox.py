import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# download annotated comments and annotations
# ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634' 
# ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637' 

# def download_file(url, fname):
#    urllib.request.urlretrieve(url, fname)
          
# download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
# download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')

len(annotations['rev_id'].unique())

# labels a comment as an atack if the majority of annoatators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments
comments['attack'] = labels

# remove newline and tab tokens
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

# comments.query('attack')['comment'].head()
print(comments.groupby('attack').count())

# fit a simple text classifier
train_comments = comments.query("split=='train'")
test_comments = comments.query("split=='test'")

train_comments_filtered = train_comments[['comment', 'attack']]
test_comments_filtered = test_comments[['comment', 'attack']]

train_comments_filtered.to_csv("train.csv", sep='\t')
test_comments_filtered.to_csv("test.csv", sep='\t')

clf = Pipeline([
    ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
    ('tfidf', TfidfTransformer(norm = 'l2')),
    ('clf', LogisticRegression()),
])

clf = clf.fit(train_comments['comment'], train_comments['attack'])
auc = roc_auc_score(test_comments['attack'], clf.predict_proba(test_comments['comment'])[:, 1])
print('Test ROC AUC: %.3f' %auc)

# correctly classify nice comment
clf.predict(['Thanks for you contribution, you did a great job!'])

# correctly classify nasty comment
clf.predict(['People as stupid as you should not edit Wikipedia!'])