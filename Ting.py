from neural import NeuralNet
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv')
dataset = df[['Post','Label']]
print(df['Label'].unqiue())



cv = TfidfVectorizer(max_df=0.9, min_df=2,max_features=300,stop_words='english')
text_counts = cv.fit_transform(dataset['Post'])

#print(text_counts.shape)
#print(cv.get_feature_names_out())
vectors = []
for i in range(0, 500):
    dic = []
    for j in range(0, 300):
        dic.append(text_counts[0, j])
    vectors.append(dic)
dataset['Vectors'] = vectors
#print("Vectorized!")
#print(str(cv.get_feature_names_out()[i]) + ": " + str(text_counts[0, i]))

dataset['Label'] = dataset['Label'].replace('Supportive','0').replace('Ideation','1').replace('Behavior','2').replace('Attempt','3').replace('Indicator','4')
#print(dataset)

accdata = ([dataset['Vectors'],[dataset['Label']]])
print(accdata)

von = NeuralNet()


