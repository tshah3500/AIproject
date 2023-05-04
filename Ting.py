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

keys = {"Supportive":0, "Ideation":1, "Behavior":2, "Attempt":3, "Indicator":4}
def output_function(str):
    output = [0, 0, 0, 0, 0]
    output[keys[str]] += 1
    return(output)

#dataset['Label'] = dataset['Label'].replace('Supportive',0).replace('Ideation',1).replace('Behavior',2).replace('Attempt',3).replace('Indicator',4)
dataset['Output'] = dataset['Label'].apply(output_function)
#print(dataset['Output'])
#print(dataset['Label'].unique())

final = []
for i in range(0, 400):
    curr_list = [dataset['Vectors'][i], dataset['Output'][i]]
    final.append(curr_list)
#print(final)

#accdata = ([dataset['Vectors'],[dataset['Label']]])
#print(accdata)

print("Creating Neural Net")
von = NeuralNet(300, 25, 5)
von.train(final)




