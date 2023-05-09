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
dataset.dropna(subset='Post', inplace=True)
dataset.reset_index(drop=True, inplace=True)


cv = TfidfVectorizer(max_df=0.9, min_df=2,max_features=100,stop_words='english')
text_counts = cv.fit_transform(dataset['Post'])

#print(text_counts.shape)
#print(cv.get_feature_names_out())
vectors = []
for i in range(0, 500):
    dic = []
    for j in range(0, 100):
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

training = []
for i in range(0, 400):
    curr_list = [dataset['Vectors'][i], dataset['Output'][i]]
    training.append(curr_list)

testing = []
for i in range(400, 500):
    #curr_list = [dataset['Vectors'][i], dataset['Output'][i]]
    testing.append(dataset['Vectors'][i])
#print(final)

#accdata = ([dataset['Vectors'],[dataset['Label']]])
#print(accdata)

print("Creating Neural Net")
von = NeuralNet(100, 5, 5)
von.train(training)

print("Testing Neural Net")
predictions = von.test(testing)
outputs = keys = {0:"Supportive", 1:"Ideation", 2:"Behavior", 3:"Attempt", 4:"Indicator"}
    
for j in range(0, 100):
    i = predictions[j]
    print("Predicted: " + str(outputs[i.index(max(i))]) + " Actual: " + str(outputs[dataset['Output'][j + 400].index(max(dataset['Output'][j + 400]))]))

