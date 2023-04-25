import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from models import multinomialNB, logisticRegression, gaussianNB

#dataset = pd.read_csv('dataset.txt', sep=";")
dataset = pd.read_csv('RegressionSetNew.txt', sep=";")
dataset['Target'] = 0
dataset['Bug Related'] = 0
#regressionSet = pd.read_csv('RegressionSetNew.txt', sep=";")
bugRelatedTests = pd.read_csv('BugRelatedTests.txt', sep=";")

#i = dataset.Key.isin(regressionSetNew.Key)
#dataset.loc[i, 'Target'] = 1

for i, row in dataset.iterrows():
    for j, row2 in bugRelatedTests.iterrows():
        if dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Inward issue link (Defect)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Outward issue link (Defect)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Outward issue link (Gantt End to End)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Inward issue link (Gantt End to Start)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Inward issue link (Parents)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Outward issue link (Relates)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Inward issue link (Tests)'] or \
                dataset.loc[i, 'Key'] == bugRelatedTests.loc[j, 'Inward issue link (Relates)']:
            dataset.loc[i, 'Bug Related'] = 1

for i, row in dataset.iterrows():
    if (dataset.loc[i, 'Priority'] == 'Critical') | (dataset.loc[i, 'Priority'] == 'High') | (dataset.loc[i, 'Priority'] == 'Medium') | (dataset.loc[i, 'Bug Related'] == 1):
        dataset.loc[i, 'Target'] = 1


# selected = dataset[
#    ((dataset['Priority'] == 'Critical') | (dataset['Priority'] == 'High') | (dataset['Priority'] == 'Medium')) & (
#               dataset['Bug Related'] == 1) & (dataset['Target'] == 1)]
# print(len(selected))

# numOfGenericTestCases = 0
# numOfManualTestCases = 0
# numOfCucumberTestCases = 0
# for i, row in dataset.iterrows():
#    if dataset.loc[i, 'Test Type'] == 'Generic':
#        numOfGenericTestCases += 1
#    elif dataset.loc[i, 'Test Type'] == 'Manual':
#        numOfManualTestCases += 1
#    else:
#        numOfCucumberTestCases += 1
# print(f"Number Of Generic Test Cases: {numOfGenericTestCases}")
# print(f"Number Of Manual Test Cases: {numOfManualTestCases}")
# print(f"Number Of Cucumber Test Cases: {numOfCucumberTestCases}")
def getGraphs(dataset):
    data = {'Selected': (len(dataset[(dataset['Target'] == 1)].index)), 'Not Selected': (len(dataset[(dataset['Target'] == 0)].index))}
    type = list(data.keys())
    nums = list(data.values())


    fig = plt.figure(figsize = (10, 5))

    plt.bar(type, nums, width = 0.4)

    plt.show()

#getGraphs(dataset)

labelencoder = LabelEncoder()
dataset['Priority'] = labelencoder.fit_transform(dataset['Priority'])
dataset['Bug Related'] = labelencoder.fit_transform(dataset['Bug Related'])

# print(dataset.head())
# print(dataset['Priority'])

corpus_title = []
stemmer = TurkishStemmer()
for i in range(dataset['Summary'].shape[0]):
    # Özel karakterler, sayılar gibi istenmeyen karakterleri çıkar
    text = re.sub("[^a-zA-Z]", ' ', dataset['Summary'][i])
    # Tüm harfleri küçük harfe çevir
    text = text.lower()
    text = text.split()
    # Stopword yani etkisiz kelimeleri çıkar
    text = [stemmer.stem(word) for word in text if not word in set(stopwords.words('turkish'))]
    text = ' '.join(text)
    # Yazının son halini corpus'a ekle
    corpus_title.append(text)

# Bag of Words modelini oluştur
cv = CountVectorizer()
text_vectors = cv.fit_transform(corpus_title).toarray()

# Text vector'lerini data frame haline getir
text_vectors_df = pd.DataFrame(text_vectors)
# print(f"**Dimension for Text features are {text_vectors_df.shape}**")

# Target değişkenini y değişkenine ata
y = dataset[['Target']].values
y = y.ravel()
y = y.astype('int')
# Target ve Summary özelliklerini datasetten çıkar
dataset = dataset.drop(['Target', 'Summary'], axis=1)
# Modelin kurulmasında işimize yaramayacak diğer özellikleri çıkar
dataset = dataset.drop(
    ['Key', 'Created', 'Reporter', 'Assignee', 'Test Repository Path', 'Test Type', 'Resolution', 'TestRunStatus'],
    axis=1)

# Geri kalan özellikleri kullanarak yeni data frame'in oluşturulması
X = pd.concat([dataset, text_vectors_df], axis=1).values
# print(f"**Dimension for features data frame are {X.shape}**")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# logisticRegression(X_train, X_test, y_train, y_test)
# gaussianNB(X_train, X_test, y_train, y_test)
multinomialNB(X_train, X_test, y_train, y_test)
