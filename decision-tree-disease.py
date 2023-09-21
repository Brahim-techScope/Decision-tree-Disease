import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree


## Prepare the data
col_names = ['disease', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'symptom_6', 'symptom_7', 'symptom_8', 'symptom_9']
data = pd.read_csv("disease_dataset/dataset.csv", header=None)
data = data.iloc[1:, :10]
data.columns = col_names

X = data.drop(['disease'], axis=1)
y = data['disease']

## split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 23)

## encode variables with ordinal encoding
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'symptom_6', 'symptom_7', 'symptom_8', 'symptom_9'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

## instantiate the DecisionTreeClassifier model with entropy criterion
clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=0) # Fell free to change entropy to gini
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
print('Testing-set accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

## Visualise decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=clf.classes_.tolist())
plt.show()

## Export the decision tree in Graphviz DOT format
'''
export_graphviz(clf, out_file='decision_tree.dot', 
                feature_names=['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'symptom_6', 'symptom_7', 'symptom_8', 'symptom_9'],
                class_names=clf.classes_,
                filled=True)
'''