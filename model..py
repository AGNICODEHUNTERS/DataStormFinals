import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = pd.get_dummies(test,columns=['gender'])
data = pd.get_dummies(data,columns=['gender'])
data = data.dropna()
test = test.dropna()
cid = test.customer_id.values
X= data.drop('Churn',axis=1).values
y = data.Churn.values
print(X)
print(y)

'''clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X, y, cv=2,scoring='f1_macro')
print(scores.mean())

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=2,scoring='f1_macro')
print(scores.mean())'''

clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y, cv=2,scoring='f1_macro')
#print(scores.mean())

clf.fit(X,y)
pr=clf.predict(test)
df = pd.DataFrame()
df.insert(0,'customer_id',cid)
df.insert(1,'churn',pd.DataFrame(pr))
df.to_csv('pr.csv',index=alse)
