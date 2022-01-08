from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump


import neptune.new as neptune

run = neptune.init(project="kaori/parctice", 
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjZiMDA2YS02MDM3LTQxZjQtOTE4YS1jODZkMTJjNGJlMDYifQ==",
                    source_files=['*.py', 'requirements.txt'])
run["sys/tags"].add(['run-organization', 'me:kaori'])  # organize things



data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.4,
                                                    random_state=1234)

params = {'n_estimators': 2,
          'max_depth': 3,
          'min_samples_leaf': 1,
          'min_samples_split': 2,
          'max_features': 3,
          }

run['parameters'] = params

clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)
y_train_pred = clf.predict_proba(X_train)
y_test_pred = clf.predict_proba(X_test)

train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
print(f'Train f1:{train_f1} | Test f1:{test_f1}')

dump(clf, 'model.pkl')

#logging
run['train/f1'] = train_f1
run['test/f1'] = test_f1
run["model"].upload('my_model.pkl')

