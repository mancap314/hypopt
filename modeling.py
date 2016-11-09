import runs
import preproc
from sklearn.preprocessing import LabelEncoder

titanic = preproc.preproc("train.csv")

variables = ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]
X = titanic[variables].astype(float)

non_numeric_col = ['Sex', 'Title']
le = LabelEncoder()
for feature in non_numeric_col:
    X[feature] = le.fit_transform(X[feature])

y = titanic["Survived"].astype('category')

methods = ['logistic_regression', 'knn', 'random_forest', 'xgb', 'decision_tree']
runs.set_X(X)
runs.set_y(y)
runs.set_methods(methods)
runs.runs()
