
import re

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import Ridge

data_train=pd.read_csv('../salary-train.csv')
data_test=pd.read_csv('../salary-test-mini.csv')
data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True,inplace=True)
data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True,inplace=True)

vectorizer = TfidfVectorizer(min_df=5)

fullDescription_train=vectorizer.fit_transform(data_train['FullDescription'])
fullDescription_test=vectorizer.transform(data_test['FullDescription'])
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

joinedMatrix_train=hstack([fullDescription_train,X_train_categ])

clf = Ridge(alpha=1.0)
clf.fit(joinedMatrix_train, data_train['SalaryNormalized'])
joinedMatrix_test=hstack([fullDescription_test,X_test_categ])
predict=clf.predict(joinedMatrix_test)
print('Прогноз')
print (' '.join(map(lambda x: str(round(x,2)), predict)))