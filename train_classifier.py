import sys
import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#used Random Forest as it gave slightly better accuracy than SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
import pickle

def load_data(data_filepath):
    engine = create_engine('sqlite:///'+data_filepath)
    df = pd.read_sql_table('dataset', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_options =list(df.iloc[:, 4:])
    return X,Y,category_options


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tok=[]
    for word in tokens:
        tok.append(lemmatizer.lemmatize(word))
    return tok


def build_model(parameter_array={}):
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                            ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(**parameter_array)))
                        ])
    return pipeline


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def implement_improved_model(pipeline,X_train,y_train):
    parameters = {
        'clf__estimator__n_estimators': [20,50],
        'clf__estimator__max_depth':[2,5],
        'clf__estimator__criterion': ['entropy', 'gini'],
        'clf__estimator__min_samples_split':[3,6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    cv.fit(X_train, y_train)
    return cv.best_params_

from sklearn.metrics import classification_report
def test_your_model(model, X_test, y_test, category_options):
    y_test = pd.DataFrame(y_test,index=y_test[:,0])
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
    num_categories = len(category_options)
    #category_wise report
    i=0
    for column in y_test.columns:
        print("For category:", category_options[i],"Accuracy = \n",classification_report(y_test[column], y_pred_df[column]))
        i+=1
    
    
    print("Overall accuracy",(y_pred == y_test).mean().mean())

#saving the model using joblib
def model_save(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))

    
def main():
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_options = load_data(database_filepath)
        X_train, X_test, y_train, y_test = split_data(X,y)

        print('Building model...')
        model = build_model()
       
       
        
        print('Grid search')
        cv = implement_improved_model(model, X_train, y_train)
        
        print("Best parameters")
        print(cv)
        
        #assigning the values of best parameters obtained from grid search to the new model
        new_params = {
                    'n_estimators': cv['clf__estimator__n_estimators'],
                    'max_depth': cv['clf__estimator__max_depth'],
                    'criterion': cv['clf__estimator__criterion'],
                    'min_samples_split' :cv['clf__estimator__min_samples_split'],
                    'verbose' : 1
                }


        print('Improvised model')
        
        new_model = build_model(new_params)
        new_model.fit(X_train, y_train)
        print('testing the model')
        test_your_model(new_model, X_test, y_test, category_options)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        model_save(new_model, model_filepath)

        print('Trained model saved successfully')


if __name__ == '__main__':
    main()