import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def tf_idf_accuracy(dataset: str, omit_ctrl = False, omit_middle = False):
    '''
    Logistic Regression model using TF-IDF for extracting hidden income traits from a dataset
    Args:
        dataset (string): Filename of the dataset to train/test model on
        omit_ctrl (boolean): Flag to ignore control data in dataset
        omit_middle (boolean): Flag to ignore middle class data in dataset
    '''
    vectorizer = TfidfVectorizer()
    data = pd.read_csv(dataset)

    if omit_ctrl == True:
        data = data[data['trait_selected'] != 0]

    if omit_middle == True:
        data = data[data['trait_selected'] != 2]


    outputs = data[['llm_output_0', 'llm_output_1', 'llm_output_2', 'llm_output_3', 'llm_output_4']]
    outputList = outputs.astype(str).agg(' '.join, axis=1).tolist()
    labels = data[['trait_selected']].astype(str).agg(' '.join, axis=1).tolist()
    data_train, data_test, label_train, label_test = train_test_split(outputList, labels, test_size=0.25, random_state=42)
    data_train_tfidf = vectorizer.fit_transform(data_train)
    data_test_tfidf = vectorizer.transform(data_test)

    model = LogisticRegression()
    model.fit(data_train_tfidf, label_train)
    label_prediction = model.predict(data_test_tfidf)
    accuracy = accuracy_score(label_test, label_prediction)
    conf_matrix = confusion_matrix(label_test, label_prediction)
    return accuracy, conf_matrix
