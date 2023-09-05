import pygal
from flask import Flask, render_template, request, session, url_for, Response
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import linear_model

import sys,os


filepath = os.getcwd()
app = Flask(__name__)
global LR1,RF1,NB1,SVM1


def f(x_train,x_test, y_train, y_test):
    global X_trains,X_tests,y_trains,y_tests
    X_trains = pd.DataFrame(x_train)
    X_tests = pd.DataFrame(x_test)
    y_trains = pd.DataFrame(y_train)
    y_tests = pd.DataFrame(y_test)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def uploaddataset():
    return render_template('uploaddataset.html')

@app.route('/traintest')
def traintestvalue():
    print("hello")
    return render_template('traintestdataset.html')

@app.route('/modelperformance')
def modelperformances():
    return render_template('modelperformance.html')

@app.route('/uploaddataset',methods=["POST","GET"])
def uploaddataset_csv_submitted():
    if request.method == "POST":
        csvfile = request.files['csvfile']
        result = csvfile.filename
        filepath.replace("\\","/")
        file = filepath +"\\" + result
        print(file)
        session['filepath'] = file
        return render_template('uploaddataset.html',msg='sucess')
    return render_template('uploaddataset.html')



@app.route('/viewdata',methods=["POST","GET"])
def viewdata():
    session_var_value = session.get('filepath')
    print("session variable is=====" + session_var_value)
    df = pd.read_csv(session_var_value)
    global x
    x = pd.DataFrame(df)
    x=x.dropna(how="any",axis=0)
    #le=preprocessing.LabelEncoder()
   # x['CHARSET']=le.fit_transform(x['CHARSET'])
    #le2 = preprocessing.LabelEncoder()
    #x['SERVER']=le2.fit_transform(x['SERVER'].astype("str"))
   # print(x.head(10))
   # session['x'] = x
    return render_template("viewdataset.html",col=x.columns.values, row_data=list(x.values.tolist()),zip=zip)

@app.route('/traintestdataset',methods=["POST","GET"])
def traintestdataset_submitted():
    if request.method == "POST":
        value = request.form['traintestvalue']
        value1=(value)
        df1=x
        df1["CONTENT_LENGTH"] = df1["CONTENT_LENGTH"].fillna(df1["CONTENT_LENGTH"].mean())
        X = df1.drop(["Result","URL"],axis=1)
        y = df1['Result']
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=float(value1))
        f(X_train,X_test, y_train, y_test)
        X_train1 = pd.DataFrame(X_train)
        X_trainlen=len(X_train)
        y_test1 = pd.DataFrame(y_test)
        y_testlen = len(y_test)
        return render_template('traintestdataset.html',msg='sucess',data=X_train1.to_html(),X_trainlenvalue=X_trainlen,y_testlenval=y_testlen)
    return render_template('traintestdataset.html')

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
@app.route('/modelperformance',methods=["POST","GET"])
def selected_model_submitted():
    global accuracyscore,precisionscore,recallscore
    if request.method == "POST":
        selectedalg =int(request.form['algorithm'])


        if selectedalg == 1:
            global p33,p31,accuracyscore
            model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            p31 = precision_score(y_tests, y_pred, average='macro')
            p33 = recall_score(y_tests, y_pred, average='macro')
            p36 = f1_score(y_true=y_tests, y_pred=y_pred, average='macro')
            p37 = roc_auc_score(y_true=y_tests, y_score=y_pred, average='macro')
            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="Decision Tree")
        elif selectedalg == 2:
            model = KNeighborsClassifier(n_neighbors=8)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            p61 = precision_score(y_tests, y_pred, average='macro')

            p63 = recall_score(y_tests, y_pred, average='macro')

            p66 = f1_score(y_true=y_tests, y_pred=y_pred, average='macro')

            p67 = roc_auc_score(y_true=y_tests, y_score=y_pred, average='macro')

            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="K Nearest Neighbour")
        elif selectedalg == 3:
            model = XGBClassifier()
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)
            p91 = precision_score(y_tests, y_pred, average='macro')

            p93 = recall_score(y_tests, y_pred, average='macro')

            p96 = f1_score(y_true=y_tests, y_pred=y_pred, average='macro')

            p97 = roc_auc_score(y_true=y_tests, y_score=y_pred, average='macro')

            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                       model="XG Boost")
        elif selectedalg==4:
            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(n_estimators=100, random_state=0)
            model.fit(X_trains, y_trains)
            y_pred = model.predict(X_tests)
            accuracyscore = accuracy_score(y_tests, y_pred)

            return render_template('modelperformance.html', msg="accuracy_score", score=accuracyscore,
                                   model="Ada Boost")




@app.route('/prediction', methods=["POST","GET"])
def prediction():
    print("hello")

    if request.method == "POST":
        print("hi")
        url = request.form['url_len']
        age = request.form['NUMBER_SPECIAL_CHARACTERS']
        length = request.form['CONTENT_LENGTH']
        traffic = request.form['SOURCE_APP_BYTES']
        anchor = request.form['APP_PACKETS']
        all_obj_vals = [[float(url), float(age), float(length), float(traffic), float(anchor)]]

        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        model.fit(X_trains, y_trains)

        predi = model.predict(all_obj_vals)
        pre = predi

        return render_template('prediction.html',msg='predictsucess',predvalue=predi)
    return render_template('prediction.html')

@app.route("/bar_chart")
def bar_chart():
    line_chart = pygal.Bar()
    line_chart.title = 'DETECTION AND IDENTIFICATION USING MACHINE LEARNING APPROACH'
    line_chart.add('PRECISION', [p33])
    line_chart.add('RECALL', [p31])
    line_chart.add('ACCURACY', [accuracyscore])
    graph_data = line_chart.render()
    return render_template('bar_chart.html', graph_data=graph_data)

if __name__=='__main__':
    app.secret_key = ".."
    app.run(debug=True)