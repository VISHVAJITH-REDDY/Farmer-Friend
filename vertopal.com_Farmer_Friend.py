

Crop Recommendation

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

Importing Dataset

    import pandas as pd
    df=pd.read_csv('/content/Crop_recommendation.csv')

    df.head(2)

        N   P   K  temperature   humidity        ph    rainfall label
    0  90  42  43    20.879744  82.002744  6.502985  202.935536  rice
    1  85  58  41    21.770462  80.319644  7.038096  226.655537  rice

#Classifiers

x & y split

    x=df.drop(['label'],axis=1)

    y=df['label'].values

    np.unique(y)

    array(['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
           'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
           'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
           'pigeonpeas', 'pomegranate', 'rice', 'watermelon'], dtype=object)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.naive_bayes import GaussianNB

    x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size = 0.20, 
                                                        random_state = 42)

    classifiers = [             ['LogisticRegression :', LogisticRegression(max_iter = 1000)],
                   
                   ['DecisionTree :',DecisionTreeClassifier()],
               
                   ['KNeighbours :', KNeighborsClassifier()],
                   ['naive bays :', GaussianNB()],
                  ]

    predictions_df = pd.DataFrame()
    predictions_df['action'] = y_test
    for name,classifier in classifiers:
        classifier = classifier
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        predictions_df[name.strip(" :")] = predictions
        print(name, accuracy_score(y_test, predictions))

    LogisticRegression : 0.95
    DecisionTree : 0.9863636363636363
    KNeighbours : 0.9704545454545455
    naive bays : 0.9954545454545455

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,

#Voting Classifier

A Voting Classifier is a machine learning model that trains on an
ensemble of numerous models and predicts an output (class) based on
their highest probability of chosen class as the output. It simply
aggregates the findings of each classifier passed into Voting Classifier
and predicts the output class based on the highest majority of voting.
The idea is instead of creating separate dedicated models and finding
the accuracy for each them, we create a single model which trains by
these models and predicts output based on their combined majority of
voting for each output class.

Soft Voting: In soft voting, the output class is the prediction based on
the average of probability given to that class. Suppose given some input
to three models, the prediction probability for class A = (0.30, 0.47,
0.53) and B = (0.20, 0.32, 0.40). So the average for class A is 0.4333
and B is 0.3067, the winner is clearly class A because it had the
highest probability averaged by each classifier.

    from sklearn.ensemble import VotingClassifier
    clf1 = LogisticRegression()
    clf2=KNeighborsClassifier()
    clf3 = DecisionTreeClassifier()
    eclf1 = VotingClassifier(estimators=[('', clf1), ('KNN', clf2), ('DecisionTree', clf3)], voting='soft')
    eclf1.fit(x_train, y_train)
    predictions = eclf1.predict(x_test)
    print(classification_report(y_test, predictions))

                  precision    recall  f1-score   support

           apple       1.00      1.00      1.00        23
          banana       1.00      1.00      1.00        21
       blackgram       0.95      1.00      0.98        20
        chickpea       1.00      1.00      1.00        26
         coconut       1.00      1.00      1.00        27
          coffee       1.00      1.00      1.00        17
          cotton       0.94      1.00      0.97        17
          grapes       1.00      1.00      1.00        14
            jute       0.92      0.96      0.94        23
     kidneybeans       1.00      1.00      1.00        20
          lentil       0.92      1.00      0.96        11
           maize       1.00      0.95      0.98        21
           mango       1.00      1.00      1.00        19
       mothbeans       1.00      0.96      0.98        24
        mungbean       1.00      1.00      1.00        19
       muskmelon       1.00      1.00      1.00        17
          orange       1.00      1.00      1.00        14
          papaya       1.00      1.00      1.00        23
      pigeonpeas       1.00      0.96      0.98        23
     pomegranate       1.00      1.00      1.00        23
            rice       0.94      0.89      0.92        19
      watermelon       1.00      1.00      1.00        19

        accuracy                           0.99       440
       macro avg       0.99      0.99      0.99       440
    weighted avg       0.99      0.99      0.99       440

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,

    import pickle
    pickle_out=open('crop.pkl','wb')
    pickle.dump(eclf1,pickle_out)
    pickle_out.close()

Fertilizer Prediction

    pd.read_csv("/content/Fertilizer_Prediction.csv")

        Temparature  Humidity   Moisture Soil Type  Crop Type  Nitrogen  \
    0            26         52        38     Sandy      Maize        37   
    1            29         52        45     Loamy  Sugarcane        12   
    2            34         65        62     Black     Cotton         7   
    3            32         62        34       Red    Tobacco        22   
    4            28         54        46    Clayey      Paddy        35   
    ..          ...        ...       ...       ...        ...       ...   
    94           25         50        32    Clayey     Pulses        24   
    95           30         60        27       Red    Tobacco         4   
    96           38         72        51     Loamy      Wheat        39   
    97           36         60        43     Sandy    Millets        15   
    98           29         58        57     Black  Sugarcane        12   

        Potassium  Phosphorous Fertilizer Name  
    0           0            0            Urea  
    1           0           36             DAP  
    2           9           30        14-35-14  
    3           0           20           28-28  
    4           0            0            Urea  
    ..        ...          ...             ...  
    94          0           19           28-28  
    95         17           17        10-26-26  
    96          0            0            Urea  
    97          0           41             DAP  
    98          0           10           20-20  

    [99 rows x 9 columns]

    df1=pd.read_csv('/content/Fertilizer_Prediction.csv')

    from sklearn import preprocessing
      
    label_encoder = preprocessing.LabelEncoder()
      

    df1['Soil Type']= label_encoder.fit_transform(df1['Soil Type'])
      
    df1['Soil Type'].unique()

    array([4, 2, 0, 3, 1])

      
    label_encoder = preprocessing.LabelEncoder()
    df1['Crop Type']= label_encoder.fit_transform(df1['Crop Type'])  
    df1['Crop Type'].unique()

    array([ 3,  8,  1,  9,  6,  0, 10,  4,  5,  7,  2])

    label_encoder = preprocessing.LabelEncoder()
    df1['Fertilizer Name']= label_encoder.fit_transform(df1['Fertilizer Name'])  
    df1['Fertilizer Name'].unique()

    array([6, 5, 1, 4, 2, 3, 0])

    xf=df1.drop(['Fertilizer Name'],axis=1)

    yf=df1['Fertilizer Name'].values

    yf

    array([6, 5, 1, 4, 6, 2, 3, 6, 4, 1, 5, 2, 6, 4, 5, 2, 6, 4, 6, 5, 3, 2,
           5, 6, 3, 4, 1, 6, 5, 3, 4, 5, 6, 1, 4, 6, 1, 5, 2, 5, 3, 1, 4, 6,
           5, 1, 3, 4, 6, 1, 4, 3, 6, 5, 2, 4, 6, 3, 2, 5, 6, 3, 4, 0, 6, 5,
           3, 1, 0, 3, 4, 6, 4, 6, 5, 1, 4, 3, 0, 5, 1, 6, 5, 3, 4, 1, 0, 6,
           0, 5, 1, 0, 6, 1, 4, 0, 6, 5, 3])

    xf_train, xf_test, yf_train, yf_test = train_test_split(xf, 
                                                        yf, 
                                                        test_size = 0.20, 
                                                        random_state = 42)

    from sklearn import preprocessing
    mm_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = mm_scaler.fit_transform(xf_train)
    mm_scaler.transform(xf_test)

    array([[ 0.69230769,  0.68181818,  0.975     ,  0.        ,  0.1       ,
             0.52777778,  0.        ,  0.47619048],
           [ 0.15384615,  0.18181818,  0.125     ,  0.25      ,  0.7       ,
             0.22222222,  0.        ,  0.30952381],
           [ 0.38461538,  0.45454545,  0.05      ,  0.75      ,  0.9       ,
            -0.02777778,  0.89473684,  0.4047619 ],
           [ 0.23076923,  0.18181818,  1.        ,  0.        ,  0.1       ,
             0.94444444,  0.        ,  0.        ],
           [ 0.84615385,  0.45454545,  0.45      ,  1.        ,  0.4       ,
             0.27777778,  0.        ,  0.97619048],
           [ 0.46153846,  0.54545455,  0.475     ,  1.        ,  0.        ,
             0.44444444,  0.        ,  0.66666667],
           [ 0.15384615,  0.13636364,  0.25      ,  0.        ,  0.5       ,
             0.88888889,  0.        ,  0.        ],
           [ 0.53846154,  0.54545455,  0.225     ,  0.75      ,  0.9       ,
             0.47222222,  0.        ,  0.57142857],
           [ 0.15384615,  0.18181818,  0.075     ,  0.25      ,  0.7       ,
             0.22222222,  0.        ,  0.95238095],
           [ 0.07692308,  0.09090909,  0.325     ,  1.        ,  0.3       ,
             0.88888889,  0.        ,  0.        ],
           [ 0.38461538,  0.45454545,  0.05      ,  0.5       ,  0.8       ,
             0.19444444,  0.        ,  0.95238095],
           [ 0.38461538,  0.45454545,  0.55      ,  1.        ,  0.3       ,
             0.47222222,  0.        ,  0.5       ],
           [ 0.30769231,  0.36363636,  0.45      ,  0.25      ,  0.6       ,
             0.52777778,  0.        ,  0.42857143],
           [ 0.38461538,  0.45454545,  0.95      ,  0.75      ,  0.1       ,
             0.11111111,  0.47368421,  0.69047619],
           [ 0.76923077,  0.77272727,  0.425     ,  1.        ,  0.        ,
             0.13888889,  0.        ,  0.83333333],
           [ 0.23076923,  0.18181818,  0.525     ,  0.25      ,  0.6       ,
             0.83333333,  0.        ,  0.        ],
           [ 0.76923077,  0.81818182,  0.2       ,  0.75      ,  0.9       ,
             0.16666667,  0.        ,  0.88095238],
           [ 0.        ,  0.        ,  1.        ,  0.5       ,  0.1       ,
             0.86111111,  0.        ,  0.        ],
           [ 0.84615385,  0.81818182,  0.625     ,  0.5       ,  1.        ,
             0.19444444,  0.94736842,  0.45238095],
           [ 0.15384615,  0.13636364,  0.225     ,  0.        ,  0.5       ,
             1.02777778,  0.        ,  0.        ]])

    classifiersf = [             ['LogisticRegression :', LogisticRegression(max_iter = 1000)],
                   
                   ['DecisionTree :',DecisionTreeClassifier()],
               
                   ['KNeighbours :', KNeighborsClassifier()],
                   ['naive bays :', GaussianNB()],
                  ]

    predictionsf_df1 = pd.DataFrame()
    predictionsf_df1['action'] = yf_test
    for name,classifierf in classifiersf:
        classifierf = classifierf
        classifierf.fit(xf_train, yf_train)
        predictionsf = classifierf.predict(xf_test)
        predictionsf_df1[name.strip(" :")] = predictionsf
        print(name, accuracy_score(yf_test, predictionsf))

    LogisticRegression : 1.0
    DecisionTree : 1.0
    KNeighbours : 0.9
    naive bays : 1.0

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,

    from sklearn.ensemble import VotingClassifier
    clf1f = LogisticRegression()
    clf2f=KNeighborsClassifier()
    clf3f = DecisionTreeClassifier()
    eclf1f = VotingClassifier(estimators=[('', clf1f), ('KNN', clf2f), ('DecisionTree', clf3f)], voting='soft')
    eclf1f.fit(xf_train, yf_train)
    predictionsf = eclf1f.predict(xf_test)
    print(classification_report(yf_test, predictionsf))

                  precision    recall  f1-score   support

               0       1.00      1.00      1.00         2
               1       1.00      1.00      1.00         1
               3       1.00      1.00      1.00         1
               4       1.00      1.00      1.00         5
               5       1.00      1.00      1.00         5
               6       1.00      1.00      1.00         6

        accuracy                           1.00        20
       macro avg       1.00      1.00      1.00        20
    weighted avg       1.00      1.00      1.00        20

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,

    import pickle
    pickle_out=open('fertilizer.pkl','wb')
    pickle.dump(eclf1f,pickle_out)
    pickle_out.close()

##Streamlit

Streamlit is an open source app framework in Python language. It helps
us create web apps for data science and machine learning in a short
time. It is compatible with major Python libraries such as scikit-learn,
Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

    pip install streamlit

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting streamlit
      Downloading streamlit-1.14.0-py2.py3-none-any.whl (9.2 MB)
    ent already satisfied: protobuf<4,>=3.12 in /usr/local/lib/python3.7/dist-packages (from streamlit) (3.17.3)
    Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (7.1.2)
    Requirement already satisfied: toml in /usr/local/lib/python3.7/dist-packages (from streamlit) (0.10.2)
    Collecting pydeck>=0.1.dev5
      Downloading pydeck-0.8.0b4-py2.py3-none-any.whl (4.7 MB)
    ent already satisfied: importlib-metadata>=1.4 in /usr/local/lib/python3.7/dist-packages (from streamlit) (4.13.0)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from streamlit) (2.8.2)
    Requirement already satisfied: requests>=2.4 in /usr/local/lib/python3.7/dist-packages (from streamlit) (2.23.0)
    Collecting validators>=0.2
      Downloading validators-0.20.0.tar.gz (30 kB)
    Collecting watchdog
      Downloading watchdog-2.1.9-py3-none-manylinux2014_x86_64.whl (78 kB)
    ent already satisfied: pandas>=0.21.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (1.3.5)
    Collecting pympler>=0.9
      Downloading Pympler-1.0.1-py3-none-any.whl (164 kB)
    ent already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (7.1.2)
    Requirement already satisfied: pyarrow>=4.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (6.0.1)
    Collecting semver
      Downloading semver-2.13.0-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: tornado>=5.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (5.1.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from streamlit) (1.21.6)
    Requirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (4.2.4)
    Requirement already satisfied: altair>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (4.2.0)
    Collecting blinker>=1.0.0
      Downloading blinker-1.5-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: typing-extensions>=3.10.0.0 in /usr/local/lib/python3.7/dist-packages (from streamlit) (4.1.1)
    Requirement already satisfied: packaging>=14.1 in /usr/local/lib/python3.7/dist-packages (from streamlit) (21.3)
    Requirement already satisfied: tzlocal>=1.1 in /usr/local/lib/python3.7/dist-packages (from streamlit) (1.5.1)
    Requirement already satisfied: toolz in /usr/local/lib/python3.7/dist-packages (from altair>=3.2.0->streamlit) (0.12.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from altair>=3.2.0->streamlit) (2.11.3)
    Requirement already satisfied: entrypoints in /usr/local/lib/python3.7/dist-packages (from altair>=3.2.0->streamlit) (0.4)
    Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.7/dist-packages (from altair>=3.2.0->streamlit) (4.3.3)
    Collecting gitdb<5,>=4.0.1
      Downloading gitdb-4.0.9-py3-none-any.whl (63 kB)
    map<6,>=3.0.1
      Downloading smmap-5.0.0-py3-none-any.whl (24 kB)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=1.4->streamlit) (3.9.0)
    Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (5.10.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (0.18.1)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (22.1.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=14.1->streamlit) (3.0.9)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.21.0->streamlit) (2022.5)
    Requirement already satisfied: six>=1.9 in /usr/local/lib/python3.7/dist-packages (from protobuf<4,>=3.12->streamlit) (1.15.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->altair>=3.2.0->streamlit) (2.0.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.4->streamlit) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.4->streamlit) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.4->streamlit) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.4->streamlit) (3.0.4)
    Requirement already satisfied: pygments<3.0.0,>=2.6.0 in /usr/local/lib/python3.7/dist-packages (from rich>=10.11.0->streamlit) (2.6.1)
    Collecting commonmark<0.10.0,>=0.9.0
      Downloading commonmark-0.9.1-py2.py3-none-any.whl (51 kB)
    ent already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.7/dist-packages (from validators>=0.2->streamlit) (4.4.2)
    Building wheels for collected packages: validators
      Building wheel for validators (setup.py) ... e=validators-0.20.0-py3-none-any.whl size=19582 sha256=e087c4a3b96323deec10722b941e1920f03e2016a1c7e165e51cf3484cfee32c
      Stored in directory: /root/.cache/pip/wheels/5f/55/ab/36a76989f7f88d9ca7b1f68da6d94252bb6a8d6ad4f18e04e9
    Successfully built validators
    Installing collected packages: smmap, gitdb, commonmark, watchdog, validators, semver, rich, pympler, pydeck, gitpython, blinker, streamlit
    Successfully installed blinker-1.5 commonmark-0.9.1 gitdb-4.0.9 gitpython-3.1.29 pydeck-0.8.0b4 pympler-1.0.1 rich-12.6.0 semver-2.13.0 smmap-5.0.0 streamlit-1.14.0 validators-0.20.0 watchdog-2.1.9

    !pip install -q  pyngrok

    !pip install -q streamlit_ace

    %%writefile app.py
    import streamlit as st

    import pandas as pd

    import numpy as np

    import pickle

    import base64

    import seaborn as sns

    import matplotlib.pyplot as plt

    Writing app.py

    !pip install -q streamlit
    !pip install -q pyngrok
    !pip install -q streamlit_ace

    !pip install pyngrok==4.1.1.

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pyngrok==4.1.1.
      Downloading pyngrok-4.1.1.tar.gz (18 kB)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from pyngrok==4.1.1.) (0.16.0)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from pyngrok==4.1.1.) (6.0)
    Building wheels for collected packages: pyngrok
      Building wheel for pyngrok (setup.py) ... e=pyngrok-4.1.1-py3-none-any.whl size=15983 sha256=82a88dbaef326658d3a333f4016b010f45f677dd0b6351c5984757b94c483ab1
      Stored in directory: /root/.cache/pip/wheels/b1/d9/12/045a042fee3127dc40ba6f5df2798aa2df38c414bf533ca765
    Successfully built pyngrok
    Installing collected packages: pyngrok
      Attempting uninstall: pyngrok
        Found existing installation: pyngrok 5.1.0
        Uninstalling pyngrok-5.1.0:
          Successfully uninstalled pyngrok-5.1.0
    Successfully installed pyngrok-4.1.1

    pip install pillow

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pillow in /usr/local/lib/python3.7/dist-packages (7.1.2)

    %%writefile app.py
    import pandas as pd
    import numpy as np
    import pickle
    import streamlit as st

    def main_page():
        st.title("👨‍🌾 Farmer Friend")
        html_temp = """
            <div style ="background-color:#4332F4;padding:20px">
          <h1 style ="color:black;text-align:center;">Farmer Friend</h1>
          </div>
          """
        st.sidebar.markdown("🏠Home Page")

    def page2():

        st.sidebar.markdown("🌾Crop Recommender")
        pickle_in = open('crop.pkl', 'rb')
        classifier = pickle.load(pickle_in)
        st.title("🌾Crop Recommender")
        html_temp = """
            <div style ="background-color:#2F5233;padding:20px">
          <h1 style ="color:black;text-align:center;">Farmer Friend</h1>
          </div>
          """
        st.markdown(html_temp, unsafe_allow_html = True)
        N = st.text_input("Nitrogen Level in soil")
        P = st.text_input("Phosphorous Level in soil")
        K = st.text_input("Potassium Level in soil")
        temperature = st.text_input("Enter Average Temperature around the field ( in Celicus)")
        humidity = st.text_input("Enter Average Percentage of Humidity around the field (1-100%)")
        ph= st.text_input("Enter Ph of the soil (1-10)")
        rainfall = st.text_input("Enter Average Amount of Rainfall around the field (in cm)")
        result =""
        if st.button("Predict"):
                   result = classifier.predict([[N,P,K,temperature,humidity,ph,rainfall]])
                   st.success('The best crop you can grow is {}'.format(result))

    def page3():
        st.sidebar.markdown("🧪Fertilizer Predictor")
        from PIL import Image
        image = Image.open('/content/cropindex.jpg')
        st.image(image, caption='index')
        model = pickle.load(open('classifier.pkl', 'rb'))
        ferti = pickle.load(open('fertilizer1.pkl', 'rb'))
        st.title("🧪Fertilizer Predictor")
        html_temp = """
            <div style ="background-color:#F96926;padding:20px">
          <h1 style ="color:black;text-align:center;">Farmer Friend</h1>
          </div>
          """
        st.markdown(html_temp, unsafe_allow_html = True)
        Temparature = st.text_input("Temperature")
        Humidity  = st.text_input("Enter Average Percentage of Humidity around the field (1-100%)")
        Moisture = st.text_input("Moisture")
        Soil_Type =st.selectbox("Soil Type",(0,1,2,3,4))
        Crop_Type = st.selectbox(
            "Crop Type",
            (0,1,2,3,4,5,6,7,8,9,10)
        )
        Nitrogen = st.text_input("Nitrogen Level in soil")
        Phosphorous = st.text_input("Phosphorous Level in soil")
        Potassium = st.text_input("Potassium Level in soil")
        input = Temparature,Humidity,Moisture,Soil_Type,Crop_Type,Nitrogen,Potassium,Phosphorous
        resultf =""
        if st.button("Predict"):
                   resultf = ferti.classes_[model.predict([input])]
                   st.success('Fertilizer is {}'.format(resultf))

    page_names_to_funcs = {
        "Home": main_page,
        "Crop Recommender": page2,
        "Fertilizer Predictor": page3,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()               

    Overwriting app.py

     !streamlit run /content/app.py & npx localtunnel --port 8501

    [############......] - finalize:string-width: sill finalize /root/.npm/_npx/409

      You can now view your Streamlit app in your browser.

      Network URL: http://172.28.0.2:8501
      External URL: http://35.192.122.203:8501

    your url is: https://yummy-berries-arrive-35-192-122-203.loca.lt
    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
      "X does not have valid feature names, but"
      Stopping...
    ^C
