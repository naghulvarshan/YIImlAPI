# Import FastAPI
import joblib
import torch
import io
import json
from PIL import Image
from fastapi import File, FastAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import requests
import haversine as hs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def latlongdist(a, b):
    distance = hs.haversine(a, b)
    return distance
def clustering(val):
        url = 'http://localhost:3001/location/fetch/all'
        x = requests.post(url)
        lst = []
        coords = np.array([])
        z = x.json()['result']
        for y in z:
                lst = lst + [[float(y['latitude']), float(y['longitude'])]]
        coords = np.array(lst)
        dbscan = DBSCAN(eps=0.3, min_samples=3, metric=latlongdist).fit(coords)
        labels = dbscan.labels_
        # print the labels for each data point
        print("Labels:", labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("Number of clusters:", n_clusters)
        x = len(coords)
        print(x)
        outliers = 0
        dic = {}
        y = 0
        for i in labels:
                y += 1
                if i == -1:
                        outliers += 1
                        y -= 1
                else:
                        if i in dic:
                                dic[i][0] += 1
                        else:
                                dic[i] = [1, 0]
        for keys in dic:
                dic[keys][1] = dic[keys][0] / y
        i=0
        x = np.where(coords == [val])
        y=0
        v = val[0].split(",")
        v[0] = float(v[0])
        v[1] = float(v[1])
        for items in coords:
                if items[0]==v[0] and items[1]==v[1]:
                        break
                y+=1
        return(dic[labels[y]][1])
        # colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
        # for i, color in zip(range(n_clusters), colors):
        #         cluster = coords[labels == i]
def predict_emotion( model,text, cv=CountVectorizer()):
  print(text)
  myvect= cv.transform(text).toarray()
  prediction=model.predict(myvect)
  prob=model.predict_proba(myvect)
  perc=dict(zip(model.classes_,prob[0]))
  print("Prediction:{}, Prediction Score:{}".format(prediction[0],np.max(prob)))
  return [prediction,np.max(prob)]
def nlp(s):
        df = pd.read_csv("Emotion_final.csv")
        # Cleaning
        df['clean'] = df['Text'].apply(nfx.remove_stopwords)
        df['clean'] = df['clean'].apply(nfx.remove_punctuations)
        df['clean'] = df['clean'].apply(nfx.remove_userhandles)
        Xfeatures = df['clean']
        ylabels = df['Emotion']
        cv = CountVectorizer()
        x = cv.fit_transform(Xfeatures)
        X_train, X_test, Y_train, Y_test = train_test_split(x, ylabels, test_size=0.3, random_state=42)
        nv_model = MultinomialNB()
        nv_model.fit(X_train, Y_train)
        nv_model.score(X_test, Y_test)
        y_predict = nv_model.predict(X_test)
        sample = [s]
        print("Check")
        ret=(predict_emotion(nv_model,sample,cv))
        print(ret[0][0])
        if ret[0][0]=="sadness" or ret[0][0]=="angry" or ret[0][0]=="frustrated":
                return ret[1]
        else:
                return(0.2)
from collections import Counter
def extract_keywords(text, num=50):
        tokens = [tok for tok in text.split()]
        most_common = Counter(tokens).most_common(num)
        return dict(most_common)

from collections import Counter
def extract_keywords(text, num=50):
        tokens = [tok for tok in text.split()]
        most_common = Counter(tokens).most_common(num)
        return dict(most_common)
        # mdl = joblib.load("emotion_classifier_nv_model_2.pkl")
        # something=predict_emotion(mdl,[s])
        # print(something[0])
        # if something[0]=="sadness" or something[0]=="anger" or something[0]=="fear":
        #         return something[1]
        # else:
        #         return 0



# Define your paths & methods for your API
@app.get('/getapi')
def getapi():
    return {"message": "GET API test"}


@app.post("/objectdetection/")
async def get_body(file: bytes = File(...),cds:list=[],s: str="" ):
        #cds=[1,2]
        input_image = Image.open(io.BytesIO(file)).convert("RGB")
        model=torch.hub.load('ultralytics/yolov5', 'custom', path="best .pt", force_reload=True)
        results = model(input_image)
        results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        print(results_json)
        l=len(results_json)
        confidence=0
        for i in results_json:
                confidence+=i['confidence']
        confidence/=l
        #print(results)
        dbscanscroe=clustering(cds)
        nlpscore=nlp(s)
        print(dbscanscroe)
        print(confidence)
        final_score=0.5*dbscanscroe + 0.2*confidence + 0.1*nlpscore
        return {"result": final_score}