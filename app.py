from flask import Flask,request,render_template,jsonify

import numpy as numpy
from keras.models import load_model
import pickle
from sklearn.feature_extraction.text import CountVectorizer

model=load_model('reviews.h5')

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')
@app.route('/y_pred' ,methods=["POST"])


def y_pred():
  print(request.form.get('feeback'))
  with open("cv.pkl",'rb')as file:
    cv=pickle.load(file) 
    entered_input=request.form.get('feeback')
    x_intent = cv.transform([entered_input] )
    y_pred=model.predict(x_intent)
    print(y_pred)
    if (y_pred>0.5):
      print("it is positive review")
      prediction = "It is a positive Review"
    else:
      print("it is negative review")
      prediction = "It is a negative Review"
  return render_template('index.html', text = prediction)

if __name__=="__main__":
    app.run(debug=True)


# with open("cv.pkl",'rb')as file:
  # cv=pickle.load(file) 
  # entered_input=" food is worst"
  # x_intent = cv.transform([entered_input] )
  # y_pred=model.predict(x_intent)
  # if (y_pred>0.5):
  #   print("it is positive review")
  # else:
  #   print("it is negative review")