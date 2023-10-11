from flask import Flask, render_template, request
import pickle
import pandas as pd

__name__ == "__main__"

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model/xgb_best_model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

 # Get input data from the form
    CreditScore = float(request.form["CreditScore"])
    Age = float(request.form["Age"])
    Tenure = float(request.form["Tenure"])
    Balance = float(request.form["Balance"])
    NumOfProducts = float(request.form["NumOfProducts"])
    HasCrCard = float(request.form["HasCrCard"])
    IsActiveMember = float(request.form["IsActiveMember"])
    EstimatedSalary = float(request.form["EstimatedSalary"])
    Geography_Germany = float(request.form["Geography_Germany"])
    Geography_Spain = float(request.form["Geography_Spain"])
    Gender_Male = float(request.form["Gender_Male"])

    # Perform prediction using your model
    prediction = model.predict(pd.DataFrame([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male]], columns= ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'] ))
    return f"Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)