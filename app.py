from flask import Flask,request, render_template
import utils.helper_models as helper_models
import pandas as pd

app = Flask(__name__)

model_pipeline = helper_models.load_pipeline()

@app.route('/')
def home():
    return render_template("Income.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features = []
    index= 0
    for x in request.form.values():
        if index in [2,3,4]:
            features.append(x)
        else:
            features.append(int(x))
        index += 1

    # Convert list to dataframe and set column names
    df_feature = pd.DataFrame(features).T
    df_feature.columns = ['age', 'education_num', 'occupation','relationship', 'race', 'capital_gain',
                          'capital_loss', 'hours_per_week']

    print(df_feature)
    prediction = model_pipeline.predict(df_feature)

    if prediction == 1:
        return render_template('Income.html',pred='Congratulations...!!!! you will earn more than 50k.')
    else:
        return render_template('Income.html',pred='Unfortunetly...!!!! you will not earn more than 50k.')




if __name__ == "__main__":
    app.run(debug=False)