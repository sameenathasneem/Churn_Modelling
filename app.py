import numpy as np
from flask import Flask, request, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = pickle.load(open('model1.pickle', 'rb'))
sc=pickle.load(open('model2.pickle','rb'))
global graph
graph = tf.get_default_graph() 

@app.route('/')
def home():
    return render_template('checkbinary.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    
    val1=request.form['SeniorCitizen']
    val1=float(val1)
    a=np.array(val1)
    
    val2=request.form['Partner']
    if(val2=='Yes'):a=np.append(a,float(1))
    else: a=np.append(a,float(0)) 
    
    val3=request.form['Dependents']
    if(val3=='Yes'):a=np.append(a,float(1))
    else: a=np.append(a,float(0))
    
    val4=request.form['tenure']
    val4=float(val4)
    a=np.append(a,val4)
    
    val5=request.form['InternetService']
    if(val5=='DSL'):a=np.append(a,float(1))
    elif(val5=='Fiber Optics'):a=np.append(a,float(2))
    else:a=np.append(a,float(0))
    
    val6=request.form['OnlineSecurity']
    if(val6=='Yes'):a=np.append(a,float(1))
    elif(val6=='No'):a=np.append(a,float(0))
    else:a=np.append(a,float(2))
    
    val7=request.form['onlinebackup']
    if(val7=='Yes'):a=np.append(a,float(1))
    elif(val7=='No'):a=np.append(a,float(0))
    else:a=np.append(a,float(2))
    
    val8=request.form['DeviceProtection']
    if(val8=='Yes'):a=np.append(a,float(1))
    elif(val8=='No'):a=np.append(a,float(0))
    else:a=np.append(a,float(2))
    
    val9=request.form['TechSupport']
    if(val9=='Yes'):a=np.append(a,float(1))
    elif(val9=='No'):a=np.append(a,float(0))
    else:a=np.append(a,float(2))
    
    val10=request.form['Contract']
    if(val10=='One Year'):a=np.append(a,float(1))
    elif(val10=='Two Year'):a=np.append(a,float(2))
    else:a=np.append(a,float(0))
    
    val11=request.form['PaperlessBilling']
    if(val11=='Yes'):a=np.append(a,float(1))
    elif(val11=='No'):a=np.append(a,float(0))
    
    val12=request.form['PaymentMethod']
    if(val12=='Electronic check'):a=np.append(a,float(1))
    elif(val12=='Mailed check'):a=np.append(a,float(2))
    elif(val12=='Bank transfer (automatic)'):a=np.append(a,float(0))
    else:np.append(a,float(3))
    
    val13=request.form['MonthlyCharges']
    val13=float(val13)
    a=np.append(a,val13)
    
    val14=request.form['TotalCharges']
    val14=float(val14)
    a=np.append(a,val14)
    

    
    a=a.reshape(1,14)
    a=sc.transform(a)
    with graph.as_default(): 
        prediction=model.predict(a)
        prediction=float(prediction)
        prediction=np.round(prediction,decimals=3)
        return render_template('checkbinary.html', prediction_text='Probability of this customer churn is {:%}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
    
    