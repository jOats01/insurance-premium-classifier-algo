from rest_framework.response import Response
from rest_framework.decorators import api_view
from joblib import load
import pandas
from tensorflow.keras.models import load_model
import numpy as np
import json
import random

model = load_model('./../saved_models/seq1.keras')

@api_view(['GET','POST'])
def getData(request):
    
    if request.method == 'POST':
        x = json.loads(request.body)
        df = pandas.DataFrame([x])
        df.columns = ['SEX', 'INSR_BEGIN', 'INSR_END', 'INSURED_VALUE', 'PROD_YEAR','SEATS_NUM', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
        y = model.predict(df)

        premiums = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
        prem = random.randint(.5*premiums[np.argmax(y)], premiums[np.argmax(y)])
        myData = {'prediction': prem}
        return Response(myData)
    else:
        return Response('Waiting...')