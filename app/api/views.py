from rest_framework.response import Response
from rest_framework.decorators import api_view
from joblib import load
import pandas
from tensorflow.keras.models import load_model
import numpy as np
import json
import random
import datetime as dt
import api.vardata as vardata
from api.vardata import find_index
from api.vardata import minBegin, maxBegin, minEnd, maxEnd
from api.vardata import makeIndex, usageIndex, vUsage, vMake, vType, usageIndex, typeIndex

model = load_model('./../saved_models/seq1.keras')

@api_view(['GET','POST'])
def getData(request):
    
    if request.method == 'POST':
        # x = json.loads(request.body)
        # df = pandas.DataFrame([x])
        # df.columns = ['SEX', 'INSR_BEGIN', 'INSR_END', 'INSURED_VALUE', 'PROD_YEAR','SEATS_NUM', 'TYPE_VEHICLE', 'MAKE', 'USAGE']

        # df['INSR_BEGIN'] = pandas.to_datetime(df['INSR_BEGIN'], format='%d-%b-%y')
        # df['INSR_BEGIN'] = df['INSR_BEGIN'].map(dt.datetime.toordinal)

        # df['INSR_END'] = pandas.to_datetime(df['INSR_END'], format='%d-%b-%y')
        # df['INSR_END'] = df['INSR_END'].map(dt.datetime.toordinal)

        # df['INSR_BEGIN'] = (df['INSR_BEGIN'] - minBegin)/(maxBegin - minBegin)
        # df['INSR_END'] = (df['INSR_END'] - minEnd)/(maxEnd - minEnd)

        # df['TYPE_VEHICLE'][0] = typeIndex(find_index(vType, df['TYPE_VEHICLE'][0]))
        # df['MAKE'][0] = vardata.makeIndex(find_index(vardata.vMake, df['MAKE'][0]))
        # df['USAGE'][0] = vardata.usageIndex(find_index(vardata.vUsage, df['USAGE'][0]))

        # y = model.predict(df)

        # premiums = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
        # prem = random.randint(.5*premiums[np.argmax(y)], premiums[np.argmax(y)])
        # myData = {'prediction': prem}
        dummy_data = {'dummy': 0}
        return Response(dummy_data)
    else:
        return Response('Waiting...')