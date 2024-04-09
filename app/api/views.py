from rest_framework.response import Response
from rest_framework.decorators import api_view
from joblib import load
import pandas
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import json
import random
import datetime as dt
import api.vardata as vardata
from api.vardata import find_index
from api.vardata import minBegin, maxBegin, minEnd, maxEnd, premiums
from api.vardata import makeIndex, usageIndex, vUsage, vMake, vType, usageIndex, typeIndex, maxValue, maxSeats, minYr, maxYr

model = load_model('./../saved_models/seq1.keras')

@api_view(['GET','POST'])
def getData(request):
    
    if request.method == 'POST':
        x = json.loads(request.body)

        print('Data received...')
        x['insuredValue'] = x['insuredValue']*57
        df = pandas.DataFrame([x])
        df.columns = ['SEX', 'INSR_BEGIN', 'INSR_END', 'INSURED_VALUE', 'PROD_YEAR','SEATS_NUM', 'TYPE_VEHICLE', 'MAKE', 'USAGE']

        df['INSR_BEGIN'] = pandas.to_datetime(df['INSR_BEGIN'], format='%d-%b-%y')
        df['INSR_BEGIN'] = df['INSR_BEGIN'].map(dt.datetime.toordinal)

        df['INSR_END'] = pandas.to_datetime(df['INSR_END'], format='%d-%b-%y')
        df['INSR_END'] = df['INSR_END'].map(dt.datetime.toordinal)

        df['INSR_BEGIN'] = (df['INSR_BEGIN'] - minBegin)/(maxBegin - minBegin)
        df['INSR_END'] = (df['INSR_END'] - minEnd)/(maxEnd - minEnd)
        df.loc[0, 'SEX'] = df['SEX'][0]*1.0/2
        df.loc[0, 'INSURED_VALUE'] = df['INSURED_VALUE'][0]/maxValue
        df.loc[0, 'PROD_YEAR'] = (df['PROD_YEAR'][0]-minYr)/maxYr
        df.loc[0, 'SEATS_NUM'] = df['SEATS_NUM'][0]/maxSeats
        df.loc[0, 'TYPE_VEHICLE'] = typeIndex[find_index(vType, df['TYPE_VEHICLE'][0])]
        df.loc[0, 'MAKE'] = makeIndex[find_index(vMake, df['MAKE'][0])]
        df.loc[0, 'USAGE'] = usageIndex[find_index(vUsage, df['USAGE'][0])]

        df = tf.convert_to_tensor(df, dtype=tf.float32)
        y = model.predict(df)

        prem = random.randint(.75*premiums[np.argmax(y)], premiums[np.argmax(y)])
        prem = prem/57
        insr_premium = {'premium': prem}
        return Response(insr_premium)
    else:
        return Response('Waiting...')

@api_view(['GET'])
def getVType(request):
    return Response(vardata.vType)

@api_view(['GET'])
def getVMake(request):
    return Response(vardata.vMake)

@api_view(['GET'])
def getVUsage(request):
    return Response(vardata.vUsage) 