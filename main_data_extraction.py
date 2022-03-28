# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:44:10 2022

@author: Featherine
"""

from urllib.request import urlopen
import json
import pandas as pd

from tqdm import tqdm

import cv2
from skimage import io


from PIL import Image
import requests
from io import BytesIO

import numpy as np

import os

import time
from PIL import Image, ImageOps

# https://www.analyticsvidhya.com/blog/2021/12/vehicle-detection-and-counting-system-using-opencv/
# cars_cascade = cv2.CascadeClassifier('car.xml')
# bus_cascade = cv2.CascadeClassifier('Bus_front.xml')
# two_cascade = cv2.CascadeClassifier('two_wheeler.xml')



# https://github.com/Kalebu/Real-time-Vehicle-Dection-Python
# cars_cascade = cv2.CascadeClassifier('Models/haarcascade_car.xml')



# https://github.com/opencv/opencv/issues/17479
# https://pysource.com/2021/07/30/count-vehicles-on-images-with-opencv-and-deep-learning/#
from Models.vehicle_detector import VehicleDetector
vd = VehicleDetector()

url_imag = 'https://api.data.gov.sg/v1/transport/traffic-images'
url_temp = 'https://api.data.gov.sg/v1//environment/air-temperature'


import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

pipeline = keras_ocr.pipeline.Pipeline()



def inpaint_text(image, pipeline):
    # read image
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([image])
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        image = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                 
    return image









latitude_fix = 1.357098686 
longitude_fix = 103.902042

gap_lat = 0.2
gap_lon = 0.05


# https://api.data.gov.sg/v1/transport/traffic-images?date_time=2022-02-10T22%3A10%3A22

# https://api.data.gov.sg/v1/environment/air-temperature?date_time=2022-02-10T22%3A10%3A22

from datetime import datetime, timedelta

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in 
       datetime_range(datetime(2021, 3, 20, 12), datetime(2022, 3, 20, 12), 
       timedelta(minutes=30))]

loop = 1

df_features = pd.DataFrame()

for date_time in dts:
    
    # date_time = dts[0]
    
    print('Loop ' + str(loop) + ' - Date time: ' + date_time)
    loop = loop + 1
    
    
    date_time_temp = date_time.replace(':','%3A')
    
    # Code executed here
    url_imag = ('https://api.data.gov.sg/v1/transport/traffic-images?date_time=' + 
                date_time_temp)
    
                
    url_temp = ('https://api.data.gov.sg/v1//environment/air-temperature?date_time=' + 
                date_time_temp)

    # url_imag = ('https://api.data.gov.sg/v1/transport/traffic-images?date_time=' + 
    #             '2022-02-10T22%3A10%3A22')
      
    # url_temp = ('https://api.data.gov.sg/v1//environment/air-temperature?date_time=' + 
    #             '2022-02-10T22%3A10%3A22')

    # Extract traffic image
    response = urlopen(url_imag)
    json_data = response.read().decode('utf-8', 'replace')
    df_imag = json.loads(json_data)
    
    try: 
    
        imag_status = df_imag['api_info']['status']
        imag_timestamp = df_imag['items'][0]['timestamp']
        
        imag_date = imag_timestamp.split('T')[0]
        imag_time_zone = imag_timestamp.split('T')[1]
        
        imag_time = imag_time_zone.split('+')[0]
        imag_time_SG = imag_time_zone.split('+')[1]
        
        imag_cameras = df_imag['items'][0]['cameras']
        
        imag_cameras = pd.DataFrame.from_dict(imag_cameras)
        
        # Convert location into latitude and longitude
        imag_cameras[['latitude', 'longitude']] = imag_cameras['location'].apply(pd.Series)
        imag_cameras = imag_cameras.drop(['location'], axis=1)
        
        imag_cameras[['height', 'width', 'md5']] = imag_cameras['image_metadata'].apply(pd.Series)
        imag_cameras = imag_cameras.drop(['image_metadata'], axis=1)
        
        imag_cameras['status'] = imag_status
        imag_cameras['date'] = imag_date
        imag_cameras['time'] = imag_time
        imag_cameras['time zone'] = imag_time_SG
        
        imag_cameras.drop_duplicates()
        
        imag_cameras = imag_cameras.sort_values(by=['camera_id'])
        
        imag_cameras = imag_cameras.loc[imag_cameras['latitude']==1.357098686].reset_index(drop=True)
        
        imag_final = imag_cameras
        
        # imag_final.to_csv('imag.csv')
        
        #print(imag_final)
        
    except:
        break
    
    
    
    
    
    # Extract air temperature data
    response = urlopen(url_temp)
    json_data = response.read().decode('utf-8', 'replace')
    df_temp = json.loads(json_data)
    # df_temp = pd.json_normalize(df_temp['items'])
    
    # Get status
    temp_status = df_temp['api_info']['status'] 
    
    # Get timestamp
    temp_timestamp = df_temp['items'][0]['timestamp']
    
    temp_date = temp_timestamp.split('T')[0]
    temp_time_zone = temp_timestamp.split('T')[1]
    
    temp_time = temp_time_zone.split('+')[0]
    temp_time_SG = temp_time_zone.split('+')[1]
    
    # Get readings
    temp_readings = pd.DataFrame(df_temp['items'][0]['readings'])
    temp_reading_type = df_temp['metadata']['reading_type']
    temp_reading_unit = df_temp['metadata']['reading_unit']
    
    # Get location
    temp_stations =  pd.DataFrame.from_dict(df_temp['metadata']['stations'])
    # Convert location into latitude and longitude
    temp_stations[['latitude', 'longitude']] = temp_stations['location'].apply(pd.Series)
    temp_stations = temp_stations.drop(['location'], axis=1)
    
    # Combine all findings into a single DF
    temp_stations['status'] = temp_status
    
    temp_stations['status'] = temp_status
    temp_stations['date'] = temp_date
    temp_stations['time'] = temp_time
    temp_stations['time zone'] = temp_time_SG
    
    temp_stations = pd.concat([temp_stations, temp_readings], axis=1)
    temp_stations['reading_type'] = temp_reading_type
    temp_stations['reading_unit'] = temp_reading_unit
    temp_stations.drop_duplicates()
    
    temp_final = temp_stations
    
    temp_final1 = temp_final[(temp_final['latitude'] < latitude_fix + gap_lat) &
                             (temp_final['latitude'] > latitude_fix - gap_lat) &
                             (temp_final['longitude'] < longitude_fix + gap_lon) &
                             (temp_final['longitude'] > longitude_fix - gap_lon)]
    
    if len(temp_final1) == 0:
        temperature = np.mean(temp_final['value'])
    else:
        temperature = np.mean(temp_final1['value'])
    
    # temp_final.to_csv('temp.csv')
    
    #print(temp_final)
    
    

    
    
    
    no_of_cars = []
    
    for i in (range(len(imag_final))):
    # for i in tqdm(range(len(imag_final))):
        
        time1 = imag_final['time'][i].replace(':','_')
        date = imag_final['date'][i].replace('-','_')
        
        camera_id = imag_final['camera_id'][i]
        url = imag_final['image'][i]
        
        if not os.path.exists('Images Original/' + camera_id):
            os.makedirs('Images Original/' + camera_id)
        
        if not os.path.exists('Images Checked/' + camera_id):
            os.makedirs('Images Checked/' + camera_id)
        
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            
            newsize = (300, 300)
            image = image.resize(newsize)
            image = np.array(image)
            
            image = Image.fromarray(image)
            image.save('Images Original/' + camera_id + 
                       '/' + camera_id + '_' + date + '__' + time1 + '.jpg')
            
            image = np.array(image)
            
            image_no_text = inpaint_text(image, pipeline)
            image_no_text = Image.fromarray(image_no_text)
            
            
            # image_grey = ImageOps.grayscale(image_no_text)
            # image_grey = np.array(image_grey)
                    
            # scaleFactor = 1.15
            # minNeighbors = 2
            # cars = cars_cascade.detectMultiScale(image_grey, scaleFactor, minNeighbors)
            
            
            image_no_text = np.array(image_no_text)
            cars = vd.detect_vehicles(image_no_text)
            
    
            for (x, y, w, h) in cars:
                cv2.rectangle(image, (x, y), (x+w,y+h), color=(0, 255, 255), thickness=2)
        
            # cv2.imshow('frame', image)
            image = Image.fromarray(image)
            image.save('Images Checked/' + camera_id + 
                       '/' + camera_id + '_' + date + '__' + time1 + '.jpg')
        
            # print('\nURL: ' + url)
            print(str(loop) + ' : ' + camera_id + ' - No of cars: ' + str(len(cars)))
            
            no_of_cars.append(len(cars))
            
        except:
            print(str(loop) + ' : ' + camera_id + ' - No of cars: 0 -  ERROR')
            
            no_of_cars.append(0)
        
    # imag_final['no of cars'] = no_of_cars
    # imag_final.to_csv('imag with no of cars.csv')
    
    columns_name = ['DateTime'] + list(imag_final['camera_id']) + ['Temp']
    
    data = [date_time] + no_of_cars + [temperature]
    data = pd.DataFrame(data)
    data = data.transpose()
    data.columns = columns_name
    
    if len(df_features) == 0:
        df_features = pd.DataFrame(columns = columns_name)
    df_features = df_features.append(data)
    
    df_features.to_csv('features.csv')

# imag_final['camera_id']


# Camera id 1006
