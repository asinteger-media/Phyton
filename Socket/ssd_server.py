import os
##os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
##os.environ['CUDA_VISIBLE_DEVICES'] = ""
import socket
import numpy as np
from PIL import Image
from PIL import ImagePalette
from io import BytesIO
import argparse
import random
import time
from pythonosc import udp_client


import cv2

import struct
import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#try:
#  tf.config.experimental.set_memory_growth(physical_devices[0], True)
#except:
#  # Invalid device or cannot modify virtual devices once initialized.
#  pass

from model import create_ssd
from utils import generate_default_boxes
from tensorflow.python.client import device_lib

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

w,h = 960,854

print(device_lib.list_local_devices())

##tf.debugging.set_log_device_placement(True)

default_boxes = generate_default_boxes()

print("Default boxes set initialised...")

NUM_CLASSES = 2

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("CUDA set...")

weights = os.path.join(os.path.realpath(os.path.dirname(__file__)), "ssd_epoch_75000.h5")


ssd = create_ssd(w,h,NUM_CLASSES,
                 'specified',
                 '',
                 weights)

# Definition of the parameters
#max_cosine_distance = 1.25
max_cosine_distance = 100
nn_budget = None
              
#initialize deep sort object
model_filename = 'C:/Users/OPERATOR-01/Desktop/nn/Final-model-0.2/saved_model.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

print("Model initialised...")

#TS SERVER
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 6000
BUFFER_SIZE = 1024

#OSC CLIENT
#CLIENT_ADDRESS = '10.0.0.5'
CLIENT_ADDRESS = '10.0.0.120'
CLIENT_PORT = 6020


cnt = 0
output = bytearray()

w,h = 960,854
x,y = 0, 0

ssd_array = []

IMAGE_SIZE = w*h
PARTS_COUNT = int(IMAGE_SIZE / BUFFER_SIZE)
LAST_BUFFER = IMAGE_SIZE - PARTS_COUNT*BUFFER_SIZE;

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((SERVER_ADDRESS, SERVER_PORT))
server.listen(10)


#client = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
#client.connect((CLIENT_ADDRESS, CLIENT_PORT))

print ('');
print ('-----------------------------------------');
print ('Starting Server: '+SERVER_ADDRESS+':'+str(SERVER_PORT))
print ('-----------------------------------------');
print ('Parts count - '+str(PARTS_COUNT));
print ('Image size - '+str(w)+'x'+str(h)+' - '+str(IMAGE_SIZE)+' байт.')
print ('LastBuffer - '+str(LAST_BUFFER));


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip", default=CLIENT_ADDRESS,help="The ip of the OSC server")
  parser.add_argument("--port", type=int, default=CLIENT_PORT,help="The port the OSC server is listening on")
  args = parser.parse_args()

  client = udp_client.SimpleUDPClient(args.ip, args.port)

print ('-----------------------------------------');
print ('Starting OSC Client: '+CLIENT_ADDRESS+':'+str(CLIENT_PORT))


#-----------------------------------------------------------------------------
# RECVALL - FUNCTION TO RECEIVE ALL DATA
#-----------------------------------------------------------------------------
def recvall(sock,count):
    buf=b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return b'#0'
        buf += newbuf
        count -= len(newbuf)
    return buf

#-----------------------------------------------------------------------------
# STARTING
#-----------------------------------------------------------------------------
while True:
    
    conn, addr = server.accept()
    print ("Client connected from: ", addr)
    receive = False

    while conn:

        
 
        start = conn.recv(4)
        if not start:
            break
        size = int.from_bytes(start, byteorder='little')
              

        if size == IMAGE_SIZE:
            

            data = b''
            output = b''
            data = recvall(conn,IMAGE_SIZE)
            if not data:
                break

            if data !=b'#0':
                output+=data
            

            if len(output) == IMAGE_SIZE:
                try:
                  #print("RECEIVED")
                  start_time = time.time()
                  image, boxes, scores, names = ssd.call(output)
                  #print(time.time()-start_time)
                  #if len(boxes) == 0:
                  #  image, boxes, scores, names = ssd_1.call(output)

                  #print(boxes, scores, names)

                  
                  #force detection on or off
                  force_detection = True

                  
                  if force_detection == True:
                    if len(boxes) != 0:
                      boxes_successful, scores_successful, names_successful = boxes, scores, names
                    elif len(boxes) == 0 and ('boxes_successful' in locals() or 'boxes_successful' in globals()):
                      boxes = boxes_successful
                      scores = scores_successful
                      names = names_successful
                
    

                  boxes = np.array(boxes) 
                  names = np.array(names)
                  scores = np.array(scores)
                  features = np.array(encoder(image, boxes))
                  detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
  
                  #print(ssd_array)
                  tracker.predict()
                  tracker.update(detections)

                  tracked_bboxes = []
                  for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                      continue
                    bbox = track.to_tlbr() # Get the corrected/predicted bounding box
                    tracking_id = track.track_id # Get the ID for the particular track
                    tracked_bboxes.append(bbox.tolist() + [tracking_id])
                  #print(time.time()-start_time)
                  #print(tracked_bboxes)

                  #force tracking on or off
                  force_tracking = True

                  if force_tracking == True:
                    if len(tracked_bboxes) != 0 and ('tracked_bboxes_successful' not in locals() or 'tracked_bboxes_successful' in globals()):
                      tracked_bboxes_successful = tracked_bboxes
                    elif len(tracked_bboxes) != 0 and ('tracked_bboxes_successful' in locals() or 'tracked_bboxes_successful' in globals()):
                      for i in range(len(tracked_bboxes)):
                        if ((((tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2)) - (tracked_bboxes_successful[i][0] + ((tracked_bboxes_successful[i][2] - tracked_bboxes_successful[i][0])/2)) > abs(30)) or
                            ((tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2)) - (tracked_bboxes_successful[i][1] + ((tracked_bboxes_successful[i][3] - tracked_bboxes_successful[i][1])/2)) > abs(30))) and
                            (len(tracked_bboxes) == len(tracked_bboxes_successful))):
                          tracked_bboxes = tracked_bboxes_successful
                          
                    elif len(tracked_bboxes_successful) == 0 and ('tracked_bboxes_successful' in locals() or 'tracked_bboxes_successful' in globals()):
                      tracked_bboxes = tracked_bboxes_successful

                  oscMessage = ""
                  answerMessage = ""

                  tracked_bboxes_len = len(tracked_bboxes)
                  
                  for i in range(tracked_bboxes_len):

                      #d = tracked_bboxes[i][4]                #Идентификатор
                      if (tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) >= 487 and
                          tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) <= 588 and
                          tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2) >= 1 and
                          tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2) <= 145):
                          d = 1
                          c = 1  #Класс
                          x = tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) #Координата по X
                          y = tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2)  #Координата по Y
                          oscMessage += (str(d) + ',' + str(c) + ',' + str(round(x)) + ',' +str(round(y)) + ',')

                          
                      elif (tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) >= 386 and
                          tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) <= 484 and
                          tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2) >= 1 and
                          tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2) <= 140):
                          d = 20
                          c = 1  #Класс
                          x = tracked_bboxes[i][0] + ((tracked_bboxes[i][2] - tracked_bboxes[i][0])/2) #Координата по X
                          y = tracked_bboxes[i][1] + ((tracked_bboxes[i][3] - tracked_bboxes[i][1])/2)  #Координата по Y
                          oscMessage += (str(d) + ',' + str(c) + ',' + str(round(x)) + ',' +str(round(y)) + ',')

                      else:
                          #tracked_bboxes.remove(tracked_bboxes[i])
                          tracked_bboxes_len -= 1
                          
                      '''if i + 1 != tracked_bboxes_len:
                          oscMessage += ','
                          '''
    

                  if oscMessage[-2:] == ",,":
                    oscMessage = oscMessage[:-2]
                  if oscMessage[-3:] == ",,,":
                    oscMessage = oscMessage[:-3]
                    
                  oscMessage = str(tracked_bboxes_len) + ',' + oscMessage
                  answerMessage = oscMessage




                  

                  conn.sendall(bytes(f"{len(answerMessage)}", 'utf-8'))
                  conn.sendall(bytes(answerMessage,'utf-8'))
                  client.send_message("/tsdata", oscMessage)
                          
                          #print(answerMessage)
                  #print(oscMessage)

                  print(answerMessage)
                      
                  #img = Image.frombytes('L',(w,h),bytes(output+data))
                  #millis = int(round(time.time() * 1000))
                  #print ('Receiving - '+str(millis));
                  #img.save('output\\'+str(millis)+'1.bmp')
                except:
                    pass

    #-----------------------------------------------------------------------------
    # CLOSE CONNECTION
    #-----------------------------------------------------------------------------
    conn.close()
    print ("Client disconnected: ", addr)
