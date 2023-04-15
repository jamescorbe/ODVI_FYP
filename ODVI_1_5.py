from __future__ import print_function
import cv2 as cv
#import detect
import time
#import tensorflow as tf
import sys
import pyttsx3
from sys import exit
import supervision as sv
from threading import Thread
from gtts import gTTS
from pysine import sine
import numpy as np
from time import sleep
import pyaudio
from ultralytics import YOLO
import playsound
import os

    
class ODVI:
    
      def __init__(self):
         os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
         self.od = objDistance()
         self.result =None
         self.lastID =None
         self.time = time.perf_counter()
         self._count_ = 0
         self.engine = pyttsx3.init()
         self.voice = self.engine.getProperty('voices')
         self.engine.setProperty("voice",self.voice[1].id)
         self.yoloResults = []
         self.formattedResults =[]
         self.alertQueue =[]
         self.trackedObjects =[['',[0,0,0,0],0,0]]
         self.yoloRun = 0
         self.lastDetection =""
         self.workingDir = os.getcwd()
         self.frameImages = os.path.join(self.workingDir,'frame_images')
         self.testVideo =  os.path.join(self.workingDir,'detect_day_2_e.mp4')
         self.boxAnnotator = sv.BoxAnnotator(
             thickness=2,
             text_thickness=1,
             text_scale=0.5
             )
         self.formatedYoloRes = []
         self.objClass =[]
         self.model = YOLO("200epc15k.pt")  # load a pretrained YOLOv8n model
         self.detectionClasses = ['Construction-barrier', 'Construction-sign', 'bin', 'bollard', 'car', 'cone', 'puddle', 'street-pole', 'street-sign', 'tree']
  
      def playTone(self,inFrequency):
          duration = 0.6
          
          normFreq = inFrequency / 640 # normalise between 0 and 1
          normFreq = (normFreq*2 ) -1  # normalise between -1 and 1
          
          outFrequency = normFreq * 5000
          samples = np.arange(duration * 44100)
          waveform = np.sin(2 * np.pi * outFrequency * samples / 44100)
        
          p = pyaudio.PyAudio()
          stream = p.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=True)
          left = (1 - normFreq) /2
          right = (1 + normFreq) /2
          stereo_sound = np.column_stack((waveform * left, waveform * right))
          stream.write(stereo_sound.astype(np.float32).tobytes())
          stream.stop_stream()
          stream.close()
          p.terminate()
    
          ##sine(frequency=outFrequency, duration=0.2)
          
        
      def ttsOut(self,outText):
          self.engine.say(outText)
          self.engine.runAndWait()
          
          
      def outImage(self):
          for item in self.yoloResults:
              if self.result.boxes.id is not None:
                  self.yoloResults.tracker_id = self.result.boxes.id.cpu().numpy().astype(int)
          
          yoloLabels =[
                f"#{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
                for _,confidence,class_id,tracker_id
                in self.yoloResults
          
                ]
          self.frame = self.boxAnnotator.annotate(scene=self.frame, detections= self.yoloResults,labels=yoloLabels)
          cv.imshow("yolo",self.frame)
          
        
      def detectAndTrack(self,showImg,detSource): #manages what process runs when 
          if detSource != "0":
              detSource = self.testVideo
          run = True        
          while run == True:
              for self.result in  self.model.track(source= detSource,imgsz=640,stream=True,save = True):##save = True):
                  self.yoloResults =""
                  self.formattedResults.clear()
                  self.frame = self.result.orig_img
                  self.yoloResults = sv.Detections.from_yolov8(self.result)
                  
                  if(showImg):
                      self.outImage()
                  
                  for xyxy, confidence, class_id, tracker_id in self.yoloResults:
                      resultFormat =[self.model.model.names[class_id],[xyxy[0],xyxy[1],xyxy[2],xyxy[3]],confidence,tracker_id]
                      self.formattedResults.append(resultFormat)

        
                  if(len(self.yoloResults) >1 or len(self.yoloResults) >0 and self.yoloResults.tracker_id  != self.lastID): #checc
                      self.customTrack()
                      
                      
                      
                  k = cv.waitKey(1) & 0xFF
                  if k == ord('q'):
                      sys.exit()
                  
                  cv.waitKey(1)
                  keyboard = cv.waitKey(30)
              
      
      def customTrack(self):# loops through new object detections and compares to existing tracks, if not found they are added to the track list. Otherwise their detection count is increased and if over 4 and close enough an alert is created.
          self.alertQueue.clear()
          print(self.formattedResults)
          for formatItem in self.formattedResults: 
             found = False
             for index,trackedItem in enumerate(self.trackedObjects):
                 if (formatItem[3] == trackedItem[3] and formatItem[3] != None):
                     trackedItem[1] = formatItem[1]
                     trackedItem[2] = formatItem[2]
                     found = True
                     if(len(trackedItem) ==4):
                         trackedItem.append(1)
                         
                     else:
                         trackedItem[4] = trackedItem[4] +1
                         
                         if(trackedItem[4] >4 and self.od.isClose(index,self.trackedObjects)):
                             if(len(trackedItem) ==6):
                                 trackedItem.append(1)
                                 print("ello")
                                 alertItem = trackedItem
                                 alertItem.append(self.od.alertItemPos)
                                 self.alertQueue.append(alertItem)
                                 self.lastID =alertItem[3]
                                 self.alertUser()
                             else:
                                 print("tracked item",trackedItem)
                                 trackedItem[6] +1
                                 
                             
             if (not found and formatItem[3] != None):
                     self.trackedObjects.append(formatItem)
         
                     
                     
                     
          
      def alertUser(self):# if detection is the same as the last play tone, otherwise start tts thread.
          for item in self.alertQueue:
              if(self.lastDetection == item[0]):
                 self.playTone(item[7])
                 break
                                                                                                                                                                                                                                                                                                      
              if(item[5] == 'forward'):
                  speech= str(item[0])+" ahead"
                  thread = Thread(target =  self.ttsOut(speech), args = (speech))
                  thread.start()
                  self.lastDetection =item[0]
                  
              else:
                  speech= str(item[0])+" on the "+str(item[5])
                  self.lastDetection =item[0]
                  thread = Thread(target = self.ttsOut(speech) ,args = (speech)).start()
                  
                  
              
              
              
         
                     
                         
                
class objDistance:
    
    def __init__(self):
        self.streetPoleWidth =20
        self.streetPoleHeight=300
        self.streetSignWidth=15
        self.alertItemPos =0
        self.streetSignHeight=250
        self.carWidth =90
        self.carHeight =350
        self.constructionBarrierWidth=80
        self.constructionBarrierHeight=350
        self.constructionSignWidth = 72
        self.binWidth =110
        self.binHeight =260
        self.bollardWidth =14
        self.bollardHeight =200
        self.coneWidth=32.2
        self.puddleWidth=244
        self.puddleHeight=290
        self.lastDetection =""
        self.treeWidth=60
        self.treeHeight=300
              
              
    def isClose(self,index,trackedObjects):
        
        print("to",trackedObjects[0])
        centerPos = trackedObjects[index][1][0] +(abs(trackedObjects[index][1][0] -trackedObjects[index][1][2])/2)
        self.alertItemPos = centerPos
        objWidth = abs(trackedObjects[index][1][0] - trackedObjects[index][1][2])
        objHeight = trackedObjects[index][1][3]
        close = False
        
        if(trackedObjects[index][0] == 'bollard'):
            if( objWidth > self.bollardWidth and objHeight > self.bollardHeight):
                if(len(trackedObjects[index]) ==5): 
                    close = True
        
        elif(trackedObjects[index][0] == 'Construction-sign'):
            if( objWidth > self.constructionSignWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
            
            
        elif(trackedObjects[index][0] == 'Construction-barrier'):
            if( objHeight > self.constructionBarrierHeight and objWidth >self.constructionBarrierWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
                    
                    
        elif(trackedObjects[index][0] == 'cone'):
            if( objWidth > self.coneWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
             
        elif(trackedObjects[index][0] == 'bin'):
            if( objWidth > self.binWidth ):
                if(len(trackedObjects[index]) ==5): 
                    close = True
            
        
        elif(trackedObjects[index][0] == 'car'):
            if( objWidth > self.carWidth):
                if(len(trackedObjects[index]) ==5):
                    close = True
       
        elif(trackedObjects[index][0] == 'puddle'):
            if( objHeight > self.puddleHeight and objWidth > self.puddleWidth):
                if(len(trackedObjects[index]) ==5):
                    close = True
        
        elif(trackedObjects[index][0] == 'street-pole'):
            if( objHeight > self.streetPoleHeight and objWidth > self.streetPoleWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
            
        elif(trackedObjects[index][0] == 'street-sign'):
            if( objHeight > self.streetSignHeight and objWidth > self.streetSignWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
                    
        elif(trackedObjects[index][0] == 'tree'):
            if( objHeight > self.treeHeight and objWidth > self.treeWidth):
                if(len(trackedObjects[index]) ==5): 
                    close = True
        
        if(close):
            print("center Position =", centerPos)
            if(centerPos <= 180 ):
                trackedObjects[index].append("left")
            
            if(centerPos > 180  and centerPos <460):
                trackedObjects[index].append("forward")
            
            elif(centerPos >= 460):
                trackedObjects[index].append("right")
            return True
        else:
            return False
                     
    
if __name__ == '__main__':
    ps = ODVI()
    ps.detectAndTrack(True,"2")