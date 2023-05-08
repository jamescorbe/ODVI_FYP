from __future__ import print_function
import cv2 as cv
import time
import sys
import pyttsx3
import supervision as sv
from threading import Thread
import numpy as np
import pyaudio
from ultralytics import YOLO
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
         self.model = YOLO("16kbest.pt")  # load a pretrained YOLOv8n model
         self.detectionClasses = ['Construction-barrier', 'Construction-sign', 'bin', 'bollard', 'car', 'cone', 'puddle', 'street-pole', 'street-sign', 'tree']
  
      def playTone(self,inFrequency):
          """
          Plays a directional tone
          
          normalised between 0 and 1 and then 1 to -1, this is in order
          to format the input for paning between the left and right channel.
          This input is then multiplied to put it into a comfortable hearing range,
          before constructing and playing a pyaudio stream and playing the tone for the user.
          
          Args: 
              arg1:   Object center position
          """
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
          """
          Text to speech output of object names and direction
          
          Args: 
              arg1:   obj name and direction
          """
          self.engine.say(outText)
          self.engine.runAndWait()
          
          
      def outImage(self): 
          """
          Outputs the test display to the user.
          
          Formats the detection bounding box, while using supervison annotation
          to apply object names confidance and track ids.
          Note ultralytics version ultralytics==8.0.68 required.
          """
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
          
        
      def detectAndTrack(self,showImg,detSource): 
          """
          Detection and tracking of input video stream
          
          Takes an input source, if its not 0 or 1 the test video will be used.
          Detection and tracking with the custom yolov8 data set will then loop for 
          each frame in an open cv live video stream, with custom track being called
          if a detection is found.
          
          
          Args: 
              arg1:   show image Boolean
              arg2:   detection source
          
          """
          if detSource != "0" or "1":
              detSource = self.testVideo
          run = True        
          while run == True:
              for self.result in  self.model.track(source= detSource,imgsz=640,stream=True):
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
              
      
      def customTrack(self):
          """
          Custom tracker used to track objects.
      
          loops through new object detections and compares to existing tracks, 
          if not found they are added to the track list. Otherwise their detection count 
          is increased and if over 4 and close enough an alert is created.
      
          """
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
                         
                         if(trackedItem[4] >4 and self.od.isClose(trackedItem)):
                             if(len(trackedItem) ==6):
                                 trackedItem.append(1)
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
         
                     
                     
                     
          
      def alertUser(self):
          """
          Loop for allerting user of hazards
          
          if detection is the same as the last play tone, 
          otherwise start tts thread based on the direction of the object.
          
          """
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
              
              
    def isClose(self,trackedObject):
        """
        Checks for object distance
        
        Takes in the tracked object and compares it width and height
        to the minimum required for detection.If this is met its area
        of the screen will be appended and true will be returned.
        
        Args:
            arg1:   tracked object to compare
        
        Returns:
            True or False
        """
        
        print("to",trackedObject)
        centerPos = trackedObject[1][0] +(abs(trackedObject[1][0] -trackedObject[1][2])/2)
        self.alertItemPos = centerPos
        objectName = trackedObject[0]
        objWidth = abs(trackedObject[1][0] - trackedObject[1][2])
        objHeight = trackedObject[1][3]
        close = False
        
        if(objectName == 'bollard'):
            if( objWidth > self.bollardWidth and objHeight > self.bollardHeight):
                if(len(trackedObject) ==5): 
                    close = True
        
        elif(objectName == 'Construction-sign'):
            if( objWidth > self.constructionSignWidth):
                if(len(trackedObject) ==5): 
                    close = True
            
            
        elif(objectName == 'Construction-barrier'):
            if( objHeight > self.constructionBarrierHeight and objWidth >self.constructionBarrierWidth):
                if(len(trackedObject) ==5): 
                    close = True
                    
        elif(objectName == 'cone'):
            if( objWidth > self.coneWidth):
                if(len(trackedObject) ==5): 
                    close = True
             
        elif(objectName == 'bin'):
            if( objWidth > self.binWidth ):
                if(len(trackedObject) ==5): 
                    close = True
            
        elif(objectName == 'car'):
            if( objWidth > self.carWidth):
                if(len(trackedObject) ==5):
                    close = True
       
       # elif(objectName == 'puddle'):
       #     if( objHeight > self.puddleHeight and objWidth > self.puddleWidth):
       #         if(len(trackedObject) ==5):
       #             close = True
        
        elif(objectName == 'street-pole'):
            if( objHeight > self.streetPoleHeight and objWidth > self.streetPoleWidth):
                if(len(trackedObject) ==5): 
                    close = True
            
        elif(objectName == 'street-sign'):
            if( objHeight > self.streetSignHeight and objWidth > self.streetSignWidth):
                if(len(trackedObject) ==5): 
                    close = True
                    
        elif(objectName == 'tree'):
            if( objHeight > self.treeHeight and objWidth > self.treeWidth):
                if(len(trackedObject) ==5): 
                    close = True
        
        if(close):
            print("center Position =", centerPos)
            if(centerPos <= 180 ):
                trackedObject.append("left")
            
            if(centerPos > 180  and centerPos <460):
                trackedObject.append("forward")
            
            elif(centerPos >= 460):
                trackedObject.append("right")
            return True
        else:
            return False
                     
    
if __name__ == '__main__':
    ps = ODVI()
    ps.detectAndTrack(True,"2")