import unittest
import ODVI_1_5
import time
import os
import threading
from pycaw.pycaw import AudioUtilities

class Test_ODVI(unittest.TestCase):
    def test_playTone(self):
        sessions = AudioUtilities.GetAllSessions()
        inFrequency = 222
        result = False 
        ODVI_1_5.ODVI.playTone(self,inFrequency)
        for session in sessions:
            if session.Process != None:
               if(session.Process.status() == 'running'):
                  result = True
        
        
        self.assertTrue(result)
        
    def test_isClose(self):
       obj = [['cone', [-8.358105, 4.5996227, 105.55864, 479.93732], 0.105199404, 1,4],[]]
       objD = ODVI_1_5.objDistance()
       result = objD.isClose(0, obj)
       
       self.assertTrue(result)
       
    def test_ttsOut(self):
        sessions = AudioUtilities.GetAllSessions()
        result = False 
        obj = ODVI_1_5.ODVI()
        
        obj.ttsOut("test working")
        for session in sessions:
            if session.Process != None:
               if(session.Process.status() == 'running'):
                  result = True
        self.assertTrue(result)
        
  
if __name__ == '__main__':
    unittest.main()