import pycurl
import requests
from io import StringIO
import json

class MotorClient():
    def __init__(self):
        self.url = "http://192.168.1.3:8000/rev/"
        self.speed = 100
        self.c = pycurl.Curl()
        self.c.setopt(pycurl.URL, self.url)
        self.c.setopt(pycurl.FORBID_REUSE,False)
        self.c.setopt(pycurl.HTTPHEADER,['Accept: application/json',
                                    'Content-Type: application/json'])
        self.c.setopt(pycurl.POST,1)
        self.c.setopt(pycurl.TCP_NODELAY,1)
        self.c.setopt(pycurl.FRESH_CONNECT, False)
        self.c.setopt(pycurl.CONNECTTIMEOUT,2)

    
    def _sendPost(self,data):
        json_string = json.dumps(data)
        file_obj = StringIO(json_string)

        self.c.setopt(pycurl.READDATA, file_obj)
        self.c.setopt(pycurl.POSTFIELDSIZE, len(json_string))

        self.c.perform()
        #status_code = self.c.getinfo(pycurl.RESPONSE_CODE)
    
    def moveForward(self,speed:int,duration:float):
        json_file = {
            "speed": speed,
            "duration":duration,
            "direction": True
        }
        connection = requests.post(url=self.url,data=json.dumps(json_file))
        connection.close()
    
    def moveBackward(self,speed:int,duration:float):
        json_file = {
            "speed": speed,
            "duration":duration,
            "direction": False
        }

        self._sendPost(json_file)
    
    def stepForward(self):
        json_file = {
            "speed": 100,
            "duration":0.01,
            "direction": True
        }
        self._sendPost(json_file)
    
    def stepBackward(self):
        json_file = {
            "speed": 100,
            "duration":0.05,
            "direction": False
        }
        self._sendPost(json_file)
    
    def close(self):
        self.c.close()
