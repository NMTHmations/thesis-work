import socket
import time

class MotorClient():
    def __init__(self):
        self.interface = "eth0"
        self.dst_mac = b'\x86\x0e\x35\x1a\x8e\x88'
        self.src_mac = b'\x2c\xcf\x67\x9d\x12\x4d'
        self.eth_type = b'\x88\xb5'
        self.s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        self.s.bind((self.interface, 0))


    def stepMotor(self,direction:bool):
        payload = str()
        if direction == False:
            payload = b"100;0;0.1"
        else:
            payload = b"100;1;0.1"


        min_payload = 46
        if len(payload) < min_payload:
            payload = payload + b'\x00' * (min_payload - len(payload))

        frame = self.dst_mac + self.src_mac + self.eth_type + payload

        try:
                n = self.s.send(frame)
                if n == 0:
                    print("Warning: send returned 0 bytes")
        except:
             print("Could not send the command to the DC motor's server!")
    
    def moveMotor(self,speed:int,direction:bool,duration:float):
        payload = str()
        if direction == False:
            payload = b"" + str(speed).encode() + b";0;" + str(duration).encode() + b""
        else:
            payload = b"" + str(speed).encode() + b";1;" + str(duration).encode() + b""


        min_payload = 46
        if len(payload) < min_payload:
            payload = payload + b'\x00' * (min_payload - len(payload))

        frame = self.dst_mac + self.src_mac + self.eth_type + payload

        try:
                n = self.s.send(frame)
                if n == 0:
                    print("Warning: send returned 0 bytes")
        except:
             print("Could not send the command to the DC motor's server!")
    
    def close(self):
         self.s.close()