from multicam import MotorClient
from time import sleep

client = MotorClient.MotorClient()

for i in range(80,100):
    client.moveMotor(100,False,duration=0.1)