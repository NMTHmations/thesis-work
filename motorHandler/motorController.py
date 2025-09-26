import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(16,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
motorSpeed = GPIO.PWM(22,1000)
motorSpeed.start(0)

def forward(speed):
    GPIO.output(16,True)
    GPIO.output(18,False)
    motorSpeed.ChangeDutyCycle(speed)

def backward(speed):
    GPIO.output(16,False)
    GPIO.output(18,True)
    motorSpeed.ChangeDutyCycle(speed)

def stop():
    GPIO.output(18,False)
    GPIO.output(16,False)
    motorSpeed.ChangeDutyCycle(0)

def moveMotor(speed:int,direction:bool,duration:float):
    if direction == True:
        forward(speed)
    else:
        backward(speed)
    time.sleep(duration)
    stop()

#moveMotor(100,False,5)
while True:
    backward(100)