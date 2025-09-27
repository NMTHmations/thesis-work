import RPi.GPIO as GPIO
import time

class motorController():
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(16,GPIO.OUT)
        GPIO.setup(18,GPIO.OUT)
        GPIO.setup(29,GPIO.OUT)
        self.motorSpeed = GPIO.PWM(16,1000)
        self.motorSpeed.start(0)

    def _forward(self,speed):
        GPIO.output(18,True)
        GPIO.output(29,False)
        self.motorSpeed.ChangeDutyCycle(speed)

    def _backward(self,speed):
        GPIO.output(18,False)
        GPIO.output(29,True)
        self.motorSpeed.ChangeDutyCycle(speed)

    def _stop(self):
        GPIO.output(18,False)
        GPIO.output(29,False)
        self.motorSpeed.ChangeDutyCycle(0)

    def moveMotor(self,speed:int,direction:bool,duration:float):
        if direction == True:
            self._forward(speed)
        else:
            self._backward(speed)
        time.sleep(duration)
        self._stop()