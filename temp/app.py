
import time
import os

global a

def f(aloc):
    global a
    a=aloc
    while True:
        print(a, os.getpid())
        time.sleep(5)