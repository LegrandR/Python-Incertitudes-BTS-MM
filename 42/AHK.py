# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:55:44 2020

@author: Romain
"""
	



import pyautogui
import time
from scipy import stats

width, height = pyautogui.size()

(x,y) = pyautogui.position()
for i in range(10):
      Xr = stats.norm.rvs(loc=0, scale = 10, size=1)
      Yr = stats.norm.rvs(loc=0, scale = 10, size=1)
      Tau = stats.poisson(1).rvs(1)
      pyautogui.moveTo(x+Xr[0], y+Yr[0], duration=Tau[0])
      pyautogui.click(x+Xr[0], y+Yr[0], button='right' )

while True:
      Tau = stats.expon(loc = 25, scale = 40).rvs(1)
      PressTime = stats.expon(loc = 0.2, scale =0.2).rvs(1)
      time.sleep(Tau)
      start = time.time()
      while (time.time() - start < PressTime[0]):
          pyautogui.press('space')
      print(PressTime[0])
      
      
pyautogui.click(3192, -127, button='left' )



while True:
    im = pyautogui.screenshot()
    (x,y) = pyautogui.position()
    print((x,y))
    print(im.getpixel((x+15,y+15)))
    time.sleep(1)
    
    
    
Xr = stats.norm.rvs(loc=0, scale = 10, size=1)
print(Xr)