import pygame
import time

pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

print(joysticks)
print(joysticks[0].get_button)
time.sleep(5)
print(joysticks[0].get_button)

while True:
    print(joysticks[0].get_button)