"""
Функции различного назначения
"""
import numpy as np
import math
import cv2 as cv
# from PIL import Image
# from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt

# ####################### Цвета RGB #######################
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (255, 0, 255)
turquoise = (255, 255, 0)
white = (255, 255, 255)


# #############################################################
#                      ФУНКЦИИ ПРОЧИЕ
# #############################################################
def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    :param n: number of distinct color
    :param name: argument name must be a standard mpl colormap name
    :return: function
    """
    return plt.cm.get_cmap(name, n)

