from FrameOperations import Frame
import numpy as np

class Marker():
    def __init__(self, name, position=(0,0,0)):
        self.name = name             # name of the marker
        self.pos = position          # position in tracker coordinate system
    
    @property
    def position(self):
        """Position in tracker system"""
        return self.pos
    

    @position.setter
    def position(self, pos):
        self.position = np.array([[pos]])


    def __repr__(self):
        return f"NavElement(name={self.name})"