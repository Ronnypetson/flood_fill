import cv2
import numpy as np
import matplotlib.pyplot as plt

im_path = 'teste_.jpg' #'../TREM-HAND.tif'

#
#gim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#plt.imshow(im)
#ptl.show()
#

class Flood:
    def __init__(self,grid):
        self.grid = grid
        self.floods = None
        self.visited = np.full(grid.shape,False)
    
    def flood_fill(self):
        r,c = self.grid.shape
        self.floods = []
        for i in range(r):
            for j in range(c):
                if not self.visited[i][j]:
                    self.fill(i,j,self.grid[i][j])
                    self.floods.append([i,j])
    
    def fill(self,i,j,t):
        r,c = self.grid.shape
        s = [[i,j]]
        while len(s) > 0:
            u = s.pop()
            k = u[0]
            l = u[1]
            if k < 0 or l < 0 or k >= r or l >= c or self.visited[k][l]:
                continue
            if self.grid[k][l] == t:
                self.visited[k][l] = True
                s.append([k+1,l])
                s.append([k,l+1])
                s.append([k-1,l])
                s.append([k,l-1])

im = cv2.imread(im_path,0)
im = cv2.medianBlur(im,5)
th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
f = Flood(th)
f.flood_fill()
print(len(f.floods))

