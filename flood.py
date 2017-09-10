import cv2
import matplotlib.pyplot as plt

im_path = '../TREM-HAND.tif'
im = cv2.imread(im_path,0)

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
    def __init__(self,grid,thresh):
        self.grid = grid
        self.floods = None
        self.thresh = thresh
        r,c = grid.shape()
        self.visited = [False*c]*r
    
    def flood_fill():
        r,c = grid.shape()
        for i in range(r):
            for j in range(c):
                if not visited[i][j]:
                    fill(i,j,grid[i][j])
                    self.floods.append([i,j])
    
    def fill(i,j,t):
        r,c = grid.shape()
        if visited[i][j] or i < 0 or j < 0 or i >= r or j >= c:
            return
        if grid[i][j] == t:
            visited[i][j] = True
            fill(i+1,j,t)
            fill(i,j+1,t)
            fill(i-1,j,t)
            fill(i,j-1,t)

