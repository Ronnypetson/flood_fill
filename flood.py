import cv2
import numpy as np
import matplotlib.pyplot as plt

im_path = 'handwriting.png' # '../TREM-HAND.tif'
blocks_fn = 'teste_'

#
#gim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#plt.imshow(im)
#ptl.show()
#

def max_(a,b):
    if a > b:
        return a
    else:
        return b

def min_(a,b):
    if a < b:
        return a
    else:
        return b

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
                    k0,l0,k1,l1 = self.fill(i,j,self.grid[i][j])
                    if self.grid[i][j] != 255:
                        self.floods.append([k0,l0,k1,l1])
    
    def fill(self,i,j,t):
        r,c = self.grid.shape
        s = [[i,j]]
        kmax = 0
        lmax = 0
        kmin = r
        lmin = c
        while len(s) > 0:
            u = s.pop()
            k = u[0]
            l = u[1]
            kmax = max_(kmax,k)
            lmax = max_(lmax,l)
            kmin = min_(kmin,k)
            lmin = min_(lmin,l)
            if k < 0 or l < 0 or k >= r or l >= c or self.visited[k][l]:
                continue
            if self.grid[k][l] == t:
                self.visited[k][l] = True
                s.append([k+1,l])
                s.append([k,l+1])
                s.append([k-1,l])
                s.append([k,l-1])
        return [kmin,lmin,kmax,lmax]

im = cv2.imread(im_path,0)
#im = cv2.medianBlur(im,5)
#th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
f = Flood(th)
f.flood_fill()
for i in range(len(f.floods)):
    ui = f.floods[i][0]
    uj = f.floods[i][1]
    li = f.floods[i][2]
    lj = f.floods[i][3]
    #print(ui,uj,li,lj)
    cv2.imwrite('blocks/'+blocks_fn+str(i)+'.jpg',im[ui:li+1,uj:lj+1])

