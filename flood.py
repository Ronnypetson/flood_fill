import cv2
import numpy as np
import matplotlib.pyplot as plt

im_path = 'TREM_.jpg'  #'test_tolerance.jpg' # 'TREM_.jpg'
blocks_fn = 'trem_'    #'test_tol_'
h_max = 2000
w_max = 2000
min_dw = 3
min_dh = 6

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
                    dk = k1-k0
                    dl = l1-l0
                    if self.grid[i][j] != 255 and dk > min_dw and dl > min_dh:
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
            kmax = max(kmax,k)
            lmax = max(lmax,l)
            kmin = min(kmin,k)
            lmin = min(lmin,l)
            if k < 0 or l < 0 or k >= r or l >= c or self.visited[k][l]:
                continue
            if self.grid[k][l] == t:
                self.visited[k][l] = True
                s.append([k+1,l])
                s.append([k,l+1])
                s.append([k-1,l])
                s.append([k,l-1])
                s.append([k+1,l+1])
                s.append([k-1,l-1])
                s.append([k-1,l+1])
                s.append([k+1,l-1])
        return [kmin,lmin,kmax,lmax]

#
# Filters: [medianBlur][cv2.THRESH_BINARY|cv2.ADAPTIVE_THRESH_MEAN_C|cv2.ADAPTIVE_THRESH_GAUSSIAN_C]
#

im = cv2.imread(im_path,0)
print(im.shape)
if im.shape[0] > h_max or im.shape[1] > w_max:
    r = min(h_max*1.0/im.shape[0],w_max*1.0/im.shape[1])
    im = cv2.resize(im,None,fx=r,fy=r,interpolation=cv2.INTER_CUBIC)
#bl = cv2.medianBlur(im,5)
#th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
_,th = cv2.threshold(im,160,255,cv2.THRESH_BINARY)
#th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
f = Flood(th)
f.flood_fill()
print(len(f.floods))
for i in range(len(f.floods)):
    ui = f.floods[i][0]
    uj = f.floods[i][1]
    li = f.floods[i][2]
    lj = f.floods[i][3]
    #print(ui,uj,li,lj)
    #cv2.imwrite('blocks2/'+blocks_fn+str(i)+'.jpg',im[ui:li+1,uj:lj+1])
    cv2.rectangle(im,(uj,ui),(lj,li),(0,255,0),1)
cv2.imwrite(blocks_fn+'.jpg',im)
cv2.imwrite(blocks_fn+'_th.jpg',th)
#
