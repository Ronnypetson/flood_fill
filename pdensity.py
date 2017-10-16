import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random
#=====================
WHITE_THRESH = 255
#=====================
def count_height(fname):
    img = cv2.imread(fname)  # Load file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Set to grayscale

    h,w = img.shape
    cnt = 0  # Row counter (blank rows are ignored)
    for i in range(h):  # height
        for j in range(w):
            if img[i,j] < WHITE_THRESH:
                cnt += 1;
                break
    return cnt

def min_(a,b):
    if a < b:
        return a
    else:
        return b

def count_min_height(fnames):
    mini = 999999
    for fname in fnames:
        w = count_height(fname)
        if w < mini:
            mini = w
    return mini

def get_density(fname):
    img = cv2.imread(fname)  # Load file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Set to grayscale

    h,w = img.shape
    ind = 0  # Row counter (blank rows are ignored)
    d = np.zeros((h))  # Setup initial density to 0.0
    for i in range(h):  # height
        wi = 0  # row length counter
        for j in range(w):  # width
            if img[i,j] < WHITE_THRESH:  # Ignore white background
                d[ind] += img[i,j]
                wi += 1
        if wi > 0:
            d[ind] /= wi
            ind += 1
    return d[0:ind+1]  # <--

def normalize_density(d,m):
    l = len(d)
    if l <= m:  # should not occur
        return d

    r = np.zeros(l)
    ind = 0
    ind2 = 0
    while ind < m and ind2 < l:
        if random.randint(0,l-1) < m:
            r[ind] = d[ind2]
            ind += 1
        ind2 += 1
    return r[0:m]  # r[0:ind]

def get_all_densities_normalized(fnames,m):
    r = []
    for fname in fnames:
        r.append(normalize_density(get_density(fname),m))
    return r

def get_means(d,mininmo):
    m = np.zeros(mininmo)
    #print(len(d))
    for j in range(mininmo):
        for i in range(len(d)):
            #print(len(d[i]))
            m[j] += d[i][j]
        m[j] /= len(d)
    return m

def plot_means(m,m2,minimo,oname):
    x = np.arange(0,minimo,1)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title(oname)
    plt.xlabel('row')
    plt.ylabel('mean pixel density')
    plt.plot(x,m,label='scribe')
    plt.plot(x,m2,label='Sue')
    plt.legend(bbox_to_anchor=(0.82,1),loc=2,borderaxespad=0.)
    plt.savefig(oname + '_mean_comparison.jpg')
    #plt.show()

def plot_densities(d,sd,names,snames,ofile):
    n = len(d)
    sn = len(sd)
    #if n%3 == 0:
    #    lin = n/3
    #else:
    #    lin = (n/3)+1
    #col = 3
    for i in range(n):
        x = np.arange(0,len(d[i]),1)
        sx = np.arange(0,len(sd[i]),1)
        #plt.subplot(lin,col,i+1)
        #plt.subplot(1,1,1)
        #plt.plot(x,d[i],label=names[i])
        #plt.title(names[i] + ' and ' + snames[i])
        plt.xlabel('row')
        plt.ylabel('pixel density')
        #plt.tight_layout()
        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(x,d[i],label=names[i])
        plt.plot(sx,sd[i],label=snames[i])
        plt.yticks(np.arange(0, 255, 50))
        plt.legend(bbox_to_anchor=(0.82,1),loc=2,borderaxespad=0.)
        plt.savefig(ofile + str(i) + '.jpg')
        #plt.show()

sue_letters = '../Sue/nontrem/h/*.jpg'
letter_files = 'nontrem/h/*.jpg'  # th, sth, nth, snth
#
fnames = glob.glob(letter_files)
snames = glob.glob(sue_letters)
#
min_len = count_min_height(fnames)
min_sue = count_min_height(snames)
min_all = min_(min_len,min_sue)
#
densities = get_all_densities_normalized(fnames,min_all)
sdensities = get_all_densities_normalized(snames,min_all)
#
#ofl = 'sue_trem/h/'
oname = 'sue_nontrem_h'
plot_means(get_means(densities,min_all),get_means(sdensities,min_all),min_all,oname)
#plot_densities(densities,sdensities,fnames,snames,ofl)

