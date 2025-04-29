import shelve, os, re
import numpy as np
g=shelve.open('library_300_infer_CI')
g['CI']=np.zeros((216000,4,3,4))
g.close()
count=0
for i in os.scandir():
    if re.match('library_300_infer_batch.*',i.name):
        g=shelve.open('library_300_infer_CI')
        h=shelve.open(i.name)
        for j in h.keys():
            index=int(j)
            g['CI'][index:index+1]=h[j]['CI']
            count+=1
        g.close()
        print(count,i.name)

	
