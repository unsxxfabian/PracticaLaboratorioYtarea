#TAREA DE IA

import numpy as np
import matplotlib.pyplot as pl

from sklearn.cluster import KMeans
#from sklearn import Metrics

v1=[3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8]
v2=[5,4,6,6,5,8,6,7,6,7,1,2,1,2,3,2,3]

print(v1)
print(v2)
x1=np.array(v1)
x2=np.array(v2)
x=np.array(list(zip(x1,x2))).reshape(len(x1),2)
pl.plot(x1,x2,'o',label="D")

pl.xlabel('x')
pl.ylabel('y')
pl.grid()
pl.legend()
pl.show()

k=3 
modelo_kmeans=KMeans(n_clusters=k).fit(x)


for i,l in enumerate(modelo_kmeans.labels_):
    print("(x1,x2)->:clase")
    print("({0},{1})->:{2}".format(x1[i],x2[i],l))
    
print("predicir: ")
predicir=modelo_kmeans.predict(x)
print(predicir)


prueba=[[3,5]]
print("prueba: ",prueba)
prediccion=modelo_kmeans.predict(prueba)
print("Prediccion: ({0},{1})".format(prueba,prediccion))

#calculando los centroides
print("centroides: ")
centroides=modelo_kmeans.cluster_centers_
print(centroides)
c1=centroides[0]
c2=centroides[1]
c3=centroides[2]
print("los centroides:")
print(c1)
print("los centroidesss: ")
cx1=c1[0]
print(cx1)
#para graficar
cx=[c1[0],c2[0],c3[0]]
cy=[c1[1],c2[1],c3[1]]


pl.plot(x1,x2,'o',label="puntos")
pl.plot(cx,cy,'o',label="centroides")
#pl.plot(c1[0],c1[1],'o',label="centroides")
#pl.plot(c2[0],c2[1],'o')
#pl.plot(c3[0],c3[1],'o')
pl.xlabel('x')
pl.ylabel('y')
pl.grid()
pl.legend()
pl.show()
print("-------------------------------------------")
pl.plot(x1,x2,'o',label="puntos")
pl.plot(c1[0],c1[1],'o',label="c1")
pl.plot(c2[0],c2[1],'o',label="c2")
pl.plot(c3[0],c3[1],'o',label="c3")
pl.xlabel('x')
pl.ylabel('y')
pl.grid()
pl.legend()
pl.show()