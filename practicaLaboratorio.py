#PRACTICA EN EL LABORATORIO
import numpy as np
import matplotlib.pyplot as pl

from sklearn.cluster import KMeans
#from sklearn import Metrics

v1=[1,2,4,5]
v2=[1,1,3,4]

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

k=2 
modelo_kmeans=KMeans(n_clusters=k).fit(x)


prueba=[[2,5]]
print("prueba: ",prueba)
prediccion=modelo_kmeans.predict(prueba)
print("Prediccion: ({0},{1})".format(prueba,prediccion))
#prueba 2
prueba=[[3,5]]
print("prueba2: ",prueba)
prediccion=modelo_kmeans.predict(prueba)
print("Prediccion2: ({0},{1})".format(prueba,prediccion))




#calculando los centroides
print("centroides: ")
centroides=modelo_kmeans.cluster_centers_
print(centroides)
c1=centroides[0]
c2=centroides[1]

print("los centroides:")
print(c1)
print("los centroidesss: ")
cx1=c1[0]
print(cx1)
#para graficar
cx=[c1[0],c2[0]]
cy=[c1[1],c2[1]]


pl.plot(x1,x2,'o',label="puntos")
pl.plot(cx,cy,'o',label="centroides")

pl.xlabel('x')
pl.ylabel('y')
pl.grid()
pl.legend()
pl.show()
print("-------------------------------------------")
pl.plot(x1,x2,'o',label="puntos")
pl.plot(c1[0],c1[1],'o',label="c1")
pl.plot(c2[0],c2[1],'o',label="c2")

pl.xlabel('x')
pl.ylabel('y')
pl.grid()
pl.legend()
pl.show()