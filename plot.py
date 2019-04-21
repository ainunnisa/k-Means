#referensi belajar = scikit-learn.org dan mubaris.com
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


Trainset = pd.read_csv('TrainsetTugas2.csv', delimiter=';')
# print(data.shape)
print(Trainset)

#ambil data berdasarkan nama kolom yang diberi
#variabel1 untuk menampung data dari kolom 'Col1'
v1 = Trainset['Col1'].values
#variabel2 untuk menampung data dari kolom 'Col2'
v2 = Trainset['Col2'].values

#menampung Trainset
X = np.array(list(zip(v1, v2)))

#visualisasi Trainset
plt.scatter(v1, v2, c='brown', s=5)
#plt.show()

#menentukan data akan dibagi menjadi berapa cluster
kmeans = KMeans(n_clusters=6)

#melakukan perhitungan / fitting kmeans clustering
fitTrain = kmeans.fit(X)

#ini kan nampung titik mana masuk ke centroid mana(dikasih label masuk ke clustering mana)
labelTrain = kmeans.predict(X)

#ini titik centroidnya, kalau di print, akan ada titik [x,y] centroid
centroids = kmeans.cluster_centers_
#print(centroids)

#visualisasi untuk masing2 cluster agar beda warna dengan kode warna rainbow
plt.scatter(v1, v2, c=labelTrain, cmap='rainbow')

#visualisasi centroid dengan warna hijau agar terlihat centroidnya dimana
plt.scatter(centroids[:,0],centroids[:,1], c='green')

#menampilkan visualisasi Trainset yang sudah di clusterisasi
plt.show()

#---ini buat Testset---
#untuk load data Test
Testset = pd.read_csv('TestsetTugas2.csv', delimiter=';')

#cek apakah data sudah berhasil ke load
print(Testset)

#variable3 untuk menampung data di kolom 'Col1' pada TestsetTuags2.csv
v3 = Testset['Col1'].values

#variable4 untuk menampung data di kolom 'Col2' pada TestsetTuags2.csv
v4 = Testset['Col2'].values

#menampung testset
Y = np.array(list(zip(v3, v4)))

#visualisasi testset
plt.scatter(v3, v4, c='red', s=5)
#plt.show()

#melakukan perhitungan / fitting kmeans clustering
fitTest = kmeans.fit(Y)

#ini kan nampung titik mana masuk ke centroid mana(dikasih label masuk ke clustering mana)
labelTest = kmeans.predict(Y)

#visualisasi untuk masing2 cluster agar beda warna dengan kode warna rainbow
plt.scatter(v3, v4, c=labelTest, cmap='rainbow')

#visualisasi centroid dengan warna biru agar terlihat centroidnya dimana
plt.scatter(centroids[:,0],centroids[:,1], c='blue')

#menampilkan visualisasi Testset yang sudah di clusterisasi
plt.show()

#menyimpan hasil Testset yang sudah diberi label
np.savetxt('hasil.txt',labelTest)