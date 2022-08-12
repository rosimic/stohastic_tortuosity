# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:18:55 2022

@author: Robert
"""
import itertools as it
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
import random as rnd
from scipy import ndimage as nd
from scipy.ndimage import label
import time
from tqdm import tqdm # loading bar
from pyevtk.hl import gridToVTK
    
class Matrica():
    
    def __init__(self, dim, mx, my, sx, sy):        
        """
        Parameters
        ----------
        dim : integer
            Dimenzije matrice.
        mx : float
            Očekivanje (mean) normalne distribucije po x osi.
        my : float
            Očekivanje (mean) normalne distribucije po y osi.
        sx : float
            Devijacija normalne distribucije po x osi.
        sy : float
            Devijacija normalne distribucije po y osi.
        Returns
        -------
        Tau.
        """
        
        # npzImgDir = 'C:\\Users\\Robert\\Desktop\\Fakultet\\Diplomski\\tort\\NPZ\\' # loadanje direktorija
        
        self.dim = dim
        self.mx = mx
        self.my = my
        self.sx = sx
        self.sy = sy
        return
    
    def set_ImgDir(self, path):
        self.ImgDir = path
        return    

    def load_Image(self, npzImgDir = None):
        m = np.zeros([self.dim, self.dim, self.dim], np.int16)    
        if npzImgDir == None: npzImgDir = self.ImgDir
        for i in tqdm(range(1, self.dim+1)):
            s = np.load(npzImgDir + str(i-1) + '.input.npz')['arr_0'] # ucitava matricu iz filea
            m[i-1, :, :] = s[:self.dim, :self.dim] # sprema 2D matricu u 3D matricu pomičući se za brojač
        self.m = m
        return
    
    def labelPores(self, myImage, pattern = 2):
        
        if pattern == 1:
            pattern_3D = np.array([
              [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]],
              [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]],
              [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]])
            
        if pattern == 2:
            pattern_3D = np.array([
              [[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]],
              [[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]],
              [[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]]])
        
        labeledArray, numFeatures = label(myImage>0, structure = pattern_3D)
        
        return (labeledArray, numFeatures)

    def connectedPores(self, myArray):     
        connected_pores = myArray.copy()
        test_elements = np.unique(connected_pores[-1].flatten()).copy()
        connected_pores = connected_pores*(1*np.isin(connected_pores, test_elements))
        test_elements = np.unique(connected_pores[0].flatten()).copy()
        connected_pores = connected_pores*(1*np.isin(connected_pores, test_elements))
        
        return (connected_pores)
    
    def koordinate(self, lcp):        
        """
        Parameters
        ----------
        lcp : numpy array
            3D matrica nula i jedinica gdje jedinice predstavljaju porni prostor.

        Returns
        -------
        x : numpy array
            Lista x koordinata svih pora predstavljenih jedinicom u matrici.
        y : numpy array
            Lista y koordinata svih pora predstavljenih jedinicom u matrici.
        z : numpy array
            Lista z koordinata svih pora predstavljenih jedinicom u matrici.
        """        
        coords = np.where(lcp == 1)
        x, y, z = coords[2], coords[1], coords[0]
        
        return (x, y, z)

    def tau_iz_tezista(self, dim, lcp):
        
        """
        racuna teziste za svaki slice (sveukupno) 
        i na temelju takvog tezista udaljenosti izmedju tezista susjednih sliceova
        
        Parameters
        ----------
        dim : integer
            Dimenzije matrice.
        lcp : numpy array
            3D matrica nula i jedinica gdje jedinice predstavljaju porni prostor.

        Returns
        -------
        tau : float
            Tortuozitet izračunat iz težišta pora.
        """
        
        self.tezista = []
        udaljenost, udaljenost_suma = 0, 0
        
        for i in range(dim):
            self.tezista.append(nd.center_of_mass(lcp[i]))
            
        for i in range(dim-1):
            udaljenost = math.sqrt((self.tezista[i][1] - self.tezista[i+1][1]) ** 2 + (self.tezista[i][0] - self.tezista[i+1][0]) ** 2 + (0 - 1) ** 2)
            udaljenost_suma += udaljenost

        tau = udaljenost_suma / dim
        
        return tau
    
    def normal2D(self, mx, my, sx, sy):
        
        """
        Parameters
        ----------
        mx : float
            Očekivanje (mean) normalne distribucije po x osi.
        my : float
            Očekivanje (mean) normalne distribucije po y osi.
        sx : float
            Devijacija normalne distribucije po x osi.
        sy : float
            Devijacija normalne distribucije po y osi.
            
        Returns
        -------
        distr: numpy array
            Vrijednosti distribucije za 3x3 matricu.

        """
        
        x = np.array([1, 0, 1])
        y = np.array([1, 0, 1])
        
        x, y = np.meshgrid(x, y)
        
        self.distr = 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))    # Formula za normalnu distribuciju
        
        return self.distr
    
    def provjera(self, z):
        
        """
        provjerava ostvarive putanje
        
        Parameters
        ----------
        z : integer
            Slice u z osi.

        Returns
        -------
        mogucnosti : array
            Mogući pomaci u x, y smjeru.

        """
        
        mogucnosti = []     # Ovdje se spremaju mogućnosti kretanja
        
        for i in range(len(self.koordinate)):   # Za sve moguće pomake provjeri
            if (self.point[1]+self.koordinate[i][1]) >= 0 and (self.point[2]+self.koordinate[i][0]) >= 0:   # Ako je unutar dimenzija max dimenzija matrice
                if (self.point[1]+self.koordinate[i][1]) < self.dim and (self.point[2]+self.koordinate[i][0]) < self.dim:   # Ako je unutar min dimenzija matrice
                    if self.kopija[z][self.point[1]+self.koordinate[i][1]][self.point[2]+self.koordinate[i][0]] == 1:   # Ako je točka neposjećena
                        mogucnosti.append(i+1)  # Dodaj ju kao mogući smjer kretanja
        
        return mogucnosti
    
    def hod(self, mogucnosti, z, mx, my, sx, sy):
        
        """
        Parameters
        ----------
        mogucnosti : array
            Mogući pomaci u x, y smjeru.
        z : integer
            Slice u z osi.
        mx : float
            Očekivanje (mean) normalne distribucije po x osi.
        my : float
            Očekivanje (mean) normalne distribucije po y osi.
        sx : float
            Devijacija normalne distribucije po x osi.
        sy : float
            Devijacija normalne distribucije po y osi.

        Returns
        -------
        self.point : tuple
            Koordinate točke koja je odabrana kao sljedeća.

        """
        
        self.sanse = self.normal2D(mx, my, sx, sy).flatten()   # Normalna distribucija unutar 3x3 grida, pretvorena u 1D listu
        self.novi = [self.sanse[mogucnosti[i]-1] for i in range(len(mogucnosti))]   # Dodaj samo one elemente distribucije na koje se može pomaknuti u 1D listu
        self.posto = [self.novi[i]/sum(self.novi) for i in range(len(self.novi))]   # Normalizacija za računanje postotnog udjela
        self.rand = np.random.choice(mogucnosti, p=self.posto)   # Random smjer kretanja izabran iz mogućih smjerova s postotcima određenim normalnom distribucijom
        
        for i in range(len(self.koordinate)):   # Za sve moguće pomake provjeri
            if self.rand == (i+1):  # Ako je smjer kretanja (i+1)
                self.point = (z, self.point[1]+self.koordinate[i][1], self.point[2]+self.koordinate[i][0])  # Nova točka
                self.kopija[self.point[0]][self.point[1]][self.point[2]] = 3    # Koordinate nove točke označene kao posjećene
        
        return self.point
    
    def stoh(self):
        
        """

        Returns
        -------
        srednji_tau : float
            Srednji tortuozitet putanja iz svih točaka nultog slicea izračunat stohastičkim hodom kroz pore.

        """

        self.sliceZ = np.zeros(len(np.where(self.z == 0)[0]), dtype=int)     # z koordinate svih točaka iz prvog (nultog) slicea
        self.sliceY = self.y[:len(self.sliceZ)]                              # y koordinate svih točaka iz prvog (nultog) slicea
        self.sliceX = self.x[:len(self.sliceZ)]                              # x koordinate svih točaka iz prvog (nultog) slicea
        self.test_tocke = []
        for i in range(len(self.sliceZ)):
            self.test_tocke.append((self.sliceZ[i], self.sliceY[i], self.sliceX[i]))
            
        self.svetocke = []     # Točke koje tvore putanju (koordinate putanje "jedne cestice" od z = 0 do z = n)
        timerpocetak = time.perf_counter()     # Inicijalizacija timera  
        self.koordinate = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]    # Koordinate svih mogućih pomaka
        self.svitau = []    # Tau za svaku putanju se sprema unutar liste

        for j in range(len(self.sliceZ)):
            timerrute1 = time.perf_counter()
            self.kopija = self.labeledConnectedPores.copy()      # Kopija lcp koja će služiti za označavanje već posjećenih pora
            self.tocke = []     # Točke koje tvore putanju (koordinate putanje "jedne cestice" od z = 0 do z = n)
            self.point = (self.sliceZ[j], self.sliceY[j], self.sliceX[j])    # Odabir početne točke
            self.tocke.append(self.point)    # Spremanje početne točke u listu odabranih točaka
            self.zpomak = 1     # Inicijalizacija pomaka po z osi
            

            while self.zpomak != self.dim:  # Dok pomak po z osi ne dođe do dimenzija matrice
                self.indeksi_pomaka = self.provjera(self.zpomak)    # Računanje mogućih točaka u sljedećem sliceu
                    
                if len(self.indeksi_pomaka) != 0:   # Ako lista mogućih pomaka nije prazna
                    self.point = self.hod(self.indeksi_pomaka, self.zpomak, self.mx, self.my, self.sx, self.sy)     # Odaberi novu točku
                    self.tocke.append(self.point)   # Dodaj novu točku u listu odabranih točaka
                    self.zpomak+=1  # Pomakni se na sljedeći slice
                    # print('Mogući pomaci: {} | Odabran pomak: {} | Nova točka: {}'.format(self.indeksi_pomaka, self.rand, self.point)) 
                    
                else:
                    self.strane = self.provjera(self.zpomak-1)  # Ako je lista mogućih pomaka prazna, pogledaj oko sebe gdje se može pomaknut u istom sliceu
                    if len(self.strane) != 0:   # Ako se može pomaknut oko sebe u istom sliceu
                        self.point = self.hod(self.strane, self.zpomak-1, self.mx, self.my, self.sx, self.sy)   # Pomakni se na točku oko sebe
                        self.tocke.pop()    # Ukloni posljednju točku iz liste odabranih točaka
                        self.tocke.append(self.point)    # Dodaj novu točku u listu odabranih točaka (kako bi bila samo jedna točka po sliceu)
                        # print('Nova bočna točka: {}'.format(self.point))
                    else:    
                        if len(self.tocke) != 1:
                            self.zpomak-=1  # Ako se ne može pomaknut ni naprijed ni oko sebe, vrati se na prošli slice
                            self.tocke.pop()     # Ukloni točku iz posljednjeg slicea
                            self.point= self.tocke[-1]  # Nova točka je zadnja odabrana
                            # print('Obrisano | Nova točka: {}'.format(self.point))
                        else:
                            # self.kopija = self.labeledConnectedPores.copy()     # U malom broju slučaja (kada postoji pomak za samo jedan 
                            break                                                    # slice prije nego dođe do potrebe za vraćanjem) ako točka 
                                                                                # skrene u stranu gdje nema puta može doći do zapinjanja 
                                                                                # točke pa je potrebno očistiti oznake posjećenih ćelija 
                                                                                # da proba opet naći put
                            # print('Kopija se očistila')
                            
            self.svetocke.append(self.tocke)    # Točke jedne putanje se dodaju u listu svih putanja
            self.zapis(self.tocke)  # Točke putanje se bilježe u .txt file
            timerrute2 = time.perf_counter()
            udaljenost_suma = 0
            
            for i in range(len(self.tocke)-1):
                udaljenost = math.sqrt((self.tocke[i][1] - self.tocke[i+1][1]) ** 2 + (self.tocke[i][2] - self.tocke[i+1][2]) ** 2 + (0 - 1) ** 2)
                udaljenost_suma += udaljenost
                
            self.svitau.append(udaljenost_suma / self.dim)  # Tau pojedine putanje se sprema u listu svih tau
            print('{}/{} | {}'.format(j+1, len(self.sliceZ), (timerrute2-timerrute1)))
                
        timerkraj = time.perf_counter()
        print('Vrijeme do pronalaska svih ruta: {} sekundi'.format(timerkraj-timerpocetak))
    
        self.srednji_tau = sum(self.svitau) / len(self.svitau)  # Srednji tau dobiven aritmetičkom sredinom tau-a svih putanja

        return self.srednji_tau

    def zapis(self, tocke):
        
        """

        Parameters
        ----------
        tocke : array
            Lista svih točaka jedne putanje.

        Returns
        -------
        .txt datoteka u koju je u svaki red zapisana putanja pojedine čestice od ulaza u jezgru do izlaska.

        """
        
        with open('koordinate_' + str(self.dim) + '.txt', 'a') as f:
            f.write(str(tocke))
            f.write('\n')
        
        return
    
    def citanje(dim, putanja = None):
        
        """

        Parameters
        ----------
        dim : integer
            Dimenzije matrice.
        putanja: string
            Putanja do .txt datoteke, po defaultu je formata koordinate_dimenzijematrice.txt, npr. 'koordinate_300.txt'.
            
        Returns
        -------
        ucitano : array
            Točke svih izračunatih putanja učitane iz .txt filea.

        """
        
        ucitano = []
        makni = '[]()\n,'
        
        if putanja == None:
            putanja = 'koordinate_' + str(dim) + '.txt'
        
        with open(putanja, 'r') as f:
            for i in f:
                ucitano.append(i)
            
            for i in range(len(ucitano)):
                for j in makni:
                    ucitano[i] = ucitano[i].replace(j, '')
                  
                ucitano[i] = ucitano[i].split()
                ucitano[i] = [int(i) for i in ucitano[i]]
                ucitano[i] = [tuple(ucitano[i][j:j+3]) for j in range(0, len(ucitano[i]), 3)]
            
        return ucitano
    
    def svi_tort(putanje):
        
        """

        Parameters
        ----------
        putanje : array
            Lista putanja iz .txt filea.

        Returns
        -------
        svi_tau : array
            Lista tortuoziteta za svaku putanju.

        """
        svi_tau = []

        for i in range(len(putanje)):
            udaljenost_suma = 0
            for j in range(len(putanje[0])-1):
                udaljenost = math.sqrt((putanje[i][j][1] - putanje[i][j+1][1]) ** 2 + (putanje[i][j][2] - putanje[i][j+1][2]) ** 2 + (0 - 1) ** 2)
                udaljenost_suma += udaljenost
            svi_tau.append(udaljenost_suma / 300)
            
        return svi_tau
    
    def avg_tort(putanje):
        
        """

        Parameters
        ----------
        putanje : array
            Lista putanja iz .txt filea.

        Returns
        -------
        tau_avg : float
            Prosječni tortuozitet uzorka.

        """
        svi_tau = Matrica.svi_tort(putanje)
        tau_avg = sum(svi_tau) / len(svi_tau)
        
        return tau_avg
        
class Vizualizacija():
    
    def povezane_tocke(dim, tocke):       
        """
        Parameters
        ----------
        dim : integer
            Dimenzije matrice.
        tocke : numpy array
            Lista parova z, y, x koordinata.

        Returns
        -------
        Graf povezanih točaka.

        """
        
        %matplotlib qt
        
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        
        ax = plt.axes(projection='3d')
        plt.xlim([0, dim])
        plt.ylim([0, dim])
        ax.set_zlim([0, dim])
        plt.locator_params(nbins=10)
        ax.view_init(None, 225)
        
        ax.set_xlabel('$Z$', fontsize=20)
        ax.set_ylabel('$X$', fontsize=20)
        ax.set_zlabel('$Y$', fontsize=20)
        
        for i in range(len(tocke)):
            x, y, z = [], [], []
            for j in range(len(tocke[i])):
                x.append(tocke[i][j][1])
                y.append(tocke[i][j][2])
                z.append(tocke[i][j][0])
            
            plt.plot(z, y, x, 'mediumblue', linestyle="-", marker = 'None')

        
        plt.show()
        
        return
    
    def paraview (dim, lcp):
        
        """

        Parameters
        ----------
        dim : integer
            Dimenzije  matrice.
        lcp : numpy array
            3D matrica nula i jedinica gdje jedinice predstavljaju porni prostor.

        Returns
        -------
        .vtr file za učitavanje u ParaView.

        """
        
        x = np.arange(0, dim+1)
        y = np.arange(0, dim+1)
        z = np.arange(0, dim+1)
        
        gridToVTK(str(dim), x, y, z, cellData = {str(dim): lcp})
        
        return
    
    def pore(dim, lcp):
        
        """

        Parameters
        ----------
        dim : integer
            Dimenzije  matrice.
        lcp : array
            3D matrica nula i jedinica gdje jedinice predstavljaju porni prostor.

        Returns
        -------
        3D graf pornog prostora.

        """
        # %matplotlib qt
        
        plt.rcParams["figure.figsize"] = [5, 5] # veličina prozora s grafom
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.add_subplot(111, projection='3d')
        
        coords = np.where(lcp == 1)
        x = coords[2]
        y = coords[1]
        z = coords[0]
        
        # ax.scatter3D(z, y, x, c = (x+y+z), cmap=plt.get_cmap('bone'), alpha = 1)
        ax.voxels(filled = lcp, facecolors = 'tab:blue', edgecolors = None)
        
        plt.xlim([0, dim])  # duljina x osi
        plt.ylim([0, dim])  # duljina y osi
        ax.set_zlim([0, dim])   # duljina z osi
        plt.locator_params(nbins=10)    # "gustoća" osi
        ax.view_init(None, 225)
        ax.set_xlabel('$Z$', fontsize=20)
        ax.set_ylabel('$Y$', fontsize=20)
        ax.set_zlabel('$X$', fontsize=20)
        plt.show()
        
        return
 
    def histogram(tau):
        
        """

        Parameters
        ----------
        tau : array
            Lista svih tortuoziteta za sve putanje.

        Returns
        -------
        Crta histogram tortuoziteta.

        """
        
        fig, ax = plt.subplots(1, 1)
        ax.hist(tau, bins=20, color = 'olivedrab', edgecolor = 'black')
        ax.set_ylabel('Broj putanja')
        ax.set_xlabel('Tortuozitet')
        
        return

    def krivulja(tau):
        
        """

        Parameters
        ----------
        tau : array
            Lista svih tortuoziteta za sve putanje.

        Returns
        -------
        Crta krivulju tortuoziteta.

        """
        
        tau_sort = np.sort(tau)[::-1]   
        x = np.linspace(1, 100, 11905)

        fig, ax = plt.subplots(1, 1)
        ax.step(x, tau_sort, linewidth = 2, color = 'olivedrab')
        ax.set_ylabel('Tortuozitet')
        ax.set_xlabel('Broj putanja, %')
        plt.xticks(np.linspace(0, 100, 11))
        
        return
    
""" Računanje putanja za sve točke iz prvog slicea i spremanje u .txt datoteku """
    
# uzorak = Matrica(300, 0, 0, 0.1, 0.1) 
# uzorak.set_ImgDir('C:\\Users\\Robert\\Desktop\\Fakultet\\Diplomski\\tort\\NPZ\\')
# # uzorak.set_ImgDir('D:\\img\\S1\\NPZ\\')
# uzorak.load_Image()
# labeledPores, poreNumber = uzorak.labelPores(uzorak.m, 2)
# uzorak.labeledConnectedPores = uzorak.connectedPores(labeledPores)
# # timer_pr1 = time.perf_counter()
# # print('Premještanje počelo')
# # for i in range(len(uzorak.labeledConnectedPores)):
# #     for j in range(len(uzorak.labeledConnectedPores[0])):
# #         for k in range(len(uzorak.labeledConnectedPores[0][0])):
# #             if uzorak.labeledConnectedPores[i][j][k] != 0:
# #                 uzorak.labeledConnectedPores[i][j][k] = 1
# # timer_pr2 = time.perf_counter()
# # print('Premještanje završilo: {} s'.format(timer_pr2-timer_pr1))
# uzorak.x, uzorak.y, uzorak.z = uzorak.koordinate(uzorak.labeledConnectedPores)
# uzorak.tau = uzorak.tau_iz_tezista(uzorak.dim, uzorak.labeledConnectedPores)
# uzorak.stoh_tau = uzorak.stoh()
# print(uzorak.stoh_tau)
# Vizualizacija.povezane_tocke(300, uzorak.svetocke)
# # Vizualizacija.pore(50, uzorak.labeledConnectedPores)

""" Loadanje i prikaz tocaka iz .txt filea, prvi je s malom devijacijom (0.1), a drugi sa standardnom (1.0) """

# devijacija = Matrica.citanje(300, 'koordinate_300 0,1.txt')
# Vizualizacija.povezane_tocke(300, devijacija)

# devijacija = Matrica.citanje(300, 'koordinate_300 1,0.txt')
# Vizualizacija.povezane_tocke(300, devijacija)

# standard_devijacija = Matrica.citanje(100, 'koordinate_100 - Copy.txt')
# Vizualizacija.povezane_tocke(100, standard_devijacija)


# tau = Matrica.avg_tort(devijacija)