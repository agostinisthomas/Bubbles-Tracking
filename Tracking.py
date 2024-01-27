import numpy as np
import cv2
import time
import csv


# Echelle Video - Réalité

scale=1.85e-4

grain=400# Taille du grain à analyser

cap = cv2.VideoCapture('/Users/Thomas/Documents/CODE/BUBBLES/Tracking/{}.mov'.format(grain))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1280,720))
print(cap)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 5000,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()  # Associe la première frame avec les variables ret,old_frame
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # Met l'ancienne frame en nuances de gris
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) # Trouve les coins dans cette image grise et met leurs coordonnées dans un vecteur p0

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)  # Juste pour dessiner les trajectoires

iteration=0
n=0
m=0
max=0

open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/data_{}.csv".format(grain),"w+", newline='').close()
with open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/data_{}.csv".format(grain),"w", newline='') as csvfile :
        writer=csv.writer(csvfile, dialect='excel' ,delimiter=';')
        writer.writerow(["iteration","Nb bulles","Coordonnées_x_bulle1","Coordonnées_y_bulle1","vitesse_bulle1","Coordonnées_x_bulle30","Coordonnées_y_bulle30","vitesse_bulle30","Coordonnées_x_bulle100","Coordonnées_y_bulle100","vitesse_bulle100"])
csvfile.close()
open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/coordonnees_y_{}.txt".format(grain),"w").close()
open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/coordonnees_x_{}.txt".format(grain),"w").close()
open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/nb_bulles_{}.txt".format(grain),"w").close()
open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/p1_{}.txt".format(grain),"w").close() 

# Initialisation des vitesses de bulles

vitesse_bulle1 = 0
vitesse_bulle30 = 0
vitesse_bulle100 = 0

y1=p0[0][0][1]
y30=p0[30][0][1]
y100=p0[100][0][1]
# ____________________________________ BOUCLE _______________________________________



while(iteration<500):
    
    iteration+=1
    ret,frame = cap.read() # Prend la frame suivante
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # La met en nuances de gris

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # Lucas-Kanade : calcul de V_x et V_y pour avoir le flot optique
    #print("Vecteur err : ",err)

    # Select good points
    good_new = p1[st==1]  # p1 contient les nouveaux points
    good_old = p0[st==1]  # p0 contient toujours les points de la frame précédente
    
    n=m
    m=len(p1)
    if m>n :
        max=m
    

    #draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)
    

    #cv2.imshow('frame',img)
    k = cv2.waitKey(500) & 0xff
    if k == 100 :
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    print("-------------------- Iteration n°",iteration," ----------------------")
    
    

    if iteration > 1 :
        vitesse_bulle1 = (abs((p1[0][0][1] - y1) / (6.94e-3)))*scale
        vitesse_bulle30 = (abs((p1[30][0][1] - y30) / (6.94e-3)))*scale
        vitesse_bulle100 = (abs((p1[100][0][1] - y100) / (6.94e-3)))*scale

    y1=p1[0][0][1]
    y30=p1[30][0][1]
    y100=p1[100][0][1]
    print ("y1 = ",y1)
    print("Vitesse bulle 1 = ",vitesse_bulle1)
   


    with open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/data_{}.csv".format(grain),"a+", newline='') as csvfile :
        writer=csv.writer(csvfile, dialect='excel' ,delimiter=';')
        writer.writerow([str(iteration),str(len(p1)),str(p1[0][0][0]).replace('.',','),str(p1[0][0][1]).replace('.',','),str(vitesse_bulle1).replace('.',','),str(p1[30][0][0]).replace('.',','),str(p1[30][0][1]).replace('.',','),str(vitesse_bulle30).replace('.',','),str(p1[100][0][0]).replace('.',','),str(p1[100][0][1]).replace('.',','),str(vitesse_bulle100).replace('.',',')])
 
    with open("/Users/Thomas/Documents/CODE/BUBBLES/Tracking/coordonnees_x_{}.txt".format(grain),"a+") as file_coordonnees_x :
        
        file_coordonnees_x.write("\n")
        file_coordonnees_x.write(str(p1[0][0][0]).replace('.',','))

    with open('/Users/Thomas/Documents/CODE/BUBBLES/Tracking/coordonnees_y_{}.txt'.format(grain),"a+") as file_coordonnees_y :
        
        file_coordonnees_y.write("\n")
        file_coordonnees_y.write(str(p1[0][0][1]).replace('.',','))


    with open('/Users/Thomas/Documents/CODE/BUBBLES/Tracking/nb_bulles_{}.txt'.format(grain),"a+") as file_nb_bulles :
        
        file_nb_bulles.write("\n")
        file_nb_bulles.write(str(len(p1)).replace('.',','))
            
    #time.sleep(1)
    csvfile.close()
    file_nb_bulles.close()
    file_coordonnees_x.close()
    file_coordonnees_y.close()
    

#print(good_new)
cv2.destroyAllWindows()
out.release()
cap.release()