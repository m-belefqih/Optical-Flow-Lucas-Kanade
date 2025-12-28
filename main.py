import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. Chargement de la vidéo
# =====================================================
# La vidéo contient un objet rigide unique (avion)
# filmé par une caméra fixe
cap = cv2.VideoCapture("airplane.mp4")

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo")
    exit()


# =====================================================
# 2. Détection des points caractéristiques (Shi-Tomasi)
# =====================================================
# Ces points seront suivis d'une image à l'autre

# ----------------------------
# 2. Paramètres Shi-Tomasi
# ----------------------------
#feature_params = dict(
#    maxCorners=100,
#    qualityLevel=0.3,
#    minDistance=7,
#    blockSize=7
#)

feature_params = dict(
    maxCorners=200,         # Nombre maximal de points détectés
    qualityLevel=0.01,      # Qualité minimale des coins
    minDistance=5,          # Distance minimale entre deux points
    blockSize=7             # Taille du voisinage
)


# =====================================================
# 3. Paramètres de l'algorithme Lucas–Kanade
# =====================================================
# Version pyramidale pour gérer les déplacements importants
lk_params = dict(
    winSize=(15, 15),     # Taille de la fenêtre locale
    maxLevel=2,           # Nombre de niveaux de la pyramide
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# =====================================================
# 4. Lecture de la première image
# =====================================================
# Conversion en niveaux de gris (nécessaire pour le flot optique)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Détection des points à suivre dans la première image
p0 = cv2.goodFeaturesToTrack(
    old_gray, mask=None, **feature_params
)

# Image masque utilisée pour dessiner les vecteurs de mouvement
mask = np.zeros_like(old_frame)

# Listes pour stocker la trajectoire globale de l'objet
trajectory_x = []
trajectory_y = []

# =====================================================
# 5. Boucle principale de traitement vidéo
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion de l'image courante en niveaux de gris
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul du flot optique entre deux images consécutives
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )

    if p1 is not None:
        # Sélection des points correctement suivis
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # =================================================
        # Estimation de la position globale de l'objet
        # =================================================
        # On approxime l'objet par le barycentre
        center_x = np.mean(good_new[:, 0])
        center_y = np.mean(good_new[:, 1])

        trajectory_x.append(center_x)
        trajectory_y.append(center_y)

        # =================================================
        # Visualisation du champ de mouvement
        # =================================================
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()

            # Tracé du vecteur de déplacement
            mask = cv2.line(
                mask,
                (int(a), int(b)),
                (int(c), int(d)),
                (0, 255, 0),
                2
            )

            # Dessin du point courant
            frame = cv2.circle(
                frame,
                (int(a), int(b)),
                3,
                (0, 0, 255),
                -1
            )

    # Superposition des vecteurs sur l'image
    img = cv2.add(frame, mask)
    cv2.imshow("Optical Flow - Object Motion", img)

    # Mise à jour pour l'itération suivante
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Sortie avec la touche ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()

# =====================================================
# 6. Affichage de la trajectoire globale de l'objet
# =====================================================
plt.figure()
plt.plot(trajectory_x, trajectory_y, '-o')
plt.title("Trajectoire globale de l'objet")
plt.xlabel("Position X (pixels)")
plt.ylabel("Position Y (pixels)")
plt.gca().invert_yaxis()  # Convention image (origine en haut à gauche)
plt.grid()
plt.show()

# =====================================================
# 7. Analyse de la vitesse et de la direction
# =====================================================
trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)

# Déplacements entre images successives
dx = np.diff(trajectory_x)
dy = np.diff(trajectory_y)

# Calcul de la vitesse instantanée (pixels/frame)
speed = np.sqrt(dx**2 + dy**2)
mean_speed = np.mean(speed)

print("Vitesse moyenne (pixels/frame) :", mean_speed)

# Calcul de la direction du mouvement
direction = np.arctan2(dy, dx)
mean_direction = np.mean(direction)

print("Direction moyenne (rad) :", mean_direction)
