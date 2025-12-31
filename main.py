import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. Video loading
# =====================================================
# The video contains a single rigid object (airplane)
# captured using a fixed camera
cap = cv2.VideoCapture("airplane.mp4")

if not cap.isOpened():
    print("Error: unable to open the video")
    exit()


# =====================================================
# 2. Feature point detection (Shi-Tomasi)
# =====================================================
# These feature points will be tracked from frame to frame
feature_params = dict(
    maxCorners=200,         # Maximum number of detected points
    qualityLevel=0.01,      # Minimum quality level for corners
    minDistance=5,          # Minimum distance between detected points
    blockSize=7             # Size of the neighborhood considered
)


# =====================================================
# 3. Lucas–Kanade optical flow parameters
# =====================================================
# Pyramidal version to handle larger motions
lk_params = dict(
    winSize=(15, 15),       # Size of the local search window
    maxLevel=2,             # Number of pyramid levels
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# =====================================================
# 4. Read the first frame
# =====================================================
# Conversion to grayscale (required for optical flow)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial feature points to track
p0 = cv2.goodFeaturesToTrack(
    old_gray, mask=None, **feature_params
)

# Mask image used to draw motion vectors
mask = np.zeros_like(old_frame)

# Lists to store the global trajectory of the object
trajectory_x = []
trajectory_y = []

# =====================================================
# 5. Main video processing loop
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow between two consecutive frames
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params
    )

    if p1 is not None:
        # Select successfully tracked points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # =================================================
        # Estimation of the global object position
        # =================================================
        # The object position is approximated by the centroid
        center_x = np.mean(good_new[:, 0])
        center_y = np.mean(good_new[:, 1])

        trajectory_x.append(center_x)
        trajectory_y.append(center_y)

        # =================================================
        # Motion field visualization
        # =================================================
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()

            # Draw the displacement vector
            mask = cv2.line(
                mask,
                (int(a), int(b)),
                (int(c), int(d)),
                (0, 255, 0),
                2
            )

            # Draw the current feature point
            frame = cv2.circle(
                frame,
                (int(a), int(b)),
                3,
                (0, 0, 255),
                -1
            )

    # Overlay motion vectors on the current frame
    img = cv2.add(frame, mask)
    cv2.imshow("Optical Flow - Object Motion", img)

    # Update variables for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Exit when ESC key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# =====================================================
# 6. Display the global object trajectory
# =====================================================
plt.figure()
plt.plot(trajectory_x, trajectory_y, '-o')
plt.title("Global Object Trajectory")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.gca().invert_yaxis()  # Image coordinate convention (origin at top-left)
plt.grid()
plt.show()

# =====================================================
# 7. Speed and direction analysis
# =====================================================
trajectory_x = np.array(trajectory_x)
trajectory_y = np.array(trajectory_y)

# Displacements between successive frames
dx = np.diff(trajectory_x)
dy = np.diff(trajectory_y)

# Instantaneous speed (pixels per frame)
speed = np.sqrt(dx**2 + dy**2)
mean_speed = np.mean(speed)

print("Average speed (pixels/frame):", mean_speed)

# Motion direction computation
direction = np.arctan2(dy, dx)
mean_direction = np.mean(direction)

print("Average direction (rad):", mean_direction)
