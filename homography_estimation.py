import numpy as np
import cv2
import matplotlib.pyplot as plt

########################################## FUNCTION TO WARP IMAGE ##########################################

def image_warp(img1, img2, H, area):
    mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [area], (255,0,0))
    final = np.copy(img1)

    for i in range(img1.shape[1]):
        for j in range(img1.shape[0]):
            if(mask[j,i] == 255):
                x, y, w = np.dot(np.linalg.inv(H), np.array([i, j, 1]))
                x_int = int(np.floor(x / w))
                y_int = int(np.floor(y / w))
                if y_int > 0 and y_int < img2.shape[0] and x_int > 0 and x_int < img2.shape[1]:
                    final[j,i] = img2[y_int, x_int]

    return final

########################################## COORDINATES ##########################################

v1_p = [300, 758]
v1_q = [2548, 850]
v1_r = [2400, 2300]
v1_s = [658, 3204]

v2_p = [492, 1436]
v2_q = [1872, 808]
v2_r = [1956, 2832]
v2_s = [448, 2688]

v3_p = [1212, 480]
v3_q = [2980, 2296]
v3_r = [1788, 3212]
v3_s = [224, 1796]

roi_p = [0, 0]
roi_q = [783, 0]
roi_r = [783, 665]
roi_s = [0,665]

m1_p = [270, 330]
m1_q = [401, 322]
m1_r = [400, 436]
m1_s = [274, 430]

m2_p = [235, 363]
m2_q = [370, 361]
m2_r = [372, 460]
m2_s = [238, 461]

m3_p = [238, 353]
m3_q = [420, 346]
m3_r = [416, 480]
m3_s = [242, 505]

r_p = [0, 0]
r_q = [785, 0]
r_r = [785, 535]
r_s = [0,535]

########################################## TRANSFORMATION OF ALEX HONNOLD IMAGES ONTO FRAMES ##########################################

# Matrix of equation coefficients 
lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0, -1*roi_p[0]*v1_p[0], -1*roi_p[1]*v1_p[0]],
                [0, 0, 0, roi_p[0], roi_p[1], 1, -1*roi_p[0]*v1_p[1], -1*roi_p[1]*v1_p[1]],
                [roi_q[0], roi_q[1], 1, 0, 0, 0, -1*roi_q[0]*v1_q[0], -1*roi_q[1]*v1_q[0]],
                [0, 0, 0, roi_q[0], roi_q[1], 1, -1*roi_q[0]*v1_q[1], -1*roi_q[1]*v1_q[1]],
                [roi_r[0], roi_r[1], 1, 0, 0, 0, -1*roi_r[0]*v1_r[0], -1*roi_r[1]*v1_r[0]],
                [0, 0, 0, roi_r[0], roi_r[1], 1, -1*roi_r[0]*v1_r[1], -1*roi_r[1]*v1_r[1]],
                [roi_s[0], roi_s[1], 1, 0, 0, 0, -1*roi_s[0]*v1_s[0], -1*roi_s[1]*v1_s[0]],
                [0, 0, 0, roi_s[0], roi_s[1], 1, -1*roi_s[0]*v1_s[1], -1*roi_s[1]*v1_s[1]]])

# Array of final values
rhs = np.array([v1_p[0], v1_p[1], v1_q[0], v1_q[1], v1_r[0], v1_r[1], v1_s[0], v1_s[1]])

h = np.linalg.solve(lhs, rhs) # Using numpy liner solver to get the8 values for the H matrix
h = np.append(h,1) # appending 1 to the 8 calculated values 
H1 = h.reshape(3,3) # Forming the H 3x3 Matrix

lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0, -1*roi_p[0]*v2_p[0], -1*roi_p[1]*v2_p[0]],
                [0, 0, 0, roi_p[0], roi_p[1], 1, -1*roi_p[0]*v2_p[1], -1*roi_p[1]*v2_p[1]],
                [roi_q[0], roi_q[1], 1, 0, 0, 0, -1*roi_q[0]*v2_q[0], -1*roi_q[1]*v2_q[0]],
                [0, 0, 0, roi_q[0], roi_q[1], 1, -1*roi_q[0]*v2_q[1], -1*roi_q[1]*v2_q[1]],
                [roi_r[0], roi_r[1], 1, 0, 0, 0, -1*roi_r[0]*v2_r[0], -1*roi_r[1]*v2_r[0]],
                [0, 0, 0, roi_r[0], roi_r[1], 1, -1*roi_r[0]*v2_r[1], -1*roi_r[1]*v2_r[1]],
                [roi_s[0], roi_s[1], 1, 0, 0, 0, -1*roi_s[0]*v2_s[0], -1*roi_s[1]*v2_s[0]],
                [0, 0, 0, roi_s[0], roi_s[1], 1, -1*roi_s[0]*v2_s[1], -1*roi_s[1]*v2_s[1]]])

rhs2 = np.array([v2_p[0], v2_p[1], v2_q[0], v2_q[1], v2_r[0], v2_r[1], v2_s[0], v2_s[1]])

h = np.linalg.solve(lhs, rhs2)
h = np.append(h,1)
H2 = h.reshape(3,3)

lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0, -1*roi_p[0]*v3_p[0], -1*roi_p[1]*v3_p[0]],
                [0, 0, 0, roi_p[0], roi_p[1], 1, -1*roi_p[0]*v3_p[1], -1*roi_p[1]*v3_p[1]],
                [roi_q[0], roi_q[1], 1, 0, 0, 0, -1*roi_q[0]*v3_q[0], -1*roi_q[1]*v3_q[0]],
                [0, 0, 0, roi_q[0], roi_q[1], 1, -1*roi_q[0]*v3_q[1], -1*roi_q[1]*v3_q[1]],
                [roi_r[0], roi_r[1], 1, 0, 0, 0, -1*roi_r[0]*v3_r[0], -1*roi_r[1]*v3_r[0]],
                [0, 0, 0, roi_r[0], roi_r[1], 1, -1*roi_r[0]*v3_r[1], -1*roi_r[1]*v3_r[1]],
                [roi_s[0], roi_s[1], 1, 0, 0, 0, -1*roi_s[0]*v3_s[0], -1*roi_s[1]*v3_s[0]],
                [0, 0, 0, roi_s[0], roi_s[1], 1, -1*roi_s[0]*v3_s[1], -1*roi_s[1]*v3_s[1]]])

rhs3 = np.array([v3_p[0], v3_p[1], v3_q[0], v3_q[1], v3_r[0], v3_r[1], v3_s[0], v3_s[1]])

h = np.linalg.solve(lhs, rhs3)
h = np.append(h,1)
H3 = h.reshape(3,3)

########################################## TRANSFORMATION OF FRAME 1 ONTO FRAME 3 USING TWO HOMOGRAPHIES ##########################################

lhs = np.array([[v1_p[0], v1_p[1], 1, 0, 0, 0, -1*v1_p[0]*v2_p[0], -1*v1_p[1]*v2_p[0]],
                [0, 0, 0, v1_p[0], v1_p[1], 1, -1*v1_p[0]*v2_p[1], -1*v1_p[1]*v2_p[1]],
                [v1_q[0], v1_q[1], 1, 0, 0, 0, -1*v1_q[0]*v2_q[0], -1*v1_q[1]*v2_q[0]],
                [0, 0, 0, v1_q[0], v1_q[1], 1, -1*v1_q[0]*v2_q[1], -1*v1_q[1]*v2_q[1]],
                [v1_r[0], v1_r[1], 1, 0, 0, 0, -1*v1_r[0]*v2_r[0], -1*v1_r[1]*v2_r[0]],
                [0, 0, 0, v1_r[0], v1_r[1], 1, -1*v1_r[0]*v3_r[1], -1*v1_r[1]*v2_r[1]],
                [v1_s[0], v1_s[1], 1, 0, 0, 0, -1*v1_s[0]*v2_s[0], -1*v1_s[1]*v2_s[0]],
                [0, 0, 0, v1_s[0], v1_s[1], 1, -1*v1_s[0]*v2_s[1], -1*v1_s[1]*v2_s[1]]])

rhsb = np.array([v2_p[0], v2_p[1], v2_q[0], v2_q[1], v2_r[0], v2_r[1], v2_s[0], v2_s[1]])

h = np.linalg.solve(lhs, rhsb)
h = np.append(h,1)
H_ab = h.reshape(3,3)

lhs = np.array([[v2_p[0], v2_p[1], 1, 0, 0, 0, -1*v2_p[0]*v3_p[0], -1*v2_p[1]*v3_p[0]],
                [0, 0, 0, v2_p[0], v2_p[1], 1, -1*v2_p[0]*v3_p[1], -1*v2_p[1]*v3_p[1]],
                [v2_q[0], v2_q[1], 1, 0, 0, 0, -1*v2_q[0]*v3_q[0], -1*v2_q[1]*v3_q[0]],
                [0, 0, 0, v2_q[0], v2_q[1], 1, -1*v2_q[0]*v3_q[1], -1*v2_q[1]*v3_q[1]],
                [v2_r[0], v2_r[1], 1, 0, 0, 0, -1*v2_r[0]*v3_r[0], -1*v2_r[1]*v3_r[0]],
                [0, 0, 0, v2_r[0], v2_r[1], 1, -1*v2_r[0]*v3_r[1], -1*v2_r[1]*v3_r[1]],
                [v2_s[0], v2_s[1], 1, 0, 0, 0, -1*v2_s[0]*v3_s[0], -1*v2_s[1]*v3_s[0]],
                [0, 0, 0, v2_s[0], v2_s[1], 1, -1*v2_s[0]*v3_s[1], -1*v2_s[1]*v3_s[1]]])

rhsc = np.array([v3_p[0], v3_p[1], v3_q[0], v3_q[1], v3_r[0], v3_r[1], v3_s[0], v3_s[1]])

h = np.linalg.solve(lhs, rhsc)
h = np.append(h,1)
H_bc = h.reshape(3,3)

H_ac = np.matmul(H_bc, H_ab)

########################################## AFFINE TRANSFORMATIONS OF ALEX HONNOLD ##########################################

lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0],
                [0, 0, 0, roi_p[0], roi_p[1], 1],
                [roi_q[0], roi_q[1], 1, 0, 0, 0],
                [0, 0, 0, roi_q[0], roi_q[1], 1],
                [roi_r[0], roi_r[1], 1, 0, 0, 0],
                [0, 0, 0, roi_r[0], roi_r[1], 1],
                [roi_s[0], roi_s[1], 1, 0, 0, 0],
                [0, 0, 0, roi_s[0], roi_s[1], 1]])

rhs = np.array([v1_p[0], v1_p[1], v1_q[0], v1_q[1], v1_r[0], v1_r[1], v1_s[0], v1_s[1]])

h = np.linalg.lstsq(lhs, rhs)[0]
h = np.append(h,[0,0,1])
H1_affine = h.reshape(3,3)

lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0],
                [0, 0, 0, roi_p[0], roi_p[1], 1],
                [roi_q[0], roi_q[1], 1, 0, 0, 0],
                [0, 0, 0, roi_q[0], roi_q[1], 1],
                [roi_r[0], roi_r[1], 1, 0, 0, 0],
                [0, 0, 0, roi_r[0], roi_r[1], 1],
                [roi_s[0], roi_s[1], 1, 0, 0, 0],
                [0, 0, 0, roi_s[0], roi_s[1], 1]])

rhs2 = np.array([v2_p[0], v2_p[1], v2_q[0], v2_q[1], v2_r[0], v2_r[1], v2_s[0], v2_s[1]])

h = np.linalg.lstsq(lhs, rhs2)[0]
h = np.append(h,[0,0,1])
H2_affine = h.reshape(3,3)

lhs = np.array([[roi_p[0], roi_p[1], 1, 0, 0, 0],
                [0, 0, 0, roi_p[0], roi_p[1], 1],
                [roi_q[0], roi_q[1], 1, 0, 0, 0],
                [0, 0, 0, roi_q[0], roi_q[1], 1],
                [roi_r[0], roi_r[1], 1, 0, 0, 0],
                [0, 0, 0, roi_r[0], roi_r[1], 1],
                [roi_s[0], roi_s[1], 1, 0, 0, 0],
                [0, 0, 0, roi_s[0], roi_s[1], 1]])

rhs3 = np.array([v3_p[0], v3_p[1], v3_q[0], v3_q[1], v3_r[0], v3_r[1], v3_s[0], v3_s[1]])

h = np.linalg.lstsq(lhs, rhs3)[0]
h = np.append(h,[0,0,1])
H3_affine = h.reshape(3,3)


########################################## TRANSFORMATION OF CUSTOM IMAGES ONTO FRAMES ##########################################

# Matrix of equation coefficients 
lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0, -1*r_p[0]*m1_p[0], -1*r_p[1]*m1_p[0]],
                [0, 0, 0, r_p[0], r_p[1], 1, -1*r_p[0]*m1_p[1], -1*r_p[1]*m1_p[1]],
                [r_q[0], r_q[1], 1, 0, 0, 0, -1*r_q[0]*m1_q[0], -1*r_q[1]*m1_q[0]],
                [0, 0, 0, r_q[0], r_q[1], 1, -1*r_q[0]*m1_q[1], -1*r_q[1]*m1_q[1]],
                [r_r[0], r_r[1], 1, 0, 0, 0, -1*r_r[0]*m1_r[0], -1*r_r[1]*m1_r[0]],
                [0, 0, 0, r_r[0], r_r[1], 1, -1*r_r[0]*m1_r[1], -1*r_r[1]*m1_r[1]],
                [r_s[0], r_s[1], 1, 0, 0, 0, -1*r_s[0]*v1_s[0], -1*r_s[1]*m1_s[0]],
                [0, 0, 0, r_s[0], r_s[1], 1, -1*r_s[0]*m1_s[1], -1*r_s[1]*m1_s[1]]])

# Array of final values
rhs = np.array([m1_p[0], m1_p[1], m1_q[0], m1_q[1], m1_r[0], m1_r[1], m1_s[0], m1_s[1]])

h = np.linalg.solve(lhs, rhs) # Using numpy liner solver to get the8 values for the H matrix
h = np.append(h,1) # appending 1 to the 8 calculated values 
HM1 = h.reshape(3,3) # Forming the H 3x3 Matrix

lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0, -1*r_p[0]*m2_p[0], -1*r_p[1]*m2_p[0]],
                [0, 0, 0, r_p[0], r_p[1], 1, -1*r_p[0]*m2_p[1], -1*r_p[1]*m2_p[1]],
                [r_q[0], r_q[1], 1, 0, 0, 0, -1*r_q[0]*m2_q[0], -1*r_q[1]*m2_q[0]],
                [0, 0, 0, r_q[0], r_q[1], 1, -1*r_q[0]*m2_q[1], -1*r_q[1]*m2_q[1]],
                [r_r[0], r_r[1], 1, 0, 0, 0, -1*r_r[0]*m2_r[0], -1*r_r[1]*m2_r[0]],
                [0, 0, 0, r_r[0], r_r[1], 1, -1*r_r[0]*m2_r[1], -1*r_r[1]*m2_r[1]],
                [r_s[0], r_s[1], 1, 0, 0, 0, -1*r_s[0]*m2_s[0], -1*r_s[1]*m2_s[0]],
                [0, 0, 0, r_s[0], r_s[1], 1, -1*r_s[0]*m2_s[1], -1*r_s[1]*m2_s[1]]])

rhs2 = np.array([m2_p[0], m2_p[1], m2_q[0], m2_q[1], m2_r[0], m2_r[1], m2_s[0], m2_s[1]])

h = np.linalg.solve(lhs, rhs2)
h = np.append(h,1)
HM2 = h.reshape(3,3)

lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0, -1*r_p[0]*m3_p[0], -1*r_p[1]*m3_p[0]],
                [0, 0, 0, r_p[0], r_p[1], 1, -1*r_p[0]*m3_p[1], -1*r_p[1]*m3_p[1]],
                [r_q[0], r_q[1], 1, 0, 0, 0, -1*r_q[0]*m3_q[0], -1*r_q[1]*m3_q[0]],
                [0, 0, 0, r_q[0], r_q[1], 1, -1*r_q[0]*m3_q[1], -1*r_q[1]*m3_q[1]],
                [r_r[0], r_r[1], 1, 0, 0, 0, -1*r_r[0]*m3_r[0], -1*r_r[1]*m3_r[0]],
                [0, 0, 0, r_r[0], r_r[1], 1, -1*r_r[0]*m3_r[1], -1*r_r[1]*m3_r[1]],
                [r_s[0], r_s[1], 1, 0, 0, 0, -1*r_s[0]*m3_s[0], -1*r_s[1]*m3_s[0]],
                [0, 0, 0, r_s[0], r_s[1], 1, -1*r_s[0]*m3_s[1], -1*r_s[1]*m3_s[1]]])

rhs3 = np.array([m3_p[0], m3_p[1], m3_q[0], m3_q[1], m3_r[0], m3_r[1], m3_s[0], m3_s[1]])

h = np.linalg.solve(lhs, rhs3)
h = np.append(h,1)
HM3 = h.reshape(3,3)

########################################## AFFINE TRANSFORMATIONS OF CUSTOM IMAGES ##########################################

lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0],
                [0, 0, 0, r_p[0], r_p[1], 1],
                [r_q[0], r_q[1], 1, 0, 0, 0],
                [0, 0, 0, r_q[0], r_q[1], 1],
                [r_r[0], r_r[1], 1, 0, 0, 0],
                [0, 0, 0, r_r[0], r_r[1], 1],
                [r_s[0], r_s[1], 1, 0, 0, 0],
                [0, 0, 0, r_s[0], r_s[1], 1]])

rhs = np.array([m1_p[0], m1_p[1], m1_q[0], m1_q[1], m1_r[0], m1_r[1], m1_s[0], m1_s[1]])

h = np.linalg.lstsq(lhs, rhs)[0]
h = np.append(h,[0,0,1])
HM1_affine = h.reshape(3,3)

lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0],
                [0, 0, 0, r_p[0], r_p[1], 1],
                [r_q[0], r_q[1], 1, 0, 0, 0],
                [0, 0, 0, r_q[0], r_q[1], 1],
                [r_r[0], r_r[1], 1, 0, 0, 0],
                [0, 0, 0, r_r[0], r_r[1], 1],
                [r_s[0], r_s[1], 1, 0, 0, 0],
                [0, 0, 0, r_s[0], r_s[1], 1]])

rhs2 = np.array([m2_p[0], m2_p[1], m2_q[0], m2_q[1], m2_r[0], m2_r[1], m2_s[0], m2_s[1]])

h = np.linalg.lstsq(lhs, rhs2)[0]
h = np.append(h,[0,0,1])
HM2_affine = h.reshape(3,3)

lhs = np.array([[r_p[0], r_p[1], 1, 0, 0, 0],
                [0, 0, 0, r_p[0], r_p[1], 1],
                [r_q[0], r_q[1], 1, 0, 0, 0],
                [0, 0, 0, r_q[0], r_q[1], 1],
                [r_r[0], r_r[1], 1, 0, 0, 0],
                [0, 0, 0, r_r[0], r_r[1], 1],
                [r_s[0], r_s[1], 1, 0, 0, 0],
                [0, 0, 0, r_s[0], r_s[1], 1]])

rhs3 = np.array([m3_p[0], m3_p[1], m3_q[0], m3_q[1], m3_r[0], m3_r[1], m3_s[0], m3_s[1]])

h = np.linalg.lstsq(lhs, rhs3)[0]
h = np.append(h,[0,0,1])
HM3_affine = h.reshape(3,3)

########################################## GOING FROM PERSPCTIVE 1 TO 2 TO 3 FOR CUSTOM IMAGE ##########################################

# Matrix of equation coefficients 
lhs = np.array([[m1_p[0], m1_p[1], 1, 0, 0, 0, -1*m1_p[0]*m2_p[0], -1*m1_p[1]*m2_p[0]],
                [0, 0, 0, m1_p[0], m1_p[1], 1, -1*m1_p[0]*m2_p[1], -1*m1_p[1]*m2_p[1]],
                [m1_q[0], m1_q[1], 1, 0, 0, 0, -1*m1_q[0]*m2_q[0], -1*m1_q[1]*m2_q[0]],
                [0, 0, 0, m1_q[0], m1_q[1], 1, -1*m1_q[0]*m2_q[1], -1*m1_q[1]*m2_q[1]],
                [m1_r[0], m1_r[1], 1, 0, 0, 0, -1*m1_r[0]*m2_r[0], -1*m1_r[1]*m2_r[0]],
                [0, 0, 0, m1_r[0], m1_r[1], 1, -1*m1_r[0]*m2_r[1], -1*m1_r[1]*m2_r[1]],
                [m1_s[0], m1_s[1], 1, 0, 0, 0, -1*m1_s[0]*m2_s[0], -1*m1_s[1]*m2_s[0]],
                [0, 0, 0, m1_s[0], m1_s[1], 1, -1*m1_s[0]*m2_s[1], -1*m1_s[1]*m2_s[1]]])

rhs2 = np.array([m2_p[0], m2_p[1], m2_q[0], m2_q[1], m2_r[0], m2_r[1], m2_s[0], m2_s[1]])

h = np.linalg.solve(lhs, rhs2)
h = np.append(h,1)
H12 = h.reshape(3,3)

lhs = np.array([[m2_p[0], m2_p[1], 1, 0, 0, 0, -1*m2_p[0]*m3_p[0], -1*m2_p[1]*m3_p[0]],
                [0, 0, 0, m2_p[0], m2_p[1], 1, -1*m2_p[0]*m3_p[1], -1*m2_p[1]*m3_p[1]],
                [m2_q[0], m2_q[1], 1, 0, 0, 0, -1*m2_q[0]*m3_q[0], -1*m2_q[1]*m3_q[0]],
                [0, 0, 0, m2_q[0], m2_q[1], 1, -1*m2_q[0]*m3_q[1], -1*m2_q[1]*m3_q[1]],
                [m2_r[0], m2_r[1], 1, 0, 0, 0, -1*m2_r[0]*m3_r[0], -1*m2_r[1]*m3_r[0]],
                [0, 0, 0, m2_r[0], m2_r[1], 1, -1*m2_r[0]*m3_r[1], -1*m2_r[1]*m3_r[1]],
                [m2_s[0], m2_s[1], 1, 0, 0, 0, -1*m2_s[0]*m3_s[0], -1*m2_s[1]*m3_s[0]],
                [0, 0, 0, m2_s[0], m2_s[1], 1, -1*m2_s[0]*m3_s[1], -1*m2_s[1]*m3_s[1]]])

rhs3 = np.array([m3_p[0], m3_p[1], m3_q[0], m3_q[1], m3_r[0], m3_r[1], m3_s[0], m3_s[1]])

h = np.linalg.solve(lhs, rhs3)
h = np.append(h,1)
H23 = h.reshape(3,3)

H13 = np.matmul(H23, H12)

########################################## OUTPUTS ##########################################
img_a = cv2.imread('img1.jpg')
img_b = cv2.imread('img2.jpg')
img_c = cv2.imread('img3.jpg')
img_d = cv2.imread('alex_honnold.jpg')

img_1 = cv2.imread('my1.jpg')
img_2 = cv2.imread('my2.jpg')
img_3 = cv2.imread('my3.jpg')
img_4 = cv2.imread('Rickroll.jpg')


final_a = image_warp(img_a, img_d, H1, np.array([[v1_p[0], v1_p[1]], [v1_q[0], v1_q[1]], [v1_r[0], v1_r[1]],[v1_s[0], v1_s[1]]]))
final_b = image_warp(img_b, img_d, H2, np.array([[v2_p[0], v2_p[1]], [v2_q[0], v2_q[1]], [v2_r[0], v2_r[1]],[v2_s[0], v2_s[1]]]))
final_c = image_warp(img_c, img_d, H3, np.array([[v3_p[0], v3_p[1]], [v3_q[0], v3_q[1]], [v3_r[0], v3_r[1]],[v3_s[0], v3_s[1]]]))

combo_c = image_warp(img_c, img_a, H_ac, np.array([[v3_p[0], v3_p[1]], [v3_q[0], v3_q[1]], [v3_r[0], v3_r[1]],[v3_s[0], v3_s[1]]]))

affine_a = image_warp(img_a, img_d, H1_affine, np.array([[v1_p[0], v1_p[1]], [v1_q[0], v1_q[1]], [v1_r[0], v1_r[1]],[v1_s[0], v1_s[1]]]))
affine_b = image_warp(img_b, img_d, H2_affine, np.array([[v2_p[0], v2_p[1]], [v2_q[0], v2_q[1]], [v2_r[0], v2_r[1]],[v2_s[0], v2_s[1]]]))
affine_c = image_warp(img_c, img_d, H3_affine, np.array([[v3_p[0], v3_p[1]], [v3_q[0], v3_q[1]], [v3_r[0], v3_r[1]],[v3_s[0], v3_s[1]]]))

custom_a = image_warp(img_1, img_4, HM1, np.array([[m1_p[0], m1_p[1]], [m1_q[0], m1_q[1]], [m1_r[0], m1_r[1]],[m1_s[0], m1_s[1]]]))
custom_b = image_warp(img_2, img_4, HM2, np.array([[m2_p[0], m2_p[1]], [m2_q[0], m2_q[1]], [m2_r[0], m2_r[1]],[m2_s[0], m2_s[1]]]))
custom_c = image_warp(img_3, img_4, HM3, np.array([[m3_p[0], m3_p[1]], [m3_q[0], m3_q[1]], [m3_r[0], m3_r[1]],[m3_s[0], m3_s[1]]]))

custom_affine_a = image_warp(img_1, img_4, HM1_affine, np.array([[m1_p[0], m1_p[1]], [m1_q[0], m1_q[1]], [m1_r[0], m1_r[1]],[m1_s[0], m1_s[1]]]))
custom_affine_b = image_warp(img_2, img_4, HM2_affine, np.array([[m2_p[0], m2_p[1]], [m2_q[0], m2_q[1]], [m2_r[0], m2_r[1]],[m2_s[0], m2_s[1]]]))
custom_affine_c = image_warp(img_3, img_4, HM3_affine, np.array([[m3_p[0], m3_p[1]], [m3_q[0], m3_q[1]], [m3_r[0], m3_r[1]],[m3_s[0], m3_s[1]]]))

custom_combo_c = image_warp(img_3, img_1, H13, np.array([[m3_p[0], m3_p[1]], [m3_q[0], m3_q[1]], [m3_r[0], m3_r[1]],[m3_s[0], m3_s[1]]]))

img_list = [final_a, final_b, final_c, combo_c, affine_a, affine_b, affine_c]
titles = ['Image on Frame A', 'Image on Frame B', 'Image on Frame C', 'Image from A on C', 'Affine a', 'Affine b', 'Affine c']

plt.figure(figsize=(15,5))

plt.figure(1)
for i, img in enumerate(img_list):
    plt.subplot(2, len(img_list), i+1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titles[i])

img_list = [custom_a, custom_b, custom_c, custom_affine_a, custom_affine_b, custom_affine_c, custom_combo_c]
titles = ['My Image 1', 'My Image 2', 'My Image 3', 'My Image 1 Affine', 'My Image 2 Affine', 'My Image 3 Affine', 'My Image combo']

plt.figure(2)
for i, img in enumerate(img_list):
    plt.subplot(2, len(img_list), i+1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(titles[i])


plt.show()