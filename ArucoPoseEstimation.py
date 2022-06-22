import numpy as np
import cv2
import cv2.aruco as aruco
import os
import math,sys,time

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



#--- Define Tag
id_to_find = 100
marker_size = 25
#--- Get the camera calibration parameters
calib_path = ""
camera_matrix = np.loadtxt(calib_path+'cameraMatrix.txt',delimiter=',')
camera_distortion = np.loadtxt(calib_path+'cameraDistortion.txt',delimiter=',')

#--- 180 deg rotation matrix around the X axis
R_flip = np.zeros((3,3), dtype = np.float32)
R_flip[0][0] = 1.0
R_flip[1][1] = 1.0
R_flip[2][2] = 1.0


#define the aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

markerImg = cv2.imread("aruco1.png")
markerImg = cv2.resize(markerImg,(1640,1232))
markerGray = cv2.cvtColor(markerImg,cv2.COLOR_BGR2GRAY)

corners,ids,rejected = aruco.detectMarkers(markerGray,aruco_dict,parameters = parameters)

if ids != None and ids[0] == id_to_find:
    #-- ret = [rvec,tvec,?]
    #-- array of rotation and position of each marker in camera frame
    #-- rvec = [[rvec_1],[rvec_2],....] attitude of marker respect to camera frame
    #-- tvec = [[tvec_1],[tvec_2],....] position of marker respect to camera frame

    ret = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,camera_distortion)

    #-- Unpack the output, get only the first
    rvec,tvec = ret[0][0,0,:],ret[1][0,0,:]
    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
    R_tc = R_ct.T
    roll_marker, pitch_marker,yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
    print(math.degrees(roll_marker),math.degrees(pitch_marker),math.degrees(yaw_marker))
    #-- Draw the detected marker and put a reference frame over it
    aruco.drawDetectedMarkers(markerImg,corners)
    #aruco.drawAxis(markerImg,camera_matrix,camera_distortion,rvec,tvec,25)
    cv2.imshow("Result",markerImg)
    cv2.waitKey(0)





















