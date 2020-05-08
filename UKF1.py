import numpy as np
import os
import sys
import json 

import cv2


# from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
# from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader



class ArgoVerseDataset(object):
    
    def __init__(self):

        self.data_path = 'Data/tracking_sample_v1.1/argoverse-tracking/sample/c6911883-1843-3727-8eaa-41dc8cda8993'

        self.camera1_path = 'ring_front_center'
        self.camera2_path = 'ring_front_right'

        self.camera_parameters = 'vehicle_calibration_info.json'
    
    def readImg(self):

        pass

    def readCameraParameters(self):

        #open the file
        path = os.path.join(self.data_path, self.camera_parameters)
        with open(path) as f:
            camera_parameters = json.load(f)
        
        for cameraData in camera_parameters['camera_data_']:

            if self.camera1_path in cameraData['key']:
                camera1_data = cameraData['value']
            
            if self.camera2_path in cameraData['key']:
                camera2_data = cameraData['value']
        
        camera1K = 0

def Homography(src_img, des_img):

    orb = cv2.ORB_create()
    kpt1, des1 = orb.detectAndCompute(src_img, None)
    kpt2, des2 = orb.detectAndCompute(des_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M 


def scalef(sx, sy):
    return np.diag([sx, sy, 1])

def transf(tx, ty):
    A = np.eye(3)
    A[0, 2] = tx
    A[1, 2] = ty
    return A
    
def getPointTransformation(H, x, y):
    c = np.matmul(H, [x, y, 1])
    return [c[0]/c[2], c[1]/c[2]]

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    c_lt = getPointTransformation(H2to1, 0, 0)
    c_lb = getPointTransformation(H2to1, 0, im2.shape[0])
    c_rt = getPointTransformation(H2to1, im2.shape[1], 0)
    c_rb = getPointTransformation(H2to1, im2.shape[1], im2.shape[0])

    height1 = max(im1.shape[0], c_lb[1]) - min(0, c_lt[1])
    height2 = max(im1.shape[0], c_rb[1]) - min(0, c_rt[1])
    width1 = max(im1.shape[1], c_rt[0]) - min(0, c_lt[0])
    width2 = max(im1.shape[1], c_rb[0]) - min(0, c_lb[0])

    scale = max(height1, height2)/max(width1, width2)
    width = im1.shape[1]
    height = int(np.ceil(width*scale))

    x_scale = width/max(width1, width2)
    y_scale = height/max(height1, height2)
    tx = -min(0, c_lt[0], c_lb[0])*x_scale
    ty = -min(0, c_lt[1], c_rt[1])*y_scale

    M = (transf(tx, ty)
     .dot(scalef(x_scale, y_scale)))

    im2_t = np.zeros((height, width, im1.shape[2]))
    im2_t[:,:,0] = cv2.warpPerspective(im2[:,:,0], np.matmul(M, H2to1), (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,1] = cv2.warpPerspective(im2[:,:,1], np.matmul(M, H2to1), (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,2] = cv2.warpPerspective(im2[:,:,2], np.matmul(M, H2to1), (im2_t.shape[1], im2_t.shape[0]))
    #cv2.imwrite('tmp2.jpg', im2_t)

    im1_t = np.zeros(im2_t.shape)
    im1_t[:,:,0] = cv2.warpPerspective(im1[:,:,0], M, (im1_t.shape[1], im1_t.shape[0]))
    im1_t[:,:,1] = cv2.warpPerspective(im1[:,:,1], M, (im1_t.shape[1], im1_t.shape[0]))
    im1_t[:,:,2] = cv2.warpPerspective(im1[:,:,2], M, (im1_t.shape[1], im1_t.shape[0]))
    #cv2.imwrite('tmp1.jpg', im1_t)
    
    pano_im = np.maximum(im1_t, im2_t)
    # cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    return pano_im


def main(data_path):

    # avm = ArgoverseMap()
    # argoverse_tracker_loader = ArgoverseTrackingLoader('argoverse-tracking/')    #simply change to your local path of the data
    # argoverse_forecasting_loader = ArgoverseForecastingLoader('argoverse-forecasting/') 

    # argoVerseDataset = ArgoVerseDataset()
    # argoVerseDataset.readCameraParameters()

    argoverse_tracker_loader = ArgoverseTrackingLoader(data_path)
    
    camera1_name = 'ring_front_center'
    camera2_name = 'ring_front_right'
    camera3_name = 'ring_front_left'

    camera1_images = argoverse_tracker_loader.image_list[camera1_name]
    camera2_images = argoverse_tracker_loader.image_list[camera2_name]
    camera3_images = argoverse_tracker_loader.image_list[camera3_name]

    camera1_calib = argoverse_tracker_loader.calib[camera1_name]
    camera2_calib = argoverse_tracker_loader.calib[camera2_name]
    camera3_calib = argoverse_tracker_loader.calib[camera3_name]


    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 600,400)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 600,400)
    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img3', 600,400)
    cv2.namedWindow('Stiched Img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stiched Img', 1200,800)

    # i = 6 #9 #10
    i = 0

    im1 = cv2.imread(camera1_images[i])

    im2 = cv2.imread(camera2_images[i])

    im3 = cv2.imread(camera3_images[i])
            
    status = stitcher.estimateTransform([im1, im2, im3]) 

    print(f'Status of Transform cal : {status}')
    
    H2to1 = Homography(im2, im1)
    H1to2 = Homography(im1, im2)


    #Merge Imgs into single
    stitcher = cv2.Stitcher_create()


    for i in range(len(camera1_images)):

        im1 = cv2.imread(camera1_images[i])

        im2 = cv2.imread(camera2_images[i])

        im3 = cv2.imread(camera3_images[i])

        
        # H2to1 = Homography(im2, im1)
        # H1to2 = Homography(im1, im2)

        # im2Wrapped = cv2.warpPerspective(im2, H2to1, (im2.shape[1], im2.shape[0]))


        # create indices of the destination image and linearize them
        h, w = im2.shape[:2]
        h = 2*h
        w = 2*w
        indy, indx = np.indices((h, w), dtype=np.float32)
        lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

        # warp the coordinates of src to those of true_dst
        # map_ind = H2to1.dot(lin_homg_ind)
        map_ind = H1to2.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
        map_x = map_x.reshape(h, w).astype(np.float32)
        map_y = map_y.reshape(h, w).astype(np.float32)

        # remap!
        dst = cv2.remap(im2, map_x, map_y, cv2.INTER_LINEAR)
        # blended = cv2.addWeighted(im1, 0.5, dst, 0.5, 0)






        myPano = imageStitching_noClip(im1, im2, H2to1=H2to1)
        # myPano = imageStitching_noClip(im2, im1, H2to1=H1to2)






        # status, stichedImg = stitcher.stitch([im1, im2, im3])


        status, stichedImg = stitcher.composePanorama([im1, im3])
        
        # if i == 0:
        #     status, stichedImg = stitcher.stitch([im1, im2, im3])   
        # else:
        #     status, stichedImg = stitcher.stitchGRG([im1, im2, im3])
        
        # cv2.imshow('img2', im2)
        # cv2.waitKey()


        if status == 0:
            cv2.imshow('Stiched Img', stichedImg)
        else:
            print(f'Skiiping img{i}!!!')

        cv2.imshow('img1', im1)
        cv2.imshow('img2', im2)
        cv2.imshow('img3', myPano/255)
        # cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cv2.imshow('imgCenter&Right', np.hstack((im1, im2)))
        # # cv2.waitKey()
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        

       

if __name__ == "__main__":
    
    data_path = 'Data/tracking_sample_v1.1/argoverse-tracking/'
    
    print('started...')
    main(data_path)
    print('finished!')
    