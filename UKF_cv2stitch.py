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

    for i in range(len(camera1_images)):

        im1 = cv2.imread(camera1_images[i])

        im2 = cv2.imread(camera2_images[i])

        im3 = cv2.imread(camera3_images[i])


        #Merge Imgs into single
        sticker = cv2.Stitcher_create()
        status, stichedImg = sticker.stitch([im1, im2, im3])

        # cv2.imshow('img2', im2)
        # cv2.waitKey()


        if status == 0:
            cv2.imshow('Stiched Img', stichedImg)


        cv2.imshow('img1', im1)
        cv2.imshow('img2', im2)
        cv2.imshow('img3', im3)
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
    