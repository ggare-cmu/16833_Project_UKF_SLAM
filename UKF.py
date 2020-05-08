import numpy as np
import os
import sys

import cv2


from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

#   Timestamp   cam0_path_to_img    cam1_path_to_img    cam2_path_to_img
#   2221156 imgs/cam0/img1.png imgs/cam1/img1.png imgs/cam2/img1.png 
def CreateDatasetFileForMultiColSLAM(camera1_images, camera2_images, camera3_images):

    # timestamp = np.arange(len(camera1_images))
    timestamp = np.array([str(i).zfill(7) for i in range(len(camera1_images))])

    dataset = np.concatenate((timestamp[:, np.newaxis], np.array(camera1_images)[:, np.newaxis], np.array(camera2_images)[:, np.newaxis], np.array(camera3_images)[:, np.newaxis]), axis=1)

    # np.savetxt(f'results/{test_name}_test_classification_results.csv', test_results, fmt=['%s','%s'], delimiter=',', header='Id,Predicted', comments='')
    np.savetxt(f'images_and_timestamps.txt', dataset, fmt=['%s','%s','%s','%s'], comments='')

def main(data_path):

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


    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img3', 600,400)
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 600,400)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 600,400)
    cv2.namedWindow('Stiched Img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stiched Img', 1200,800)

    #Generate Dataloader for MultiCol-SLAM
    CreateDatasetFileForMultiColSLAM(camera1_images, camera2_images, camera3_images)

    #Create times.txt file for ORB-SLAM2
    np.savetxt(f'Data/argoverseWithTurnsUKF/times.txt', np.arange(len(camera1_images)), fmt=['%s'], comments='')
    np.savetxt(f'Data/argoverseWithTurnsCenter/times.txt', np.arange(len(camera1_images)), fmt=['%s'], comments='')

    #Merge Imgs into single
    stitcher = cv2.Stitcher_create()

    status = -1

    str_idx = 0
    # i = 6 #9 #10
    for i in range(str_idx, len(camera1_images)):

        im1 = cv2.imread(camera1_images[i])

        im2 = cv2.imread(camera2_images[i])

        im3 = cv2.imread(camera3_images[i])
                
        status = stitcher.estimateTransform([im1, im2, im3]) 

        if status == 0 : # and i > str_idx : #and i != 2 and i != 3:
            print(f'Calculated Transform for img{i}')
            break
  

    for i in range(len(camera1_images)):

        im1 = cv2.imread(camera1_images[i])

        im2 = cv2.imread(camera2_images[i])

        im3 = cv2.imread(camera3_images[i])

        cv2.imwrite( f'Data/argoverseWithTurnsCenter/image_0/{str(i).zfill(6)}.png', im1)

        # stichedImg.shape = (1108, 3727, 3)        
        status, stichedImg = stitcher.composePanorama([im1, im2, im3])

        # stichedImg = stichedImg[70:-70, 60:-60, :]
        if status == 0:
            cv2.imshow('Stiched Img', stichedImg)
            # cv2.imwrite( f'Data/arggoverseUKF/image_0/{str(i).zfill(6)}.png', stichedImg)
            cv2.imwrite( f'Data/argoverseWithTurnsUKF/image_0/{str(i).zfill(6)}.png', stichedImg)
        else:
            print(f'Skiiping img{i}!!!')

        cv2.imshow('img1', im1)
        cv2.imshow('img2', im2)
        cv2.imshow('img3', im3)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        
        

       

if __name__ == "__main__":
    
    # data_path = 'Data/tracking_sample_v1.1/argoverse-tracking/'
    # data_path = 'Data/has_turn_273c1883-673a-36bf-b124-88311b1a80be/'
    data_path = 'Data/argoverseWithTurns1/'

    print('started...')
    main(data_path)
    print('finished!')
    