import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def grgimageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    
    c_lt = getPointTransformation(H2to1, 0, 0)
    c_lb = getPointTransformation(H2to1, 0, im2.shape[0])
    c_rt = getPointTransformation(H2to1, im2.shape[1], 0)
    c_rb = getPointTransformation(H2to1, im2.shape[1], im2.shape[0])

    width1 = max(im1.shape[1], c_rt[0]) - min(0, c_lt[0])
    width2 = max(im1.shape[1], c_rb[0]) - min(0, c_lb[0])

    width = int(np.ceil(max(width1, width2)))

    im2_t = np.zeros((im1.shape[0], width, im1.shape[2]))
    im2_t[:,:,0] = cv2.warpPerspective(im2[:,:,0], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,1] = cv2.warpPerspective(im2[:,:,1], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,2] = cv2.warpPerspective(im2[:,:,2], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    #cv2.imwrite('im2_t.jpg', im2_t)
    cv2.imwrite('../results/6_1.jpg', im2_t)

    im1 = np.pad(im1, ((0,0), (0,im2_t.shape[1]-im1.shape[1]), (0,0)), mode='constant')
    pano_im = np.maximum(im1, im2_t)
    #cv2.imwrite('pano_im.jpg', pano_im)
    cv2.imwrite('../results/6_1_pan.jpg', pano_im)

    return pano_im

def imageStitching_mask(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    
    c_lt = getPointTransformation(H2to1, 0, 0)
    c_lb = getPointTransformation(H2to1, 0, im2.shape[0])
    c_rt = getPointTransformation(H2to1, im2.shape[1], 0)
    c_rb = getPointTransformation(H2to1, im2.shape[1], im2.shape[0])

    width1 = max(im1.shape[1], c_rt[0]) - min(0, c_lt[0])
    width2 = max(im1.shape[1], c_rb[0]) - min(0, c_lb[0])

    width = int(np.ceil(max(width1, width2)))

    im2_t = np.zeros((im1.shape[0], width, im1.shape[2]))
    im2_t[:,:,0] = cv2.warpPerspective(im2[:,:,0], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,1] = cv2.warpPerspective(im2[:,:,1], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    im2_t[:,:,2] = cv2.warpPerspective(im2[:,:,2], H2to1, (im2_t.shape[1], im2_t.shape[0]))
    cv2.imwrite('im2_t.jpg', im2_t)

    mask2 = np.zeros((im2.shape[0], im2.shape[1]))
    mask2[0,:] = 1
    mask2[-1,:] = 1
    mask2[:,0] = 1
    mask2[:,-1] = 1
    mask2 = distance_transform_edt(1-mask2)
    mask2 = mask2/mask2.max()
    mask2_t = cv2.warpPerspective(mask2, H2to1, (im2_t.shape[1], im2_t.shape[0]))
    

    mask1 = np.zeros((im1.shape[0], im1.shape[1]))
    mask1[0,:] = 1
    mask1[-1,:] = 1
    mask1[:,0] = 1
    mask1[:,-1] = 1
    
    mask2 = distance_transform_edt(1-mask2)
    mask1 = mask1/mask1.max()
    
    
    im1 = np.pad(im1, ((0,0), (0,im2_t.shape[1]-im1.shape[1]), (0,0)), mode='constant')
    mask1 = np.pad(mask1, ((0,0), (0,im2_t.shape[1]-mask1.shape[1])), mode='constant')
    
    mask1 = np.repeat(np.expand_dims(mask1, axis=-1), 3, axis=-1)
    mask2_t = np.repeat(np.expand_dims(mask2_t, axis=-1), 3, axis=-1)
    
    mask_inter = np.array(np.multiply(im2_t, im1) > 0)*1
    mask1 = np.multiply(mask1, mask_inter)
    mask2_t = np.multiply(mask2_t, mask_inter)
    pano_im = np.multiply(im1, (1-mask_inter)) + np.multiply(im2_t, (1-mask_inter)) - np.multiply(mask2_t, im2_t) - np.multiply(mask1, im1)
    pano_im = im1 + im2_t - np.multiply(mask2_t*1.5, im2_t) - np.multiply(mask1*1.5, im1)
    


    # pano_im = np.empty(im2_t.shape)
    # pano_im[:,:,0]= np.multiply(im1[:,:,0], mask1) + np.multiply(im2_t[:,:,0], mask2_t)
    # pano_im[:,:,1]= np.multiply(im1[:,:,1], mask1) + np.multiply(im2_t[:,:,1], mask2_t)
    # pano_im[:,:,2]= np.multiply(im1[:,:,2], mask1) + np.multiply(im2_t[:,:,2], mask2_t)
    cv2.imwrite('pano_im_mask.jpg', pano_im)
    
    return pano_im

# define some helper functions
# to create affine transformations
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
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    return pano_im


def generatePanorama(im1, im2):
    '''
    Returns a panorama of im1 and im2 without cliping.
    ''' 
    ######################################
    # TO DO ...

    locs1, desc1 = briefLite(im1)
    print(f'No of desc1 is {desc1.shape[0]}')

    locs2, desc2 = briefLite(im2)
    print(f'No of desc2 is {desc2.shape[0]}')
    
    matches = briefMatch(desc1, desc2)
    print(f'No of matches is {matches.shape[0]}')
    
    H2to1 = ransacH(matches, locs1, locs2, num_iter=10000, tol=1)

    #Save result
    np.save('../results/q6_1.npy', H2to1)

    #pano_im = imageStitching_mask(im1, im2, H2to1)
    
    pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    cv2.imwrite('../results/q6_3.jpg', pano_im)

    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    #TODO - GRG : Testing
    #im2 = im1
    #H2to1 = np.load('../results/q6 1.npy')

    print('generating panorama...')

    pano_im = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)

    print('Done - panorama created')