'''
General Utility Files 
- Consisting of helper functions 
for supporting visualisations
of model results
'''

import os
import math
import cv2
import copy 
import numpy as np
from empatches import EMPatches
from einops import rearrange
from plantcv import plantcv as pcv


global mean , std 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]



def merge_images_horizontally(img1, img2, img3):
    assert img1.shape == img2.shape==img3.shape , "Error merging the images"
    # Resize images if necessary to make them the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, img1.shape[:2])
    if img1.shape != img3.shape:
        img3 = cv2.resize(img3, img1.shape[:2])
    # Combine the images horizontally
    merged_img = np.hstack((img1, img2, img3))
    return merged_img



def imvisualize(settings,imdeg, imgt, impred, ind, epoch=0,threshold=0.4):
    """
    Visualize the predicted images along with the degraded and clean gt ones
    Args:
        imdeg (tensor): degraded image
        imgt (tensor): gt clean image
        impred (tensor): prediced cleaned image
        ind (str): index of images (name)
        epoch (str): current epoch
        setting (str): experiment name
    """

    # unnormalize data
    imdeg = imdeg.numpy()
    imgt = imgt.numpy()
    impred = impred.numpy()

    impred = np.squeeze(impred, axis=0)
    imgt = np.squeeze(imgt, axis=0)
    imdeg = np.squeeze(imdeg, axis=0)
    
    imdeg = np.transpose(imdeg, (1, 2, 0))
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))

    # Only for the input image 
    for ch in range(3):
        imdeg[:,:,ch] = (imdeg[:,:,ch] *std[ch]) + mean[ch]
        
    # avoid taking values of pixels outside [0, 1]
    impred[np.where(impred>1.0)] = 1
    impred[np.where(impred<0.0)] = 0

    # thresholding now 
    # binarize the predicted image taking 0.5 as threshold
    impred = (impred>threshold)*1

    # Change to 0-255 range 
    imdeg=imdeg*255
    imgt=imgt*255
    impred=impred*255
    impred= impred.astype(np.uint8)

    # save images
    if not settings['enabledWandb']:
        base_dir = os.path.join(settings['visualisation_folder'],'epoch_{}'.format(epoch))
        epoch=str(epoch)
        os.makedirs(base_dir,exist_ok=True)
        imdeg_ = imdeg[:,:,0].reshape(*imdeg.shape[:-1], 1)
        out = merge_images_horizontally(imdeg_,imgt,impred)
        cv2.imwrite(os.path.join(base_dir,str(ind)+'_combined.png'),out)

    return imdeg,imgt,impred



def preprocess(deg_img):
    deg_img = (np.array(deg_img) /255).astype('float32')
    # normalize data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    out_deg_img = np.zeros([3, *deg_img.shape[:-1]])
    for i in range(3):
        out_deg_img[i] = (deg_img[:,:,i] - mean[i]) / std[i]
    return out_deg_img

def readFullImage(path,PDIM=256,DIM=256,OVERLAP=0.25):
    input_patches=[]
    emp = EMPatches()
    try:
        img = cv2.imread(path)
        img = preprocess(img)
        img = np.transpose(img)
        img_patches, indices = emp.extract_patches(img,patchsize=PDIM,overlap=OVERLAP)
        for i,patch in enumerate(img_patches):
              resized=[DIM,DIM]
              if patch.shape[0]!= DIM or patch.shape[1]!= DIM :
                  resized=[patch.shape[0],patch.shape[1]]
                  patch = cv2.resize(patch,(DIM,DIM),interpolation = cv2.INTER_AREA)
              # cv2_imshow(patch)
              patch = np.asarray(patch,dtype=np.float32)
              patch =  np.transpose(patch)
              patch= np.expand_dims(patch,axis=0)
              sample={'img':patch,'resized':resized}
              input_patches.append(sample)
    except Exception as exp :
        print('ImageReading Error ! :{}'.format(exp))
        return None,None
    return input_patches,indices


'''
Reconstruct from pred_pixels to patches 
'''
def reconstruct(pred_pixel_values,patch_size,target_shape,image_size):
    rec_patches = copy.deepcopy(pred_pixel_values)
    output_image = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
    output_image = output_image.cpu().numpy().squeeze()
    output_image =  output_image.T
    # Resizing to get desired output
    output_image = cv2.resize(output_image,target_shape, interpolation = cv2.INTER_AREA)
    # Basic Thresholding
    output_image[np.where( output_image>1)] = 1
    output_image[np.where( output_image<0)] = 0
    return output_image





'''
Calculate patch/image wise PSNR given two input 
'''
def computePSNR(gt_img, pred_img, PIXEL_MAX=None):
    # Detach from computation graph, move calculations to CPU
    pred_img = pred_img.detach().cpu()
    gt_img = gt_img.detach().cpu()

    # Convert torch tensor to number arrays
    pred_img = pred_img.numpy()[0]
    gt_img = gt_img.numpy()[0]

    # calculate Pixel Max and normalise it
    if not PIXEL_MAX:
        PIXEL_MAX = np.max(gt_img)
    gt_img = gt_img/PIXEL_MAX

    # Convert prediction values to 0 and 1
    pred_img[np.where(pred_img>1)] = 1
    pred_img[np.where(pred_img<0)] = 0
    pred_img = (pred_img>0.5)*1

    # Calculate MSE
    mse = np.mean( (pred_img - gt_img) ** 2 )
    if (mse == 0):
        return (100)

    # Calculate PSNR
    p = (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    return p    




def polygon_to_distance_mask(pmask,threshold=60):
    # Read the polygon mask image as a binary image
    polygon_mask = cv2.cvtColor(pmask,cv2.COLOR_BGR2GRAY)

    # Ensure that the mask is binary (0 or 255 values)
    _, polygon_mask = cv2.threshold(polygon_mask, 128, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    distance_mask = cv2.distanceTransform(polygon_mask, cv2.DIST_L2, cv2.DIST_MASK_5)

    # Normalize the distance values to 0-255 range
    distance_mask = cv2.normalize(distance_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Threshold the image
    src = copy.deepcopy(distance_mask)
    src[src<threshold]=0
    src[src>=threshold]=255
    src = np.uint8(src)
    return src

def average_coordinates(hull):
    # Calculate the average x and y coordinates of all points in the hull/contour.
    # Format : [[x1,y1], [x2,y2],[x3,y3]...[xn,yn]]
    num_points = len(hull)
    avg_x = sum(pt[0][0] for pt in hull) / num_points
    avg_y = sum(pt[0][1] for pt in hull) / num_points
    return avg_x, avg_y


# You send set of clean contours to this function
# you obtain a list of hulls and merge the ones on the same horizontal level.
def combine_hulls_on_same_level(contours,tolerance=50):
    combined_hulls = []
    hulls = [cv2.convexHull(np.array(contour)) for contour in contours]

    # Sort the hulls by the average y-coordinate of all points
    sorted_hulls = sorted(hulls, key=lambda hull: average_coordinates(hull)[1])

    # Starting with 0th  contour 
    current_combined_hull = sorted_hulls[0]
    # Starts from 1st contour and keeps track of which hull is merging and breaks
    for hull in sorted_hulls[1:]:
        # Check if the current hull is on the same horizontal level as the combined hull
        if abs(average_coordinates(hull)[1] - average_coordinates(current_combined_hull)[1]) < tolerance:
            # Merge the hulls by extending the current_combined_hull with hull (combining points)
            current_combined_hull = np.vstack((current_combined_hull, hull))
        else:
            # Hull is on a different level, add the current combined hull to the result
            combined_hulls.append(current_combined_hull)
            current_combined_hull = hull

    # Add the last combined hull
    combined_hulls.append(current_combined_hull)
    # Returning them as hulls again
    nethulls = [cv2.convexHull(np.array(contour,dtype=np.int32)) for contour in combined_hulls]
    # finalHulls = [ hull.reshape((-1,2)).tolist() for hull in nethulls ]
    return nethulls


# Text Dilation 
def text_dilate(image, kernel_size, iterations=1):
    # Create a structuring element (kernel) for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

# Horizontal Dilation
def horizontal_dilation(image, kernel_width=5,iterations=1):
    # Create a horizontal kernel for dilation
    kernel = np.ones((1, kernel_width), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations)
    return dilated_image


'''
Generates Scribble given a polygon inputs and img dimensions (pmask)
'''
from scipy.interpolate import interp1d

def uniformly_sampled_line(points):
    num_points = min(len(points),100)
    # Separate x and y coordinates from the given points
    x_coords, y_coords = zip(*points)

    # Calculate the cumulative distance along the original line
    distances = np.cumsum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
    distances = np.insert(distances, 0, 0)  # Add the initial point (0, 0) distance

    # Create a linear interpolation function for x and y coordinates
    interpolate_x = interp1d(distances, x_coords, kind='linear')
    interpolate_y = interp1d(distances, y_coords, kind='linear')

    # Calculate new uniformly spaced distances
    new_distances = np.linspace(0, distances[-1], num_points)

    # Interpolate new x and y coordinates using the uniformly spaced distances
    new_x_coords = interpolate_x(new_distances)
    new_y_coords = interpolate_y(new_distances)

    # Create a list of new points
    new_points = [[np.int32(new_x_coords[i]), np.int32(new_y_coords[i])] for i in range(num_points)]
    return new_points

def draw_line_through_midpoints(box_points):
    # Ensure points are ordered clockwise starting from the top-left corner
    x_coords, y_coords = zip(*box_points)

    top_left_index = np.argmin(x_coords + y_coords)
    ordered_points = box_points[top_left_index:] + box_points[:top_left_index]
    
    # Calculate midpoints of the opposite sides
    top_midpoint = ((ordered_points[0][0] + ordered_points[1][0]) / 2, (ordered_points[0][1] + ordered_points[1][1]) / 2)
    bottom_midpoint = ((ordered_points[2][0] + ordered_points[3][0]) / 2, (ordered_points[2][1] + ordered_points[3][1]) / 2)
    left_midpoint = ((ordered_points[0][0] + ordered_points[3][0]) / 2, (ordered_points[0][1] + ordered_points[3][1]) / 2)
    right_midpoint = ((ordered_points[1][0] + ordered_points[2][0]) / 2, (ordered_points[1][1] + ordered_points[2][1]) / 2)

    return [int(left_midpoint[0]), int(left_midpoint[1])], [int(right_midpoint[0]), int(right_midpoint[1])]

def draw_line_through_midpoints_of_obb(polygon_points):
    # Compute the oriented bounding box (OBB)
    rect = cv2.minAreaRect(np.array(polygon_points, dtype=np.float32))

    # Get the four corner points of the OBB
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points).tolist()

    # Sort the points to ensure they are arranged from top-left to bottom-right
    x_coords, y_coords = zip(*box_points)
    top_left_index = np.argmin(x_coords + y_coords)
    ordered_points = box_points[top_left_index:] + box_points[:top_left_index]

    # Calculate midpoints of the opposite sides
    top_midpoint = ((ordered_points[0][0] + ordered_points[1][0]) / 2, (ordered_points[0][1] + ordered_points[1][1]) / 2)
    bottom_midpoint = ((ordered_points[2][0] + ordered_points[3][0]) / 2, (ordered_points[2][1] + ordered_points[3][1]) / 2)
    left_midpoint = ((ordered_points[0][0] + ordered_points[3][0]) / 2, (ordered_points[0][1] + ordered_points[3][1]) / 2)
    right_midpoint = ((ordered_points[1][0] + ordered_points[2][0]) / 2, (ordered_points[1][1] + ordered_points[2][1]) / 2)

    # Determine the orientation (horizontal or vertical)
    is_horizontal = abs(ordered_points[0][0] - ordered_points[1][0]) > abs(ordered_points[0][1] - ordered_points[3][1])
    if is_horizontal:
        return [ [int(left_midpoint[0]), int(left_midpoint[1])], [int(right_midpoint[0]), int(right_midpoint[1])]]
    else:
        return [ [int(top_midpoint[0]), int(top_midpoint[1])] , [int(bottom_midpoint[0]), int(bottom_midpoint[1])] ]


def find_index_of_max_length_list(list_of_2d_lists):
    return max(enumerate(list_of_2d_lists), key=lambda x: len(x[1]), default=(None, None))[0]


def calculate_average_orientation(points):
    # Calculate the centroid (mean) of the points
    centroid_x = np.mean([point[0] for point in points])
    centroid_y = np.mean([point[1] for point in points])

    # Calculate the orientation (angle) relative to the centroid
    orientations = [np.arctan2(point[1] - centroid_y, point[0] - centroid_x) for point in points]

    # Calculate the average orientation
    average_orientation = np.mean(orientations)

    return average_orientation,centroid_x,centroid_y

def remove_oriented_and_sharp_turn_points(points, angle_threshold_deg):
    filtered_points = []
    average_orientation,centroid_x,centroid_y = calculate_average_orientation(points)
    angle_threshold_rad = np.radians(angle_threshold_deg)

    for point in points:
        # Calculate the orientation (angle) of the point relative to the centroid
        point_orientation = np.arctan2(point[1] - centroid_y, point[0] - centroid_x)

        # Calculate the angular difference between the point's orientation and the average orientation
        angle_difference = abs(point_orientation - average_orientation)

        # Check if the angle difference is below the threshold (not oriented or sharp turn)
        if angle_difference <= angle_threshold_rad:
            filtered_points.append(point)

    return filtered_points

def find_corner_points_polygon(points):
    if not points:
        return None, None
    # Sort the points based on their x-coordinate
    sorted_points = sorted(points, key=lambda point: point[0])
    leftmost_point = list(sorted_points[0])
    rightmost_point = list(sorted_points[-1])
    return leftmost_point, rightmost_point


def generateScribble(H, W, polygon, isBox=False):
    # Generate Canvas
    canvas = np.zeros((H,W))
    # Mark the polygon on the canvas
    if isBox is False:
      leftmost_point, rightmost_point = draw_line_through_midpoints_of_obb(polygon)
    else:
      leftmost_point,rightmost_point = draw_line_through_midpoints(polygon)
    poly_arr = np.asarray(polygon,dtype=np.int32).reshape((-1,1,2))
    canvas = cv2.fillPoly(canvas,[poly_arr],(255,255,255))
    # Scribble generation
    skeleton = pcv.morphology.skeletonize(canvas)
    pruned_skeleton,_,segment_objects = pcv.morphology.prune(skel_img=skeleton,size=100)
    index = find_index_of_max_length_list(segment_objects)
    scribble = np.asarray(segment_objects[index],dtype=np.int32).reshape((-1,2))
    scribble=scribble.tolist()
    scribble = sorted(scribble, key=lambda point: point[0])
    scribble = remove_oriented_and_sharp_turn_points(scribble, angle_threshold_deg=30)

     # return scribble
    if leftmost_point is not None and rightmost_point is not None :
      scribble.append(leftmost_point)
      scribble.append(rightmost_point)
    scribble = sorted(scribble, key=lambda point: point[0])
    return scribble

# New helper function
def deformat(listofpoints):
    # Input : [[[x1,y1],[[x2,y2]],[[x3,y3]]....]
    # Output : [ [x1,y1], [x2,y2],[x3,y3]....]
    output = [ pt[0].tolist() for pt in listofpoints ]
    return output

# Extracts final contours with a specific area threshold 
def cleanImageFindContours(patch,threshold):
  try:
    patch = np.uint8(patch)
    patch = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(patch,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
      print('No contours in the raw image!')
      return patch
    # Else sort them
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    areaList = [cv2.contourArea(c) for c in cntsSorted]
    maxArea = max(areaList)
    sortedContours = [deformat(c) for c in cntsSorted if cv2.contourArea(c)>np.int32(threshold*maxArea)]
    return sortedContours

  except Exception as exp :
    print('Error in figuring out the clean contours : {} '.format(exp))
    return None