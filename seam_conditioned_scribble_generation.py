import cv2
import json
import numpy as np
import copy
from PIL import Image
from scipy.spatial import distance
from scipy.signal import convolve2d
from numba import jit
from sys import maxsize
import itertools
import json
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from numpy import array


# Global Settings 
sobel_kernels = {
    "x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 
    "y": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
}
gaussian_kernel = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
DIAGONAL_LEFT = np.intp(1)
DOWN = np.intp(2)
DIAGONAL_RIGHT = np.intp(3)


def polyGeneration(img, upperSeam, lowerSeam, scribble=None, display=False):
    firstPoint = np.asarray(scribble[0]).tolist()
    lastPoint = np.asarray(scribble[-1]).tolist()

    xLimit = np.maximum(lastPoint[0], firstPoint[0]) + 50
    yLimit = np.maximum(lastPoint[1], firstPoint[1]) + 50

    upperSeam = [
        point for point in upperSeam if (point[0] < xLimit and point[1] < yLimit)
    ]
    lowerSeam = [
        point for point in lowerSeam if (point[0] < xLimit and point[1] < yLimit)
    ]

    upperSeam = np.asarray(upperSeam, dtype=np.int32)
    lowerSeam = np.asarray(lowerSeam, dtype=np.int32)

    upperSeam = upperSeam.reshape((upperSeam.shape[0], 1, 2))
    lowerSeam = lowerSeam.reshape((lowerSeam.shape[0], 1, 2))

    # upperSeam=np.append(upperSeam,lastPoint,axis=0)
    # lowerSeam=np.append(lowerSeam,firstPoint,axis=0)

    region = np.concatenate((upperSeam, lowerSeam))
    if display and img is not None:
        canvas = copy.deepcopy(img)
        canvas = cv2.drawContours(canvas, [region], 0, (255, 0, 0), 2)
        # cv2_imshow(canvas)

    return region


def converttoPolySeam(seams):
    x_axis = np.expand_dims(np.array(range(0, len(seams[0]))), -1)
    seams = [
        np.concatenate((x, np.expand_dims(seam, -1)), axis=1)
        for seam, x in zip(seams, itertools.repeat(x_axis))
    ]
    seamfinal = np.asarray(seams[0])
    return seamfinal

def Reverse(lst):
    new_lst = lst[::-1]
    return new_lst

def get_energy_in_rgb(energy_grayscale: np.array):
    return np.array(Image.fromarray(energy_grayscale).convert("RGB"))


@jit
def is_in_image(position: tuple, rows: int, cols: int) -> bool:
    row, col = position
    return row >= 0 and col >= 0 and row < rows and col < cols


@jit
def trace_seam(
    original_image, energy_image, seam_start, next_seam_position, seam, endSeam
):
    seam_pos = seam_start
    while True:
        row, col = seam_pos
        seam.append([original_image.shape[1] - col, row])
        if next_seam_position[row][col] == 0 or row > endSeam:
            break
        if next_seam_position[row][col] == DIAGONAL_LEFT:
            seam_pos = [row + 1, col - 1]
        elif next_seam_position[row][col] == DIAGONAL_RIGHT:
            seam_pos = [row + 1, col + 1]
        else:
            seam_pos = [row + 1, col]


def carve_column_and_mark_seam(
    original_image: np.array,
    energy_image: np.array,
    seam_start: list,
    next_seam_position: tuple,
    endSeam: int,
):
    seam = list()
    trace_seam(
        original_image, energy_image, seam_start, next_seam_position, seam, endSeam
    )
    return original_image, energy_image, seam


@jit
def compute_optimal_seam2(energy, region):
    energy[np.where(region == 0)] = 1000000
    rows, cols = energy.shape
    infinity = maxsize / 10
    dp = energy.copy()

    next_seam_position = np.zeros_like(dp, dtype=np.intp)

    for row in range(rows):
        dp[row][cols - 1] = energy[row][cols - 1]

    for col in range(cols - 2, -1, -1):
        for row in range(rows):
            optimal_adjacent_cost = infinity
            optimal_choice = -1            
            adjacents = [
                ((row, col + 1), 0),
                ((row - 1, col + 1), 1),
                ((row + 1, col + 1), 2),
            ]
            for adjacent, choice in adjacents:
                adjacent_row, adjacent_col = adjacent
                if not is_in_image(adjacent, rows, cols):
                    continue
                if dp[adjacent_row][adjacent_col] < optimal_adjacent_cost:
                    optimal_adjacent_cost = dp[adjacent_row][adjacent_col]
                    optimal_choice = choice

            next_seam_position[row][col] = optimal_choice
            dp[row][col] = energy[row][col] + optimal_adjacent_cost

    seam_start_col = np.argmin(dp[0, :])
    seam_start = [0, seam_start_col]
    seam_cost = dp[0][seam_start_col]
    return (seam_start, seam_cost, next_seam_position)


@jit
def compute_optimal_seam_down(energy, region):
    energy[np.where(region == 0)] = 255
    rows, cols = energy.shape
    infinity = maxsize / 10
    dp = energy.copy()

    next_seam_position = np.zeros_like(dp, dtype=np.intp)

    for row in range(rows):
        dp[row][cols - 1] = energy[row][cols - 1]

    for col in range(cols - 2, -1, -1):
        for row in range(rows):
            optimal_adjacent_cost = infinity
            optimal_choice = -1
            adjacents = [
                ((row, col + 1), 0),
                ((row + 1, col + 1), 1),
                ((row - 1, col + 1), 2),
            ]
            for adjacent, choice in adjacents:
                adjacent_row, adjacent_col = adjacent
                if not is_in_image(adjacent, rows, cols):
                    continue
                if dp[adjacent_row][adjacent_col] < optimal_adjacent_cost:
                    optimal_adjacent_cost = dp[adjacent_row][adjacent_col]
                    optimal_choice = choice

            next_seam_position[row][col] = optimal_choice
            dp[row][col] = energy[row][col] + optimal_adjacent_cost

    seam_start_col = np.argmin(dp[0, :])
    seam_start = [0, seam_start_col]
    seam_cost = dp[0][seam_start_col]
    return (seam_start, seam_cost, next_seam_position)


# In[372]:


def crop_down(
    energy_image: np.array,
    points: list,
    regions: list,
    endPoints: list,
):
    image_energy = energy_image
    seams = list()
    for i in range(len(regions)):
        (seam_start, seam_cost, next_seam_position) = compute_optimal_seam_down(
            image_energy.copy(), regions[i]
        )
        final_seam = []
        curr = [points[i][0], points[i][1]]
        while (curr[0] < endPoints[i][0]):
            final_seam.append(curr)
            if next_seam_position[curr[1]][curr[0]] == 0:
                curr = [curr[0] + 1, curr[1]]
            elif next_seam_position[curr[1]][curr[0]] == 1:
                curr = [curr[0] + 1, curr[1] + 1]
            else:
                curr = [curr[0] + 1, curr[1] - 1]
        seams.append(final_seam)

    energy_image = Image.fromarray(energy_image).convert("RGB")

    return None, energy_image, seams


def apply_sobel(image: np.array):
    blurred = convolve2d(image, gaussian_kernel, mode="same", boundary="symm")
    grad_x = convolve2d(blurred, sobel_kernels["x"], mode="same", boundary="symm")
    grad_y = convolve2d(blurred, sobel_kernels["y"], mode="same", boundary="symm")
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    normalised_grad = grad * (255.0 / np.max(grad))
    return normalised_grad


# node -> centroid , nodes = list of scribblePoints
def closest_node(node, nodes):
    node = np.asarray(node)
    nodes = np.asarray(nodes).reshape(-1, 2)
    closest_index = np.argmin(distance.cdist([node], nodes))
    return nodes[closest_index]


# Collate the stats-centroids-labels in correct fashion , and apply areaThreshold
def processStats(image, labels, stats, centroids, areaThreshold=70, vis=False):
    H, W, C = image.shape
    image1 = image.copy()
    stats = np.asarray(stats, dtype=np.int32).tolist()
    centroids = np.asarray(centroids, dtype=np.int32).tolist()
    # label,x,y,w,h,centroid
    stats = [
        [i] + stat + [centroids[i][0], centroids[i][1]] for i, stat in enumerate(stats)
    ]
    stats = sorted(stats, key=lambda x: x[5], reverse=True)
    # Remove the first two label (bg)
    stats.pop(0)
    stats.pop(0)
    stats = [stat for stat in stats if stat[5] > areaThreshold]
    netstats = []
    for i, lab in enumerate(stats):
        trueLabel = lab[0]
        labelMask = np.zeros((H, W, 1))
        labelMask[labels == trueLabel] = 255
        labelMask = np.uint8(labelMask)
        # Find contour for this .
        contours, hierarchy = cv2.findContours(
            labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        polygon = [contours[0]]

        x = lab[1]
        y = lab[2]
        w = lab[3]
        h = lab[4]
        cX = lab[-2]
        cY = lab[-1]

        # label,polygon,centroidx,centroidy
        netstats.append([trueLabel, polygon, cX, cY])

    return netstats


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img

# Region Mask for each scribble
def regionMask(scribble, img, gap=30):
    # Generate Pseudo Scribbles
    upperPseudoScribbles = generatePseudoScribble(scribble, gap, isUp=True)
    lowerPseudoScribbles = generatePseudoScribble(scribble, gap, isUp=False)
    N = len(lowerPseudoScribbles)
    bottom = np.flipud(np.asarray(lowerPseudoScribbles, dtype=np.int32))
    top = np.asarray(upperPseudoScribbles, dtype=np.int32)
    region = np.concatenate((top, bottom))
    polynomialgon = np.zeros((img.shape[0], img.shape[1]))
    polynomialgon = cv2.fillPoly(polynomialgon, [region], color=[1])
    rmask = np.expand_dims(polynomialgon, axis=2)
    return rmask


def ccl(img, conn=8, inv=False):
    # Convert it to 255 scale
    img = np.asarray(img, dtype=np.int32)
    if (np.max(img)) != 255:
        img = img / np.max(img)
        img = 255 * img
    # Inverting it
    if inv:
        bin_uint8 = (255 - img).astype(np.int32)
    else:
        bin_uint8 = np.asarray(img, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_uint8, conn, cv2.CV_32S
    )
    markedImage = imshow_components(labels)
    return markedImage, num_labels, labels, stats, centroids


def genDiacriticMap(binImage, scribbles, gap, thickness=5):
    newScribbles = []
    # Take the binary image
    vcanvas = np.zeros((binImage.shape[0], binImage.shape[1], 3))
    for scribble in scribbles:
        canvas = cv2.polylines(
            binImage.copy(), np.int32([scribble]), False, (255, 255, 255), thickness
        )
        rMask = regionMask(scribble, canvas, gap)
        netRegion = np.uint8(canvas * (rMask))
        netRegion = cv2.cvtColor(netRegion, cv2.COLOR_BGR2GRAY)
        cImg, num_labels, labels, stats, centroids = ccl(netRegion, conn=4, inv=False)
        netstats = processStats(
            cImg, labels, stats, centroids, areaThreshold=70, vis=False
        )
        # Connected points
        connectedPoints = []
        for i, stat in enumerate(netstats):
            basePoint = [stat[-2], stat[-1]]
            basePolygon = stat[1][0]
            sclosestPoint = list(closest_node(basePoint, scribble))
            closestPoint = list(closest_node(sclosestPoint, basePolygon))
            connectedPoints.append(
                {"basePoint": sclosestPoint, "connectPoint": closestPoint}
            )

        # Map plotting
        for i, pdict in enumerate(connectedPoints):
            startPoint = pdict["basePoint"]
            endPoint = pdict["connectPoint"]
            vcanvas = cv2.line(
                vcanvas, startPoint, endPoint, (255, 255, 255), thickness
            )
        vcanvas = cv2.polylines(
            vcanvas, np.int32([scribble]), False, (255, 255, 255), thickness
        )

    bMapNew = 255 * cv2.bitwise_or(vcanvas / 255, binImage / 255)    
    return vcanvas, bMapNew


# Mark scribble on images.
def markScribbleList(img, scribbleList, thickness=5):
    # blank canvas
    scribbleThickness = 5
    h, w, c = img.shape
    for i, scribble in enumerate(scribbleList):
        img = cv2.polylines(
            img, np.int32([scribble]), False, (255, 255, 255), scribbleThickness
        )
    return img


# Mark scribble on images.
def markScribbleListGray(img, scribbleList, thickness=5):
    # blank canvas
    h, w = img.shape
    scribbleThickness = 5
    for i, scribble in enumerate(scribbleList):
        img = cv2.polylines(
            img, np.int32([scribble]), False, (255, 255, 255), scribbleThickness
        )
    return img


def get_energy_image_with_blur_masks(
    image_to_crop: Image, mask1, mask2, scribbleList
):
    grayscale_to_crop = image_to_crop.convert("1")
    grayscale_to_crop_bytes = np.array(grayscale_to_crop)
    grayscale_to_crop_energy = apply_sobel(np.array(grayscale_to_crop_bytes))
    grayscale_to_crop_energy = np.int32(
        0.6 * grayscale_to_crop_energy + 0.4 * np.float32(mask2)
    )
    grayscale_to_crop_energy = markScribbleListGray(
        grayscale_to_crop_energy, scribbleList
    )
    return grayscale_to_crop_energy


def crop(
    energy_image: np.array,
    points: list,
    regions: list,
    endPoints: list,
):
    image_energy = energy_image
    # cv2.imwrite("crop_energy.jpg", image_energy)
    seams = list()
    for i in range(len(regions)):
        (seam_start, seam_cost, next_seam_position) = compute_optimal_seam2(
            image_energy.copy(), regions[i]
        )
        final_seam = []
        curr = [points[i][0], points[i][1]]
        while (curr[0] < endPoints[i][0]):
            final_seam.append(curr)
            if next_seam_position[curr[1]][curr[0]] == 0:
                curr = [curr[0] + 1, curr[1]]
            elif next_seam_position[curr[1]][curr[0]] == 1:
                curr = [curr[0] + 1, curr[1] - 1]
            else:
                curr = [curr[0] + 1, curr[1] + 1]
        seams.append(final_seam)
    energy_image = Image.fromarray(energy_image).convert("RGB")
    return None, energy_image, seams

def avgHeight(scribble):
    scribble = np.asarray(scribble, dtype=np.float32).reshape(-1, 2)
    avgY = np.mean(scribble, axis=0)
    val = int(np.mean(avgY))
    return val

# Get average height
def avgYHeight(scribble):
    scribble = np.asarray(scribble, dtype=np.float32).reshape(-1, 2)
    avgY = np.mean(scribble, axis=0)
    val = np.mean(avgY)
    return val

# Gap
def getInterlineGap(img, scribbles, BUFFER=5):
    avgDist = []
    for i in range(len(scribbles) - 1):
        img2 = cv2.polylines(img.copy(), [np.array(scribbles[i])], False, (255, 0, 0), 5)
        # cv2.imwrite(f"scribble{i}.jpg", img2)
        scribble = np.array(scribbles[i], np.int32)
        scribble2 = np.array(scribbles[i + 1], dtype=np.int32)
        avgDist.append(np.mean(scribble2[:, 1]) - np.mean(scribble[:, 1]))
    img2 = cv2.polylines(img.copy(), [np.array(scribbles[len(scribbles) - 1])], False, (255, 0, 0), 5)
    gap = np.mean(avgDist)
    print(avgDist)
    return gap


# Pseudo Scribbles
def generatePseudoScribble(scribble, gap, isUp):
    pseudoScribble = np.empty((scribble).shape, dtype=np.int32)
    if isUp:
        gap = -gap
    for i in range(len(scribble)):
        pseudoScribble[i] = np.array([scribble[i][0], scribble[i][1] + gap])
    pseudoScribble[:, 0] = scribble[:, 0]
    pseudoScribble[:, 1] = scribble[:, 1] + gap
    return pseudoScribble


def generateSeams(
    imgSource, binImage, scribbleList, showImg=False, save=True, omega=0.70
):
    n = len(scribbleList)
    if n == 0:
        print("Error message , No Scribbles are present ...")
        return None, None

    initPoints = np.empty((n, 2), dtype=np.int32)
    endPoints = np.empty((n, 2), dtype=np.int32)
    for i in range(n):
        initPoints[i] = np.array(scribbleList[i][0], dtype=np.int32)
        endPoints[i] = np.array(scribbleList[i][-1], dtype=np.int32)

    if n > 1:
        gap = getInterlineGap(imgSource, scribbleList, 10)
    else:
        gap = 35

    regions = np.empty((n, binImage.shape[0], binImage.shape[1]), dtype=np.int8)
    regions2 = np.empty((n, binImage.shape[0], binImage.shape[1]), dtype=np.int8)
    sList = scribbleList
    imgNow = imgSource

    polygons = []

    for i, currscr in enumerate(scribbleList):
        seam_current = generatePseudoScribble(scribbleList[i], gap, True)
        seam_next = scribbleList[i]

        seam_next = np.flipud(seam_next)
        region = np.concatenate((seam_current, seam_next))

        polygons.append(region)

        polynomialgon = np.zeros((imgNow.shape[0], imgNow.shape[1]))
        cv2.fillPoly(polynomialgon, [region], color=[255])
        regions[i] = polynomialgon

    for i, currscr in enumerate(scribbleList):
        seam_next = np.asarray(generatePseudoScribble(scribbleList[i], gap, False))
        seam_current = np.asarray(scribbleList[i])

        seam_current = seam_current.reshape((seam_current.shape[0], 1, 2))
        seam_next = seam_next.reshape((seam_next.shape[0], 1, 2))

        seam_next = np.flipud(seam_next)

        region = np.concatenate((seam_current, seam_next))

        polynomialgon = np.zeros((imgNow.shape[0], imgNow.shape[1]))
        cv2.fillPoly(polynomialgon, [region], color=[255])
        regions2[i] = polynomialgon


    coors = initPoints
    ends = endPoints
    if len(coors) == 0:
        print("Cannot process furthur")
        return None, None

    # First Map - ScribbleMap - Generate via Diacritic Map Function , not from raw scribbles
    markScribbleList(binImage, scribbleList)
    _, bMapNew = genDiacriticMap(binImage, scribbleList, gap)
    mask1 = copy.deepcopy(bMapNew)

    # Gaussian Blur
    mask2 = cv2.GaussianBlur(mask1, (5, 11), 0)
    mask2 = markScribbleList(mask2, scribbleList)
   
    mask2 = cv2.cvtColor(mask2.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    binImage = Image.fromarray(binImage, "RGB")

    original_energy_image = get_energy_image_with_blur_masks(
        binImage, mask1, mask2, scribbleList
    )

    marked_original_image, marked_energy_image, seams = crop(
        original_energy_image, coors, regions, ends
    )

    save = True
    img = copy.deepcopy(imgSource)
    if save or showImg:
        for i in range(len(seams)):
            img = cv2.polylines(img, [np.array(seams[i])], False, (0, 0, 255), 2)

    marked_original_image, marked_energy_image, seams2 = crop_down(
        original_energy_image, coors, regions2, ends)
    
    img_down = copy.deepcopy(imgSource)
    if save or showImg:
        for i in range(len(seams2)):
            img_down = cv2.polylines(img_down, [np.array(seams2[i])], False, (0, 0, 255), 2)

    polygons = []
    for i in range(len(seams2)):
        fseam = seams[i]
        revseam = np.flipud(seams2[i])
        polygons.append(np.concatenate((fseam, revseam)).tolist())    
    return polygons

# Calling Function
def imageTask(img, bimg, scribbles):
    img = np.asarray(img, dtype=np.uint8)
    bimg = np.asarray(bimg, dtype=np.uint8)
    for i in range(len(scribbles)):
        scribbles[i] = np.array(scribbles[i], dtype=np.int32)
        scribbles[i] = np.array(sorted(scribbles[i], key=lambda k: [k[0], k[1]]))
    # Sorting of scribbles
    scribbles = sorted(scribbles, key=lambda k: [k[0][1], k[0][0]])
    predpolygons = generateSeams(img, bimg, scribbles, showImg=True)
    if predpolygons is None:
        return None
    return predpolygons
