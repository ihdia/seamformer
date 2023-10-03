'''
Given the dataset 
in the desired structure
mentioned in README .

Args: 
  > Input  : Configurations for image and dataset JSON file . 
            > Ground Truth Binary Folder.
            > Image Folder 
  > Output : Set of final folders where everything is structured as per the input to code.

'''


# Library Imports
import sys 
import os 
import cv2 
import json
import argparse 
import itertools
import numpy as np 
from empatches import EMPatches
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
from utils import generateScribble

#File Import 
sys.path.append('..') 

# Global Parameters 
THICKNESS = 5
PATCHSIZE = 256 
OVERLAP = 0.25 

# Argument Parser 
def argumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputjsonPath',type=str,default=None)
    parser.add_argument('--datafolder',type=str,default=None)
    parser.add_argument('--patchsize',type=int,default=PATCHSIZE)
    parser.add_argument('--overlap',type=float,default=OVERLAP)
    parser.add_argument('--binaryFlag',type=bool,default=True)
    parser.add_argument('--binaryFolderPath',type=str,default=None)
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--outputfolderPath',type=str,default=None,required=True)
    args = parser.parse_args()
    return args 
      
# Helper Functions

# Make the respective folders 
def createFolders(args):
    os.makedirs(args.outputfolderPath, exist_ok=True)
    smPath = os.path.join(args.outputfolderPath,'scribbleMap/')
    if args.binaryFlag:
        bmPath = os.path.join(args.outputfolderPath,'binaryImages/')
    imPath = os.path.join(args.outputfolderPath,'images/')
    # Prepare a key point image folder 
    try : 
        os.makedirs(smPath,exist_ok=True)
        if args.binaryFlag:
            os.makedirs(bmPath,exist_ok=True)
        os.makedirs(imPath,exist_ok=True)
    except FileExistsError: 
        print('Error in Folder Creation !')
    print('~Folder Creation Completed !')

# Binarisation technique 
def sauvola_niblack_threshold(image,window_size=7):
    h,w,c=image.shape
    if(c>1):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_global = image > threshold_otsu(image)
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)
    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola
    return(binary_sauvola,binary_niblack)

# Cleaning while binarisation
def cleanImage(img):
    _, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 30:   #filter small dotted regions
            img2[labels == i + 1] = 255
    res = cv2.bitwise_not(img2)
    return res 

# Preproceesing : Binarisation of documents 
def preprocess(image):
    bs,bn = sauvola_niblack_threshold(image,window_size=15)
    bs = bs * 255 
    bn = bn * 255 
    # Reshape it 
    bs = bs.reshape((bs.shape[0],bs.shape[1],1))    
    bs =np.uint8(bs)
    clean_image=255-cleanImage(bs)
    # Copy it to 3 channels..
    clean_image = np.stack((clean_image,) * 3, axis=-1)
    return(clean_image)

# Drawing scribble on canvas
def drawScribble(canvas,scribble, thickness=THICKNESS):
    canvas=cv2.polylines(canvas,np.int32([scribble]),False,(255,255,255),thickness)
    return canvas

# Binarisation Function Call 
def get_channel_binary(image):
    binImage = preprocess(image)
    graybinImage = cv2.cvtColor(binImage ,cv2.COLOR_BGR2GRAY)
    graybinImage = graybinImage/255 
    graybinImage = np.asarray(graybinImage,dtype=np.int32)
    return graybinImage

# Scribble Map Generation 
def get_channel_scibbles(img,scribbleList,thickness=THICKNESS):
    # blank canvas 
    h,w, _=img.shape 
    canvas_0 = np.zeros((h,w))
    for i in range(0,len(scribbleList)):
        scribble = scribbleList[i]
        canvas_0 = drawScribble(canvas_0, scribble)
    return canvas_0

def datasetPrepare(args):
    # create folders
    createFolders(args)
    # Read the json file from the arguments
    try:
        with open(args.inputjsonPath,'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('JSON File does not exist.')
        sys.exit() 
    emp = EMPatches()
    # For every datapoints 
    errors=0
    count=0

    for i,datapoint in enumerate(data):
        path = datapoint['imgPath'].replace('./',args.datafolder)
        print('Processing .. {}'.format(path))
        img = cv2.imread(path)
        imgName=os.path.basename(path)
        lower_path = path.lower()

        try:
            if  'scribbles' not in datapoint:
                H = img.shape[0]
                W = img.shape[1]
                # if the ground truth polygons are in rectangular shape set isBox=True, else isBox=False
                scribbles = [generateScribble(H, W, polygon, isBox=False) for polygon in datapoint['gdPolygons'] ]
            else:   
                scribbles = [scr for scr in  datapoint['scribbles']]
            sMap = get_channel_scibbles(img,scribbles,thickness=THICKNESS)
            if sMap is None or img is None: 
                print('Nothing to process..')
                continue

            # Get all patches 
            spatches,indices = emp.extract_patches(sMap,patchsize=args.patchsize,overlap=args.overlap)
            ipatches,indices = emp.extract_patches(img,patchsize=args.patchsize,overlap=args.overlap)

            if args.binaryFlag:
                # Either get the binary image via Sauvola-Niblack Binarisation Method
                bMap = get_channel_binary(img) * 255 # patches --> White text, black background
                if args.binaryFolderPath is not None : 
                    binImage = 255 - cv2.imread(os.path.join(args.binaryFolderPath,imgName.replace('.jpg', '_binarized.jpg')))
                    graybinImage = cv2.cvtColor(binImage ,cv2.COLOR_BGR2GRAY)
                    bMap = np.asarray(graybinImage,dtype=np.int32) 

                # Go ahead and compute patches 
                bpatches,indices = emp.extract_patches(bMap,patchsize=args.patchsize,overlap=args.overlap)
            N = len(spatches)
            for i in range(0,N,1):
                count = count + 1
                # Resizing of the patches to 256 x 256 pixels
                ipatch=  cv2.resize(ipatches[i], (args.patchsize,args.patchsize), interpolation = cv2.INTER_AREA)
                spatch=  cv2.resize(spatches[i], (args.patchsize,args.patchsize), interpolation = cv2.INTER_AREA)
                if args.binaryFlag:
                    bpatch = cv2.resize(bpatches[i], (args.patchsize,args.patchsize), interpolation = cv2.INTER_AREA)
                # List of indices to name the patch
                lindices = list(indices[i])
                imageName_i = imgName.split('.')[0]+'_{}_{}_{}_{}'.format(str(lindices[0]),str(lindices[1]),str(lindices[2]),str(lindices[3]))
                try:
                    # Save the image patch to respective folders
                    cv2.imwrite(os.path.join(args.outputfolderPath,'scribbleMap/sm_{}.jpg'.format(imageName_i)),spatch)
                    cv2.imwrite(os.path.join(args.outputfolderPath,'images/im_{}.jpg'.format(imageName_i)),ipatch)
                    if args.binaryFlag:
                        cv2.imwrite(os.path.join(args.outputfolderPath,'binaryImages/bm_{}.jpg'.format(imageName_i)),bpatch)
                except Exception as exp:
                    print('Error : Saving the patch {}'.format(exp))
                    errors+=1
                    continue

        except Exception as exp:
            print('Error:{}-{}'.format(imgName,exp))
            errors+=1
            continue

# Main 
if __name__ == "__main__":
    args = argumentParser()
    print('Invoking dataset preparation function...')
    datasetPrepare(args)
    print('~Competed!')

