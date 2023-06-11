from data_aug.data_aug import Sequence, RandomHorizontalFlip, RandomScale, RandomTranslate, \
        RandomRotate, RandomShear
#from data_aug.bbox_util import *
import cv2 
import numpy as np 
import pandas as pd
from PIL import Image

train = pd.read_csv('train.csv', header=None)

a = train.iloc[:, :].values

h = 267   # number of unique images in train
Matrix = [[] for y in range(h)] 

# Segregrating entries per image
k = 0
img_name = []
labels = []
for i in range(len(a)):
    labels.append(a[i][1])
img_name.append(a[0][0])
Matrix[k].append(a[0][2:7])
for i in range(1, len(a)):
    if a[i][0] == a[i-1][0]:
        Matrix[k].append(a[i][2:7])
    else:
        k = k+1
        Matrix[k].append(a[i][2:7])
        img_name.append(a[i][0])

#Matrix[0], Matrix[1] etc are the bounding boxes


# Flipping
k = h + 1
csv_ready_list = []
j = 0
for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)
    
    transforms = Sequence([RandomHorizontalFlip(1)])
    img, bboxes = transforms(img, bboxes)
    
#    img = Image.fromarray(img)
#    img.save('train_images/{}.jpg'.format(k))
#    print("\r" + "{}.jpg =-------= {}              ".format(k, i + 1), end="\r")
    
    bboxes = [list(x) for x in bboxes]

    for bbox in bboxes:
        bbox.insert(0, k)
        bbox.insert(1, labels[j])
        j+=1

    csv_ready_list.extend([*bboxes])
    k+=1

df = pd.DataFrame(data=csv_ready_list)
df.to_csv('FLIP.csv', header=None, index=None)
    

# Scaling
csv_ready_list = []
j = 0
for i in range(0, img_name[-1]):
#    print(i)
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    scale = RandomScale(0.2, diff = True)
    img, bboxes = scale(img, bboxes)
    
#    img = Image.fromarray(img)
#    img.save('train_images/{}.jpg'.format(k))
#    print("\r" + "{}.jpg =-------= {}".format(k, i + 1), end="\r")
    
    bboxes = [list(x) for x in bboxes]

    for bbox in bboxes:
        bbox.insert(0, k)
        bbox.insert(1, labels[j])
        j+=1

    csv_ready_list.extend([*bboxes])
    k+=1
    
df = pd.DataFrame(data=csv_ready_list)
df.to_csv('SCALED.csv', header=None, index=None)

    
# Translation
csv_ready_list = []
j = 0
for i in range(0, img_name[-1]):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    translate = RandomTranslate(0.4, diff = True)
    img, bboxes = translate(img, bboxes)
    
#    img = Image.fromarray(img)
#    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i + 1), end="\r")
    
    bboxes = [list(x) for x in bboxes]

    for bbox in bboxes:
        bbox.insert(0, k)
        bbox.insert(1, labels[j])
        j+=1

    csv_ready_list.extend([*bboxes])
    k+=1
    

df = pd.DataFrame(data=csv_ready_list)
df.to_csv('TRANSLATED.csv', header=None, index=None)
    
# Rotation
   
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    rotate = RandomRotate(10)  ## rotating by 10 degrees
    img, bboxes = rotate(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('ROTATION.csv', index = False)
    
    
    
# Shearing
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    shear = RandomShear(0.7)  
    img, bboxes = shear(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('SHEAR.csv', index = False)
    
    
    
# Deterministic Scaling
    
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    scale = RandomScale((0.4))  
    img, bboxes = scale(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('DET_SCALE.csv', index = False)
    
    
    
# Flip and Scale
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('FLIP_SCALE.csv', index = False)
    
    

# Flip and Translation
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomTranslate(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('FLIP_TRANSLATE.csv', index = False)
    
# Flip and Rotation
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomRotate(20)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('FLIP_ROTATE.csv', index = False)

# Flip and Shear
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomShear(0.6)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)

    img = Image.fromarray(img)
    img.save('train_images/{}.jpg'.format(k))
    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
    bounding_boxes.to_csv('FLIP_SHEAR.csv', index = False)
    

# Flip, Scale and Rotate
#k=k+1
#for i in range(0, len(img_name)):
#    img = cv2.imread("train_images/{}.jpg".format(img_name[i]))[:,:,::-1]
#    bboxes = Matrix[i]
#    bboxes = np.asarray(bboxes)
#
#    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff = True), RandomRotate(30)])
#    img, bboxes = transforms(img, bboxes)
#    
#    if i == 0:
#        bounding_boxes = pd.DataFrame(bboxes)
#        bounding_boxes.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
#    
#    if i>0:
#        df = pd.DataFrame(bboxes)
#        k = k+1
#        df.insert(loc = 0, column = 'image_name', value = '{}'.format(k))
#        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
#
#    img = Image.fromarray(img)
#    img.save('train_images/{}.jpg'.format(k))
#    print("\r" + "{}.jpg =-------= {}".format(k, i), end="\r")
#    bounding_boxes.to_csv('FLIP_SCALE_ROTATE.csv', index = False)
