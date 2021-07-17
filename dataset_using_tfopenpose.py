import argparse
import logging
import sys
import time
import os

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from PIL import Image

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()
    # change done
    list_dir='/content/drive/My Drive/clothing_seg/lower_body_images/'
    list_categories = ['Skirt','Sweatpants','Shorts']
    directory_name = '/content/drive/My Drive/clothing_seg/images_new_tfopenpose/'
    
    count=0
    num = 0 
    for folder in os.listdir(list_dir):
        if 'Skirt' in folder:
            print(folder)
            #f=open(directory_name + 'category.txt',"a")
            #f.write(folder+"\n")
            
            if not os.path.exists(directory_name + 'skirts/' + folder):
                os.mkdir(directory_name + 'skirts/' + folder)
                num = num +1
            else:
                num = num +1
                continue
                
            for filename in os.listdir(list_dir + "/"+ folder):
                with Image.open(list_dir + folder + "/" + filename) as img:
                    width,height=img.size
                    image = common.read_imgfile(list_dir + folder + "/" + filename, None, None)
                    w, h = model_wh(args.resize)
                    if w == 0 or h == 0:
                        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
                    else:
                        e = TfPoseEstimator(get_graph_path(args.model))
                    # if image is None:
                    #     logger.error('Image can not be read, path=%s' % args.image)
                    #     sys.exit(-1)
                    # t = time.time()
                    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                    # elapsed = time.time() - t


                    # logger.info('inference image: %s in %.4f seconds.' % (list_dir + folder + "/" + filename, elapsed))
                       
                    image2=np.copy(image)
                    image1,coord_hip, max_coord_lower, min_coord_lower = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                    # print(max_coord_lower,min_coord_lower)
                    # diff_coord = max_coord_lower - min_coord_lower
                    # print(diff_coord)
                    # x1 = ( min_coord_lower - 2*diff_coord)
                    # x2 = ( max_coord_lower + 2*diff_coord)  
                    # if x1 < 0:
                    #     x1 = 0
                    # if x2 > width:
                    #     x2 = width
                    x1 = 0
                    x2 = width
                    y1 = coord_hip[1]
                    y2 = height

                    print(x1,x2)
                    
                    print(width,height)

                    cv2.rectangle(image2, (x1,y1) , (x2,y2), (12,150,100),2)
    
                    lower_body= image2[y1:y2,x1:x2]
    
                    Lower_image="lower"+ str(num) + "_" +  str(count) + ".jpg"
    
                    cv2.imwrite(directory_name + 'skirts/' + folder + '/' +  Lower_image,lower_body)

                    count=count+1

            

                    


                    
                

    
       
     
    

    # estimate human poses from a single image !
    
    

   
    
    # logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    # #change done
    # #image,coordRhip,coordLhip,maxcoordUp,mincoordUp,maxcoordLo,mincoordLo,maxAnkle = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # image2=np.copy(image)
    # image1,coordRhip,coordLhip = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    # from PIL import Image
    # with Image.open(image) as img:
    #     width,height=img.size
    # print(width,height)

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # a = fig.add_subplot(2, 2, 1)
    # a.set_title('Result')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #change done
    #for (x,y,w,h) in faces:
        #print(x,y,w,h)
    #print(maxcoord,mincoord)
    # cv2.rectangle(image2, (0,0) , (width,coordRhip[1]), (12,150,100),2)
    # cv2.rectangle(image2, (0,coordRhip[1]) , (width,height), (12,150,100),2)
    # upper_body=image2[0:coordRhip[1],0:width]
    # lower_body= image2[coordRhip[1]:height,0:width]
    # Upper_image= "croppedimageUp" + ".jpg"
    # Lower_image="croppedimageLo" + ".jpg"
    # cv2.imwrite(Upper_image, upper_body)
    # cv2.imwrite(Lower_image,lower_body)

#second change from here
    # str_label_upper,maxi_upper=predicted_upper(Upper_image)
    # str_label,maxi=predicted_lower(Lower_image)
    # print("\n\n\n\n")
    # print("upper body",str_label_upper,maxi_upper)
    # print("lower body",str_label,maxi)
    # print("\n\n\n\n")


    # cv2.imshow("image",image2)

    


    """print(coordLhip)
    print("maxAnkle",maxAnkle,maxcoordLo)
    cv2.rectangle(image, (mincoordUp,0) , (maxcoordUp,coordRhip[1]), (12,150,100),2)
    cv2.rectangle(image, (mincoordLo,coordRhip[1]) , (maxcoordLo,maxAnkle), (12,150,100),2)
    cv2.imwrite('/home/nihali/Desktop/cprmi/cprmi-fashion/newimage.jpg',image)
    upper_body = image[0:coordRhip[1], mincoordUp:maxcoordUp]
    
    lower_body = image[coordRhip[1]:maxAnkle, mincoordLo:maxcoordLo]
    Upper_image= "/home/nihali/Desktop/cprmi/cprmi-fashion/" + "croppedimageUp" + ".jpg"
    Lower_image="/home/nihali/Desktop/cprmi/cprmi-fashion/" + "croppedimageLo" + ".jpg"
    cv2.imwrite(Upper_image, upper_body)
    cv2.imwrite(Lower_image,lower_body)"""
#heat map part of the code
    
    # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

