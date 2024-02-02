import sys
import os
import numpy as np
import cv2
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d, gaussian_filter, distance_transform_edt
import matplotlib.pyplot as plt
from common import *
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math
import pandas as pd
import argparse

def preprocessFrame(image):
   w,h = image.shape[:2]
   k = max(w,h)
   k = int(math.log2(k))
   k -= 1-k%2

   image = (image/image.max() * 255).astype(np.uint8)
   image = cv2.blur(image, (k,k))
 
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.equalizeHist(gray)
   image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

   grad = cv2.Laplacian(gray, cv2.CV_16S, ksize=k)

   return image, grad

def drawFeatures(image, features, color, ids=None):
   for i,p in enumerate(features.reshape(-1,2)):
      position = (int(p[0]),int(p[1]))
      id = str(-i)

      if ids != None:
         id = str(ids[i])

      image = cv2.putText(image, id, position, font, font_scale, color, font_thickness)

   return image


class Frames:
   def __init__(self):
      self.frames = []
      self.splits = []
      self.merges = []
      self.birth = []
      self.death = []
      self.id_count = 0
   
   def append(self, frame):
      self.frames.append(frame)

   def _estimateFlow(self, img1, img2):
      gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
      return cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

   def estimateFlow(self):
      flows = []

      ref_shape = self.frames[0].nuclei.shape
      zeros = np.zeros((ref_shape[0], ref_shape[1], 1))

      print("estimate flow")
      for current_frame, next_frame in tqdm(zip(self.frames[:-1],self.frames[1:]), total=len(self.frames)-1):
         next_frame.nuclei_flow   = self._estimateFlow(current_frame.nuclei,   next_frame.nuclei)

   def adjustFrames(self):
      print("transform frames to motionless state")

      adjust = self.frames[1:]
      flows = [a.nuclei_flow for a in adjust]

      for i,frame in tqdm(enumerate(adjust), total=len(adjust)):
         nflow = flows[:(i+1)][::-1]

         for j,flow in enumerate(nflow):

            h, w = flow.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            remapped_x = (x + flow[...,0]).astype(np.float32)
            remapped_y = (y + flow[...,1]).astype(np.float32)

            frame.nuclei        = cv2.remap(frame.nuclei,        remapped_x, remapped_y, interpolation=cv2.INTER_LINEAR)
            frame.nuclei_grad   = cv2.remap(frame.nuclei_grad,   remapped_x, remapped_y, interpolation=cv2.INTER_LINEAR)

   def adjustContours(self):
      print("transform frames back")

      adjust = self.frames[1:]
      flows = [a.nuclei_flow for a in adjust]

      for i,frame in tqdm(enumerate(adjust), total=len(adjust)-1):
         nflow = flows[:(i+1)]
         frame.contours = self.frames[0].contours.copy()
         frame.ids      = self.frames[0].ids.copy()

         for j,flow in enumerate(nflow):

            h, w = flow.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            remapped_x = (x + flow[...,0]).astype(np.float32)
            remapped_y = (y + flow[...,1]).astype(np.float32)
            remapped = np.dstack([remapped_x, remapped_y])

            for i,contour in enumerate(frame.contours):
               contour = np.clip(contour, (0,0), (h-1,w-1))
               contour = np.transpose(remapped, (1,0,2))[contour[:,0], contour[:,1]]
               frame.contours[i] = np.int32(np.round(contour))

   def averageField(self, field):
      fields = []

      for frame in self.frames:
         fields.append(getattr(frame, field, None))

      fields = np.stack(fields)
      fields = np.mean(fields, axis=0)

      for frame in self.frames:
         setattr(frame, field, fields)

   def extractContours(self):
      frame = self.frames[0]
      kernel = np.ones((3,3),np.uint8)

      gray = cv2.cvtColor(np.uint8(frame.nuclei), cv2.COLOR_BGR2GRAY)
      mu = np.mean(gray)
      _, thresh = cv2.threshold(gray, mu, 255, 0)
      sure_bg = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kernel,iterations=2)

      grad = np.uint8(frame.nuclei_grad < 0)*255
      sure_fg = cv2.morphologyEx(grad,cv2.MORPH_ERODE,kernel,iterations=2)

      unknown = cv2.subtract(sure_bg,sure_fg)

      ret, markers = cv2.connectedComponents(sure_fg)
      markers = markers+1
      markers[unknown==255] = 0

      markers = cv2.watershed(np.uint8(frame.nuclei), markers)

      #fig,ax = plt.subplots(2,3)
      #ax[0,0].imshow(frame.nuclei_grad)
      #ax[1,0].imshow(gray)
      #ax[0,1].imshow(sure_fg)
      #ax[1,1].imshow(sure_bg)
      #ax[0,2].imshow(unknown)
      #ax[1,2].imshow(markers)
      #plt.show()

      contours = []
      
      for i in range(2, np.max(markers)+1):
         mask = (markers==i).astype(np.uint8)
         contour,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         contours.extend(contour)
      
      for poly in contours:
         cand = poly.reshape(-1,2)
      
         if polygonArea(cand) < polygon_min_area:
            continue
      
         frame.addContour(cand)

   def exportVideo(self, output_name):
      # Get video properties
      width  = self.frames[0].nuclei.shape[0]
      height = self.frames[0].nuclei.shape[1]
      fps = 1
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
      
      # Create a VideoWriter object to save the processed video
      out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
      
      for i,frame in enumerate(self.frames):
         centroids = []
         
         for poly in frame.contours:
            centroid = polygonCentroid(poly)
            centroids.append(centroid)

         centers = np.asarray(centroids)
      
         image = np.uint8(frame.image)
         #image = np.uint8(frame.membrane)

         cv2.drawContours(image, frame.contours, -1, (0, 255, 0), 2) 
         cv2.drawContours(image, frame.removed, -1, (0, 0, 255), 2) 

         image = drawFeatures(image, centers, (255,0,0), frame.ids)

         image = cv2.putText(image, "frame number: " + str(i), [0,40], font, 1.5*font_scale, [255,255,255], font_thickness)
         
         out.write(image)
      
      out.release()

   def exportContours(self, file_path, meta_path):
      di = {
         "frame_id":    np.array([], dtype=np.int32),
         "contour_id":  np.array([], dtype=np.int32),
         "x[px]":           np.array([], dtype=np.int32),
         "y[px]":           np.array([], dtype=np.int32)
      }

      meta = {
         "frame_id":     [],
         "contour_id":   [],
         "center_x[px]": [],
         "center_y[px]": [],
         "area[mu^2]":   [],
         "vx[mu/s]":     [],
         "vy[mu/s]":     []
      }

      for frame_id,frame in enumerate(self.frames):
         for contour_id,contour in enumerate(frame.contours):
            di["frame_id"]   = np.hstack([di["frame_id"],   np.full(len(contour), frame_id)])
            di["contour_id"] = np.hstack([di["contour_id"], np.full(len(contour), contour_id)])
            di["x[px]"]      = np.hstack([di["x[px]"],      contour[:,0]])
            di["y[px]"]      = np.hstack([di["y[px]"],      contour[:,1]])

            centroid = polygonCentroid(contour)

            meta["frame_id"].append(frame_id)
            meta["contour_id"].append(contour_id)
            meta["center_x[px]"].append(centroid[0])
            meta["center_y[px]"].append(centroid[1])
            contour_ = np.copy(contour)
            contour_[:,0] = contour_[:,0] * x_res
            contour_[:,1] = contour_[:,1] * y_res
            meta["area[mu^2]"].append(polygonArea(contour_))

            if frame_id != 0:
               flow = frame.nuclei_flow
               mask = np.zeros(flow.shape[:2], dtype=np.uint8)
               cv2.drawContours(mask, [np.array(contour)], 0, 255, thickness=cv2.FILLED)
               v = np.mean(flow[mask == 255], axis=0)

               meta["vx[mu/s]"].append(v[0]*x_res / t_res)
               meta["vy[mu/s]"].append(v[1]*y_res / t_res)
            else:
               meta["vx[mu/s]"].append(0)
               meta["vy[mu/s]"].append(0)

      df = pd.DataFrame(di)
      df.to_csv(file_path, index=False)  
      df = pd.DataFrame(meta)
      df.to_csv(meta_path, index=False)  

def preprocess(inp):
   image_set     = inp[0]
   frame_counter = inp[1]

   if frame_counter < frame_start:
      return None

   if frame_counter > frame_end:
      return None

   # set debug roi
   #image_set = image_set[:,:,0:400,0:400]

   nuclei, nuclei_grad     = preprocessFrame(image_set)

   return Frame(nuclei, nuclei_grad, None, None)


if __name__ == '__main__':

   parser = argparse.ArgumentParser(description="Contour finder for nuclei images using optical flow")
   
   parser.add_argument('videos', nargs='+', type=str,                      help='list of videos to process')
   parser.add_argument('--x_res',           type=float,  default=0.214198, help='resolution in x direction [mu/px]')
   parser.add_argument('--y_res',           type=float,  default=0.152812, help='resolution in y direction [mu/px]')
   parser.add_argument('--t_res',           type=float,  default=60,       help='resolution in time [s]')
   parser.add_argument('--cell_diag_min',   type=float,  default=3,        help='minimum expected cell diagonal [mu]')
   parser.add_argument('--output',          type=str,    default="",       help='output name prefix of the result files')

   args = parser.parse_args()

   if not hasattr(args, 'videos') or args.videos is None or args.videos is []:
       print("Error: Missing required argument videos")
       parser.print_help()
       exit(1)

   videos = args.videos

   global x_res, y_res, t_res, cell_diag_min, cell_diag_max, polygon_min_area, polygon_max_area
   x_res = args.x_res 
   y_res = args.y_res 
   
   cell_diag_min = args.cell_diag_min 
   cell_diag_max = 10 #unused
   
   polygon_min_area = (cell_diag_min / max(x_res,y_res))**2
   polygon_max_area = (cell_diag_max / max(x_res,y_res))**2

   output_name = args.output

   if output_name == "":
      output_name, _ = os.path.splitext(videos[0])

   frame_counter = 0 
   frames = Frames()
   pool = Pool(cpu_count()) if threading else Pool(1)

   # ----------------- preprocess data -----------------------------
   print(f"preprocessing files: {videos}")

   for arg in tqdm(videos):   
   
      read = []

      cap = cv2.VideoCapture(arg)
      
      if not cap.isOpened():
         print("Error: Could not open video.")
         exit(1)
      
      while cap.isOpened():
         ret, frame = cap.read()
          
         if ret:
            read.append(frame)              
         else:
            break
      
      cap.release()


      result = pool.map(preprocess, zip(read, [i for i in range(frame_counter,frame_counter+len(read))]))

      for frame in result:
         if frame != None:
            frames.append(frame)

      frame_counter += len(read)


   frames.estimateFlow()
   frames.adjustFrames()
   frames.averageField("nuclei_grad")
   frames.averageField("nuclei")
   frames.extractContours()
   frames.adjustContours()
   frames.exportVideo(output_name + "_contour_movie.avi")
   frames.exportContours(output_name + "_contours.csv", output_name + "_metadata.csv")

