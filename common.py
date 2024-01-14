import numpy as np
import matplotlib.patches as patches
import cv2
from scipy.spatial import cKDTree

# ----------------- globals -----------------------------
threading = True

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5 
font_thickness = 2

frame_start = 0
frame_end = np.inf

x_res = 0.214198 # um
y_res = 0.152812 # um

cell_diag_min = 3
cell_diag_max = 10

polygon_min_area = (cell_diag_min / max(x_res,y_res))**2
polygon_max_area = (cell_diag_max / max(x_res,y_res))**2

# ----------------- classes -----------------------------
class Frame:
   def __init__(self, nuclei, nuclei_grad, membrane, membrane_grad): 
      self.image = nuclei
      self.nuclei = nuclei   
      self.nuclei_grad = nuclei_grad
      self.nuclei_flow = None
      self.membrane = membrane
      self.membrane_grad = membrane_grad
      self.membrane_flow = None
      self.sim = None
      self.contours = []
      self.kdtrees  = []
      self.estimated_forward = []
      self.estimated_backward = []
      self.ids = []
      self.removed = []

   def addContour(self, contour):
      self.contours.append(contour)
      self.kdtrees.append(cKDTree(contour))
      self.ids.append(len(self.ids))

   def deleteContour(self, i):
      result = self.contours[i]
      self.contours.pop(i)
      self.kdtrees.pop(i)
      self.ids.pop(i)
      return result

   def addContours(self, contours):
      for contour in contours:
         self.addContour(contour)

   def removeIndices(self, remove):
      remove = reversed(np.sort(np.unique(remove)))

      for i in remove:
         contour = self.contours[i]
         #frame.removed.append(contour)
         self.deleteContour(i)
  

# ----------------- functios -----------------------------
def polygonBounds(poly):
   return (np.min(poly, axis=0), np.max(poly, axis=0))

def plotPolyExtent(plt, poly):
   mi,ma = polygonBounds(poly)
   mi -= 20
   ma += 20
   plt.set_xlim((mi[0],ma[0]))
   plt.set_ylim((mi[1],ma[1]))

def plotPoly(plt, poly, color):
   return plt.add_patch(patches.Polygon(poly, facecolor=color, alpha=0.5))

def computePairwiseDelta(polygon):
   return (polygon-np.roll(polygon,-1,axis=0))

def computePairwiseDistances(polygon):
   distances = (polygon-np.roll(polygon,-1,axis=0))**2
   distances = np.sqrt(distances.sum(axis=1))
   return distances

def computePerimeter(polygon):
   return np.sum(computePairwiseDistances(polygon))

def computeBoundaryDistances(polygon):
   distances = computePairwiseDistances(polygon)
   distances = np.cumsum(distances)
   distances = np.insert(distances, 0, 0)

   return distances

def polygonCentroid(poly):
   xy = np.int32(np.round(np.sum(poly,axis=0)/poly.shape[0]))
   return (xy[0], xy[1]) 

def AABBextend(a, sz):
   mi,ma = a
   c = 0.5*(ma+mi)
   e = 0.5*(ma-mi)
   return (
      (int(c[0]-sz[0]*e[0]),int(c[1]-sz[1]*e[1])),
      (int(c[0]+sz[0]*e[0]),int(c[1]+sz[1]*e[1]))
   )

def AABBintersect(a,b):
   ami,ama = a
   bmi,bma = b
   return ((ami <= bma) & (bmi <= ama)).all()

def polygon2Contour(poly):
   mi = np.min(poly,axis=0)
   ma = np.max(poly,axis=0)
   result = np.zeros([ma[1]-mi[1]+2,ma[0]-mi[0]+2], dtype=np.uint8)
   cv2.drawContours(result, [poly-mi], -1, (255, 255, 255), 1)  
   return result
   
def polygonArea(poly):
   x = poly[:,0]
   y = poly[:,1]
   return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
