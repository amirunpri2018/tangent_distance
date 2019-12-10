import cv2 as cv
import numpy as np
import copy
class GradientDescent():
    def __init__(self,T):
        self.T = T #(2,784)
    def __call__(self,x,y):
        r,d = self.T.shape # (2,784)
        a = np.ones(shape = (r,1)) # (2,1)
        t = 0
        while True:
            b = copy.copy(a)
            # (784,2).dot (2,1) -> (784,1) -> (2,1)
            a = a - 0.0005 * self.T.dot(x + self.T.T.dot(a) - y)
            t += 1
            #print(a,b)
            if np.sqrt(np.mean((b-a)**2)) < 0.0001 or t > 5000:
                break
        return a,self.T


class TanhDistance():
    def __init__(self,frame,transforms = None):
        self.vectors = []
        h,w = frame.shape
        self.hw = h*w
        if transforms is not None:
            for transform in transforms:
                t = transform(frame) # (h,w)
                self.vectors.append(np.reshape(t,(h*w,)) - np.reshape(frame,(h*w,)))
        self.gradientDescent = GradientDescent(np.array(self.vectors)) # r,28*28
    def __call__(self,x,y):
        x = np.reshape(x,(self.hw,1))
        y = np.reshape(y,(self.hw,1))
        a,T = self.gradientDescent(x,y) # (28*28,1)
        
        return np.sqrt(np.mean((x + T.T.dot(a) - y)**2)) 
def get_transforms(frame):
    h,w = frame.shape
    transformations = []
    # rotate
    delta_theta = 5
    M = cv.getRotationMatrix2D(((w-1)/2.0,(h-1)/2.0),delta_theta,1)
    transformations.append(lambda x:cv.warpAffine(x,M,(w,h)))

    # shift
    delta_x = 2
    delta_y = 0
    M = np.float32([[1,0,delta_x],[0,1,delta_y]])
    transformations.append(lambda x:cv.warpAffine(x,M,(w,h)))
    
    return transformations

if __name__ == "__main__":
    img = cv.imread("/home/xueaoru/图片/0000.jpg")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray,(28,28))/255
    transforms = get_transforms(gray)
    metric = TanhDistance(gray,transforms)

    img2 = cv.imread("/home/xueaoru/图片/000.jpg")
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    gray2 = cv.resize(gray2,(28,28))/255 

    print("tan distance:{}".format(metric(gray,gray2)))
    print("l2 distance:{}".format(np.sqrt(np.mean((gray - gray2)**2))))
    #for transform in transforms:
    #    print(transform(gray))
