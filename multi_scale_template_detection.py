# import the necessary packages(we used only numpy and matplotlib :D )
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

def rectangle(img,x,y,w,h,t):
    img[y-t:y,x-t:x+w+t,:]=255
    img[y-t:y+h+t,x+w:x+w+t,:]=255
    img[y+h:y+h+t,x-t:x+w+t,:]=255
    img[y-t:y+h+t,x-t:x,:]=255
    return img 


def GetBilinearPixel(imArr, posX, posY):
        out = []
        #Get integer and fractional parts of numbers
        modXi = int(posX)
        modYi = int(posY)
        modXf = posX - modXi
        modYf = posY - modYi
        modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
        modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)

        #Get pixels in four corners
        #for chan in range(imArr.shape[2]):     
        bl = imArr[modYi, modXi]
        br = imArr[modYi, modXiPlusOneLim]
        tl = imArr[modYiPlusOneLim, modXi]
        tr = imArr[modYiPlusOneLim, modXiPlusOneLim]

        #Calculate interpolation
        b = modXf * br + (1. - modXf) * bl
        t = modXf * tr + (1. - modXf) * tl
        pxf = modYf * t + (1. - modYf) * b
        out.append(int(pxf+0.5))
 
        return out[0]
 
def re_size(im,x):
        enlargedShape = tuple(map(int, [im.shape[0]*x, im.shape[1]*x]))
        enlargedImg = np.zeros(enlargedShape)
        rowScale = float(im.shape[0]) / float(enlargedImg.shape[0])
        colScale = float(im.shape[1]) / float(enlargedImg.shape[1])
 
        for r in range(enlargedImg.shape[0]):
                for c in range(enlargedImg.shape[1]):
                        orir = r * rowScale #Find position in original image
                        oric = c * colScale
                        enlargedImg[r, c] = GetBilinearPixel(im, oric, orir)
        return enlargedImg

def rgbtogray(img):
    #R,G and B are seperately multiplied with respective weights
    #and added together to give a 2D image
    img=img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114
    return img


#edge-detection
#########################################
#########################################

###########
def convolution(image, kernel):
    #height and width of image are collected
    image_row, image_col = image.shape

    #height and width of kernel are collected
    kernel_row, kernel_col = kernel.shape

    #creating an empty(black) image sized matrix
    output = np.zeros(image.shape)

    #obtaining padding dimensions from kernel shape
    pad_height = kernel_row//2
    pad_width = kernel_col//2

    #adding the padding around the image
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    #actual 2D convolution as follows
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            
    return output

###########

def sobel_edge_detection(image, filter, convert_to_degree=False):
    #convolution in x direction
    new_image_x = convolution(image, filter)
    #convolution in y direction
    new_image_y = convolution(image, np.flip(filter.T, axis=0))
    #calculating magnitude
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    gradient_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180
    return gradient_magnitude, gradient_direction
#########
def Gauss(img):
    height,width = img.shape
    img_out=np.zeros(img.shape)
    #convolution with a 5*5 gaussian filter
    gauss = (1.0 / 273) * np.array(
         [[1, 4, 7, 4, 1],
         [4, 16, 26, 16, 4],
         [7, 26, 41, 26, 7],
         [4, 16, 26, 16, 4],
         [1, 4, 7, 4, 1]])
    img_out=convolution(img,gauss)
    return img_out


###########
def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
    return output

###########
def threshold(image,low, high,weak):
    output = np.zeros(image.shape)
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
    output[strong_row, strong_col] = 255
    output[weak_row, weak_col] = weak
    return output
###########


def hysteresis(img, weak, strong):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

#######
def edge_detection(img):
    img_gauss=Gauss(img)
    edge_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    gradient_magnitude, gradient_direction = sobel_edge_detection(img_gauss, edge_filter, convert_to_degree=True)
    n_max_sup=non_max_suppression(gradient_magnitude, gradient_direction)
    img_threshold=threshold(n_max_sup,5,20,25)
    new_image = hysteresis(img_threshold,25,255)
    return new_image

#####
def corrcoef2(a, b):
    return np.sum(a*b) / np.sqrt(np.sum(a*a) * np.sum(b*b))


def matchtemplate(img, kernel):
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    h, w = kernel_h // 2, kernel_w // 2
    image_conv = np.zeros(img.shape)
    coeff = 0
    coeff2=0
    (x,y)=(0,0)
    patched=np.zeros(kernel.shape)
    for i in range(h, img_h - h):
        for j in range(w, img_w - w):
            patch = img[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            coeff = corrcoef2(patch, kernel)
            if coeff >= coeff2:
                (x,y)=(i-h,j-w)
                coeff2=coeff
                patched=patch
    return (x,y),coeff2,patched
#####
#####

if __name__=='__main__':
    st=time.time()
    template1=mpimg.imread('logo.jpg')
    template = np.array(template1)
    template = rgbtogray(template)
    template = edge_detection(template)
    print('completed the edge detection of template!')
    for j in range(1,4):
            image = np.array(mpimg.imread('image'+str(j)+'.jpg'))
            if j==1 :
                    show1=image
            elif j==2 :
                    show2=image
            else :
                    show3=image
            gray1 = rgbtogray(image)
            gray = edge_detection(gray1)
            print('completed the edge detection of image'+str(j))
            (tH,tW)=template.shape
            found=(0,(0,0),0)
            print('started template-matching for image'+str(j))
            for scale in np.linspace(0.2,1.0,5)[::-1]:
                resized=re_size(gray,scale)
                r = gray.shape[1] / float(resized.shape[1])
    
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break
    
                (maxLoc,maxVal,patch)=matchtemplate(resized,template)
                
                if found==(0,(0,0),0) or maxVal > found[0]:
                    found,patch2,final_scale = (maxVal, maxLoc, r),patch,scale
            print('completed template-matching for image'+str(j))

            (_, maxLoc, r) = found
            if j==1:
                    show1=rectangle(show1,int(maxLoc[1]*r),int(maxLoc[0]*r),int(tW*r),int(tH*r),5)
            elif j==2:
                    show2=rectangle(show2,int(maxLoc[1]*r),int(maxLoc[0]*r),int(tW*r),int(tH*r),5)
            else:
                    show3=rectangle(show3,int(maxLoc[1]*r),int(maxLoc[0]*r),int(tW*r),int(tH*r),5)
                    


    #display
    plt.subplot(221),plt.imshow(template1)
    plt.title('template')
    plt.subplot(222),plt.imshow(show1)
    plt.title('image1')
    plt.subplot(223),plt.imshow(show2)
    plt.title('image2')
    plt.subplot(224),plt.imshow(show3)
    plt.title('image3')
    sp=time.time()
    print('Time-taken : '+str(sp-st)+' sec')
    print('Thankyou!')
    plt.show()
    
