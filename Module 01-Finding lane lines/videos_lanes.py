import math
import matplotlib.pyplot as plt
#Function 1
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Function 2
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

#Function 3
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#Function 4
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#Function 5
"""
Thickness has been increased
Kind a like mid Blue.For further HSV values and colors:
http://stackoverflow.com/questions/22499663/detect-only-rgb-blue-in-image-using-opencv
"""
def draw_lines(img, lines, color=[0,0,245], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

#Function 6
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

#Function 7
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
test_cases = os.listdir("E:/Courses/Udacity/[FCO] Udacity - Self-Driving Car Engineer v1.0.0/Part 01-Module 01-Lesson 04_Finding Lane Lines Project/CarND-LaneLines-P1/test_images/")
import matplotlib.image as mpimg

def heart_lanelines(image):
    # Ideas from Quiz 12:Canny Edges
    # Use Function 1
    gray_scaled_image = grayscale(image)
    # Define a kernel size for Gaussian smoothing / blurring
    kernal_size = 7 # Must be an odd number (3, 5, 7...)
    # A larger kernel_size implies averaging, or smoothing, over a larger area.
    # Use Function 3
    blur_gray = gaussian_blur(gray_scaled_image, kernal_size)  
    # Lesson 6 Region Selection
    vertices = np.array([[
        (135, 540), 
        (960, 540),
        (540, 335), 
        (435, 335)]])
    
    #As far as a ratio of low_threshold to high_threshold, John Canny himself recommended a low to high ratio of 1:2 or 1:3.
    low_threshold = 50
    high_threshold = 150
    # Apply Function 2 which is Canny
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Use Function 4
    region_fit = region_of_interest(edges, vertices)
    
    # Lesson 14
    # Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/360 # angular resolution in radians of the Hough grid
    threshold = 40 # minimum number of votes (intersections in a given grid cell) 
    min_line_length = 100 # minimum length of a line (in pixels) that we will accept in the output
    max_line_gap = 300 # maximum distance(in pixels)between segments that we will be allowed to connect into a single line.
    
    # Function 6
    # Run Hough on edge detected image
    lines = hough_lines(region_fit, rho, theta, threshold,
                        min_line_length, max_line_gap)
    
    # Function 7        
    output_image = weighted_img(lines, image)
    
    return output_image

def display_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    
def final_touchup(image_path):
    image = mpimg.imread(image_path)
    processed_image = heart_lanelines(image)
    display_image(processed_image)

for image in test_cases:
    path = 'E:/Courses/Udacity/[FCO] Udacity - Self-Driving Car Engineer v1.0.0/Part 01-Module 01-Lesson 04_Finding Lane Lines Project/CarND-LaneLines-P1/test_images/' + image
    final_touchup(path)

from moviepy.editor import VideoFileClip
from IPython.display import HTML
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    return heart_lanelines(image)

white_output = 'E:/Courses/Udacity/[FCO] Udacity - Self-Driving Car Engineer v1.0.0/Part 01-Module 01-Lesson 04_Finding Lane Lines Project/CarND-LaneLines-P1/test_videos/white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
