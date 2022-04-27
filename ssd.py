import numpy as np
import cv2

def process_img(img, ratio):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    return img.astype(np.float32)

def ssd(img1, img2, w, ndisp, direction="right"):
    '''
    https://piazza.com/class/kjliq19wrwi2rj?cid=544_f24
    use boxfilter instead of double for loops
    '''
    r, c = img1.shape
    kernel = np.ones((w,w))
    disparities = np.zeros((r, c, ndisp))
    
    if direction == "right":
        shift = img2
        original = img1
    else:
        shift = img1
        original = img2
    
    for x in range(ndisp):
        ## shift to right/left by x
        y = x if (direction == "right") else -x
        M = np.float32([
            [1, 0, y],
            [0, 1, 0]])
        shifted = cv2.warpAffine(shift, M, (c, r))
        
        ## calculate sum of squared differences with boxFilter (filter2D)
        ssd = (original - shifted)**2
        ssd = cv2.filter2D(ssd, -1, kernel)
        
        ## store disparities
        disparities[:, :, x] = ssd
    
    ## find disparities with smallest SSD
    disp = np.argmin(disparities, axis=2)
    disp = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return disp

def run_ssd(folder, size, ndisp, color=False):
    ## load+read+process images
    f1 = f"images/{folder}/im0.png"
    f2 = f"images/{folder}/im1.png"
    
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    
    img1 = process_img(img1, 0.25)
    img2 = process_img(img2, 0.25)
    
    right = ssd(img1, img2, size, ndisp, "right")
    left = ssd(img1, img2, size, ndisp, "left")
    cv2.imwrite(f"results/{folder}/{folder}_right_{size}.png", right)
    cv2.imwrite(f"results/{folder}/{folder}_left_{size}.png", left)

    if color:
        right_color = cv2.applyColorMap(right, cv2.COLORMAP_JET)
        left_color = cv2.applyColorMap(left, cv2.COLORMAP_JET)
        cv2.imwrite(f"results/{folder}/{folder}_rightcolor_{size}.png", right_color)
        cv2.imwrite(f"results/{folder}/{folder}_leftcolor_{size}.png", left_color)
    
    return right
