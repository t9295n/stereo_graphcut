import numpy as np
import maxflow
import graphcut
import cv2
from ssd import run_ssd

def process_img(img, ratio):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    return img.astype(np.float32)

def process_disparity(disp, n):
    disp[disp == graphcut.OCCLUDED] = 255
    disp = disp.astype(np.uint8)

    res = np.zeros(shape=disp.shape, dtype=np.uint8)
    for r in range(disp.shape[0]):
        for c in range(disp.shape[1]):
            d = disp[r, c]
            if d != 255:
                x = 255 * (d + n/2) / n
                res[r, c] = x
    return res

if __name__ == "__main__":
    ## ## 30, 50, 40, 20
    folders = ["motorcycle", "jadeplant", "newkaba", "chair"]
    # folders = ["chair"]
    for name in folders:

        print(f"Running {name} ssd...")
        for size in [3, 10]:
            run_ssd(name, size, 160)
        print(f"Done with {name} ssd")

        ## 30, 40, 50
        
        print(f"Running {name} graphcut...")
        f1 = f"images/{name}/im1.png"
        f2 = f"images/{name}/im0.png"
        
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        
        img1 = process_img(img1, 1/8)
        img2 = process_img(img2, 1/8)
        
        g = graphcut.Graphcut(img1, img2)
        iters, mindisp, maxdisp = 1, 0, 35
        disp = g.run(iters, mindisp, maxdisp)
        disp = process_disparity(disp, maxdisp - mindisp)
        cv2.imwrite(f"results/{name}/{name}_graphcut.png", disp)
        print(f"Done with {name} graphcut")