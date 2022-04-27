Stereo Correspondence with Energy Minimization via Graph Cuts

The use of graph cuts as an optimization method is useful for many computer vision tasks, a prominent
one being the stereo correspondence problem. The correspondence problem is
when pixels in one image are matched to pixels in another from a different
view, which can be formulated as an energy minimization problem and solved using graph cuts

To generate results run main.py

To individually run the SSD function, run proces_img, followed by the ssd file
- ssd file takes in two images, the window size, and the number of disparities as required params

To run the graphcuts method individually, run process_img, followed by instantiating the Graphcut class with two images
- the first img should be the left image, the second image the right image