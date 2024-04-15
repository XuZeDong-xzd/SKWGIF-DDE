# SKWGIF-DDE
# Introduction
Image detail enhancement and contrast enhancement are of great significance for infrared imaging systems, and are prerequisites for the further application of low-quality infrared images. Addressing the limitations of infrared image enhancement algorithms based on Guided Image Filtering (GIF), such as insufficient detail layer information, we propose a new method based on Weighted Guided Image Filtering with Steering Kernel (SKWGIF) in this paper. By utilizing the steering kernel to adaptively learn edge directions, the edge preservation ability of filter is enhanced.
Initially, we employ SKWGIF to obtain the base layer of the original infrared image. Subsequently, we apply Gaussian blur to this base layer, subtracting the base layer processed by Gaussian blur from the input image to derive the detail layer. Following the stratification process, Contrast Limited Adaptive Histogram Equalization (CLAHE) is utilized to enhance the contrast of the base layer, while the Automatic Gain Control (AGC) method with SKWGIF linear coefficient as a parameter is employed to enhance the detail layer. Finally, the two layers are fused to generate an infrared image with high contrast and distinct details. Experimental results demonstrate that the algorithm efficiently enhances infrared images across various scenes while effectively preserving image edges.
# flowchart
![image](https://github.com/XuZeDong-xzd/SKWGIF-DDE/blob/main/doc/1.png)
# Used library
OpenCV
# code
We have provided the C++ code and Python code for the algorithm
