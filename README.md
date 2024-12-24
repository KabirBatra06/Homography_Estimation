# Intro

This program calculates a homography between two images by manually recording the pixel coordinates of a set of corresponding points in two images and using these coordinates to calculate the unknown elements of the homography that relates the images. 
For this I use 4 corresponding points between two images to map one onto the other. To demostrate the code, I map an image of Alex Honnold onto another images of a photoframe. This is done so that the image of Alex fits perfectly into the Photo Frame. I estimate both non-affine and affine homographies to compare and contrast between them. The math I use is detailed below. All the code I wrote is in the python file that is a part of this repositiry. My results and outputs have also been added to this repository. 

# Homography and Affine Transformations

We know that homographies are linear mappings that can be represented by a $\( 3 \times 3 \)$ matrix:

```math
H = 
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}.
```

If we have the coordinates of original points and the corresponding mapped points, we can construct a system of linear equations to solve for the 8 unknown elements of \( H \) (the 9th element can be set to 1 to fix the scale).

---

## 1. Getting Coordinates

1. **Open Gimp (or any image editing tool).**  
2. Collect the pixel coordinates of 4 corner points of the *photo frame*.  
3. Collect the corresponding 4 corner points of the *climbing image* (or the image of Alex).  

We will call the original points 
```math
((x_p, y_p), (x_q, y_q), (x_r, y_r), (x_s, y_s)
```
 and their mapped points 
 ```math 
 ((x'_p, y'_p), (x'_q, y'_q), (x'_r, y'_r), (x'_s, y'_s)
```

---

## 2. Finding Equations

From the homogenous coordinates relationship:

```math
\begin{bmatrix}
x'_1 \\
x'_2 \\
x'_3
\end{bmatrix}
=
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix},
```

we get:

```math
\begin{aligned}
x_1' &= h_{11} x_1 + h_{12} x_2 + h_{13} \\
x_2' &= h_{21} x_1 + h_{22} x_2 + h_{23} \\
x_3' &= h_{31} x_1 + h_{32} x_2 + x_3.
\end{aligned}
```

Because these are *homogeneous* coordinates:

```math
x = \frac{x_1}{x_3}, \quad y = \frac{x_2}{x_3},
```

we rewrite:

```math
\begin{aligned}
x' &= \frac{h_{11} x + h_{12} y + h_{13}}{h_{31} x + h_{32} y + 1}, \\
y' &= \frac{h_{21} x + h_{22} y + h_{23}}{h_{31} x + h_{32} y + 1}.
\end{aligned}
```

Then, by rearranging:

```math
\begin{aligned}
x' (h_{31} x + h_{32} y + 1) &= h_{11} x + h_{12} y + h_{13}, \\
y' (h_{31} x + h_{32} y + 1) &= h_{21} x + h_{22} y + h_{23}.
\end{aligned}
```

These lead to the simplified linear equations:

```math
\begin{aligned}
h_{11} x + h_{12} y + h_{13} - h_{31} x\,x' - h_{32} y\,x' &= x', \\
h_{21} x + h_{22} y + h_{23} - h_{31} x\,y' - h_{32} y\,y' &= y'.
\end{aligned}
```

We do this for each of the 4 point pairs $\{(x_p, y_p),(x'_p, y'_p)\}\), \(\{(x_q, y_q),(x'_q, y'_q)\}$, etc.

---

## 3. Solving Equations to Get \( H \)

For each pair $\bigl((x_p, y_p), (x'_p, y'_p)\bigr)$ , we have two equations:

```math
\begin{aligned}
x'_p &= h_{11} x_p + h_{12} y_p + h_{13} - h_{31} x_p x'_p - h_{32} y_p x'_p, \\
y'_p &= h_{21} x_p + h_{22} y_p + h_{23} - h_{31} x_p y'_p - h_{32} y_p y'_p.
\end{aligned}
```

Similarly for the other three points $(x_q, y_q), (x_r, y_r), (x_s, y_s)$ and their mapped coordinates. This gives us 8 equations in 8 unknowns (since we set $ h_{33} = 1 $ to fix scale).

We can write these equations in matrix form A $\mathbf{h}$ = $\mathbf{b}$ (where $\mathbf{h}$ contains the 8 unknowns), and solve using any linear algebra solver. After finding these 8 values, we **append 1** to form the full 9 elements of the homography matrix:

```math
H = 
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & 1
\end{bmatrix}.
```

---

## 4. Affine Transformation

An **affine transformation** enforces parallel lines to remain parallel. Mathematically, this means the last row of $\(H\)$ is $\([0, 0, 1]\)$. Thus, we rewrite:

```math
\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix}
=
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
0      & 0      & 1
\end{bmatrix}
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
```

So the equations reduce to:

```math
\begin{aligned}
x'_p &= h_{11} x_p + h_{12} y_p + h_{13}, \\
y'_p &= h_{21} x_p + h_{22} y_p + h_{23}.
\end{aligned}
```

For 4 pairs of points, we get 8 equations, which is an overdetermined system for the 6 unknown parameters $\{h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}\}$. We can solve this system in a **least squares** sense. After finding these 6 values, we **append \([0, 0, 1]\)** to shape the final affine matrix:

```math
H_{\text{affine}} =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
0      & 0      & 1
\end{bmatrix}.
```

---

## 5. Warping Images

1. **Create a mask** that highlights the pixels in the *frame image* where we want to place the new image (Alex’s image).  
2. For each pixel $(x', y')$ **in the masked region of the frame image**:
   - Compute the **inverse** of $\(H\)$, call it $\(H^{-1}\)$.  
   - Multiply $\([x', y', 1]^T\)$ by $\(H^{-1}\)$ to find the corresponding pixel $\((x, y)\)$ in the *Alex image*:  
```math
     \begin{bmatrix}
     x \\ 
     y \\ 
     1
     \end{bmatrix}
     =
     H^{-1}
     \begin{bmatrix}
     x' \\ 
     y' \\ 
     1
     \end{bmatrix}.
```
   - Use $\((\lfloor x \rfloor, \lfloor y \rfloor)\)$ (or a suitable interpolation) to **map the RGB values** from the *Alex image* to the frame image.  
3. The result is the *Alex image* warped into the region defined by the mask in the frame image.

## Results

<p align="center">
  <img src="https://github.com/KabirBatra06/Homography_Estimation/blob/main/img1.jpg" width="350" title="Frame A">
  <img src="https://github.com/KabirBatra06/Homography_Estimation/blob/main/img2.jpg" width="350" title="Frame B">
 <br>
  <img src="https://github.com/KabirBatra06/Homography_Estimation/blob/main/img3.jpg" width="350" title="Frame C">
  <img src="https://github.com/KabirBatra06/Homography_Estimation/blob/main/alex_honnold.jpg" width="350" title="Alex">
 <br>
  <img src="https://github.com/KabirBatra06/Homography_Estimation/blob/main/results.png" width="350" title="Results">
</p>
<!-- ![alt text](https://github.com/KabirBatra06/Homography_Estimation/blob/main/img1.jpg)
![alt text](https://github.com/KabirBatra06/Homography_Estimation/blob/main/img2.jpg)
![alt text](https://github.com/KabirBatra06/Homography_Estimation/blob/main/img3.jpg)
![alt text](https://github.com/KabirBatra06/Homography_Estimation/blob/main/alex_honnold.jpg)
![alt text](https://github.com/KabirBatra06/Homography_Estimation/blob/main/results.png) -->

---
