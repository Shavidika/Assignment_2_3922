# EC7212 – Computer Vision and Image Processing

## Take Home Assignment 2

## Assignment Overview

This assignment implements two fundamental image processing techniques:

1. **Otsu's Thresholding Algorithm**: Automatic threshold selection for binary image segmentation
2. **Region-Growing Segmentation**: Seed-based segmentation technique for object extraction

## Implementation Details

### 1. Synthetic Image Creation

Created a synthetic grayscale image with exactly 3 distinct pixel values:

- **Background**: Pixel value = 50
- **Object 1** (Circle): Pixel value = 150
- **Object 2** (Rectangle): Pixel value = 250

**Image Specifications**:

- Dimensions: 200 × 200 pixels
- Object 1: Circular shape with radius 30 pixels
- Object 2: Rectangular shape (50 × 70 pixels)

### 2. Gaussian Noise Addition

Added Gaussian noise with variance = 0.01 to simulate real-world image acquisition conditions. The noise corrupts the original discrete pixel values, making segmentation more challenging.

### 3. Otsu's Thresholding Implementation

Implemented Otsu's algorithm from scratch with the following steps:

1. **Histogram Calculation**: Computed pixel intensity distribution
2. **Probability Calculation**: Normalized histogram to get probabilities
3. **Inter-class Variance Maximization**:
   - For each possible threshold t (1 to 254):
   - Calculate background class probability (w₀) and foreground class probability (w₁)
   - Calculate class means (μ₀ and μ₁)
   - Compute inter-class variance: σ²ᵦ = w₀ × w₁ × (μ₀ - μ₁)²
4. **Optimal Threshold Selection**: Choose threshold that maximizes inter-class variance

**Key Formula**:

```
σ²ᵦ(t) = w₀(t) × w₁(t) × [μ₀(t) - μ₁(t)]²
```

### 4. Region-Growing Segmentation Implementation

Implemented two variants of region-growing:

#### Standard Region Growing:

- **Input**: Seed points and threshold range
- **Process**:
  1. Start from seed points
  2. Add neighboring pixels if their intensity difference from seed is within threshold
  3. Continue until no more pixels can be added
- **Connectivity**: 8-connected neighborhood

#### Adaptive Region Growing:

- **Enhancement**: Dynamically adjusts threshold based on region statistics
- **Adaptive Threshold**: max(initial_threshold, 2 × region_std)
- **Advantage**: Better handles regions with varying intensities

## Results and Analysis

### Performance Metrics

| Method                  | Accuracy | Precision | Recall | F1-Score | Jaccard Index |
| ----------------------- | -------- | --------- | ------ | -------- | ------------- |
| Otsu Thresholding       | 0.984    | 0.995     | 0.906  | 0.948    | 0.902         |
| Region Growing          | 0.877    | 0.984     | 0.224  | 0.366    | 0.224         |
| Adaptive Region Growing | 0.244    | 0.142     | 0.752  | 0.239    | 0.136         |

### Key Findings

1. **Noise Impact**: Gaussian noise significantly affects global thresholding methods like Otsu's algorithm
2. **Otsu Performance**: Achieved 98.4% accuracy and 94.8% F1-score, making it the best performing method
3. **Region Growing Trade-offs**:
   - Regular region growing: High precision (98.4%) but low recall (22.4%)
   - Adaptive region growing: Higher recall (75.2%) but lower precision (14.2%)
4. **Threshold Selection**:
   - Original image: Optimal threshold = 50 (perfect separation)
   - Noisy image: Optimal threshold = 129 (reasonable compromise)
5. **Computational Complexity**:
   - Otsu's: O(L × N) where L is intensity levels, N is pixels
   - Region Growing: O(N) in best case, O(N²) in worst case

### Visual Results

The notebook generates the following visualizations:

1. **Original vs Noisy Images**: Shows the effect of Gaussian noise
2. **Otsu's Results**: Demonstrates automatic threshold selection
3. **Region Growing Results**: Compares different parameter settings
4. **Performance Comparison**: Side-by-side comparison of all methods
5. **Error Analysis**: Visualizes segmentation errors using difference images

## Files Structure

```
Assignment_2_3922/
├── assignment_2_cv_image_processing.ipynb  # Main notebook
├── README.md                               # This file
└── outputs/                               # Generated outputs
    ├── 01_original_image.png              # Original synthetic image
    ├── 02_noisy_images.png                # Noise comparison
    ├── 03_otsu_results.png                # Otsu algorithm results
    ├── 04_region_growing_basic.png        # Basic region growing
    ├── 05_segmentation_comparison.png     # Method comparison
    ├── 06_performance_comparison.png      # Performance analysis
    ├── original_image.png                 # Individual result images
    ├── noisy_image.png
    ├── otsu_result.png
    ├── region_growing_result.png
    └── analysis_summary.txt               # Numerical results summary
```

## How to Run

1. **Prerequisites**:

   ```bash
   pip install numpy matplotlib scikit-image opencv-python scipy
   ```

2. **Execution**:

   - Open `assignment_2_cv_image_processing.ipynb` in Jupyter Notebook/Lab or VS Code
   - Run all cells sequentially
   - Check the `outputs/` directory for generated images and analysis

3. **Python Environment**:
   - Python 3.7+ recommended
   - All dependencies are standard scientific computing libraries

## Technical Implementation Notes

### Otsu's Algorithm Details:

- **Validation**: Compared our implementation with scikit-image's `threshold_otsu()`
- **Accuracy**: Typically within 1-2 intensity levels of reference implementation
- **Optimization**: Vectorized operations for efficient computation

### Region Growing Details:

- **Seed Selection**: Manual selection of representative points for each region
- **Threshold Strategy**: Experimented with fixed and adaptive thresholds
- **Boundary Handling**: Proper edge case management for image boundaries
- **Memory Efficiency**: Used deque for breadth-first search implementation

## Conclusion

This assignment successfully demonstrates:

1. **Theoretical Understanding**: Proper implementation of both algorithms from mathematical foundations
2. **Practical Application**: Effective testing on realistic noisy data
3. **Comparative Analysis**: Quantitative evaluation using multiple metrics
4. **Performance Results**: Otsu's method achieved the best overall performance (98.4% accuracy)

**Key Results:**

- **Otsu's Algorithm**: Best for this specific noisy image (F1-score: 94.8%)
- **Region Growing**: Good precision but struggles with recall in noisy conditions
- **Adaptive Region Growing**: Higher recall but lower precision, needs parameter tuning

The results show that while region-growing methods offer local adaptability, global methods like Otsu can be very effective when the noise characteristics are well-understood. The choice of method depends on the specific requirements of precision vs. recall for the application.

## References

1. Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
2. Adams, R., & Bischof, L. (1994). Seeded region growing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(6), 641-647.
3. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.

---

_This implementation is part of the EC7212 Computer Vision and Image Processing course assignment._
