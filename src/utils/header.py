IMAGE_DEST_PATH = "images/pair_plot.png"

LABEL_COLORS = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Ravenclaw": "blue",
    "Hufflepuff": "orange"
}

# histogram: a course is "homogeneous" when the per-house means and stds barely
# vary, i.e. both coefficients of variation stay under these thresholds.
MEAN_CV_THRESHOLD = 6
STD_CV_THRESHOLD = 0.1

# scatter_plot: |corr| above which two features count as "similar".
CORRELATION_THRESHOLD = 0.9
