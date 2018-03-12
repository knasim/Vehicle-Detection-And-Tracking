# MAIN PROGRAM
# AUTHOR: Khurrum Nasim

import matplotlib.pyplot as plt
import glob
import time
import collections
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip
from function import *

cars = glob.glob('vehicles/vehicles/*/*.png')
notcars = glob.glob('temp-non-vehicles/non-vehicles/*/*.png')

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 720] # Min and max in y to search in slide_window()
x_start_stop = [400, 1280]


car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
noncar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)


# create an array stack of feature vectors
X = np.vstack((car_features, noncar_features)).astype(np.float64)

# Define the labels vector
Y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

# split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=rand_state)


# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))


# Use a linear SVC
svc = LinearSVC()
# check SVC training time
t=time.time()
svc.fit(X_train, Y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, Y_test), 4))
# Check the prediction time for a single sample
t=time.time()

image = mpimg.imread('test_images/test4.jpg')
draw_image = np.copy(image)
img = image
heat_img = image

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)


ystart = 400
ystop = 656
scale = 1.5

out_img, bboxes = detect_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                              spatial_size, hist_bins)
print("show 1")
plt.show()
#plt.imshow(out_img)


image = out_img
box_list = bboxes

heat = np.zeros_like(image[:,:,0]).astype(np.float)
# Add heat to each box in box list
heat = add_heat(heat, box_list)

# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)

# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(heat_img), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
plt.show()
fig.tight_layout()


heatmaps = collections.deque(maxlen=30)


def pipeline(image):
    global heats

    img = image
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    out_img, bboxes = detect_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins)
    box_list = bboxes
    heat = add_heat(heat, box_list)
    heatmaps.append(heat)
    heatmap_sum = sum(heatmaps)
    # Apply threshold to help remove false positives

    heat = apply_threshold(heatmap_sum, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap_sum)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img



video = VideoFileClip("project_video.mp4")
clip = video.fl_image(pipeline) #NOTE: this function expects color images!!
output = "output.mp4"
clip.write_videofile(output, audio=False)
