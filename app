import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from flaskwebgui import FlaskUI

# theme = dbc.themes.BOOTSTRAP
# theme = dbc.themes.UNITED
# theme = dbc.themes.QUARTZ
theme = dbc.themes.SUPERHERO
icon_lib = dbc.icons.FONT_AWESOME

app = Dash(
    name=__name__,
    external_stylesheets=[theme, icon_lib],
    use_pages=True
)
OUR_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app.layout = html.Div(children=[
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Process Optical", active='exact', href="/")),
            dbc.NavItem(dbc.NavLink("Process SEM", active='exact', href="/sem")),
            dbc.NavItem(dbc.NavLink("Comparison (optical)", active='exact', href="/compare-optical")),
            dbc.NavItem(dbc.NavLink("Comparison (SEM)", active='exact', href="/compare-sem")),
        ],
        pills=True,
        justified=True,
    ),
    dash.page_container
])

if __name__ == '__main__':
    # app.run_server(debug=True)
    FlaskUI(app=app, server='flask', port=3000).run()
## SEM analysis utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import cv2
import glob
import os
import time

from skimage.io import imread
from skimage.filters import sobel, threshold_otsu, threshold_multiotsu
from skimage.segmentation import watershed
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb, rgb2gray
from skimage.feature import peak_local_max


# Optical image analysis functions
# Core segmentation founction (uses blue channel)
def segment_from_image(image_blue,
                       background_bound,
                       foreground_bound,
                       min_distance):
    
    # Sobel edge detection
    edges = sobel(image_blue)
    edges[edges < 0.01] = 0
    edges[edges > 0.1] = 0.1
    # Thresholding
    markers = np.zeros_like(image_blue)
    foreground, background = 1, 2
    markers[image_blue < background_bound] = background
    markers[image_blue > foreground_bound] = foreground
    # 1st watershed
    ws = watershed(edges, markers)
    # Distance transform to get local maxima
    to_dist = (ws == 1)
    distance = ndi.distance_transform_edt(to_dist)
    coords = peak_local_max(ndi.distance_transform_edt(to_dist), min_distance = min_distance,
                            footprint=np.ones((3, 3)),
                            labels=to_dist)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    # 2nd watershed
    ws2 = watershed(edges, markers, mask=to_dist)
    # Label
    seg1 = label(ws2)
    return seg1

# wrapper around segment_from_image
# returns image parameters, input image names and the corresponding labelled images
def segment_from_file(file_to_segment,min_distance=80):
    # Read file
    image = imread(fname = file_to_segment)
    # Use blue channel to segment
    image[:,:,0] = 0
    image[:,:,1] = 0
    image_blue = rgb2gray(image)
    image_gray = rgb2gray(imread(fname = file_to_segment))
    # threshold
    thresholds = threshold_multiotsu(image_blue)
    threshold = threshold_otsu(image_blue)
    
    # Segment image
    seg1 = segment_from_image(image_blue,
                              background_bound=threshold,
                              foreground_bound=thresholds[1],
                              min_distance=min_distance)
    # Color labelled image
    color1 = label2rgb(seg1, image=image_gray, bg_label=0)
    
    to_return = pd.DataFrame(regionprops_table(seg1, intensity_image = image_gray, properties=('intensity_image', 'image', 'mean_intensity', 'area', 'convex_area', 'solidity', 'eccentricity')))
    to_return["intensity_image_flat"] = to_return["intensity_image"].map(lambda x: np.array([a for b in x for a in b]))
    to_return["image_flat"] = to_return["image"].map(lambda x: np.array([a for b in x for a in b]))
    to_return["intensities"] = to_return.apply(lambda x: x["intensity_image_flat"][x["image_flat"]], axis=1)
    to_return["var"] = to_return["intensities"].map(lambda x: x.var())
    to_return["fano"] = to_return["var"] / to_return["mean_intensity"]
    to_return["file_name"] = pd.Series([file_to_segment.split("/")[1] for i in range(len(to_return))])
    return seg1, color1, to_return[["mean_intensity", "area", "solidity", "eccentricity", "var", "fano", "file_name"]]

# entry point for optical image analysis
# returns time taken
# rest of the outputs written to disk
def analyse_optical_images(user_upload_dir, analysis_type, output_dir, dpi=300):
    start_time = time.time()
    # Segment files, output images
    output_images=[]
    output_df=[]
    input_image_paths = sorted(glob.glob(user_upload_dir + '/' + analysis_type + "/*.bmp"))
    if len(input_image_paths)==0:
        print(user_upload_dir)
        raise f"Error: there are no .bmp files in specified input folder. Check base or sample condition directories."
    for input_image_path in input_image_paths:
        seg1, color_labelled, subdf = segment_from_file(input_image_path)
        height, width, depth = color_labelled.shape
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(color_labelled)
        ax.axis("off")
        for region in regionprops(seg1):
            if (region.area > 8000) and (region.area < 20000):
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)
                ax.annotate(region.label-1, (minc, minr), ha = "left", va = "bottom", color = "white")
        output_images.append(fig)
        output_df.append(subdf)
    parameters_df = pd.concat(output_df).assign(condition=analysis_type).reset_index(drop=True)
    figsavepath=f'{output_dir}/{analysis_type}'
    if not os.path.exists(figsavepath):
        os.makedirs(figsavepath, exist_ok=True)
    for input_image_path,fig in zip(input_image_paths,output_images):
        fig.savefig(f'{figsavepath}/{os.path.basename(input_image_path)[:-4]}.png')
    parameters_df.to_csv(f'{output_dir}/{analysis_type}/{analysis_type}.csv')
    return time.time()-start_time, input_image_paths

# SEM image analysis functions

# step 1: load images given folder containing *tif files
def load_image_object(image_dir):
    image_paths = sorted(glob.glob(image_dir + '/*.tif'))
    return [cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# step 2: 
def get_gradient(img):
    # find gradient
    ksize = 3 #if args["scharr"] > 0 else 3
    gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    # the gradient magnitude images are now of the floating point data
    # type, so we need to take care to convert them back a to unsigned
    # 8-bit integer representation so other OpenCV functions can operate
    # on them and visualize them
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    # combine the gradient representations into a single image
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    
    return combined


def get_contours(gradient):
    markers = np.zeros_like(gradient)
    markers[gradient >= threshold_otsu(gradient)] = 1
    markers[gradient < threshold_otsu(gradient)] = 2
    markers = watershed(gradient, markers)

    markers = 2-markers

    # denoise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    dilated = cv2.dilate(markers.astype(np.uint8), kernel)

    contours, hierarchy = cv2.findContours(markers.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    return contours, hierarchy


def get_ellipse(img, contours):
    rboxes = []
    for cont in contours:
        if cv2.arcLength(cont, True) > 100 and len(cont) > 100:
            rbox = cv2.fitEllipseAMS(cont)
            if rbox[1][1] > 80 and max(rbox[1])/min(rbox[1]) < 1.5:
                rbox = list(rbox)
                rbox[1] = (rbox[1][0]*0.9, rbox[1][1]*0.9)
                rboxes.append(rbox)
            # cv2.ellipse(img, rbox, (255,100,255), 2, cv2.LINE_AA)
    # plt.figure()
    # plt.imshow(img)
    return rboxes

def filter_ellipse(img, rboxes):
    similar = set()
    for i in range(len(rboxes)-1):
        mask1 = np.zeros_like(img)
        cv2.ellipse(mask1, rboxes[i], (255,255,255), -1, cv2.LINE_AA)
        for j in range(i+1, len(rboxes)):
            a,b = rboxes[i],rboxes[j]
            if np.sqrt((a[0][0]-b[0][0])**2 + (a[0][1]-b[0][1])**2) < 0.2 * min(a[1][0],a[1][1],b[1][0],b[1][1]):
                similar.add(tuple(sorted([i,j])))
            else:
                mask2 = np.zeros_like(img)
                cv2.ellipse(mask2, rboxes[j], (255,255,255), -1, cv2.LINE_AA)
                sum1 = np.sum(mask1 > 0)
                sum2 = np.sum(mask2 > 0)
                sum3 = np.sum(mask1*mask2 > 0)
                if sum3 > min(sum1, sum2)*0.6:
                    similar.add((i,j))
    # remove overlapping
    # print(similar)
    selected = []
    used = set()
    while len(similar) > 0:
        for i in similar:
            a,b = rboxes[i[0]], rboxes[i[1]]
            if max(a[1]) / min(a[1]) <= max(b[1])/min(b[1]):
                selected.append(i[0])
            else:
                selected.append(i[1])
            used.add(i[0])
            used.add(i[1])
        
        selected = sorted(set(selected))
        new_similar = set()
        retry = set()
        for idx1 in range(len(selected)-1):
            for idx2 in range(idx1, len(selected)):
                if (selected[idx1], selected[idx2]) in similar:
                    new_similar.add((selected[idx1], selected[idx2]))
                    retry.add(idx1)
                    retry.add(idx2)
        
        similar = new_similar
        selected = [selected[i] for i in range(len(selected)) if i not in retry]
    for i in range(len(rboxes)):
        if i not in used:
            selected.append(i)
    rboxes = [rboxes[i] for i in selected]
    
    # size filtering
    sizes = []
    for rbox in rboxes:
        sizes.append(rbox[1][1])
    sizes = (sizes-np.mean(sizes))/np.std(sizes)
    rboxes = [rboxes[i] for i in range(len(sizes)) if np.abs(sizes[i])<3]
    
    return rboxes

class Predictor :
    def train( self, img ):
        self.em = cv2.ml.EM_create()
        self.em.setClustersNumber( 3 )
        self.em.setTermCriteria( ( cv2.TERM_CRITERIA_COUNT,4,0 ) )
        samples = np.reshape( img, (img.shape[0]*img.shape[1], -1) ).astype('float')
        self.em.trainEM( samples )

    def predict( self, img ):
        samples = np.reshape( img, (img.shape[0]*img.shape[1], -1) ).astype('float')
        labels = np.zeros( samples.shape, 'uint8' )
        for i in range ( samples.shape[0] ):
            retval, probs = self.em.predict2( samples[i] )
            labels[i] = retval[1] # make it [0,255] for imshow
        return np.reshape( labels, img.shape )

def process_one_sem_image(oimg, vimg = None, fig_outpath=None, dpi=300):
    img = oimg.copy()
    med_img = cv2.medianBlur(img, ksize=5)

    combined = get_gradient(med_img)
    
    contours, hierarchy = get_contours(combined)

    rboxes = get_ellipse(img, contours)
    rboxes = filter_ellipse(img, rboxes)
    
    # classify pixels
    # gmm = GaussianMixture(n_components=3)
    gmm = Predictor()
    
    img = oimg.copy() if vimg is None else vimg.copy()
    v_offset = int(img.shape[0] * 0.2)
    h_offset = int(img.shape[1] * 0.2)
    # med_img = cv2.medianBlur(img, ksize=5)
    gmm.train(img[v_offset:-v_offset, h_offset:-h_offset].reshape((-1,1)))
    labels = gmm.predict(img.reshape((-1,1)))
    labels = labels.reshape(img.shape)

    label_values = []
    if vimg is None:
        for l in np.unique(labels):
            label_values.append([img[labels==l].mean(), l])
    else:
        mask = np.zeros_like(img)
        for rbox in rboxes:                
            cv2.ellipse(mask, rbox, (255,255,255), -1, cv2.LINE_AA)
        for l, c in zip(*np.unique(labels[mask>0], return_counts=True)):
            label_values.append([c, l])
    label_values.sort(key=lambda k:-k[0])
    target_label = label_values[0][1]
    
    # print(target_label, len(rboxes), label_values)
    # img *= 120
    output = []
    variances = []
    means = []
    final_rboxes = []
    for rbox in rboxes:
        mask = np.zeros_like(img)
        cv2.ellipse(mask, rbox, (255,255,255), -1, cv2.LINE_AA)

        ratio = (mask>0).sum() / (np.pi * rbox[1][0] * rbox[1][1] / 4)
        value = (labels[mask > 0] == target_label).sum() / labels[mask>0].shape[0]
        mean_intensity = np.mean(img[mask>0])
        var = np.var(img[mask>0])
        if ratio > 0.8 and (vimg is not None or value > 0.2):
            output.append(value)
            means.append(mean_intensity)
            variances.append(var)
            final_rboxes.append(rbox)
            cv2.ellipse(img, rbox, (255,100,255), 2, cv2.LINE_AA)

    fig = plt.figure()
    plt.imshow(img, cmap='Greys_r', vmin=0, vmax=255)
    for v, e in zip(output, final_rboxes):
        plt.text(e[0][0],e[0][1],f'{v:.2f}', color='red')
    plt.title(f'{np.mean(output):.3f}')
    if fig_outpath is not None:
        fig.savefig(fig_outpath, dpi=dpi)
        plt.cla()
        plt.close()
    out_df = pd.DataFrame({
        'homogeneity': output,
        'mean_intensity': means,
        'var_intensity': variances
    })
    return out_df

def process_sem_folder(user_upload_dir,analysis_type,output_dir):
    start_time = time.time()
    image_dir = f'{user_upload_dir}/{analysis_type}'
    imgs = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    cimgs = sorted(glob.glob(os.path.join(image_dir, '*.bmp')))
    imgs = imgs + cimgs
    output_df = []

    if not os.path.exists(f'{output_dir}/{analysis_type}'):
        os.mkdir(f'{output_dir}/{analysis_type}')

    for img_path in imgs:
        print(img_path)
        img_base_name = os.path.basename(img_path)
        fig_outpath = os.path.join(output_dir,analysis_type,img_base_name)
        fig_outpath = fig_outpath[:-4] + '.png'

        if img_path.endswith('bmp'):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            df = process_one_sem_image(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2], vimg=img[:, :, 2].copy(),
                                       fig_outpath=fig_outpath, dpi=300)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            df = process_one_sem_image(img, fig_outpath=fig_outpath, dpi=300)
        # print(img.shape)
        output_df.append(df)

    pd.concat(output_df).reset_index(drop=True).to_csv(f'{output_dir}/{analysis_type}/{analysis_type}.csv',index=False)
    return time.time() - start_time, imgs