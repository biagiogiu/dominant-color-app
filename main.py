from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import flask
from flask import Flask, render_template, redirect, url_for, request, flash
import os

import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans, vq
import pandas as pd

app = Flask(__name__, static_url_path='', static_folder='static')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def find_frequent_colors(uploaded_img):
    # resize image to speed up process
    pil_img = Image.open(uploaded_img).convert('RGB')
    newsize = (250, pil_img.height * 250 // pil_img.width)
    resized_img = pil_img.resize(newsize)

    # convert to nparray
    py_img = np.asarray(resized_img)

    # Reshape the image to a 2D array of pixels
    pixels = py_img.reshape(-1, 3)

    # Use KMeans to divide pixel colors in 10 clusters
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_clusters_indices = np.argsort(counts)[-10:]

    # find the pixel at the centre of each cluster
    cluster_centroids = []
    for cluster_index in largest_clusters_indices:
        cluster_mask = labels.reshape(py_img.shape[:2]) == cluster_index
        cluster_coords = np.column_stack(np.where(cluster_mask))
        centroid = cluster_coords.mean(axis=0)
        cluster_centroids.append(centroid)

    # find the RGB values of the centre pixels
    rgb_values = []
    for centroid in cluster_centroids:
        x, y = int(centroid[0]), int(centroid[1])
        rgb = py_img[x, y]
        rgb_values.append(rgb)
    return [('#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])) for rgb in rgb_values]

    # # resize image to speed up process
    # pil_img = Image.open(uploaded_img).convert('RGB')
    # newsize = (250, pil_img.height * 250 // pil_img.width)
    # resized_img = pil_img.resize(newsize)
    #
    # # convert to nparray
    # py_img = np.asarray(resized_img)
    # mpl_image = py_img
    # # mpl_image = img.imread(uploaded_img)
    # r = []
    # g = []
    # b = []
    # for row in mpl_image:
    #     for temp_r, temp_g, temp_b in row:
    #         r.append(temp_r)
    #         g.append(temp_g)
    #         b.append(temp_b)
    #
    # mpl_df = pd.DataFrame({'red': r, 'green': g, 'blue': b})
    #
    # mpl_df['scaled_color_red'] = whiten(mpl_df['red'])
    # mpl_df['scaled_color_blue'] = whiten(mpl_df['blue'])
    # mpl_df['scaled_color_green'] = whiten(mpl_df['green'])
    #
    # cluster_centers, _ = kmeans(mpl_df[['scaled_color_red',
    #                                     'scaled_color_blue',
    #                                     'scaled_color_green']], 10)
    #
    # cluster_labels, _ = vq(mpl_df[['scaled_color_red',
    #                                'scaled_color_blue',
    #                                'scaled_color_green']], cluster_centers)
    #
    # cluster_sizes = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)
    # dominant_colors = []
    #
    # red_std, green_std, blue_std = mpl_df[['red',
    #                                        'green',
    #                                        'blue']].std()
    #
    # for cluster_index in cluster_sizes.index:
    #     cluster_center = cluster_centers[cluster_index]
    #     red_scaled, green_scaled, blue_scaled = cluster_center
    #     dominant_colors.append((
    #         int(red_scaled * red_std),
    #         int(green_scaled * green_std),
    #         int(blue_scaled * blue_std)
    #     ))
    # return [('#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])) for rgb in dominant_colors]


@app.route("/", methods=['GET', 'POST'])
def home():
    if flask.request.method == 'POST':
        if 'imgFile' not in request.files:
            return 'No file part', 400
        file = request.files['imgFile']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filename = file.filename
            if filename.split('.')[-1] in ['jpg', 'jpeg']:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                uploaded_img = f"static/uploads/{filename}"
                color_hex_codes = find_frequent_colors(uploaded_img)
                return render_template("home.html", uploaded_image=f"uploads/{filename}", frequent_colors=color_hex_codes)
            else:
                return 'Please upload an image file', 400
    else:
        default_img = 'static/images/little_manuel.jpg'
        color_hex_codes = find_frequent_colors(default_img)

        return render_template("home.html", uploaded_image="images/little_manuel.jpg", frequent_colors=color_hex_codes)


if __name__ == '__main__':
    app.run(debug=True, port=5000)