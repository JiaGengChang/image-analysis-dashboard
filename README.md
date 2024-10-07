# Python dashboard
Image analysis project for an internal company hackathon in December 2022.

# Summary
Our team wanted to make something useful for the reagent technicians to analyse the coating efficiency of microbeads (dry reagent particles, a bit like instant coffee granules, except they need to be coated with a protective inert substance)

Our technicians use optical microscope and scanning electron microscope to do quality control of the coating process.

However, the technicians lack a way to efficiently quantify the coating efficiency of hundreds of microbeads images. Especially when the process becomes more efficient, it becomes hard to tell by the human eye whether there is an improvement.

We produce a desktop application that can annotate/analyse microbeads images, compare the coating homogeneity across experimental conditions, and exports results in plots and csv files.

This windows application is lightweight (100mb), and uses scikit learn algorithms to analyse both optical microscope images (colored) and scanning electron microscope (SEM) images.

# Roles 
Guo-Liang: wrote the original version of the image analysis algorithm
Cao Fan: wrote a newer version of the core plotting function (the `.ipynb`) with scikit-learn
__Me__ (__Jia Geng__, data dashboard developer): developed the user interface of our desktop application with dash/plotly
Daniel: guided and provided Jia Geng with the dash/plotly template, presenter
Claire: collected imaging samples and relayed it to us developers
Johan: team lead

# Demo 1: Analysis of optical microscope images 
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-optical-demo.gif)


# Demo 2: Comparison of experiments, optical microscopy
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-compare-optical-demo.gif)


# Demo 3: Analysis of SEM images
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-sem-demo.gif)

# Results

We won third place among 15 teams!

This internal tool is currently in use by imaging technicians over in the United Kingdom.
