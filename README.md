# Image analysis dashboard
Full stack web application converted into a desktop application. The data dashboard is built with dash and backend is served using flask. Electron is used to convert it 

# Summary
Our team wanted to make something useful for the reagent technicians to analyse the coating efficiency of microbeads (dry reagent particles, a bit like instant coffee granules, except they need to be coated with a protective inert substance)

Our technicians use optical microscope and scanning electron microscope to do quality control of the coating process.

However, the technicians lack a way to efficiently quantify the coating efficiency of hundreds of microbeads images. Especially when the process becomes more efficient, it becomes hard to tell by the human eye whether there is an improvement.

We produce a desktop application that can annotate/analyse microbeads images, compare the coating homogeneity across experimental conditions, and exports results in plots and csv files.

This windows application is lightweight (100mb), and uses scikit learn algorithms to analyse both optical microscope images (colored) and scanning electron microscope (SEM) images.

# Roles 
* Guo-Liang: wrote the original version of the image analysis algorithm
* Johan: team lead
* Cao Fan: wrote a newer version of the image analysis algorithm
* __Me__: developed the user interface of our desktop application with dash/plotly
* Daniel: guided and provided Jia Geng with the dash/plotly template, presenter
* Claire: collected imaging samples and relayed it to us

# Demo 1: Analysis of optical microscope images 
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-optical-demo.gif)


# Demo 2: Comparison of experiments, optical microscopy
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-compare-optical-demo.gif)


# Demo 3: Analysis of SEM images
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-sem-demo.gif)
