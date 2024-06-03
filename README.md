# Micro-Image
Image analysis project for an internal company hackathon in December 2022.

# Summary
Our team wanted to make something useful for the microbead technicians to analyse coating effeciency of reagent microbeads.

Our technicians use optical microscope and scanning electron microscope to do quality control of their coating efficiency.

However, the technicians lacked a way to measure quantitatively coating efficiency across hundreds of microbead snapshots.

We produce a desktop application that can analyse microbead images for them, compare coating efficiency across experimental conditions, and save results in plots and csv files.

This desktop application is lightweight (100mb), and uses scikit learn algorithms to analyse both optical microscope images (colored) and scanning electron microscope (SEM) images.

This internal tool is currently in use by our imaging technicians over at Hayward, Cambridge, United Kingdom.

# Roles 
Cao Fan (main developer): wrote the core plotting function (the `.ipynb`)

Jia Geng (main developer): develoepd the user interface of our desktop application

Daniel Martana (mentor developer): guided and provided Jia Geng with the dash/plotly template, presenter

Claire Tang (data provider): provided us with imaging samples

Johan Basuki: team lead

# Results
We won third place among 15 teams!

# Demo 1: Analysis of optical microscope images 
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-optical-demo.gif)





# Demo 2: Comparison of experiments, optical microscopy
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-compare-optical-demo.gif)





# Demo 3: Analysis of SEM images
![](https://github.com/JiaGengChang/microimage/blob/main/micro-image-sem-demo.gif)

