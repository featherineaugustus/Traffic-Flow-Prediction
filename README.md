# Traffic-Flow-Prediction
Using government available public data of CCTV cameras, predict the traffic flow at a particular road at a given time.

We extract the images of the CCTV camera and deployed keras_ocr to remove words from the images, and YOLOv4 to detect the number of vehicles at different time instances.

Next, we trained linear regression models with different polynominal degress to best fit the number of vehicles against time (in hour 0-23).

Also, we develop a best fit line table based on the mean number of vehicles at any given hour to predict the number of vehicles given a specific hour (0-23).

The best performance is apparently the best fit line table.
