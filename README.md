# Introduction
Our goal in this application is to create models for fashion-mnist dataset which can be found <a href="https://github.com/zalandoresearch/fashion-mnist">here</a>.
In this repository you can find application that can learn 5 models:
  <li> KNN with gabor
  <li> KNN without gabor
  <li> Neural network (3 x Dense)
  <li> Neural network (Conv + Pooling + Dropout + Dense)
  <li> Neural network (Conv + Pooling + Dropout + Conv)

# Methods
#### Features extraction:
  <li> Gabor
    
#### Models:
  <li> KNearestNeighbours
  <li> Convolutional-Dense Neural Network
  <li> Convolutional Neural Network
  <li> Dense Neural Network

### Neural network (3 x dense):
<table>
  <tr>
    <th>Layer</th>
    <th>Output Shape</th>
    <th>Param</th>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 1000)</td>
    <td>785000</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 1000)</td>
    <td>1001000</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 500)</td>
    <td>500500</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 10)</td>
    <td>5010</td>
  </tr>
</table>
    
### Neural network (Conv + Pooling + Dropout + Dense):
<table>
  <tr>
    <th>Layer</th>
    <th>Output Shape</th>
    <th>Param</th>
  </tr>
  <tr>
    <td>Conv2D</td>
    <td>(None, 28, 28, 50)</td>
    <td>1300</td>
  </tr>
  <tr>
    <td>MaxPooling2D</td>
    <td>(None, 14, 14, 50)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Dropout</td>
    <td>(None, 14, 14, 50)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>(None, 9800)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 1000)</td>
    <td>9801000</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 10)</td>
    <td>10010</td>
  </tr>
</table>
    
### Neural network (Conv + Pooling + Dropout + Conv):
<table>
  <tr>
    <th>Layer</th>
    <th>Output Shape</th>
    <th>Param</th>
  </tr>
  <tr>
    <td>Conv2D</td>
    <td>(None, 28, 28, 50)</td>
    <td>1300</td>
  </tr>
  <tr>
    <td>MaxPooling2D</td>
    <td>(None, 14, 14, 50)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Dropout</td>
    <td>(None, 14, 14, 50)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Conv2D</td>
    <td>(None, 14, 14, 50)</td>
    <td>22550</td>
  </tr>
  <tr>
    <td>Flatten</td>
    <td>(None, 9800)</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Dense</td>
    <td>(None, 10)</td>
    <td>10010</td>
  </tr>
</table>    
    
# Results
<table>
  <tr>
    <th>Model</th>
    <th>Time to learn*</th>
    <th>Accuracity</th>
    <th>Model size</th>
  </tr>
  <tr>
    <td>KNN without gabor (k=11)</td>
    <td>~15s * Number of K-values</td>
    <td>0.8495</td>
    <td>45,32MB</td>
  </tr>
  <tr>
    <td>KNN with gabor (k=11)</td>
    <td>~15s * Number of filters * Number of K-values</td>
    <td>0.8511</td>
    <td>45,32MB</td>
  </tr>  
  <tr>
    <td>Neural network (3 x Dense)</td>
    <td>~13s for an epoch</td>
    <td>0.8884</td>
    <td>26,26MB</td>
  </tr>
  <tr>
    <td>Neural network (Conv + Pooling + Dropout + Conv)</td>
    <td>~20s for an epoch</td>
    <td>0.9238</td>
    <td>1,43MB</td>
  </tr>
  <tr>
    <td>Neural network (Conv + Pooling + Dropout + Dense)</td>
    <td>~60-80s for an epoch</td>
    <td>0.9287</td>
    <td>112,33MB</td>
  </tr>
</table>
*On i9-9900K CPU
    
### My models vs Selected Zalando Research Benchmark:
<table>
  <tr>
    <th>My Model</th>
    <th>Zalando Model</th>
    <th>Time to learn (my model)</th>
    <th>Time to learn (zalando model)</th>
    <th>Accuracity (my model)</th>
    <th>Accuracity (zalando model)</th>
  </tr>
  <tr>
    <td>KNN with gabor (k=11)</td>
    <td>KNeighborsClassifier	{"n_neighbors":5,"p":1,"weights":"distance"}</td>
    <td>~48 min</td>
    <td>~42 min</td>
    <td>0.8511</td>
    <td>0.860</td>
  </tr>
  <tr>
    <td>Neural network (Conv + Pooling + Dropout + Conv)</td>
    <td>2 Conv + Pooling</td>
    <td>~4 min</td>
    <td>-------</td>
    <td>0.9238</td>
    <td>0.916</td>
  </tr>
</table>
    
# Usage
<ol>
  <li>Run <code>python3 App.py</code>
  <li>Choose option in application you want:
  <ol>
    <li> <b>Learn models</b> - Use this to learn one of the models
      <li> <b>Generate filters for fashion mnist images</b> - Use this to apply gabor filters to mnist images
      <li> <b>Test accuracy</b> - Use this to test accuracy of learned models (requires to have trained both models)
      <li> <b>Launch app</b> - This will switch application to mode where you can provide 28x28 Images to predict category using one of the models
  </ol>
</ol>
    
# Used modules:
<li> scikit_image==0.18.1
<li> numpy==1.19.5
<li> opencv_python==4.5.2.54
<li> keras_nightly==2.5.0.dev2021032900
<li> tensorflow==2.5.0
<li> pandas==1.2.4
<li> keras==2.4.3
<li> scikit_learn==0.24.2
<li> skimage==0.0

