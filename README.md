# Introduction
Our goal in this application is to create two models for fashion-mnist dataset which can be found <a href="https://github.com/zalandoresearch/fashion-mnist">here</a>.
In this repository you can find application that can learn 5 models:
  <li> KNN with gabor
  <li> KNN without gabor
  <li> Neural network (3 x Dense)
  <li> Neural network (Conv + Pooling + Dropout + Dense)
  <li> Neural network (Conv + Pooling + Dropout + Conv)

# Methods
### KNN model:
To learn this model we create gabor filters that we use to filter out features from the image. It creates us a new set of images we we save and use later in traing a model.
To train a model we used sklearn module. We have to find the best filter and the best k value.
### Neural network:
This time we used keras library to create multiple neural network models. In this case we don't have to extract features from images we can just put our mnist-dateset into it.
    
# Results
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
    <td>(None, 9800)</td>
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

### Summary:
<table>
  <tr>
    <th>Model</th>
    <th>Time to learn*</th>
    <th>Accuracity</th>
    <th>Model size</th>
  </tr>
  <tr>
    <td>KNN without gabor</td>
    <td>~15s * Number of K-values</td>
    <td>0.8495</td>
    <td>45,32MB</td>
  </tr>
  <tr>
    <td>KNN with gabor</td>
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
