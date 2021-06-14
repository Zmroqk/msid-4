# Introduction
Our goal in this application is to create two models for fashion-mnist dataset which can be found <a href="https://github.com/zalandoresearch/fashion-mnist">here</a>.
In this repository you can find application that can learn 2 models:
  <li> KNN (k-nearest neighbors)
  <li> Neural network (6 layers)

# Methods
### KNN model:
To learn this model we create gabor filters that we use to filter out features from the image. It creates us a new set of images we we save and use later in traing a model.
To train a model we used sklearn module. We have to find the best filter and the best k value.
### Neural network:
This time we used keras library to create a 6 - layer neural network. In this case we don't have to extract features from images we can just put our mnist-dateset into it.
    
# Results
<table>
  <tr>
    <th>Model</th>
    <th>Time to learn*</th>
    <th>Accuracity</th>
  </tr>
  <tr>
    <td>KNN</td>
    <td>~15s * Number of filters * Number of K-values</td>
    <td>0.8511</td>
  </tr>
  <tr>
    <td>Neural network</td>
    <td>~60-80s for an epoch</td>
    <td>0.9287</td>
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
      <li> <b>Launch app</b> - This will switch application to mode where you can provide 28x28 Images to predict category using both models (requires to have trained both models)
  </ol>
</ol>
