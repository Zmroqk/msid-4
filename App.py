import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import learnModel as lm
import generateGaborData as ggd
import testAccuracy as ta
import input as inp
import application as app

if __name__ == '__main__':
    response = inp.menu(["Learn models", "Generate filters for fashion mnist images", "Test accuracy", "Launch app"])
    if response == 1:
        response = inp.menu(["Learn KNN", "Learn KNN without filter",
                             "Learn Neural Network (Conv + Pulling + Dropout + Dense)",
                             "Learn Neural Network (3 x Dense)",
                             "Learn Neural Network (Conv + Pulling + Dropout + Conv)"])
        if response == 1:
            lm.learnKNN()
        if response == 2:
            lm.learnKNN_without_filter()
        elif response == 3:
            lm.learn_neural_network(1)
        elif response == 4:
            lm.learn_neural_network(2)
        elif response == 5:
            lm.learn_neural_network(3)
    elif response == 2:
        ggd.GenerateGaborData()
        if inp.ask_y_n("Do you want to learn KNN model? (y or n): "):
            lm.learnKNN()
    elif response == 3:
        ta.test_models()
    elif response == 4:
        knn_model, model_network, model_knn, model_kernel = app.init_app()
        while True:
            if knn_model:
                app.run_predict_knn(model_knn, model_kernel)
            else:
                app.run_predict_neural(model_network)
            if inp.ask_y_n("Do you want to exit application? (y or n): "):
                exit(0)
