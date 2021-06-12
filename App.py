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
        response = inp.menu(["Learn KNN", "Learn Neural Network"])
        if response == 1:
            lm.LearnKNN()
        elif response == 2:
            lm.LearnNeuralNetwork()
    elif response == 2:
        ggd.GenerateGaborData()
        if inp.ask_y_n("Do you want to learn KNN model? (y or n): "):
            lm.LearnKNN()
    elif response == 3:
        ta.test_models()
    elif response == 4:
        while True:
            app.run_predict()
            if inp.ask_y_n("Do you want to exit application? (y or n): "):
                exit(0)
