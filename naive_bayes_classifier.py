import pandas as pd
import math

class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self, continuous: list):
        """
        :param continuous: list containing a bool for each feature column in the input data. True if a feature column
                           contains a continuous feature, False if discrete.
        """
        self.continuous = continuous


    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities. Depending on self.continuous the
        features can be treated with the respective algorithm.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """

        # First created empty dictionaries
        self.class_probs = {}       # Stores prior probabilities of classes (P(Class))
        self.gaussian_params = {}   # Stores mean and std for continuous features
        self.feature_probs = {}     # Stores probability of discrete values given class

        # Separate feature and Target
        # Drop the label column to get features and store target separately
        feature = data.drop(columns=[target_name])
        target=data[target_name]

        # Calculate prior probabilities(Classes : True & False), Probability of classes is saved in dictionary
        self.class_probs = target.value_counts(normalize=True).to_dict()

        # Train feature probabilities per class
        # keys[True,False] go in for loop one by one through classes variable
        for classes in self.class_probs.keys():
            print('\nReturn class : ', classes)

            # Index based selection. Pandas uses that Boolean Series to filter rows from feature.
            # feature and target come from the same DataFrame and share the same index numbers
            # For example, if classes = True, it picks all rows where target == True.
            class_data = feature[target == classes]
            print('Print Feature table : \n', class_data.head(10))

            # Initialize dictionaries for this class
            self.gaussian_params[classes] = {}
            self.feature_probs[classes] = {}

            # Columns from feature go inside the loop one by one
            for i, column in enumerate(feature.columns):

                # If column type is continuous (like numbers) then go inside the if and calculate mean & standard deviation
                # These two values tell the model what is average and how much data spreads around the average for this feature
                if self.continuous[i]:
                    mean = class_data[column].mean()
                    std = class_data[column].std()

                    # Save mean and standard deviation for this column under current class (True/False)
                    self.gaussian_params[classes][column] = (mean, std)

                else:
                    # If column type is discrete (like yes/no), then calculate how many times each value appears
                    # Value_counts(normalize=True) gives percentage (probability) of each value inside the current class
                    value_counts = class_data[column].value_counts(normalize=True)
                    self.feature_probs[classes][column] = value_counts.to_dict()

            print('\n Prior Probabilities : \n', self.class_probs)
            print('\n Gaussian Probabilities (For Continious Type : mean, std): \n', self.gaussian_params)
            print('\n Feature Probabilities (For Discrete Type) : \n', self.feature_probs)

            pass

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted (not containing label columns)
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        # Create an empty list to store final predicted class (True/False) for each test row
        predictions = []


        # Go row by row in the test dataset and calculate class probability for each row
        # Each row from test data will be classified as either True or False (disease or no disease)
        for index, row in data.iterrows():
            print('\nIndex of Rows :', index)
            print('Row :\n', row)

            # Empty dictionary for each row to store total probability for both classes (True/False)
            # Example structure: {True: 0.00123, False: 0.00056}
            class_scores = {}

            # Go through all classes (True and False) one by one
            # Store in variable 'prob', print the class and its starting prior probability
            for c in self.class_probs.keys():
                prob = self.class_probs[c] # Start with prior probability (P(Class))
                print('\nStarting probability : ', c, ' : ', prob)

                # Save the starting probability (prior) of this class in dictionary
                class_scores[c] = prob

                # Now multiply with each feature probability for this row
                for i, column in enumerate(data.columns):
                    value = row[column] # Take the feature value for this row
                    print('\nFeature : ', column, ', Value : ', value)

                    ## If feature is discrete: multiply with P(value | class), if value unseen then skip or use small value
                    if not self.continuous[i]: 
                        if value in self.feature_probs[c][column]:  # Check if that value exists under this class in training
                            prob *= self.feature_probs[c][column][value] # Multiply by conditional probability P(feature=value | class)
                            print('Multiply by P', column, ' = ', value, ' / ', c , '=', self.feature_probs[c][column][value])
                        else:
                            prob *= 1e-6 # If unseen discrete value comes (not in training) → multiply by small value to avoid zero
                            print('Value 1e-6: ', value)
                    else:
                        # For continuous column multiply by tiny number first just to avoid underflow
                        prob *= 1e-6
                        print('Value 1e-6 : ', value)

                    # If continuous feature → calculate Gaussian probability
                    if self.continuous[i]:
                        mean, std = self.gaussian_params[c][column]# fetch mean and std for this column under this class

                        # If std = 0 (means all values same), change to small value to avoid division by zero
                        if std == 0:
                            std = 1e-6

                        # Apply Gaussian probability density function formula
                        # Formula: (1 / sqrt(2πσ²)) * exp(-((x - μ)²) / (2σ²))
                        exponent = math.exp(-((value - mean) ** 2) / (2 * (std ** 2)))
                        prob_density = (1 / (math.sqrt(2 * math.pi) * std)) * exponent
                        prob *= prob_density  # Multiply with current total probability for this class

                        print('Multiply by Gaussian (continuous feature) : ', column, value, c, prob_density)

            # Print total score for both classes for this row
            print('Class score : ', class_scores)  

            # Pick the class with maximum probability → final predicted class for this row
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            print('Predicted class for this row : ', predicted_class)

        # Convert predictions list into DataFrame for easy reading
        results = pd.DataFrame({"Predicted_Class": predictions})
        print('Final prediction : \n', results.head())

        # Return final prediction DataFrame
        return results

        pass

    def evaluate_on_data(self, data: pd.DataFrame, test_labels: str):
        """
        Predicts a test DataFrame (including labels) and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels: str Name of the label column in data
        :return: tuple of overall accuracy and confusion matrix values
        """

        # Split data into features (X_test) and actual labels (y_true)
        X_test = data.drop(columns=[test_labels])
        y_true = data[test_labels]

        # Get predictions for test features
        predictions_df = self.predict_probability(X_test)
        y_predicted = predictions_df["Predicted_Class"]

        # Compare predicted labels with actual labels and count how many are correct
        correct = (y_predicted.reset_index(drop=True) == y_true.reset_index(drop=True)).sum()
        total = len(y_true)
        accuracy = correct / total # overall accuracy in decimal (correct / total)

        # Calculate values for confusion matrix (True Positive, False Positive, etc.)
        True_Positive = ((y_true == True) & (y_predicted == True)).sum()
        True_Negative = ((y_true == False) & (y_predicted == False)).sum()
        False_Positive = ((y_true == False) & (y_predicted == True)).sum()
        False_Negative = ((y_true == True) & (y_predicted == False)).sum()

        # Print final evaluation results on screen
        print('\nEvalueation Results : ')
        print('Accuracy : ', accuracy)
        print('Confusion Matrix ')
        print('True_Positive : ',True_Positive, 'True_Negative : ',True_Negative, 'False_Positive : ',False_Positive, 'False_Negative : ',False_Negative)

        # Return both accuracy and confusion matrix as dictionary
        return accuracy, {"True_Positive": True_Positive, "False_Positive": False_Positive, "True_Negative": True_Negative, "False_Negative": False_Negative}

        pass
