import math
import pandas as pd
from kolmogorov_gabor_polinomial import KolmogorovGaborPolynomial

class PolynomialNetwork:
    """
    Class for constructing and using a polynomial network.

    The polynomial network consists of two layers:
      1. The first layer contains several polynomial models:
         - The first polynomial is trained on all columns of the input DataFrame.
         - The remaining polynomials are trained on groups of two columns.
      2. The second layer consists of a single polynomial model trained on the outputs of the first layer.

    Attributes:
        architecture (dict): A dictionary defining the network architecture in the format {layer_number: number_of_polynomials}.
        outputs (dict): A dictionary storing the outputs of the polynomials with keys formatted as "layer_polynomial".
        models (dict): A dictionary of trained models with keys formatted as "layer_polynomial".
        predicts (dict): A dictionary to store intermediate predictions during the predict method.
    """
    def __init__(self):
        """
        Initializes an instance of the PolynomialNetwork class.

        Creates empty dictionaries for:
            - the network architecture,
            - the outputs of the polynomials,
            - the trained models,
            - intermediate predictions.
        """
        self.architecture = {}  # {layer number: number of polynomials in that layer}
        self.outputs = {}       # {"layer_polynomial": predictions after training}
        self.models = {}        # {"layer_polynomial": instance of KolmogorovGaborPolynomial}
        self.predicts = {}      # {"layer_polynomial": predictions during the predict method}

    def fit(self, X, Y):
        """
        Trains the polynomial network on the input features X and target variable Y.

        Training steps:
          1. Define the network architecture:
             - First layer: number of polynomials = ceil(number_of_features / 2) + 1.
             - Second layer: 1 polynomial.
          2. Train the first layer polynomials:
             - The first polynomial is trained on all features.
             - The remaining polynomials are trained on subsets of features, grouping them in pairs.
          3. Train the second layer polynomial:
             - The second layer is trained on the outputs of the first layer.

        Args:
            X (pandas.DataFrame): DataFrame containing the input features.
            Y (pandas.Series or pandas.DataFrame): The target variable.
        """
        num_columns = X.shape[1]  # Determine the number of features

        # Define the network architecture:
        # First layer: ceil(number_of_features / 2) + 1 polynomials.
        self.architecture[1] = math.ceil(num_columns / 2) + 1  
        # Second layer: 1 polynomial.
        self.architecture[2] = 1  

        # Train polynomials in the first layer.
        for p in range(1, self.architecture[1] + 1):
            if p == 1:
                # The first polynomial is trained on all features.
                current_data = X
            else:
                # The remaining polynomials are trained on groups of 2 features.
                # Determine the starting and ending indices for slicing the columns.
                start_idx = (p - 2) * 2
                end_idx = min(start_idx + 2, num_columns)
                current_data = X.iloc[:, start_idx:end_idx]

            # Create an instance of the polynomial model.
            polynomial = KolmogorovGaborPolynomial()
            # Train the model on the selected features and the target variable.
            polynomial.fit(current_data, Y)
            # Get predictions from the trained model on the training data.
            predictions = polynomial.predict(current_data)

            # Create a key to store the model and its predictions (e.g., "1_1", "1_2", etc.).
            key = f"1_{p}"
            # Convert predictions to a one-dimensional array and store as a Series with Y's index.
            self.outputs[key] = pd.Series(predictions.to_numpy().ravel(), index=Y.index)
            # Store the trained model.
            self.models[key] = polynomial

        # Train the second layer.
        # Form the input for the second layer by concatenating the outputs of all first layer polynomials.
        layer_2_input = pd.concat(
            [self.outputs[f"1_{p}"] for p in range(1, self.architecture[1] + 1)],
            axis=1
        )
        # Create and train the polynomial model for the second layer.
        polynomial = KolmogorovGaborPolynomial()
        polynomial.fit(layer_2_input, Y)
        predictions = polynomial.predict(layer_2_input)

        key = "2_1"  # Key for the second layer model.
        # Convert predictions to a one-dimensional array and store them.
        self.outputs[key] = pd.Series(predictions.to_numpy().ravel(), index=Y.index)
        self.models[key] = polynomial  # Save the second layer model.

    def predict(self, X):
        """
        Makes predictions for new input data X using the trained polynomial network.

        Prediction steps:
          1. Obtain predictions from the first layer polynomials:
             - The first polynomial uses all features.
             - The remaining polynomials use groups of 2 features.
          2. Form the input for the second layer by concatenating the first layer predictions.
          3. Obtain the final predictions from the second layer model.

        Args:
            X (pandas.DataFrame): DataFrame containing the new input features.

        Returns:
            pandas.Series: The final predictions as a one-dimensional Series with the same index as X.
        """
        self.predicts = {}  # Reset the temporary predictions dictionary.
        num_columns = X.shape[1]  # Determine the number of features in the input data.

        # Obtain predictions from the first layer.
        for p in range(1, self.architecture[1] + 1):
            if p == 1:
                # The first polynomial uses all features.
                current_data = X
            else:
                # The remaining polynomials use groups of 2 features.
                start_idx = (p - 2) * 2
                end_idx = min(start_idx + 2, num_columns)
                current_data = X.iloc[:, start_idx:end_idx]

            key = f"1_{p}"  # Key corresponding to the first layer model.
            polynomial = self.models[key]  # Retrieve the trained model.
            predictions = polynomial.predict(current_data)  # Get predictions.

            # Convert predictions to a one-dimensional array and store as a Series with X's index.
            self.predicts[key] = pd.Series(predictions.to_numpy().ravel(), index=X.index)

        # Form the input for the second layer by concatenating all first layer predictions.
        layer_2_input = pd.concat(
            [self.predicts[f"1_{p}"] for p in range(1, self.architecture[1] + 1)],
            axis=1
        )
        key = "2_1"  # Key for the second layer model.
        polynomial = self.models[key]  # Retrieve the second layer model.
        final_predictions = polynomial.predict(layer_2_input)  # Get the final predictions.

        # Return the final predictions as a one-dimensional Series with the same index as the input data.
        return pd.Series(final_predictions.to_numpy().ravel(), index=X.index)


# Print a message indicating that the Polynomial Network class has been successfully added.
print('Polynomial network class added successfully!')
