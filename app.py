import pickle
import pandas as pd

# Load the best model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_salary(input_data: pd.DataFrame):
    """Predicts salary based on input features."""
    # Ensure input_data has the same columns and order as X_train
    # (This is a simplified example; actual preprocessing for new data
    #  would need to replicate the training pipeline, including LabelEncoding
    #  with the same encoders used during training for categorical features).
    
    # For demonstration, assuming input_data is already preprocessed
    # and matches the format expected by the model.
    
    prediction = model.predict(input_data)
    return prediction

if __name__ == '__main__':
    print("app.py created successfully. You can import 'predict_salary' or run this file for testing.")
    # Example usage (this part would usually be in a separate test script or API endpoint)
    # from sklearn.preprocessing import LabelEncoder

    # Create a dummy DataFrame matching the structure of X used for training
    # This assumes you have the original label encoders for each categorical column.
    # For a real application, these encoders would need to be saved and loaded.
    # For this example, let's assume `df` from the notebook is available for structure.
    
    # You would need to provide actual preprocessed input data here.
    # dummy_input = pd.DataFrame([[28, 8129, 28, 2, 0, 1, 0]], 
    #                          columns=['Rating', 'Company Name', 'Job Title', 
    #                                   'Salaries Reported', 'Location', 
    #                                   'Employment Status', 'Job Roles'])
    # 
    # predicted_salary = predict_salary(dummy_input)
    # print(f"Predicted Salary: {predicted_salary[0]}")
