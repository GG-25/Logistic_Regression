#from typing import BinaryIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

#load dataset
data = pd.read_csv('/Users/gayathrigurram/Desktop/logreg/bankmarket/data/bank-full.csv', sep=';')
le_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/label_encoder.pkl'
ohe_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/ohe_encoder.pkl'
model_fpath = '/Users/gayathrigurram/Desktop/logreg/bankmarket/model/log_reg_model.pkl'
#print(data.shape)
#print(data.columns)
#print(data.head(5))


#preproces the data
def pre_process(data):
    """
     : param data:
     :return:
    """
    global encoded_cat_df
    cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
    target_col = 'y'


    if target_col in data.columns:
        print(f"Encoding target column '{target_col}'...")
        label_encoder = LabelEncoder()
        data[target_col] = label_encoder.fit_transform(data[target_col])

        # Save LabelEncoder
        try:
            with open(le_fpath, 'wb') as le_file:
                pickle.dump(label_encoder, le_file)  #type: ignore
            print(f"LabelEncoder saved successfully at: {le_fpath}")
        except Exception as e:
            print(f"Failed to save LabelEncoder: {e}")
    else:
        print(f"Target column '{target_col}' not found in data!")
# try:
#     pre_process(data)
# except Exception as e:
#     y = data[target_col]
#     return X, y

#one hot encoding
    encoded_cat = []
    if all(col in data.columns for col in cat_cols):
        print(f"Encoding cat column '{cat_cols}'...")
        ohe_encoder = OneHotEncoder(sparse_output= False)
        encoded_cat = ohe_encoder.fit_transform(data[cat_cols])
        encoded_cat_cols = ohe_encoder.get_feature_names_out(cat_cols)
        encoded_cat_df = pd.DataFrame(encoded_cat,columns= encoded_cat_cols)

        # Save LabelEncoder
        try:
            with open(ohe_fpath, 'wb') as le_file:
                pickle.dump(ohe_encoder, le_file)  #type: ignore
            print(f"LabelEncoder saved successfully at: {le_fpath}")
        except Exception as e:
            print(f"Failed to save LabelEncoder: {e}")
    else:
        print(f"Cat column '{cat_cols}' not found in data!")
    #return data, encoded_cat, num_cols
    #try:
    # pre_process(data)
    # except Exception as e:
    #     print(f"Error during processing: {str(e)}")
    X = pd.concat([encoded_cat_df, data[num_cols].reset_index(drop=True)], axis=1)
    y = data[target_col].values  # target variable
    return X, y


try:
    X,y = pre_process(data)
     #print(X.shape)
    # print(y.shape)
except Exception as e:
    print(f"Error during processing: {str(e)}")

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train.columns)
# print(y_train.shape)
# print(X_test.shape)

# fit the model
log_reg_model = LogisticRegression(solver='saga',max_iter= 5000)
log_reg_model.fit(X_train,y_train)

# predict the output
y_prd = log_reg_model.predict(X_test)
accuracy = accuracy_score(y_test,y_prd)
classification_report = classification_report(y_test,y_prd)
print(f"Classification report: {classification_report}")
print(f"Accuracy: {accuracy:.3f}")

#save the model
with open(model_fpath,'wb') as model_file:
    pickle.dump(log_reg_model,model_file) #type 'ignore'