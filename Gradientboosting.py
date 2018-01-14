#author - abhishek jagtap
#liberty mutual group property insepection prediction
#model used- gradient boosting regressor
#total runtime observed- 185secs on intel 6th generation 6500HQ edition 2.4GHZ CPU 
#GPU - 960M # 8GB DDR4 RAM

#import necessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# Load the data set
df = pd.read_csv("train.csv")
tf= pd.read_csv("test.csv")

# This is to remove the unnecessary feature and creating the problem of overfitting
# This removal is done by checking the importance of each feature using feature selection file 
df.drop('T2_V10', axis=1, inplace=True)
df.drop('T2_V7', axis=1, inplace=True)
df.drop('T1_V13', axis=1, inplace=True)
df.drop('T1_V10', axis=1, inplace=True)

# Alternative to remove the fields from the data set that we don't want to include in our model
#del df['T2_V10']
#del df['T2_V7']
#del df['T1_V13']
#del df['T1_V10']

#same thing for test data
tf.drop('T2_V10', axis=1, inplace=True)
tf.drop('T2_V7', axis=1, inplace=True)
tf.drop('T1_V13', axis=1, inplace=True)
tf.drop('T1_V10', axis=1, inplace=True)

labels = df.Hazard
df.drop('Hazard', axis=1, inplace=True)

columns = df.columns

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8',
                                          'T1_V9','T1_V11','T1_V12','T1_V15','T1_V16',
                                          'T1_V17','T2_V3','T2_V4','T2_V5','T2_V11','T2_V12','T2_V13'])
features_tf = pd.get_dummies(tf, columns=['T1_V4','T1_V5','T1_V6','T1_V7','T1_V8',
                                          'T1_V9','T1_V11','T1_V12','T1_V15','T1_V16',
                                          'T1_V17','T2_V3','T2_V4','T2_V5','T2_V11','T2_V12','T2_V13'])

features_df = features_df.astype(float)
features_tf = features_tf.astype(float)
test_ind=features_tf.index
#print the dimensions of the data after performing one-hot encoding
print("\n \n The dimensions of the train data are:")
print(features_df.shape)

print("\n \n The dimensions of the test data are:")
print(features_tf.shape)

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.3, random_state=0)
#n_estimator=1000
# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs for updating purpose
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
m1 = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % m1)

# Find the error rate on the test set
m2 = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % m2)

#abhi=joblib.load('trained_house_classifier_model.pkl')

model.fit(features_df,labels)
yp=model.predict(features_tf)
#predict the values and save it to csv file
preds = pd.DataFrame({"Id": test_ind, "Hazard": yp})
preds = preds.set_index('Id')
preds.to_csv('Abhishek_jagtap_SVM.csv')


