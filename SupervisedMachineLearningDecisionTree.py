# Description: This program predicts if a foreign object that is 
# floating around space is hazardous using a decision tree classifier
# Developed by Jonathan Espedal 26 June 2022 CS379-2203A-1

import pandas as pd # Data processing, CSV file I/O
df = pd.read_csv('C:\\Users\\BigDickJ\\Desktop\\CS379_JonathanEspedal_IP5\\neo_v2.csv')

# Plot the data
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
fig = px.scatter_matrix(df, dimensions=['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude'],height=1000,width=1200,color='hazardous')
fig.show()

X = df.drop(['id','name','orbiting_body','sentry_object','hazardous'],axis=1)
y = df.hazardous.astype('int')

# Normalising the data
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(X)
X_scaled

# Model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
print("Model Training Score using Gradient Boosting Classifier")
print(model.score(X_train, y_train))

# Create the prediction using sklearn model
y_pred = model.predict(X_test)

# Model test score
print("Model Test Score")
print(model.score(X_test,y_test))

# Confusion matrix, classification report, and accuracy score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
cm1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (cm1)
cm2 = accuracy_score(y_test,y_pred)
print("Accuracy:",cm2)

# Find area under curve
from sklearn import metrics
y_pred_proba = model.predict_proba(X_test)[::,1]
#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print AUC score
print("Area under curve, closer to 1 the better")
print(auc)