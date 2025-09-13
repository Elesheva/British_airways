import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier

#DAY_1
data = pd.read_csv("C:/Users/vikto/Downloads/customer_booking.csv", encoding="cp1251")
data.booking_complete.sum()
data.isna().sum()
data.dtypes
data.head(5)
data.columns

#Предобработка
from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
data["sales_channel_encoded"] = la.fit_transform(data.sales_channel)
data["trip_type_encoded"] = la.fit_transform(data.trip_type)
data["route_encoded"] = la.fit_transform(data.route)
data["booking_origin_encoded"] = la.fit_transform(data.booking_origin)
#done
purchase_lead_bins = pd.DataFrame(data.purchase_lead)
purchase_lead_bins["0-3_purchase_lead"] = np.where(purchase_lead_bins.purchase_lead <= 3, 1, 0)
purchase_lead_bins["4-14_purchase_lead"] = np.where((purchase_lead_bins.purchase_lead >= 4) & (purchase_lead_bins.purchase_lead <= 14), 1, 0)
purchase_lead_bins["15-60_purchase_lead"] = np.where((purchase_lead_bins.purchase_lead >= 15) & (purchase_lead_bins.purchase_lead <= 60), 1, 0)
purchase_lead_bins[">60_purchase_lead"] = np.where((purchase_lead_bins.purchase_lead > 60), 1, 0)
#done
length_of_stay_bins = pd.DataFrame(data.length_of_stay)
length_of_stay_bins["0-3_length_of_stay"] = np.where(length_of_stay_bins.length_of_stay <= 3, 1, 0)
length_of_stay_bins["4-7_length_of_stay"] = np.where((length_of_stay_bins.length_of_stay>= 4) & (length_of_stay_bins.length_of_stay <= 7), 1, 0)
length_of_stay_bins["8-14_length_of_stay"] = np.where((length_of_stay_bins.length_of_stay>= 8) & (length_of_stay_bins.length_of_stay <= 14), 1, 0)
length_of_stay_bins[">14_length_of_stay"] = np.where(length_of_stay_bins.length_of_stay > 14, 1, 0)
#done
is_weekend_flight = pd.DataFrame(data.flight_day)
is_weekend_flight["weekend"] = np.where((is_weekend_flight.flight_day == "Sat") | (is_weekend_flight.flight_day == "Sun"), 1, 0)


X = data.drop(["sales_channel", "length_of_stay", "purchase_lead", "trip_type", "flight_day", "route", "booking_origin", "booking_complete"], axis = 1)
y = data.booking_complete


#НАРАЩИВАЕМ 
X = pd.merge(X, purchase_lead_bins, left_index=True, right_index=True)
X = X.drop(["purchase_lead"], axis = 1)
X = pd.merge(X, length_of_stay_bins, left_index=True, right_index=True)
X = X.drop(["length_of_stay"], axis = 1)
X = pd.merge(X, is_weekend_flight, left_index=True, right_index=True)
X = X.drop(["flight_day"], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state = 4)

#ОБУЧЕНИЕ
clf = RandomForestClassifier(class_weight="balanced", random_state=41)
parametrs = {'n_estimators':range(10,100,10),
'max_depth':range(1,12,2),
'min_samples_leaf':range(1,5),
'min_samples_split':range(2,9,2)}

grid_searh_cv = GridSearchCV(clf, parametrs, cv = 5)
grid_searh_cv.fit(X_train, y_train)
best_clf = grid_searh_cv.best_estimator_
pred = best_clf.predict(X_test)
feature_importances = best_clf.feature_importances_
y_proba = best_clf.predict_proba(X_test)[:, 1] 

#самые значимые признаки
imp = pd.DataFrame(feature_importances, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()

#confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=[0,1], yticklabels=[0,1])
plt.title("Матрица ошибок")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.show()

#precision_recall_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(7,5))
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall кривая")
plt.grid()
plt.show()


#Меняем порог
# например, хотим recall >= 0.80 и максимизировать precision при этом
target_recall = 0.80
idxs = np.where(recall >= target_recall)[0]
if len(idxs) > 0:
    idx = idxs[-1]   # берем последнюю точку (максимальный precision при recall>=target)
    best_thresh = thresholds[idx] if idx < len(thresholds) else 1.0
    print("Выбран порог:", best_thresh)
    print("precision:", precision[idx], "recall:", recall[idx])
else:
    print("Нельзя достигнуть желаемого recall с этой моделью.")


proba_predict=  pd.DataFrame({
    'probability': y_proba,
    'class': pred
})
pred = (y_proba >= 0.406).astype(int)

#correlation_matrix
data_for_corr = pd.merge(X, y, left_index=True, right_index=True)
correlation_matrix = data_for_corr.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Корреляционная матрица')
plt.show()

X.columns











