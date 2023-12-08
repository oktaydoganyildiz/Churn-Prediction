import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_csv("VBO//Alıştırmalar//6. Hafta ML - Regresyon ve KNN//Telco-Customer-Churn.csv")
df = df_.copy()
df.head()


df.info()
# Boşluk karakterlerini NaN olarak değiştirme
df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.isnull().sum()
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
df['Churn'] = pd.to_numeric(df['Churn'])


def check_df(dataframe, head=5):
    print(" SHAPE ".center(90, '~'))
    print("%d rows and %d columns" % dataframe.shape)
    print(" SIZE ".center(90, '~'))
    print(dataframe.size)
    print(" TYPES ".center(90, '~'))
    print(dataframe.dtypes)
    print(" HEAD ".center(90, '~'))
    print(dataframe.head(head))
    print(" TAIL ".center(90, '~'))
    print(dataframe.tail(head))
    print(" NA ".center(90, '~'))
    print(dataframe.isna().sum())
    print(" DP ".center(90, '~'))
    print(dataframe.duplicated().sum())
    print(" DESC ".center(90, '~'))
    quantiles = dataframe.describe([0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95]).T
    quantiles['IQR'] = quantiles['75%'] - quantiles['25%']
    print(quantiles)
check_df(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
df.head()

#HedefDeğişkeninSayısalDeğişkenlerileAnalizi
#######################


#Hedef değişken analizi


#Hedef değişkene göre numerik değişkenlerin ortalaması

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
    print('##########################')
for col in num_cols:
    target_summary_with_num(df, "Churn", col)

#Kategorik değişkenlere göre hedef değişkenin ortalaması


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
    print('##########################')
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)




def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25) #aslında 25 ve 75 kullanılır biz datayı bilidğimizden çok outlier olmadıgından bunları kullandık öbür türlü çok data silebilir.
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



#aykırı değer varmı yok mu ?

def check_outlier(dataframe, col_names):
    outlier_dict = {}
    for col_name in col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
        if outliers.shape[0] > 0:  # Eğer sütunda outlier varsa
            outlier_dict[col_name] = outliers

    return outlier_dict

outliers_dict = check_outlier(df, num_cols)

if outliers_dict:
    print("Outlier Bulunan Sütunlar:")
    for col_name, outliers in outliers_dict.items():
        print(f"{col_name}: {outliers.shape[0]} outlier")
else:
    print("Outlier bulunan sütun yok.")


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(df, na_name=True)


#MİSSING VALUE DOLDUMRA#

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols ]

#get_dummies kategorik değişkenleri ML anlicak şekilde 0,1 li şekilde ayarladı
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması. 1. satırında yaşında eksik var mesela gider 5en yakın komsuuna  bakar ve o 5in ortalması ile doldurur
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
df.isnull().sum()
df["TotalCharges_imputed_knn"] = dff[["TotalCharges"]]

df.loc[df["TotalCharges"].isnull(), ["TotalCharges", "TotalCharges_imputed_knn"]]
df.loc[df["TotalCharges"].isnull()]

df["TotalCharges"] = dff[["TotalCharges"]]
df.head()
df.drop("TotalCharges_imputed_knn",inplace=True,axis=1)
df.isnull().sum()
#######################################


#FEATURE ENG#


# 1. Tenure Grubu
def tenure_group(tenure):
    if tenure <= 12:
        return 'Kısa Süreli'
    elif 12 < tenure <= 24:
        return 'Orta Süreli'
    else:
        return 'Uzun Süreli'

df['New_TenureGroup'] = df['tenure'].apply(tenure_group)

# 2. Yaş Grubu
def age_group(senior_citizen):
    return 'Yaşlı' if senior_citizen == 1 else 'Genç'

df['New_AgeGroup'] = df['SeniorCitizen'].apply(age_group)

# 3. İnternet Hizmeti ve Güvenlik Durumu
df['New_InternetSecurityStatus'] = df['InternetService'] + '-' + df['OnlineSecurity']

# 4. Fatura Ödeme Durumu
#df['New_PaymentStatus'] = pd.cut(df['MonthlyCharges'] - df['TotalCharges'], bins=[-float('inf'), 0, float('inf')], labels=['Ödenmedi', 'Ödendi'])

# 5. Müşteri Profili
df['New_CustomerProfile'] = df['New_AgeGroup'] + '-' + df['Partner']

# 6. Çeşitli Hizmetlere Abonelik Durumu
services = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['New_ServicesSubscribed'] = df[services].apply(lambda row: ', '.join(row.index[row == 'Yes']), axis=1)
df['New_ServicesSubscribed_2'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(lambda row: sum(row == 'Yes'), axis=1)

#df['New_TotalSpending'] = df['MonthlyCharges'] + df['TotalCharges']
df['New_ContractBilling'] = df['Contract'] + '_' + df['PaperlessBilling']
df['New_ServiceSubscription'] = df['MonthlyCharges'] * (df['OnlineSecurity'] == 'Yes') + df['MonthlyCharges'] * (df['OnlineBackup'] == 'Yes') + df['MonthlyCharges'] * (df['DeviceProtection'] == 'Yes') + df['MonthlyCharges'] * (df['TechSupport'] == 'Yes') + df['MonthlyCharges'] * (df['StreamingTV'] == 'Yes') + df['MonthlyCharges'] * (df['StreamingMovies'] == 'Yes')
df['New_ContractTenure'] = df['Contract'] + '_' + df['tenure'].astype(str)
df['New_PaymentBilling'] = df['PaymentMethod'] + '_' + df['PaperlessBilling']

df.info()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# kategorikleri 0,1 çevirme
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
#erkek kadın gibi 2 uniqli şeyleri eliyor veya 10 dan küçük olanları

df = one_hot_encoder(df, ohe_cols)
df.head()

#Standartlaştırma işlemi

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.head()
#MODEL

y = df["Churn"]
#X = df.drop(["Churn","customerID","New_ServicesSubscribed","New_ContractTenure",'New_ServicesSubscribed_2_6', 'New_PaymentStatus', 'New_ContractBilling_Two year_Yes', 'StreamingTV_No internet service', 'New_ContractBilling_Two year_No', 'DeviceProtection_No internet service', 'InternetService_No', 'New_PaymentBilling_Credit card (automatic)_No', 'New_ContractBilling_One year_No', 'New_ServicesSubscribed_2_5'], axis=1)
X = df.drop(["Churn","customerID","New_ServicesSubscribed","New_ContractTenure"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = (RandomForestClassifier(random_state=46))
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)



def plot_importance(model, features, num=len(X), save=False):
    global feature_imp
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    # top_features = feature_imp.sort_values(by="Value", ascending=False)
    # print(top_features)
    worst_features = feature_imp.sort_values(by="Value", ascending=True).head(10)['Feature'].tolist()
    print(worst_features)

    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


#KNN
knn_model = KNeighborsClassifier().fit(X, y)

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



#TÜM ML
primitive_success=[]
model_names=[]
y = df["Churn"]
#X = df.drop(["Churn","customerID","New_ServicesSubscribed","New_ContractTenure",'New_ServicesSubscribed_2_6', 'New_PaymentStatus', 'New_ContractBilling_Two year_Yes', 'StreamingTV_No internet service', 'New_ContractBilling_Two year_No', 'DeviceProtection_No internet service', 'InternetService_No', 'New_PaymentBilling_Credit card (automatic)_No', 'New_ContractBilling_One year_No', 'New_ServicesSubscribed_2_5'], axis=1)
X = df.drop(["Churn","customerID","New_ServicesSubscribed","New_ContractTenure"],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

def ML(algName):

    # Model Building / Training
    model=algName().fit(X_train,y_train)
    model_name=algName.__name__
    model_names.append(model_name)
    # Prediction
    y_pred=model.predict(X_test)
    # primitive-Success / Verification Score
    from sklearn.metrics import accuracy_score
    primitiveSuccess=accuracy_score(y_test,y_pred)
    primitive_success.append(primitiveSuccess)
    return  primitive_success,model_names,model


models = [KNeighborsClassifier, SVC, MLPClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier]
for i in models:
    ML(i)

classification = pd.DataFrame(primitive_success, columns=['accuracy_Score'], index=model_names).sort_values(by='accuracy_Score', ascending=False)
print(classification)



def plot_importance(model, features,modelName, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title('Features'+ ' - ' + modelName.__name__ )
    plt.tight_layout()
    plt.show()

for i in models[3:]:
    model=i().fit(X_train,y_train)
    plot_importance(model, X_train,i)




###########




