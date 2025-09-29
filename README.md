## EXNO-3-DS

# AIM:

To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
3. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
4. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
5. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:

  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("C:/Users/admin/Downloads/Encoding Data.csv")
print(df)
```
<img width="1440" height="429" alt="image" src="https://github.com/user-attachments/assets/50ef6bfa-ebfa-47d7-820b-ae80f78d7a59" />

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm', 'Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="1434" height="438" alt="image" src="https://github.com/user-attachments/assets/649d01d2-4d56-4478-92f5-85e146d6bbe3" />

df['bo2']=e1.fit_transform(df[["ord_2"]])
print(df)
<img width="1434" height="438" alt="image" src="https://github.com/user-attachments/assets/0afdc669-b3cf-4dbd-9ece-1435627a9f85" />

```
le=LabelEncoder() 
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
print(dfc)
```
<img width="1454" height="463" alt="image" src="https://github.com/user-attachments/assets/ae3b50a8-265e-4398-b40b-06985cf048cf" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
print(df2)
```
<img width="1450" height="886" alt="image" src="https://github.com/user-attachments/assets/680ae324-fc90-4d5e-a1ea-93e65d19e21c" />

pd.get_dummies(df2,columns=["nom_0"])
<img width="1431" height="453" alt="image" src="https://github.com/user-attachments/assets/26802860-8ff6-4549-8b5d-0b129dca5860" />

pip install upgrade category_encoders
<img width="1433" height="194" alt="image" src="https://github.com/user-attachments/assets/1fe225ba-45d5-4b95-85a6-ae836b106219" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("C:/Users/admin/Downloads/data.csv")
print(df)
```
<img width="1446" height="399" alt="image" src="https://github.com/user-attachments/assets/7780fc53-ffd1-4fdb-92c6-7f563758bea4" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
print(df)
```
<img width="1452" height="417" alt="image" src="https://github.com/user-attachments/assets/f61ad710-1be4-4a99-ae91-24631bca3aef" />

```
dfb=pd.concat([df,nd],axis=1)
print(dfb)
```
<img width="1484" height="626" alt="image" src="https://github.com/user-attachments/assets/1e1f9778-77bf-4872-802b-e63d65d1fb38" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
print(CC)
```
<img width="1469" height="588" alt="image" src="https://github.com/user-attachments/assets/f59ef648-652e-48be-93b6-b21ea6e933d8" />

```
import pandas as pd 
from scipy import stats
import numpy as np
df=pd.read_csv("C:/Users/admin/Downloads/Data_to_Transform.csv")
print(df)
```
<img width="1494" height="849" alt="image" src="https://github.com/user-attachments/assets/64bb5fb0-e9d3-4c59-9487-02e1b8b1f275" />

df.skew()
<img width="1449" height="191" alt="image" src="https://github.com/user-attachments/assets/e52ff6b0-399f-4867-a0d7-9566011f8c99" />

np.log(df["Highly Positive Skew"])
<img width="1454" height="335" alt="image" src="https://github.com/user-attachments/assets/9d9655ab-3180-479b-8682-b6310e5bb030" />

np.reciprocal(df["Moderate Positive Skew"])
<img width="1440" height="341" alt="image" src="https://github.com/user-attachments/assets/0da83c0e-a3ea-47a4-809d-8f8406b95eb6" />

np.sqrt(df["Highly Positive Skew"])
<img width="1438" height="341" alt="image" src="https://github.com/user-attachments/assets/5b5dc716-21a1-4490-96d5-9c8b85afa3c8" />

np.square(df["Highly Positive Skew"])
<img width="1439" height="331" alt="image" src="https://github.com/user-attachments/assets/c98f1b95-1355-4170-9af2-3be8c98b27cc" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
print(df)
```
<img width="1448" height="696" alt="image" src="https://github.com/user-attachments/assets/31875363-d0bf-4997-b7fb-febde06cc834" />

df.skew()
<img width="1446" height="213" alt="image" src="https://github.com/user-attachments/assets/a6036354-8465-4c87-a3df-15b1b2f4adee" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
print(df)
```
<img width="1447" height="894" alt="image" src="https://github.com/user-attachments/assets/b29faf25-e035-43a6-91c5-bb2857f698a5" />

df.skew()
<img width="1441" height="235" alt="image" src="https://github.com/user-attachments/assets/69d88644-c36e-4a35-947b-1f775a65fe90" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
print(df)
```
<img width="1452" height="895" alt="image" src="https://github.com/user-attachments/assets/dec7c700-f81e-4333-8c2a-61574604d257" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="1432" height="689" alt="image" src="https://github.com/user-attachments/assets/2affcd0f-d603-4be8-b748-9d9164778b26" />
<img width="1447" height="618" alt="image" src="https://github.com/user-attachments/assets/7756b6d7-2cbe-4bcd-a0c7-62a18090a581" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="1441" height="605" alt="image" src="https://github.com/user-attachments/assets/e69a575b-d673-4df6-a209-2686db0f2970" />
<img width="1449" height="537" alt="image" src="https://github.com/user-attachments/assets/ef2d5c6b-b908-441d-b099-a262f55d1768" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="1456" height="742" alt="image" src="https://github.com/user-attachments/assets/b0f2cec5-f7ee-435b-b2b8-ad195f7236e4" />
<img width="1451" height="616" alt="image" src="https://github.com/user-attachments/assets/b4edd02c-8437-48dc-80e4-331b1796cb67" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="1453" height="629" alt="image" src="https://github.com/user-attachments/assets/f8764f20-9576-469e-9555-2deb739077ab" />
<img width="1468" height="568" alt="image" src="https://github.com/user-attachments/assets/02746fcc-0a5d-497a-bb2f-3444f2558b37" />

```
dt=pd.read_csv("C:/Users/admin/Downloads/titanic_dataset
print(dt)
```
<img width="1458" height="891" alt="image" src="https://github.com/user-attachments/assets/b4e3bc09-a5a6-4a9e-ac9c-500225785f54" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="1461" height="748" alt="image" src="https://github.com/user-attachments/assets/941b713e-486c-4a2d-8936-e08bfd55532b" />
<img width="1448" height="626" alt="image" src="https://github.com/user-attachments/assets/2936a3b5-518b-48de-aae8-1c04bacbba24" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="1437" height="589" alt="image" src="https://github.com/user-attachments/assets/f7478baf-8b22-4288-9306-ae410e61122d" />
<img width="1467" height="701" alt="image" src="https://github.com/user-attachments/assets/4bcb1b36-e56f-4873-9cbe-96a4a429a616" />

# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file
 was performed successfully.

       
