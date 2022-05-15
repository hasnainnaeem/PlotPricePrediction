# Plot Price Prediction
This repo contains the analysis & visualization of a dataset containing the prices & other attributes of plots in Pakistan. 
Plus, it contains results established using various classical machine learning algorithms as well as deep learning ANN models.
This contribution was made to a project at [TUKL-NUST Research & Development Lab.](https://tukl.seecs.nust.edu.pk/), 
during my internship.

## Description of Repo Files
- `PricePredictionModel-Rethought.ipyn` contains the implications of visualization & analysis along with the baseline results
using various ML models. Moreover, it describes the measures to validate & improve the dataset.
- `/images` contains plots & other images used in `PricePredictionModel-Rethought.ipyn`.
- `PlotPricePrediction-Visualization and Analysis.pptx` Presentation on this work.

-------

# Notebook Preview

## Imports & Basic Setup


```python
# Common imports
import sys
import os
import numpy as np
import pandas as pd

import seaborn as sns

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Define directory to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

# function to save the figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# To get same results during each run
np.random.seed(42)

# Ignoring useless warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
```


```python
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
```

## Dataset Setup


```python
df = pd.read_csv('dataset.csv')
df.shape
```




    (43628, 18)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43628 entries, 0 to 43627
    Data columns (total 18 columns):
    id                  43628 non-null int64
    disc                43628 non-null object
    region              43628 non-null object
    cityID              43628 non-null int64
    area                43628 non-null float64
    price               43628 non-null int64
    Lat                 43628 non-null float64
    Lng                 43628 non-null float64
    bank                43628 non-null int64
    mosque              43628 non-null int64
    bus                 43628 non-null int64
    park                43628 non-null int64
    department_store    43628 non-null int64
    school              43628 non-null int64
    supermarket         43628 non-null int64
    cemetary            43628 non-null int64
    hospital            43628 non-null int64
    restaurant          43628 non-null int64
    dtypes: float64(3), int64(13), object(2)
    memory usage: 6.0+ MB
    


```python
# some region names end with comma, removing those commas
df['region'] = df['region'].str.strip(',')
df.shape
```




    (43628, 18)




```python
def cleanData():
    global df
    # Removing outliers
    indexNames = df[ (df['price'] >= 250000000 )].index
    df.drop(indexNames , inplace=True)
    print(df.shape)
    indexNames = df[(df['price'] <= 10000)].index
    df.drop(indexNames , inplace=True)
    print(df.shape)
    indexNames = df[(df['area']>= 1000)].index
    df.drop(indexNames , inplace=True)
    print(df.shape)
    indexNames = df[(df['area'] <= 3 )].index
    df.drop(indexNames , inplace=True)
    print(df.shape)
    
    # Removing places with the same latitude and longitude
    df.drop_duplicates(subset=['Lat','Lng'],keep='first',inplace = True)
    print(df.shape)
```


```python
cleanData()
df.head()
```

    (43591, 18)
    (43591, 18)
    (43590, 18)
    (43587, 18)
    (2158, 18)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>disc</th>
      <th>region</th>
      <th>cityID</th>
      <th>area</th>
      <th>price</th>
      <th>Lat</th>
      <th>Lng</th>
      <th>bank</th>
      <th>mosque</th>
      <th>bus</th>
      <th>park</th>
      <th>department_store</th>
      <th>school</th>
      <th>supermarket</th>
      <th>cemetary</th>
      <th>hospital</th>
      <th>restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7 Marla Plot for Sale.</td>
      <td>B-17</td>
      <td>1</td>
      <td>7.62400</td>
      <td>2300000</td>
      <td>33.669341</td>
      <td>72.844890</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>41</td>
      <td>5 Marla Residential Land for Sale in Islamabad...</td>
      <td>Top City-1</td>
      <td>1</td>
      <td>5.44504</td>
      <td>1000000</td>
      <td>33.586108</td>
      <td>72.866789</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>54</td>
      <td>600 Square Yard Plot for Sale</td>
      <td>F-16</td>
      <td>1</td>
      <td>21.60000</td>
      <td>5500000</td>
      <td>33.656936</td>
      <td>72.888470</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>74</td>
      <td>200 Square Yards Plot for Sale</td>
      <td>Faisal Hills</td>
      <td>1</td>
      <td>7.20000</td>
      <td>2185000</td>
      <td>33.729388</td>
      <td>73.093146</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>88</td>
      <td>10 Marla Plot for Sale</td>
      <td>Bahria Town</td>
      <td>1</td>
      <td>10.89200</td>
      <td>9500000</td>
      <td>33.692555</td>
      <td>73.219032</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
# remaining dataset entries are 2158. 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cityID</th>
      <th>area</th>
      <th>price</th>
      <th>Lat</th>
      <th>Lng</th>
      <th>bank</th>
      <th>mosque</th>
      <th>bus</th>
      <th>park</th>
      <th>department_store</th>
      <th>school</th>
      <th>supermarket</th>
      <th>cemetary</th>
      <th>hospital</th>
      <th>restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2.158000e+03</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
      <td>2158.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27620.110287</td>
      <td>1.963855</td>
      <td>18.053462</td>
      <td>1.529253e+07</td>
      <td>32.572422</td>
      <td>73.643578</td>
      <td>0.863763</td>
      <td>0.501854</td>
      <td>0.246525</td>
      <td>0.495366</td>
      <td>0.401761</td>
      <td>0.994903</td>
      <td>0.995366</td>
      <td>0.981928</td>
      <td>0.988879</td>
      <td>0.982854</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18338.542450</td>
      <td>0.999578</td>
      <td>30.080210</td>
      <td>2.349205e+07</td>
      <td>1.114469</td>
      <td>0.671395</td>
      <td>0.343120</td>
      <td>0.500112</td>
      <td>0.431087</td>
      <td>0.500094</td>
      <td>0.490368</td>
      <td>0.071230</td>
      <td>0.067931</td>
      <td>0.133244</td>
      <td>0.104894</td>
      <td>0.129844</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.267000</td>
      <td>3.300000e+05</td>
      <td>24.993758</td>
      <td>67.307440</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15779.500000</td>
      <td>1.000000</td>
      <td>7.623040</td>
      <td>4.000000e+06</td>
      <td>31.455539</td>
      <td>73.043821</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>25207.500000</td>
      <td>1.000000</td>
      <td>10.890040</td>
      <td>7.575000e+06</td>
      <td>33.521323</td>
      <td>73.218711</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36617.250000</td>
      <td>3.000000</td>
      <td>21.780000</td>
      <td>1.650000e+07</td>
      <td>33.632623</td>
      <td>74.284503</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>66744.000000</td>
      <td>3.000000</td>
      <td>784.080000</td>
      <td>2.300000e+08</td>
      <td>33.744962</td>
      <td>74.499221</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Digging the Dataset and Visualizing to Gain Insights

### Histograms of Dataset Attributes


```python
df.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
```

    Saving figure attribute_histogram_plots
    


    
![png](./images/readme/output_13_1.png)
    


**Attribute Histograms without removing the duplicate (latitude, longitude) records:**
![attribute_histogram_plots_without_removing_duplicates](images/attribute_histogram_plots_without_removing_duplicates.png)


```python
df['price'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x195d0c18cc0>




    
![png](./images/readme/output_15_1.png)
    



```python
df['bank'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x195d7df3898>




    
![png](./images/readme/output_16_1.png)
    


#### Observations and deductions from the Attribute Histograms
- **Most of variables are not continuous**. Variables are not suitable for predicting a continuous variable: price. So, binary attributes must be replaced with suitable continuous attributes. For example, **distance to bank, restaurant, school, and other places can be added to the dataset** instead of simply having a flag variable indicating their presence.

- **Useless Attributes**: *Cemetery, hospital, restaurant, school and supermarket* are present in almost all the records. So, these will not be helpful in predictive models. To increase generalizability of the model during test and deployment phase, records with absence of these places should be added to the dataset. Moreover, replacing current values (1 or 0) with distances can help in better use of current dataset.
    
- **Remaining potential attributes**: *Area, bank, bus, mosque, department_store and park* may prove helpful in predictive models. Nonetheless, adding more attributes will help with getting better results.



### Visualizing Attributes w.r.t Price

#### Area


```python
df.plot(kind="scatter", x="area", y="price",
             alpha=1)
save_fig("price_vs_area_scatterplot")
```

    Saving figure price_vs_area_scatterplot
    


    
![png](./images/readme/output_20_1.png)
    


### Preprocessing Dataset


```python
# converting to string variable "region" to numerical variables 
df['region'] = pd.Categorical(df['region'])

dfDummies = pd.get_dummies(df['region'], prefix = 'category')
df = pd.concat([df, dfDummies], axis=1)
```

### Location Scatter Plots


```python
df.plot(kind="scatter", x="Lat", y="Lng")
save_fig("overall_scatter_plot")

```

    Saving figure overall_scatter_plot
    


    
![png](./images/readme/output_24_1.png)
    



```python
df.plot(kind="scatter", x="Lng", y="Lat", alpha=1,
    figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```

    WARNING: Logging before flag parsing goes to stderr.
    W1201 15:39:05.701071  6008 legend.py:1282] No handles with labels found to put in legend.
    

    Saving figure housing_prices_scatterplot
    


    
![png](./images/readme/output_25_2.png)
    


#### Implications of scatter plots:
- Dataset points are very closely placed. So, most of the entries are from the same or nearly situated towns/colonies. 
- **To better predict price differences in cities, attributes which count to difference of prices in cities should present in the dataset. There is not even a single such attribute in the dataset.**	
    - Possible new attributes to differentiate cities can be can be population, presence of people of certain social class (can be differentiated on the basis of their income, job, etc), presence of industries, number of malls, number of cinemas, number of parks, etc. 
    - Presence of places should be counted within a certain radius around the plots - preferably dataset should be collected with different radius values, best one can be found during training. Dataset collection with different radius values does not need much effort, same scraping algorithm can be used multiple times with little changes.
- Last but not the least, **to predict the price differences within the towns/colonies**, we need more attributes which count to differences in prices within same town, colony, or city. Currently, such attributes in the dataset are: area, bank, bus, mosque and park. Only area is a continuous variable here. Problems with the non-continuous variables are discussed earlier in "Attributes Histograms" section.


### Correlation Matrix and Plots


```python
corr_matrix = df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corr_matrix, vmax=0.9, square=True)
save_fig("correlation_matrix_heatmap")
```

    Saving figure correlation_matrix_heatmap
    


    
![png](./images/readme/output_28_1.png)
    


**Correlation matrix heatmap without converting region attribute to categorical variable:**
![correlation matrix heatmap without categorical region variable](images/correlation_matrix_heatmap_without_categorical_region_variable.png)

**Correlation Matrix without Category Variables**

![correlation_matrix_without_categorical_region_variable](images/correlation_matrix_without_categorical_region_variable.png)


```python
# Run this to generate full correlation matrix. It needs significant processing and memory. 
# Chrome may become unresponsive due to the amount of output data
# corr_matrix.style.background_gradient(cmap='coolwarm', axis=None)
```


```python
corr_matrix["price"].sort_values(ascending=False)
```




    price                                  1.000000
    area                                   0.462654
    category_F-7                           0.282082
    category_F-8                           0.244955
    category_E-7                           0.196835
    category_Emporium Mall                 0.187667
    category_F-10                          0.174314
    category_F-11                          0.173792
    category_Peco Road                     0.169332
    category_DHA                           0.157106
    category_Japan Road                    0.132662
    category_Model Town                    0.128494
    category_Kalma Chowk Lahore            0.118910
    category_Tarlai                        0.114327
    bank                                   0.084902
    category_Ali Pur                       0.078118
    category_F-6                           0.077428
    department_store                       0.075604
    Lng                                    0.075235
    category_F-9                           0.069864
    category_Koral Chowk                   0.066197
    category_E-11                          0.064487
    category_Revenue Society               0.063905
    category_Pwd Housing Scheme            0.062385
    cemetary                               0.061637
    bus                                    0.061583
    category_I-8                           0.058287
    restaurant                             0.057722
    category_Mohlanwal Road                0.057488
    cityID                                 0.056059
                                             ...   
    category_I-15                         -0.028610
    category_Soan Garden                  -0.028661
    category_F-16                         -0.029860
    category_F-18                         -0.030197
    category_E-16                         -0.030511
    category_Top City-1                   -0.030555
    category_Capital Smart City           -0.030572
    category_E-12                         -0.031914
    category_F-17                         -0.032631
    category_E-17                         -0.033513
    category_Jinnah Garden                -0.033758
    category_Lda Avenue                   -0.035191
    category_E-19                         -0.036075
    category_I-12                         -0.037636
    category_Cbr Town                     -0.039286
    category_G-16                         -0.040106
    id                                    -0.040678
    category_Pechs                        -0.040794
    category_I-14                         -0.040902
    category_Fateh Jang Road              -0.041034
    category_State Life Housing Society   -0.041305
    category_University Town              -0.041393
    category_D-18                         -0.042296
    category_I-16                         -0.042367
    park                                  -0.043135
    Lat                                   -0.043254
    category_Ghauri Town                  -0.056375
    category_Bahria Town                  -0.060152
    category_B-17                         -0.061401
    category_Bahria Orchard               -0.065297
    Name: price, Length: 368, dtype: float64



#### Deductions from the correlations
- As expected, there is a strong correlation between the price and area.
- Category variables do not have considerable correlation with the prices. Because, area and other attributes are not being considered in correlation calculation.
- *Correlation of price with previously separated potential attributes (area, bank, bus, mosque, park) is insignificant*. **This points to possibility of defects in the dataset collection, selection of dataset attributes and default attribute values/types.** We can't certainly say that the correlation doesn't exist because there are other attributes affecting the price at the same time. For example, decrease in price is not significant due to absence of bank if the area of plot is high.
    
    **Presence of defects can be pointed out by** checking correlation between pair of attributes with the price. **For example**, *correlation between (mosque, area) and price should be checked*. One attribute in the pair should be area because it is most influential attribute in the dataset.


## Preprocessing Dataset for Predictive Models


```python
del_col_count = 4
X = df.drop(columns=['price','id','disc','region'])
Y = df[['price']]
```


```python
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
```


```python
X_scale
```




    array([[0.        , 0.00558008, 0.99135876, ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.00278945, 0.98184766, ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.02347937, 0.98994124, ..., 0.        , 0.        ,
            0.        ],
           ...,
           [1.        , 0.05160391, 0.73929733, ..., 0.        , 0.        ,
            0.        ],
           [1.        , 0.00976295, 0.73383047, ..., 0.        , 0.        ,
            0.        ],
           [1.        , 0.00278945, 0.74342956, ..., 0.        , 0.        ,
            0.        ]])




```python
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.2, random_state=10)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cityID</th>
      <th>area</th>
      <th>Lat</th>
      <th>Lng</th>
      <th>bank</th>
      <th>mosque</th>
      <th>bus</th>
      <th>park</th>
      <th>department_store</th>
      <th>school</th>
      <th>...</th>
      <th>category_Uet Housing Society</th>
      <th>category_University Road</th>
      <th>category_University Town</th>
      <th>category_Valencia Housing Society</th>
      <th>category_Valencia Housing Society Block A-1</th>
      <th>category_Vital Homes Housing Scheme</th>
      <th>category_Walton Road</th>
      <th>category_Wapda Town</th>
      <th>category_Zaraj Housing Scheme</th>
      <th>category_Zone-5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7.62400</td>
      <td>33.669341</td>
      <td>72.844890</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>5.44504</td>
      <td>33.586108</td>
      <td>72.866789</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>21.60000</td>
      <td>33.656936</td>
      <td>72.888470</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>7.20000</td>
      <td>33.729388</td>
      <td>73.093146</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1</td>
      <td>10.89200</td>
      <td>33.692555</td>
      <td>73.219032</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 366 columns</p>
</div>




```python
Y.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2300000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5500000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2185000</td>
    </tr>
    <tr>
      <th>57</th>
      <td>9500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
```

## Predictive ML Models

### Defining Models

#### Model 1 - Dense Layers


```python
model_1 = Sequential([
    Dense(len(list(df)) - del_col_count, activation='relu', input_shape=(len(list(df)) - del_col_count,)),
    Dense(500, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1),
])
```


```python
model_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
```

#### Model 2 - 3 Dense Layers


```python
model_2 = Sequential([
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(len(list(df)) - del_col_count,)),
    Dropout(0.3),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, kernel_regularizer=regularizers.l2(0.01)),
])
```


```python
model_2.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])
```

### Training


```python
# Easily keep track of loss values by counting number of 10s.
# returns a "num" and "count" of 10s
# get actual number by: num * 10^count
def countTens(num):
    count = 0
    while num >= 10:
        num = num/10.0
        count += 1
    # return reduced number too
    return (num, count)
```

#### Linear Regression


```python
from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



#### Decision Tree Regression


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg_model = DecisionTreeRegressor(random_state=42)
tree_reg_model.fit(X_train, Y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=42, splitter='best')



#### Random Forest Regressor


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
forest_reg_model.fit(X_train, Y_train)
```

    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\ipykernel_launcher.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      after removing the cwd from sys.path.
    




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=150,
                          n_jobs=None, oob_score=False, random_state=42, verbose=0,
                          warm_start=False)




```python
# Using Grid Search for Optimal Hypyerparameters
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [50, 70, 100, 150, 200], 'max_features': [2, 4, 6, 8, 10, 12]},
    {'bootstrap': [False], 'n_estimators': [50, 70, 100, 150, 200], 'max_features': [2, 3, 4, 8, 10, 12]},
  ]

forest_reg_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg_model, param_grid, cv=12,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train, Y_train)
```

    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_validation.py:514: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      estimator.fit(X_train, y_train, **fit_params)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    C:\Users\G3NZ\Anaconda3\envs\tf-2.0\lib\site-packages\sklearn\model_selection\_search.py:714: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      self.best_estimator_.fit(X, y, **fit_params)
    




    GridSearchCV(cv=12, error_score='raise-deprecating',
                 estimator=RandomForestRegressor(bootstrap=True, criterion='mse',
                                                 max_depth=None,
                                                 max_features='auto',
                                                 max_leaf_nodes=None,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators='warn', n_jobs=None,
                                                 oob_score=False, random_state=42,
                                                 verbose=0, warm_start=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'max_features': [2, 4, 6, 8, 10, 12],
                              'n_estimators': [50, 70, 100, 150, 200]},
                             {'bootstrap': [False],
                              'max_features': [2, 3, 4, 8, 10, 12],
                              'n_estimators': [50, 70, 100, 150, 200]}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
                 scoring='neg_mean_squared_error', verbose=0)




```python
price_predictions = grid_search.predict(X_train)
forest_reg_mse = mean_squared_error(price_predictions, Y_train)
forest_reg_rmse = np.sqrt(forest_reg_mse)
reduced_num, ten_count = countTens(forest_reg_rmse)
print("RMSE on training set using Forest Tree Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on training set using Forest Tree Regression is: 6.579669884270951 * 10^6
    

#### Deep Learning Model 1


```python
hist = model_1.fit(X_train, Y_train,
          batch_size=32, epochs=1000,
          validation_data=(X_val, Y_val))
```

    Train on 1726 samples, validate on 216 samples
    Epoch 1/1000
    1726/1726 [==============================] - 1s 806us/sample - loss: 831902638043169.2500 - mean_squared_error: 831902684545024.0000 - val_loss: 594396421437667.5000 - val_mean_squared_error: 594396428894208.0000
    Epoch 2/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 828949228689339.2500 - mean_squared_error: 828949290549248.0000 - val_loss: 585798088140269.0000 - val_mean_squared_error: 585798105694208.0000
    Epoch 3/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 801758647251205.1250 - mean_squared_error: 801758590795776.0000 - val_loss: 533185677200952.8750 - val_mean_squared_error: 533185662287872.0000
    Epoch 4/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 714507562991862.8750 - mean_squared_error: 714507538137088.0000 - val_loss: 413004964881294.2500 - val_mean_squared_error: 413004927598592.0000
    Epoch 5/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 590066418941610.2500 - mean_squared_error: 590066296553472.0000 - val_loss: 318234586464862.8125 - val_mean_squared_error: 318234595164160.0000
    Epoch 6/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 542046773806840.5625 - mean_squared_error: 542046850908160.0000 - val_loss: 303690430963408.5625 - val_mean_squared_error: 303690426613760.0000
    Epoch 7/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 535248453052579.6875 - mean_squared_error: 535248387440640.0000 - val_loss: 295993226947849.5000 - val_mean_squared_error: 295993241239552.0000
    Epoch 8/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 530003255444462.1875 - mean_squared_error: 530003292848128.0000 - val_loss: 288327371781764.7500 - val_mean_squared_error: 288327362150400.0000
    Epoch 9/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 524326782974576.1250 - mean_squared_error: 524326688260096.0000 - val_loss: 281672359103450.0625 - val_mean_squared_error: 281672377434112.0000
    Epoch 10/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 519488650375049.3125 - mean_squared_error: 519488642482176.0000 - val_loss: 274702212831156.1562 - val_mean_squared_error: 274702215938048.0000
    Epoch 11/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 514992276955658.0625 - mean_squared_error: 514992281485312.0000 - val_loss: 267866626355655.1250 - val_mean_squared_error: 267866607714304.0000
    Epoch 12/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 510602275683402.8125 - mean_squared_error: 510602321592320.0000 - val_loss: 261248072742532.7188 - val_mean_squared_error: 261248063111168.0000
    Epoch 13/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 506353617381957.3750 - mean_squared_error: 506353558749184.0000 - val_loss: 256061469428091.2812 - val_mean_squared_error: 256061470670848.0000
    Epoch 14/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 502248673422349.0625 - mean_squared_error: 502248543092736.0000 - val_loss: 250699841536000.0000 - val_mean_squared_error: 250699824758784.0000
    Epoch 15/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 499004238353622.7500 - mean_squared_error: 499004332834816.0000 - val_loss: 245250256914204.4375 - val_mean_squared_error: 245250266234880.0000
    Epoch 16/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 495173756544827.0625 - mean_squared_error: 495173792432128.0000 - val_loss: 240904260162294.5312 - val_mean_squared_error: 240904279425024.0000
    Epoch 17/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 492485271173744.1250 - mean_squared_error: 492485243568128.0000 - val_loss: 236519946734554.0625 - val_mean_squared_error: 236519939899392.0000
    Epoch 18/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 489871963982661.6875 - mean_squared_error: 489871990849536.0000 - val_loss: 232947518207620.7188 - val_mean_squared_error: 232947516964864.0000
    Epoch 19/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 487891009697599.8125 - mean_squared_error: 487891037847552.0000 - val_loss: 230320547940124.4375 - val_mean_squared_error: 230320540483584.0000
    Epoch 20/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 486094436705328.6250 - mean_squared_error: 486094332231680.0000 - val_loss: 227920091215492.7188 - val_mean_squared_error: 227920073195520.0000
    Epoch 21/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 484667042015889.3750 - mean_squared_error: 484666960248832.0000 - val_loss: 226242872931972.7188 - val_mean_squared_error: 226242871689216.0000
    Epoch 22/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 483032991625780.7500 - mean_squared_error: 483032993628160.0000 - val_loss: 223589470782046.8125 - val_mean_squared_error: 223589471092736.0000
    Epoch 23/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 482339620218455.1875 - mean_squared_error: 482339624845312.0000 - val_loss: 221767409511082.6562 - val_mean_squared_error: 221767415103488.0000
    Epoch 24/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 482057152477846.0625 - mean_squared_error: 482057163636736.0000 - val_loss: 220315104721123.5625 - val_mean_squared_error: 220315095400448.0000
    Epoch 25/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 481523393846337.2500 - mean_squared_error: 481523379732480.0000 - val_loss: 219612603540366.2188 - val_mean_squared_error: 219612616589312.0000
    Epoch 26/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 480783814739571.6875 - mean_squared_error: 480783806496768.0000 - val_loss: 218745850308077.0312 - val_mean_squared_error: 218745821724672.0000
    Epoch 27/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 480013667992387.3750 - mean_squared_error: 480013665173504.0000 - val_loss: 218131682412316.4375 - val_mean_squared_error: 218131691732992.0000
    Epoch 28/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 479799985146048.1875 - mean_squared_error: 479799889887232.0000 - val_loss: 217767759120763.2812 - val_mean_squared_error: 217767760363520.0000
    Epoch 29/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 479891659450778.5625 - mean_squared_error: 479891560595456.0000 - val_loss: 216885447195458.3750 - val_mean_squared_error: 216885429796864.0000
    Epoch 30/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 479213407717169.5625 - mean_squared_error: 479213391970304.0000 - val_loss: 217218427863646.8125 - val_mean_squared_error: 217218423980032.0000
    Epoch 31/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 479468636996343.3750 - mean_squared_error: 479468606980096.0000 - val_loss: 216443563909423.4062 - val_mean_squared_error: 216443568259072.0000
    Epoch 32/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 479698541505378.1875 - mean_squared_error: 479698589057024.0000 - val_loss: 216548878222525.6250 - val_mean_squared_error: 216548878843904.0000
    Epoch 33/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 478590762306620.5000 - mean_squared_error: 478590655266816.0000 - val_loss: 216494093777123.5625 - val_mean_squared_error: 216494084456448.0000
    Epoch 34/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 479547248158276.2500 - mean_squared_error: 479547258568704.0000 - val_loss: 216445357207400.2812 - val_mean_squared_error: 216445380198400.0000
    Epoch 35/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 477668904366341.0000 - mean_squared_error: 477668881465344.0000 - val_loss: 215883513720073.4688 - val_mean_squared_error: 215883511234560.0000
    Epoch 36/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 477785563172830.7500 - mean_squared_error: 477785550225408.0000 - val_loss: 215945510193531.2812 - val_mean_squared_error: 215945503047680.0000
    Epoch 37/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 478267153177551.3750 - mean_squared_error: 478267257651200.0000 - val_loss: 216056446427439.4062 - val_mean_squared_error: 216056467554304.0000
    Epoch 38/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 477225915173151.1250 - mean_squared_error: 477225962962944.0000 - val_loss: 215775231548529.7812 - val_mean_squared_error: 215775247859712.0000
    Epoch 39/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 477606747522182.0625 - mean_squared_error: 477606738657280.0000 - val_loss: 214197996617728.0000 - val_mean_squared_error: 214197988229120.0000
    Epoch 40/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 477636102848360.0625 - mean_squared_error: 477636065230848.0000 - val_loss: 214678464704360.2812 - val_mean_squared_error: 214678470918144.0000
    Epoch 41/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 477203685308489.5625 - mean_squared_error: 477203749928960.0000 - val_loss: 215248728489984.0000 - val_mean_squared_error: 215248728489984.0000
    Epoch 42/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 475994360250519.9375 - mean_squared_error: 475994381090816.0000 - val_loss: 215753880832379.2812 - val_mean_squared_error: 215753890463744.0000
    Epoch 43/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 478634851644393.5000 - mean_squared_error: 478634947117056.0000 - val_loss: 215771428868247.7188 - val_mean_squared_error: 215771422654464.0000
    Epoch 44/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 476214201695971.1875 - mean_squared_error: 476214196174848.0000 - val_loss: 214932369848547.5625 - val_mean_squared_error: 214932377305088.0000
    Epoch 45/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 476775506494332.3125 - mean_squared_error: 476775461158912.0000 - val_loss: 214388393474882.3750 - val_mean_squared_error: 214388376076288.0000
    Epoch 46/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 477357099758379.5625 - mean_squared_error: 477357093683200.0000 - val_loss: 215856667067581.6250 - val_mean_squared_error: 215856667688960.0000
    Epoch 47/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 475159255291797.1875 - mean_squared_error: 475159177723904.0000 - val_loss: 213481025088170.6562 - val_mean_squared_error: 213481047457792.0000
    Epoch 48/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 477707302961873.4375 - mean_squared_error: 477707301289984.0000 - val_loss: 214625760633742.2188 - val_mean_squared_error: 214625756905472.0000
    Epoch 49/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 475069780318727.7500 - mean_squared_error: 475069822271488.0000 - val_loss: 213051319575438.2188 - val_mean_squared_error: 213051315847168.0000
    Epoch 50/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 476295010916207.2500 - mean_squared_error: 476294995247104.0000 - val_loss: 213583012230788.7188 - val_mean_squared_error: 213583019376640.0000
    Epoch 51/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 475092744079666.1250 - mean_squared_error: 475092739948544.0000 - val_loss: 215275634173421.0312 - val_mean_squared_error: 215275639144448.0000
    Epoch 52/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 475631126447750.6875 - mean_squared_error: 475631154364416.0000 - val_loss: 214483249368632.8750 - val_mean_squared_error: 214483234455552.0000
    Epoch 53/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 474376224459229.0000 - mean_squared_error: 474376185053184.0000 - val_loss: 214183882784768.0000 - val_mean_squared_error: 214183878590464.0000
    Epoch 54/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 474951483196683.0000 - mean_squared_error: 474951408680960.0000 - val_loss: 214594845817059.5625 - val_mean_squared_error: 214594836496384.0000
    Epoch 55/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 474727301040263.3125 - mean_squared_error: 474727365738496.0000 - val_loss: 213251897095812.7188 - val_mean_squared_error: 213251904241664.0000
    Epoch 56/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 474164274079148.3125 - mean_squared_error: 474164255260672.0000 - val_loss: 213219461455568.5938 - val_mean_squared_error: 213219457105920.0000
    Epoch 57/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 474546219221257.7500 - mean_squared_error: 474546238914560.0000 - val_loss: 215029925475972.7188 - val_mean_squared_error: 215029920038912.0000
    Epoch 58/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 474556361952316.5000 - mean_squared_error: 474556305244160.0000 - val_loss: 213658952434877.6250 - val_mean_squared_error: 213658969833472.0000
    Epoch 59/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 473332964231663.9375 - mean_squared_error: 473333011316736.0000 - val_loss: 213356992994417.7812 - val_mean_squared_error: 213356996722688.0000
    Epoch 60/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 473659953901715.1250 - mean_squared_error: 473659965702144.0000 - val_loss: 214148449459920.5938 - val_mean_squared_error: 214148461887488.0000
    Epoch 61/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 474010906247756.5625 - mean_squared_error: 474010877952000.0000 - val_loss: 214058300662670.2188 - val_mean_squared_error: 214058334683136.0000
    Epoch 62/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 474281019473958.0000 - mean_squared_error: 474281058238464.0000 - val_loss: 213475717895509.3438 - val_mean_squared_error: 213475695525888.0000
    Epoch 63/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 472636030708388.3750 - mean_squared_error: 472635951546368.0000 - val_loss: 213716680349620.1562 - val_mean_squared_error: 213716683456512.0000
    Epoch 64/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 472834977812921.3750 - mean_squared_error: 472835029991424.0000 - val_loss: 213142880299690.6562 - val_mean_squared_error: 213142885892096.0000
    Epoch 65/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 472781296456690.9375 - mean_squared_error: 472781208682496.0000 - val_loss: 211977078007883.8438 - val_mean_squared_error: 211977070706688.0000
    Epoch 66/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 473083224756950.1875 - mean_squared_error: 473083097907200.0000 - val_loss: 212465708986823.1250 - val_mean_squared_error: 212465707122688.0000
    Epoch 67/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 471738638109319.8750 - mean_squared_error: 471738672480256.0000 - val_loss: 212031864162076.4375 - val_mean_squared_error: 212031881871360.0000
    Epoch 68/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 473044427279705.3125 - mean_squared_error: 473044376092672.0000 - val_loss: 212578484190018.3750 - val_mean_squared_error: 212578483568640.0000
    Epoch 69/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 474110405335218.0000 - mean_squared_error: 474110400397312.0000 - val_loss: 211993342120694.5312 - val_mean_squared_error: 211993344606208.0000
    Epoch 70/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 471476375366289.3750 - mean_squared_error: 471476343930880.0000 - val_loss: 212804900797857.1875 - val_mean_squared_error: 212804892098560.0000
    Epoch 71/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 471725238790383.6875 - mean_squared_error: 471725250707456.0000 - val_loss: 213157183963742.8125 - val_mean_squared_error: 213157196857344.0000
    Epoch 72/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 472334372957747.6250 - mean_squared_error: 472334397865984.0000 - val_loss: 212321375218953.4688 - val_mean_squared_error: 212321389510656.0000
    Epoch 73/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 470996940611726.3750 - mean_squared_error: 470996884652032.0000 - val_loss: 212253898498958.2188 - val_mean_squared_error: 212253911547904.0000
    Epoch 74/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 471079719580155.8750 - mean_squared_error: 471079764099072.0000 - val_loss: 212371364487471.4062 - val_mean_squared_error: 212371385614336.0000
    Epoch 75/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 471871691891229.0625 - mean_squared_error: 471871648694272.0000 - val_loss: 211948871934862.2188 - val_mean_squared_error: 211948884983808.0000
    Epoch 76/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 470480698861485.0000 - mean_squared_error: 470480716824576.0000 - val_loss: 212600300163223.7188 - val_mean_squared_error: 212600277172224.0000
    Epoch 77/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 469930917541562.8750 - mean_squared_error: 469930927456256.0000 - val_loss: 211235536400839.1250 - val_mean_squared_error: 211235534536704.0000
    Epoch 78/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 471023008040981.3750 - mean_squared_error: 471023057108992.0000 - val_loss: 211855146328367.4062 - val_mean_squared_error: 211855150678016.0000
    Epoch 79/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 470596799840123.1250 - mean_squared_error: 470596748050432.0000 - val_loss: 212622189459076.7188 - val_mean_squared_error: 212622171439104.0000
    Epoch 80/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 469944640876558.1875 - mean_squared_error: 469944550555648.0000 - val_loss: 211211137357748.1562 - val_mean_squared_error: 211211140464640.0000
    Epoch 81/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 469322353250346.6875 - mean_squared_error: 469322384277504.0000 - val_loss: 211550386335592.2812 - val_mean_squared_error: 211550375772160.0000
    Epoch 82/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 469987455048712.3125 - mean_squared_error: 469987433119744.0000 - val_loss: 211285724665780.1562 - val_mean_squared_error: 211285731966976.0000
    Epoch 83/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 469385163842153.0625 - mean_squared_error: 469385097510912.0000 - val_loss: 211283387506384.5938 - val_mean_squared_error: 211283399933952.0000
    Epoch 84/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 470151168172231.3750 - mean_squared_error: 470151145193472.0000 - val_loss: 210664967207746.3750 - val_mean_squared_error: 210664958197760.0000
    Epoch 85/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 468693274098231.1875 - mean_squared_error: 468693372895232.0000 - val_loss: 211235903635456.0000 - val_mean_squared_error: 211235903635456.0000
    Epoch 86/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 468086107234408.3750 - mean_squared_error: 468086071230464.0000 - val_loss: 211102654636335.4062 - val_mean_squared_error: 211102658985984.0000
    Epoch 87/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 468311142968143.2500 - mean_squared_error: 468311120805888.0000 - val_loss: 211015396335919.4062 - val_mean_squared_error: 211015400685568.0000
    Epoch 88/1000
    1726/1726 [==============================] - 0s 128us/sample - loss: 468148018505227.3125 - mean_squared_error: 468148046266368.0000 - val_loss: 211604419845233.7812 - val_mean_squared_error: 211604431962112.0000
    Epoch 89/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 468620282360790.4375 - mean_squared_error: 468620291342336.0000 - val_loss: 211702443836605.6250 - val_mean_squared_error: 211702461235200.0000
    Epoch 90/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 467369856350756.1250 - mean_squared_error: 467369784770560.0000 - val_loss: 210957771259297.1875 - val_mean_squared_error: 210957770948608.0000
    Epoch 91/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 467371980262707.3125 - mean_squared_error: 467372032917504.0000 - val_loss: 210247098020901.9375 - val_mean_squared_error: 210247088078848.0000
    Epoch 92/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 467723223235620.8125 - mean_squared_error: 467723247157248.0000 - val_loss: 210265871104227.5625 - val_mean_squared_error: 210265845006336.0000
    Epoch 93/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 467126788123110.5000 - mean_squared_error: 467126884237312.0000 - val_loss: 210280533614288.5938 - val_mean_squared_error: 210280541847552.0000
    Epoch 94/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 466943341990006.6875 - mean_squared_error: 466943274385408.0000 - val_loss: 210356084058794.6562 - val_mean_squared_error: 210356089651200.0000
    Epoch 95/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 466422077533994.4375 - mean_squared_error: 466422073393152.0000 - val_loss: 210812600184073.4688 - val_mean_squared_error: 210812580921344.0000
    Epoch 96/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 466768926956457.3750 - mean_squared_error: 466768992665600.0000 - val_loss: 210048691285864.2812 - val_mean_squared_error: 210048680722432.0000
    Epoch 97/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 466246415980513.1875 - mean_squared_error: 466246348832768.0000 - val_loss: 210374716710608.5938 - val_mean_squared_error: 210374712360960.0000
    Epoch 98/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 466012927096070.1875 - mean_squared_error: 466012877094912.0000 - val_loss: 210791406210616.8750 - val_mean_squared_error: 210791408074752.0000
    Epoch 99/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 466851098649933.4375 - mean_squared_error: 466851033251840.0000 - val_loss: 209899464163328.0000 - val_mean_squared_error: 209899464163328.0000
    Epoch 100/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 466442082915751.5625 - mean_squared_error: 466442004725760.0000 - val_loss: 209603162372930.3750 - val_mean_squared_error: 209603144974336.0000
    Epoch 101/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 466035207744373.1250 - mean_squared_error: 466035190792192.0000 - val_loss: 210809786582812.4375 - val_mean_squared_error: 210809795903488.0000
    Epoch 102/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 465095507089073.4375 - mean_squared_error: 465095498924032.0000 - val_loss: 210449963767125.3438 - val_mean_squared_error: 210449958174720.0000
    Epoch 103/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 466221227372159.5625 - mean_squared_error: 466221283672064.0000 - val_loss: 210714920436318.8125 - val_mean_squared_error: 210714920747008.0000
    Epoch 104/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 464941599791452.8750 - mean_squared_error: 464941618298880.0000 - val_loss: 210133071361668.7188 - val_mean_squared_error: 210133086896128.0000
    Epoch 105/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 465037680399650.6875 - mean_squared_error: 465037751746560.0000 - val_loss: 210250115434268.4375 - val_mean_squared_error: 210250107977728.0000
    Epoch 106/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 465114943484089.0625 - mean_squared_error: 465114926940160.0000 - val_loss: 209779936441381.9375 - val_mean_squared_error: 209779960053760.0000
    Epoch 107/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 465055123970510.7500 - mean_squared_error: 465055166496768.0000 - val_loss: 210067066997722.0625 - val_mean_squared_error: 210067068551168.0000
    Epoch 108/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 464178452360755.6250 - mean_squared_error: 464178523406336.0000 - val_loss: 210135453104962.3750 - val_mean_squared_error: 210135452483584.0000
    Epoch 109/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 465537630821896.8750 - mean_squared_error: 465537578565632.0000 - val_loss: 209886759771856.5938 - val_mean_squared_error: 209886763810816.0000
    Epoch 110/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 463711322770701.3125 - mean_squared_error: 463711277940736.0000 - val_loss: 209602598316714.6562 - val_mean_squared_error: 209602591326208.0000
    Epoch 111/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 463902078083613.0625 - mean_squared_error: 463902068441088.0000 - val_loss: 210019216824244.1562 - val_mean_squared_error: 210019203153920.0000
    Epoch 112/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 463173380508068.0000 - mean_squared_error: 463173400395776.0000 - val_loss: 209196260825012.1562 - val_mean_squared_error: 209196247154688.0000
    Epoch 113/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 464230232885630.1250 - mean_squared_error: 464230230786048.0000 - val_loss: 208347410124951.7188 - val_mean_squared_error: 208347387133952.0000
    Epoch 114/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 462904126642752.6250 - mean_squared_error: 462904159633408.0000 - val_loss: 209015547217692.4375 - val_mean_squared_error: 209015539761152.0000
    Epoch 115/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 463435685126096.5625 - mean_squared_error: 463435661836288.0000 - val_loss: 208914077372567.7188 - val_mean_squared_error: 208914071158784.0000
    Epoch 116/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 462888692206690.5000 - mean_squared_error: 462888657485824.0000 - val_loss: 208692014762970.0625 - val_mean_squared_error: 208692007927808.0000
    Epoch 117/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 462120666523413.0625 - mean_squared_error: 462120697200640.0000 - val_loss: 207562338322204.4375 - val_mean_squared_error: 207562330865664.0000
    Epoch 118/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 462541524250410.4375 - mean_squared_error: 462541570441216.0000 - val_loss: 209091361903350.5312 - val_mean_squared_error: 209091356000256.0000
    Epoch 119/1000
    1726/1726 [==============================] - 0s 123us/sample - loss: 462566494268941.6250 - mean_squared_error: 462566568493056.0000 - val_loss: 209067559072805.9375 - val_mean_squared_error: 209067549130752.0000
    Epoch 120/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 462302910956829.9375 - mean_squared_error: 462302830657536.0000 - val_loss: 208277437328042.6562 - val_mean_squared_error: 208277442920448.0000
    Epoch 121/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 461409078539374.3750 - mean_squared_error: 461409175470080.0000 - val_loss: 208461741880964.7188 - val_mean_squared_error: 208461740638208.0000
    Epoch 122/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 462031525694942.1875 - mean_squared_error: 462031475965952.0000 - val_loss: 208616757143021.0312 - val_mean_squared_error: 208616745336832.0000
    Epoch 123/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 461680736076231.6250 - mean_squared_error: 461680765042688.0000 - val_loss: 210647631683280.5938 - val_mean_squared_error: 210647627333632.0000
    Epoch 124/1000
    1726/1726 [==============================] - 0s 123us/sample - loss: 461009260024458.1875 - mean_squared_error: 461009307303936.0000 - val_loss: 208070181543632.5938 - val_mean_squared_error: 208070160416768.0000
    Epoch 125/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 460236449607485.3750 - mean_squared_error: 460236448071680.0000 - val_loss: 209182392902542.2188 - val_mean_squared_error: 209182389174272.0000
    Epoch 126/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 460636921539559.0625 - mean_squared_error: 460636920217600.0000 - val_loss: 207599702424993.1875 - val_mean_squared_error: 207599710502912.0000
    Epoch 127/1000
    1726/1726 [==============================] - 0s 138us/sample - loss: 460547607223399.1875 - mean_squared_error: 460547698982912.0000 - val_loss: 207784285976803.5625 - val_mean_squared_error: 207784276656128.0000
    Epoch 128/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 460037413795609.8125 - mean_squared_error: 460037369626624.0000 - val_loss: 207032887910703.4062 - val_mean_squared_error: 207032892260352.0000
    Epoch 129/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 459666721907950.5000 - mean_squared_error: 459666794479616.0000 - val_loss: 208065900246660.7188 - val_mean_squared_error: 208065899003904.0000
    Epoch 130/1000
    1726/1726 [==============================] - 0s 128us/sample - loss: 459369760344047.3750 - mean_squared_error: 459369770647552.0000 - val_loss: 207941142849005.0312 - val_mean_squared_error: 207941160402944.0000
    Epoch 131/1000
    1726/1726 [==============================] - 0s 133us/sample - loss: 459986942564283.1875 - mean_squared_error: 459986903760896.0000 - val_loss: 207193416663040.0000 - val_mean_squared_error: 207193416663040.0000
    Epoch 132/1000
    1726/1726 [==============================] - 0s 136us/sample - loss: 459029678788309.0000 - mean_squared_error: 459029662924800.0000 - val_loss: 207519557042782.8125 - val_mean_squared_error: 207519532187648.0000
    Epoch 133/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 458648294837270.5000 - mean_squared_error: 458648350359552.0000 - val_loss: 207704096010695.1250 - val_mean_squared_error: 207704098340864.0000
    Epoch 134/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 458768463872816.3125 - mean_squared_error: 458768441671680.0000 - val_loss: 207075092550997.3438 - val_mean_squared_error: 207075103735808.0000
    Epoch 135/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 459146533886281.8125 - mean_squared_error: 459146533011456.0000 - val_loss: 206826350439689.4688 - val_mean_squared_error: 206826364731392.0000
    Epoch 136/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 458516695987974.8750 - mean_squared_error: 458516682768384.0000 - val_loss: 206033492765961.4688 - val_mean_squared_error: 206033490280448.0000
    Epoch 137/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 458162276308505.5000 - mean_squared_error: 458162314412032.0000 - val_loss: 207736819504469.3438 - val_mean_squared_error: 207736830689280.0000
    Epoch 138/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 457426959703276.1250 - mean_squared_error: 457426969034752.0000 - val_loss: 207999704808865.1875 - val_mean_squared_error: 207999696109568.0000
    Epoch 139/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 457591297569105.0000 - mean_squared_error: 457591285088256.0000 - val_loss: 207839100559056.5938 - val_mean_squared_error: 207839087820800.0000
    Epoch 140/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 457365308936158.7500 - mean_squared_error: 457365262434304.0000 - val_loss: 206652984010221.0312 - val_mean_squared_error: 206652988981248.0000
    Epoch 141/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 456499368854149.4375 - mean_squared_error: 456499356762112.0000 - val_loss: 206712124628688.5938 - val_mean_squared_error: 206712128667648.0000
    Epoch 142/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 457058660658758.6250 - mean_squared_error: 457058709143552.0000 - val_loss: 205978011910750.8125 - val_mean_squared_error: 205978008027136.0000
    Epoch 143/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 457231084481719.9375 - mean_squared_error: 457231078260736.0000 - val_loss: 206501578956496.5938 - val_mean_squared_error: 206501574606848.0000
    Epoch 144/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 456394628305850.0000 - mean_squared_error: 456394599825408.0000 - val_loss: 206377154151613.6250 - val_mean_squared_error: 206377154772992.0000
    Epoch 145/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 456151819804282.8125 - mean_squared_error: 456151799955456.0000 - val_loss: 206378359625652.1562 - val_mean_squared_error: 206378362732544.0000
    Epoch 146/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 455355823119273.3750 - mean_squared_error: 455355754610688.0000 - val_loss: 206369381174234.0625 - val_mean_squared_error: 206369403699200.0000
    Epoch 147/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 455145533528052.1250 - mean_squared_error: 455145603203072.0000 - val_loss: 206242299044750.2188 - val_mean_squared_error: 206242282733568.0000
    Epoch 148/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 454865553170099.7500 - mean_squared_error: 454865557913600.0000 - val_loss: 206148419181074.9688 - val_mean_squared_error: 206148414210048.0000
    Epoch 149/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 454793702827683.1250 - mean_squared_error: 454793751429120.0000 - val_loss: 206511903158120.2812 - val_mean_squared_error: 206511875817472.0000
    Epoch 150/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 455096902682474.5000 - mean_squared_error: 455096848613376.0000 - val_loss: 206825441207789.0312 - val_mean_squared_error: 206825441984512.0000
    Epoch 151/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 454934093326974.3750 - mean_squared_error: 454934042509312.0000 - val_loss: 205638866858439.1250 - val_mean_squared_error: 205638873382912.0000
    Epoch 152/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 453844884027655.4375 - mean_squared_error: 453844966309888.0000 - val_loss: 206619753627041.1875 - val_mean_squared_error: 206619736539136.0000
    Epoch 153/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 453820742109566.1250 - mean_squared_error: 453820773564416.0000 - val_loss: 205708734953092.7188 - val_mean_squared_error: 205708733710336.0000
    Epoch 154/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 453692017043790.5625 - mean_squared_error: 453692025208832.0000 - val_loss: 205313238805162.6562 - val_mean_squared_error: 205313261174784.0000
    Epoch 155/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 453176946481254.0625 - mean_squared_error: 453176998232064.0000 - val_loss: 206588316852527.4062 - val_mean_squared_error: 206588279259136.0000
    Epoch 156/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 453885936193597.6875 - mean_squared_error: 453885969825792.0000 - val_loss: 205553814140548.7188 - val_mean_squared_error: 205553812897792.0000
    Epoch 157/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 453786560524625.0000 - mean_squared_error: 453786548043776.0000 - val_loss: 204487700357423.4062 - val_mean_squared_error: 204487687929856.0000
    Epoch 158/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 453430396364622.0000 - mean_squared_error: 453430334193664.0000 - val_loss: 204558274027216.5938 - val_mean_squared_error: 204558269677568.0000
    Epoch 159/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 452910924698971.6250 - mean_squared_error: 452910945140736.0000 - val_loss: 205889889117677.0312 - val_mean_squared_error: 205889877311488.0000
    Epoch 160/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 452760733671338.5625 - mean_squared_error: 452760688394240.0000 - val_loss: 204823199316802.3750 - val_mean_squared_error: 204823198695424.0000
    Epoch 161/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 453319867404634.4375 - mean_squared_error: 453319805894656.0000 - val_loss: 204417265876385.1875 - val_mean_squared_error: 204417273954304.0000
    Epoch 162/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 450608040474204.0000 - mean_squared_error: 450607936700416.0000 - val_loss: 206125987421904.5938 - val_mean_squared_error: 206125983072256.0000
    Epoch 163/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 451860055206887.0625 - mean_squared_error: 451860053884928.0000 - val_loss: 204814027150677.3438 - val_mean_squared_error: 204814004781056.0000
    Epoch 164/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 451118205732600.5625 - mean_squared_error: 451118198947840.0000 - val_loss: 204908774303213.0312 - val_mean_squared_error: 204908779274240.0000
    Epoch 165/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 450861862527934.7500 - mean_squared_error: 450861943750656.0000 - val_loss: 204703098062620.4375 - val_mean_squared_error: 204703090606080.0000
    Epoch 166/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 450866053702002.2500 - mean_squared_error: 450866104500224.0000 - val_loss: 205411933574485.3438 - val_mean_squared_error: 205411944759296.0000
    Epoch 167/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 450265238793248.0625 - mean_squared_error: 450265312395264.0000 - val_loss: 204549311420946.9688 - val_mean_squared_error: 204549327421440.0000
    Epoch 168/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 449719150093854.2500 - mean_squared_error: 449719180460032.0000 - val_loss: 203684223327345.7812 - val_mean_squared_error: 203684227055616.0000
    Epoch 169/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 449872650850175.8750 - mean_squared_error: 449872691986432.0000 - val_loss: 203540129722216.2812 - val_mean_squared_error: 203540127547392.0000
    Epoch 170/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 449219169111024.5625 - mean_squared_error: 449219185868800.0000 - val_loss: 204369470694893.0312 - val_mean_squared_error: 204369475665920.0000
    Epoch 171/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 449131173293524.6875 - mean_squared_error: 449131139039232.0000 - val_loss: 204914840820242.9688 - val_mean_squared_error: 204914852626432.0000
    Epoch 172/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 448226187939844.7500 - mean_squared_error: 448226209562624.0000 - val_loss: 204573391541589.3438 - val_mean_squared_error: 204573385949184.0000
    Epoch 173/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 448540722446772.6250 - mean_squared_error: 448540748808192.0000 - val_loss: 204886404371190.5312 - val_mean_squared_error: 204886381690880.0000
    Epoch 174/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 447898967855241.6250 - mean_squared_error: 447899020296192.0000 - val_loss: 204449992477051.2812 - val_mean_squared_error: 204449989525504.0000
    Epoch 175/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 447677679438092.1875 - mean_squared_error: 447677661708288.0000 - val_loss: 203155522919537.7812 - val_mean_squared_error: 203155509870592.0000
    Epoch 176/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 447782520785771.6875 - mean_squared_error: 447782452199424.0000 - val_loss: 202867039933174.5312 - val_mean_squared_error: 202867008864256.0000
    Epoch 177/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 447043060295672.8750 - mean_squared_error: 447043180953600.0000 - val_loss: 203336574556690.9688 - val_mean_squared_error: 203336569585664.0000
    Epoch 178/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 446830572646148.4375 - mean_squared_error: 446830479409152.0000 - val_loss: 204908363261421.0312 - val_mean_squared_error: 204908359843840.0000
    Epoch 179/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 446966770591091.3750 - mean_squared_error: 446966777511936.0000 - val_loss: 203691235582255.4062 - val_mean_squared_error: 203691223154688.0000
    Epoch 180/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 445973071523227.6875 - mean_squared_error: 445973130117120.0000 - val_loss: 202204960036788.1562 - val_mean_squared_error: 202204979920896.0000
    Epoch 181/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 445893195839390.6875 - mean_squared_error: 445893169905664.0000 - val_loss: 203794119684551.1250 - val_mean_squared_error: 203794134597632.0000
    Epoch 182/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 446220441677895.1875 - mean_squared_error: 446220459835392.0000 - val_loss: 203831569692899.5625 - val_mean_squared_error: 203831547789312.0000
    Epoch 183/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 445492860525956.0000 - mean_squared_error: 445492932640768.0000 - val_loss: 203494953961244.4375 - val_mean_squared_error: 203494946504704.0000
    Epoch 184/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 446069107962439.8125 - mean_squared_error: 446069095792640.0000 - val_loss: 202485561149212.4375 - val_mean_squared_error: 202485562081280.0000
    Epoch 185/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 444555919722688.1875 - mean_squared_error: 444556025790464.0000 - val_loss: 202989858474970.0625 - val_mean_squared_error: 202989851639808.0000
    Epoch 186/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 444636989621590.8750 - mean_squared_error: 444637059743744.0000 - val_loss: 205182900963707.2812 - val_mean_squared_error: 205182918983680.0000
    Epoch 187/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 443905802174995.6250 - mean_squared_error: 443905808007168.0000 - val_loss: 203854672385365.3438 - val_mean_squared_error: 203854650015744.0000
    Epoch 188/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 443152749661263.5000 - mean_squared_error: 443152745889792.0000 - val_loss: 203118442475975.1250 - val_mean_squared_error: 203118432223232.0000
    Epoch 189/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 443206816475489.6250 - mean_squared_error: 443206835634176.0000 - val_loss: 203572221429532.4375 - val_mean_squared_error: 203572222361600.0000
    Epoch 190/1000
    1726/1726 [==============================] - 0s 84us/sample - loss: 443109572805747.1250 - mean_squared_error: 443109628444672.0000 - val_loss: 203037313762493.6250 - val_mean_squared_error: 203037314383872.0000
    Epoch 191/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 443053674038121.3125 - mean_squared_error: 443053659652096.0000 - val_loss: 201359567928737.1875 - val_mean_squared_error: 201359559229440.0000
    Epoch 192/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 441823652466521.8750 - mean_squared_error: 441823621283840.0000 - val_loss: 203388878459638.5312 - val_mean_squared_error: 203388897722368.0000
    Epoch 193/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 441693000820349.1875 - mean_squared_error: 441693027434496.0000 - val_loss: 202649269805359.4062 - val_mean_squared_error: 202649257377792.0000
    Epoch 194/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 441705985374623.2500 - mean_squared_error: 441706012999680.0000 - val_loss: 203330043248640.0000 - val_mean_squared_error: 203330043248640.0000
    Epoch 195/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 441459902574815.0625 - mean_squared_error: 441459958349824.0000 - val_loss: 202762601142196.1562 - val_mean_squared_error: 202762587471872.0000
    Epoch 196/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 441422659609667.6250 - mean_squared_error: 441422712930304.0000 - val_loss: 202730610098176.0000 - val_mean_squared_error: 202730610098176.0000
    Epoch 197/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 440828349108969.1875 - mean_squared_error: 440828430385152.0000 - val_loss: 202673280176279.7188 - val_mean_squared_error: 202673265573888.0000
    Epoch 198/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 440518751203076.4375 - mean_squared_error: 440518756532224.0000 - val_loss: 201479138991824.5938 - val_mean_squared_error: 201479147225088.0000
    Epoch 199/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 439703677542827.1250 - mean_squared_error: 439703618715648.0000 - val_loss: 201551890592805.9375 - val_mean_squared_error: 201551876456448.0000
    Epoch 200/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 440140122547685.3125 - mean_squared_error: 440140128321536.0000 - val_loss: 202581693043446.5312 - val_mean_squared_error: 202581695528960.0000
    Epoch 201/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 439652267176216.0625 - mean_squared_error: 439652313989120.0000 - val_loss: 202883390262234.0625 - val_mean_squared_error: 202883383427072.0000
    Epoch 202/1000
    1726/1726 [==============================] - 0s 83us/sample - loss: 438623059908311.3125 - mean_squared_error: 438623031787520.0000 - val_loss: 202492488585974.5312 - val_mean_squared_error: 202492474294272.0000
    Epoch 203/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 438063140474467.0625 - mean_squared_error: 438063176089600.0000 - val_loss: 202705090088504.8750 - val_mean_squared_error: 202705091952640.0000
    Epoch 204/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 438870122697771.8750 - mean_squared_error: 438870126624768.0000 - val_loss: 201929662699444.1562 - val_mean_squared_error: 201929649029120.0000
    Epoch 205/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 437216621507181.7500 - mean_squared_error: 437216597770240.0000 - val_loss: 202069404481156.7188 - val_mean_squared_error: 202069403238400.0000
    Epoch 206/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 437168277332319.1875 - mean_squared_error: 437168279388160.0000 - val_loss: 202349919221987.5625 - val_mean_squared_error: 202349935067136.0000
    Epoch 207/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 436700061310605.7500 - mean_squared_error: 436700060844032.0000 - val_loss: 200620524728775.1250 - val_mean_squared_error: 200620522864640.0000
    Epoch 208/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 436172639645482.4375 - mean_squared_error: 436172719390720.0000 - val_loss: 201920086016000.0000 - val_mean_squared_error: 201920086016000.0000
    Epoch 209/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 436059286261593.8750 - mean_squared_error: 436059238301696.0000 - val_loss: 200722172288113.7812 - val_mean_squared_error: 200722159239168.0000
    Epoch 210/1000
    1726/1726 [==============================] - 0s 83us/sample - loss: 436504610748964.1250 - mean_squared_error: 436504673386496.0000 - val_loss: 199451691508622.2188 - val_mean_squared_error: 199451704557568.0000
    Epoch 211/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 435544565330503.8125 - mean_squared_error: 435544479760384.0000 - val_loss: 200486700333283.5625 - val_mean_squared_error: 200486707789824.0000
    Epoch 212/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 435186011464581.7500 - mean_squared_error: 435185984208896.0000 - val_loss: 202016854512374.5312 - val_mean_squared_error: 202016856997888.0000
    Epoch 213/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 435015090631334.6875 - mean_squared_error: 435015057932288.0000 - val_loss: 202173222825832.2812 - val_mean_squared_error: 202173220651008.0000
    Epoch 214/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 435365898990759.3125 - mean_squared_error: 435365869518848.0000 - val_loss: 200919493475138.3750 - val_mean_squared_error: 200919492853760.0000
    Epoch 215/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 433556175499468.1250 - mean_squared_error: 433556211892224.0000 - val_loss: 200604986541245.6250 - val_mean_squared_error: 200604970385408.0000
    Epoch 216/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 433069011259871.3750 - mean_squared_error: 433069001539584.0000 - val_loss: 200453255574565.9375 - val_mean_squared_error: 200453237243904.0000
    Epoch 217/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 432502057108253.3750 - mean_squared_error: 432502032302080.0000 - val_loss: 202776481181544.2812 - val_mean_squared_error: 202776495783936.0000
    Epoch 218/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 432172088150682.8125 - mean_squared_error: 432172091572224.0000 - val_loss: 200995899402315.8438 - val_mean_squared_error: 200995896295424.0000
    Epoch 219/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 432102545501690.6875 - mean_squared_error: 432102566789120.0000 - val_loss: 201511467754989.0312 - val_mean_squared_error: 201511476920320.0000
    Epoch 220/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 431016702133761.7500 - mean_squared_error: 431016711815168.0000 - val_loss: 198728019966786.3750 - val_mean_squared_error: 198728019345408.0000
    Epoch 221/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 431697875601068.6250 - mean_squared_error: 431697732567040.0000 - val_loss: 201367303468069.9375 - val_mean_squared_error: 201367310303232.0000
    Epoch 222/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 430574372914063.3125 - mean_squared_error: 430574330183680.0000 - val_loss: 200876203597520.5938 - val_mean_squared_error: 200876224413696.0000
    Epoch 223/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 430133852970334.0625 - mean_squared_error: 430133827600384.0000 - val_loss: 200021692451195.2812 - val_mean_squared_error: 200021676916736.0000
    Epoch 224/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 429805077178038.1875 - mean_squared_error: 429805094830080.0000 - val_loss: 199507818131304.2812 - val_mean_squared_error: 199507807567872.0000
    Epoch 225/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 429597886842123.0000 - mean_squared_error: 429597829103616.0000 - val_loss: 199030028496137.4688 - val_mean_squared_error: 199030042787840.0000
    Epoch 226/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 429852386710932.6250 - mean_squared_error: 429852473688064.0000 - val_loss: 198926692651766.5312 - val_mean_squared_error: 198926678360064.0000
    Epoch 227/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 429733290600706.6875 - mean_squared_error: 429733254791168.0000 - val_loss: 197924789313232.5938 - val_mean_squared_error: 197924776574976.0000
    Epoch 228/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 427417332521384.8125 - mean_squared_error: 427417395003392.0000 - val_loss: 199805485556015.4062 - val_mean_squared_error: 199805485711360.0000
    Epoch 229/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 428884985180686.8125 - mean_squared_error: 428885032304640.0000 - val_loss: 200488213389615.4062 - val_mean_squared_error: 200488200962048.0000
    Epoch 230/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 427230860841914.0000 - mean_squared_error: 427230865915904.0000 - val_loss: 199343833893850.0625 - val_mean_squared_error: 199343827058688.0000
    Epoch 231/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 427321939827644.3750 - mean_squared_error: 427321966198784.0000 - val_loss: 198985827522484.1562 - val_mean_squared_error: 198985834823680.0000
    Epoch 232/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 426321622245271.6250 - mean_squared_error: 426321574363136.0000 - val_loss: 199352333728578.3750 - val_mean_squared_error: 199352333107200.0000
    Epoch 233/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 425874978053925.6875 - mean_squared_error: 425874931318784.0000 - val_loss: 200922960766445.0312 - val_mean_squared_error: 200922965737472.0000
    Epoch 234/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 425359547976840.4375 - mean_squared_error: 425359602352128.0000 - val_loss: 200109015996340.1562 - val_mean_squared_error: 200109035880448.0000
    Epoch 235/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 425464131678582.9375 - mean_squared_error: 425464191516672.0000 - val_loss: 199209224697476.7188 - val_mean_squared_error: 199209240231936.0000
    Epoch 236/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 424148249953262.1875 - mean_squared_error: 424148220248064.0000 - val_loss: 199352357651645.6250 - val_mean_squared_error: 199352383438848.0000
    Epoch 237/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 423844870394658.1250 - mean_squared_error: 423844854628352.0000 - val_loss: 200519201218256.5938 - val_mean_squared_error: 200519171702784.0000
    Epoch 238/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 423696301148371.2500 - mean_squared_error: 423696275603456.0000 - val_loss: 198663492929915.2812 - val_mean_squared_error: 198663494172672.0000
    Epoch 239/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 423304931196187.5625 - mean_squared_error: 423304963817472.0000 - val_loss: 201621688937737.4688 - val_mean_squared_error: 201621686452224.0000
    Epoch 240/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 423980395734596.2500 - mean_squared_error: 423980515196928.0000 - val_loss: 200810481960732.4375 - val_mean_squared_error: 200810457726976.0000
    Epoch 241/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 421710799324066.2500 - mean_squared_error: 421710792753152.0000 - val_loss: 199129730209109.3438 - val_mean_squared_error: 199129733005312.0000
    Epoch 242/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 421134866018109.3750 - mean_squared_error: 421134763819008.0000 - val_loss: 200788046473291.8438 - val_mean_squared_error: 200788043366400.0000
    Epoch 243/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 420560560976930.3750 - mean_squared_error: 420560479715328.0000 - val_loss: 198128684683415.7188 - val_mean_squared_error: 198128670081024.0000
    Epoch 244/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 419515000871052.0000 - mean_squared_error: 419514957168640.0000 - val_loss: 199362861120929.1875 - val_mean_squared_error: 199362835644416.0000
    Epoch 245/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 420028762441369.6875 - mean_squared_error: 420028776185856.0000 - val_loss: 200762054216059.2812 - val_mean_squared_error: 200762038681600.0000
    Epoch 246/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 419297534675022.3125 - mean_squared_error: 419297423785984.0000 - val_loss: 199755214337061.9375 - val_mean_squared_error: 199755204395008.0000
    Epoch 247/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 419280258652755.6875 - mean_squared_error: 419280311025664.0000 - val_loss: 198606279516463.4062 - val_mean_squared_error: 198606283866112.0000
    Epoch 248/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 417592611540958.7500 - mean_squared_error: 417592657313792.0000 - val_loss: 199969618768933.9375 - val_mean_squared_error: 199969617215488.0000
    Epoch 249/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 417937603873358.8750 - mean_squared_error: 417937596874752.0000 - val_loss: 198279707278753.1875 - val_mean_squared_error: 198279715356672.0000
    Epoch 250/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 417678185500259.0625 - mean_squared_error: 417678221115392.0000 - val_loss: 199943598860363.8438 - val_mean_squared_error: 199943612530688.0000
    Epoch 251/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 416002795913842.5000 - mean_squared_error: 416002747662336.0000 - val_loss: 198868486897057.1875 - val_mean_squared_error: 198868478197760.0000
    Epoch 252/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 416079504173998.1250 - mean_squared_error: 416079553757184.0000 - val_loss: 200939266977640.2812 - val_mean_squared_error: 200939256414208.0000
    Epoch 253/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 415590067039605.8125 - mean_squared_error: 415590028148736.0000 - val_loss: 196912183353647.4062 - val_mean_squared_error: 196912187703296.0000
    Epoch 254/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 415207832981394.8125 - mean_squared_error: 415207843168256.0000 - val_loss: 199379424582769.7812 - val_mean_squared_error: 199379445088256.0000
    Epoch 255/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 415291977222507.0625 - mean_squared_error: 415291930574848.0000 - val_loss: 196561326095777.1875 - val_mean_squared_error: 196561325785088.0000
    Epoch 256/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 413284012484790.7500 - mean_squared_error: 413283966255104.0000 - val_loss: 197956154007855.4062 - val_mean_squared_error: 197956149968896.0000
    Epoch 257/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 413355358791426.0625 - mean_squared_error: 413355403640832.0000 - val_loss: 199453873789458.9688 - val_mean_squared_error: 199453885595648.0000
    Epoch 258/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 412571929854768.3125 - mean_squared_error: 412571974762496.0000 - val_loss: 197376406434853.9375 - val_mean_squared_error: 197376413270016.0000
    Epoch 259/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 412960301331585.3125 - mean_squared_error: 412960333758464.0000 - val_loss: 197428082745647.4062 - val_mean_squared_error: 197428070318080.0000
    Epoch 260/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 412308351674968.4375 - mean_squared_error: 412308371144704.0000 - val_loss: 198046338689706.6562 - val_mean_squared_error: 198046344282112.0000
    Epoch 261/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 411849907978183.0625 - mean_squared_error: 411849950494720.0000 - val_loss: 195665126053205.3438 - val_mean_squared_error: 195665154015232.0000
    Epoch 262/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 411716825779013.6875 - mean_squared_error: 411716772954112.0000 - val_loss: 196389691758516.1562 - val_mean_squared_error: 196389661310976.0000
    Epoch 263/1000
    1726/1726 [==============================] - 0s 84us/sample - loss: 409519790647858.4375 - mean_squared_error: 409519863627776.0000 - val_loss: 198530241445281.1875 - val_mean_squared_error: 198530249523200.0000
    Epoch 264/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 408947202345512.9375 - mean_squared_error: 408947190136832.0000 - val_loss: 197371987968606.8125 - val_mean_squared_error: 197371984084992.0000
    Epoch 265/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 408051382955514.6875 - mean_squared_error: 408051320356864.0000 - val_loss: 199643181991708.4375 - val_mean_squared_error: 199643182923776.0000
    Epoch 266/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 407694163852723.4375 - mean_squared_error: 407694066319360.0000 - val_loss: 198451006382383.4062 - val_mean_squared_error: 198451010732032.0000
    Epoch 267/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 408054435106308.1250 - mean_squared_error: 408054474473472.0000 - val_loss: 201598046112123.2812 - val_mean_squared_error: 201598030577664.0000
    Epoch 268/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 406553247951315.5625 - mean_squared_error: 406553316294656.0000 - val_loss: 198521400473827.5625 - val_mean_squared_error: 198521407930368.0000
    Epoch 269/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 405505578902728.5000 - mean_squared_error: 405505545601024.0000 - val_loss: 198386915553128.2812 - val_mean_squared_error: 198386921766912.0000
    Epoch 270/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 406609597575030.3750 - mean_squared_error: 406609553522688.0000 - val_loss: 200796469257102.2188 - val_mean_squared_error: 200796465528832.0000
    Epoch 271/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 404657115644682.3750 - mean_squared_error: 404657155342336.0000 - val_loss: 196246523075090.9688 - val_mean_squared_error: 196246518104064.0000
    Epoch 272/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 404130884797384.2500 - mean_squared_error: 404130887630848.0000 - val_loss: 197679371886895.4062 - val_mean_squared_error: 197679342682112.0000
    Epoch 273/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 403454475228336.7500 - mean_squared_error: 403454564499456.0000 - val_loss: 196056931834993.7812 - val_mean_squared_error: 196056952340480.0000
    Epoch 274/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 403047311194318.4375 - mean_squared_error: 403047247249408.0000 - val_loss: 199714128177834.6562 - val_mean_squared_error: 199714133770240.0000
    Epoch 275/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 401834708042335.5000 - mean_squared_error: 401834724294656.0000 - val_loss: 197614636688270.2188 - val_mean_squared_error: 197614632960000.0000
    Epoch 276/1000
    1726/1726 [==============================] - 0s 84us/sample - loss: 400954636689170.6875 - mean_squared_error: 400954658652160.0000 - val_loss: 200709439002548.1562 - val_mean_squared_error: 200709425332224.0000
    Epoch 277/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 400921432668670.2500 - mean_squared_error: 400921506873344.0000 - val_loss: 200816194292091.2812 - val_mean_squared_error: 200816178757632.0000
    Epoch 278/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 400455202370442.5000 - mean_squared_error: 400455200931840.0000 - val_loss: 197435432409012.1562 - val_mean_squared_error: 197435435515904.0000
    Epoch 279/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 401208861430106.4375 - mean_squared_error: 401208799920128.0000 - val_loss: 195697056512834.3750 - val_mean_squared_error: 195697064280064.0000
    Epoch 280/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 399093850151716.5000 - mean_squared_error: 399093830516736.0000 - val_loss: 198290745910158.2188 - val_mean_squared_error: 198290721210368.0000
    Epoch 281/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 398233113961634.5625 - mean_squared_error: 398233159335936.0000 - val_loss: 200234290225455.4062 - val_mean_squared_error: 200234277797888.0000
    Epoch 282/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 397500029950889.3750 - mean_squared_error: 397499961442304.0000 - val_loss: 197623924430772.1562 - val_mean_squared_error: 197623927537664.0000
    Epoch 283/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 397292441316479.0000 - mean_squared_error: 397292427280384.0000 - val_loss: 197724080215988.1562 - val_mean_squared_error: 197724070739968.0000
    Epoch 284/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 396227457821329.3750 - mean_squared_error: 396227543826432.0000 - val_loss: 198307853854795.8438 - val_mean_squared_error: 198307867525120.0000
    Epoch 285/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 395448121125654.2500 - mean_squared_error: 395448141479936.0000 - val_loss: 200698019621546.6562 - val_mean_squared_error: 200698016825344.0000
    Epoch 286/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 394836981706747.2500 - mean_squared_error: 394837014609920.0000 - val_loss: 196285999242960.5938 - val_mean_squared_error: 196286011670528.0000
    Epoch 287/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 394136286908553.6250 - mean_squared_error: 394136263852032.0000 - val_loss: 200034104639488.0000 - val_mean_squared_error: 200034108833792.0000
    Epoch 288/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 393871898008185.5625 - mean_squared_error: 393871888482304.0000 - val_loss: 202896483947254.5312 - val_mean_squared_error: 202896503209984.0000
    Epoch 289/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 393432946704991.5000 - mean_squared_error: 393432929402880.0000 - val_loss: 198684245258695.1250 - val_mean_squared_error: 198684247588864.0000
    Epoch 290/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 392947707475965.6250 - mean_squared_error: 392947765870592.0000 - val_loss: 201902711344696.8750 - val_mean_squared_error: 201902721597440.0000
    Epoch 291/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 391337517120472.8750 - mean_squared_error: 391337522233344.0000 - val_loss: 198048925487862.5312 - val_mean_squared_error: 198048911196160.0000
    Epoch 292/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 390904891313725.0625 - mean_squared_error: 390904938496000.0000 - val_loss: 201504834385540.7188 - val_mean_squared_error: 201504833142784.0000
    Epoch 293/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 390379891079108.6875 - mean_squared_error: 390379845189632.0000 - val_loss: 200705820716297.4688 - val_mean_squared_error: 200705818230784.0000
    Epoch 294/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 390155646450402.0625 - mean_squared_error: 390155668029440.0000 - val_loss: 196130202286914.3750 - val_mean_squared_error: 196130201665536.0000
    Epoch 295/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 389402209479863.9375 - mean_squared_error: 389402169704448.0000 - val_loss: 200322620403560.2812 - val_mean_squared_error: 200322626617344.0000
    Epoch 296/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 388974092992189.2500 - mean_squared_error: 388974182924288.0000 - val_loss: 202567041562851.5625 - val_mean_squared_error: 202567015464960.0000
    Epoch 297/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 387907053758324.0000 - mean_squared_error: 387907118432256.0000 - val_loss: 200791222959521.1875 - val_mean_squared_error: 200791231037440.0000
    Epoch 298/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 387380913999838.7500 - mean_squared_error: 387380917829632.0000 - val_loss: 204220418179678.8125 - val_mean_squared_error: 204220426878976.0000
    Epoch 299/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 386209633155164.5625 - mean_squared_error: 386209666826240.0000 - val_loss: 201337720886613.3438 - val_mean_squared_error: 201337732071424.0000
    Epoch 300/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 385550843342045.8750 - mean_squared_error: 385550825553920.0000 - val_loss: 200139417554488.8750 - val_mean_squared_error: 200139436195840.0000
    Epoch 301/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 384786084161339.0625 - mean_squared_error: 384786120048640.0000 - val_loss: 200179973367277.0312 - val_mean_squared_error: 200179969949696.0000
    Epoch 302/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 384110561709458.2500 - mean_squared_error: 384110535114752.0000 - val_loss: 198962518843088.5938 - val_mean_squared_error: 198962514493440.0000
    Epoch 303/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 383648216396245.8750 - mean_squared_error: 383648222150656.0000 - val_loss: 198384015580273.7812 - val_mean_squared_error: 198384019308544.0000
    Epoch 304/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 383666997498088.5000 - mean_squared_error: 383666945523712.0000 - val_loss: 199149216945493.3438 - val_mean_squared_error: 199149194575872.0000
    Epoch 305/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 381190044738855.4375 - mean_squared_error: 381190058016768.0000 - val_loss: 200390775358046.8125 - val_mean_squared_error: 200390758891520.0000
    Epoch 306/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 380827211703534.5000 - mean_squared_error: 380827200389120.0000 - val_loss: 205417022663338.6562 - val_mean_squared_error: 205417028255744.0000
    Epoch 307/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 380750863200934.6875 - mean_squared_error: 380750864056320.0000 - val_loss: 202954803414205.6250 - val_mean_squared_error: 202954804035584.0000
    Epoch 308/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 381132659136633.0000 - mean_squared_error: 381132713492480.0000 - val_loss: 205678501788482.3750 - val_mean_squared_error: 205678501167104.0000
    Epoch 309/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 380921535078181.6875 - mean_squared_error: 380921521897472.0000 - val_loss: 199376606321170.9688 - val_mean_squared_error: 199376609738752.0000
    Epoch 310/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 378256596558915.6250 - mean_squared_error: 378256595353600.0000 - val_loss: 198966335970417.7812 - val_mean_squared_error: 198966356475904.0000
    Epoch 311/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 377642276652261.0000 - mean_squared_error: 377642280812544.0000 - val_loss: 207127316296135.1250 - val_mean_squared_error: 207127347986432.0000
    Epoch 312/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 377315174372680.6875 - mean_squared_error: 377315125100544.0000 - val_loss: 196792147810228.1562 - val_mean_squared_error: 196792146722816.0000
    Epoch 313/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 376849894028417.3125 - mean_squared_error: 376849859346432.0000 - val_loss: 200633027172276.1562 - val_mean_squared_error: 200633038667776.0000
    Epoch 314/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 375527751388239.5000 - mean_squared_error: 375527814725632.0000 - val_loss: 201589956697808.5938 - val_mean_squared_error: 201589943959552.0000
    Epoch 315/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 374137026560664.5000 - mean_squared_error: 374137017073664.0000 - val_loss: 202094432669847.7188 - val_mean_squared_error: 202094434844672.0000
    Epoch 316/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 374046749002907.4375 - mean_squared_error: 374046688542720.0000 - val_loss: 203260091578595.5625 - val_mean_squared_error: 203260099035136.0000
    Epoch 317/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 373380331660849.2500 - mean_squared_error: 373380297523200.0000 - val_loss: 198820472990378.6562 - val_mean_squared_error: 198820478582784.0000
    Epoch 318/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 373989320106525.0625 - mean_squared_error: 373989276909568.0000 - val_loss: 200276531216384.0000 - val_mean_squared_error: 200276539604992.0000
    Epoch 319/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 370936166568187.5625 - mean_squared_error: 370936125587456.0000 - val_loss: 206812382475301.9375 - val_mean_squared_error: 206812372533248.0000
    Epoch 320/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 371368312076095.8125 - mean_squared_error: 371368340226048.0000 - val_loss: 204805664640568.8750 - val_mean_squared_error: 204805649727488.0000
    Epoch 321/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 369997566871516.4375 - mean_squared_error: 369997541015552.0000 - val_loss: 204178544734056.2812 - val_mean_squared_error: 204178550947840.0000
    Epoch 322/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 369462176398164.0000 - mean_squared_error: 369462180052992.0000 - val_loss: 200517503923237.9375 - val_mean_squared_error: 200517510758400.0000
    Epoch 323/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 369539409582998.3750 - mean_squared_error: 369539388801024.0000 - val_loss: 200403953239836.4375 - val_mean_squared_error: 200403945783296.0000
    Epoch 324/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 367655073797676.4375 - mean_squared_error: 367655039008768.0000 - val_loss: 206340619899638.5312 - val_mean_squared_error: 206340613996544.0000
    Epoch 325/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 368917008841802.8125 - mean_squared_error: 368916954087424.0000 - val_loss: 206720130467612.4375 - val_mean_squared_error: 206720131399680.0000
    Epoch 326/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 366839438662037.8125 - mean_squared_error: 366839364321280.0000 - val_loss: 211479598290261.3438 - val_mean_squared_error: 211479592697856.0000
    Epoch 327/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 365469866171248.3750 - mean_squared_error: 365469873733632.0000 - val_loss: 208014530586093.0312 - val_mean_squared_error: 208014527168512.0000
    Epoch 328/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 363831774017609.5625 - mean_squared_error: 363831712808960.0000 - val_loss: 200877741819676.4375 - val_mean_squared_error: 200877734363136.0000
    Epoch 329/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 363465809095977.8125 - mean_squared_error: 363465835282432.0000 - val_loss: 214249911227126.5312 - val_mean_squared_error: 214249913712640.0000
    Epoch 330/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 363625337033126.4375 - mean_squared_error: 363625319497728.0000 - val_loss: 206449904199149.0312 - val_mean_squared_error: 206449917558784.0000
    Epoch 331/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 362268860578587.0000 - mean_squared_error: 362268780920832.0000 - val_loss: 210508045867538.9688 - val_mean_squared_error: 210508040896512.0000
    Epoch 332/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 361638434826460.7500 - mean_squared_error: 361638460915712.0000 - val_loss: 212456766109316.7188 - val_mean_squared_error: 212456781643776.0000
    Epoch 333/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 361564803027537.2500 - mean_squared_error: 361564808937472.0000 - val_loss: 209536751938218.6562 - val_mean_squared_error: 209536757530624.0000
    Epoch 334/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 360691307735526.5000 - mean_squared_error: 360691319963648.0000 - val_loss: 206767477635299.5625 - val_mean_squared_error: 206767459926016.0000
    Epoch 335/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 358883233086861.5000 - mean_squared_error: 358883205840896.0000 - val_loss: 206745443247597.0312 - val_mean_squared_error: 206745448218624.0000
    Epoch 336/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 358492745202155.2500 - mean_squared_error: 358492732915712.0000 - val_loss: 207494254515541.3438 - val_mean_squared_error: 207494265700352.0000
    Epoch 337/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 358074221704966.8750 - mean_squared_error: 358074174930944.0000 - val_loss: 209273431047585.1875 - val_mean_squared_error: 209273439125504.0000
    Epoch 338/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 355897734877359.6250 - mean_squared_error: 355897767362560.0000 - val_loss: 205628674078340.7188 - val_mean_squared_error: 205628656058368.0000
    Epoch 339/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 356753739484277.5000 - mean_squared_error: 356753740922880.0000 - val_loss: 214906175643344.5938 - val_mean_squared_error: 214906188070912.0000
    Epoch 340/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 355180471625777.8750 - mean_squared_error: 355180440715264.0000 - val_loss: 211743788178545.7812 - val_mean_squared_error: 211743783518208.0000
    Epoch 341/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 354056143433647.3125 - mean_squared_error: 354056165916672.0000 - val_loss: 207023575623755.8438 - val_mean_squared_error: 207023547351040.0000
    Epoch 342/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 353783834608375.3750 - mean_squared_error: 353783871700992.0000 - val_loss: 211728485804107.8438 - val_mean_squared_error: 211728482697216.0000
    Epoch 343/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 352522339193406.3125 - mean_squared_error: 352522225057792.0000 - val_loss: 210240315831333.9375 - val_mean_squared_error: 210240326860800.0000
    Epoch 344/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 351251772261927.7500 - mean_squared_error: 351251753598976.0000 - val_loss: 204735490517067.8438 - val_mean_squared_error: 204735487410176.0000
    Epoch 345/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 351225943436566.8125 - mean_squared_error: 351225916686336.0000 - val_loss: 203798551355088.5938 - val_mean_squared_error: 203798547005440.0000
    Epoch 346/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 350659410639965.7500 - mean_squared_error: 350659417210880.0000 - val_loss: 215485056082830.2188 - val_mean_squared_error: 215485069131776.0000
    Epoch 347/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 349175584895104.1250 - mean_squared_error: 349175606673408.0000 - val_loss: 209232601517624.8750 - val_mean_squared_error: 209232620158976.0000
    Epoch 348/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 347129637468651.2500 - mean_squared_error: 347129625182208.0000 - val_loss: 215616468287488.0000 - val_mean_squared_error: 215616468287488.0000
    Epoch 349/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 346777696530503.1875 - mean_squared_error: 346777773408256.0000 - val_loss: 207283470234055.1250 - val_mean_squared_error: 207283476758528.0000
    Epoch 350/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 347814178986370.8125 - mean_squared_error: 347814202703872.0000 - val_loss: 205252853565819.2812 - val_mean_squared_error: 205252846419968.0000
    Epoch 351/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 345382914756309.0000 - mean_squared_error: 345382882115584.0000 - val_loss: 210867192002180.7188 - val_mean_squared_error: 210867190759424.0000
    Epoch 352/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 344647222325888.7500 - mean_squared_error: 344647234748416.0000 - val_loss: 208260970801227.8438 - val_mean_squared_error: 208260967694336.0000
    Epoch 353/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 344251196317173.9375 - mean_squared_error: 344251124678656.0000 - val_loss: 209264501219024.5938 - val_mean_squared_error: 209264513646592.0000
    Epoch 354/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 343255134043324.6250 - mean_squared_error: 343255094919168.0000 - val_loss: 218787832494914.3750 - val_mean_squared_error: 218787798319104.0000
    Epoch 355/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 342352264402533.4375 - mean_squared_error: 342352279371776.0000 - val_loss: 222361865420800.0000 - val_mean_squared_error: 222361848643584.0000
    Epoch 356/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 341591445379193.0000 - mean_squared_error: 341591466180608.0000 - val_loss: 207479074630314.6562 - val_mean_squared_error: 207479065542656.0000
    Epoch 357/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 339759971850645.8125 - mean_squared_error: 339759931064320.0000 - val_loss: 209487200120073.4688 - val_mean_squared_error: 209487180857344.0000
    Epoch 358/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 339141547066635.0000 - mean_squared_error: 339141556436992.0000 - val_loss: 213183108024699.2812 - val_mean_squared_error: 213183100878848.0000
    Epoch 359/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 338211258448951.7500 - mean_squared_error: 338211259809792.0000 - val_loss: 209112133650204.4375 - val_mean_squared_error: 209112142970880.0000
    Epoch 360/1000
    1726/1726 [==============================] - ETA: 0s - loss: 342751522964111.3750 - mean_squared_error: 342751543558144.00 - 0s 113us/sample - loss: 338132204945901.6250 - mean_squared_error: 338132239122432.0000 - val_loss: 204914228141169.7812 - val_mean_squared_error: 204914215092224.0000
    Epoch 361/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 336617614564417.2500 - mean_squared_error: 336617592061952.0000 - val_loss: 214260877312606.8125 - val_mean_squared_error: 214260869234688.0000
    Epoch 362/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 335543100298340.8750 - mean_squared_error: 335543112040448.0000 - val_loss: 212344670072680.2812 - val_mean_squared_error: 212344676286464.0000
    Epoch 363/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 334930507599269.2500 - mean_squared_error: 334930475220992.0000 - val_loss: 214969080261290.6562 - val_mean_squared_error: 214969085853696.0000
    Epoch 364/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 333808072199817.0625 - mean_squared_error: 333808113025024.0000 - val_loss: 208859305976111.4062 - val_mean_squared_error: 208859293548544.0000
    Epoch 365/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 334492439432784.1250 - mean_squared_error: 334492422111232.0000 - val_loss: 214085668496270.2188 - val_mean_squared_error: 214085664768000.0000
    Epoch 366/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 331905448053459.8125 - mean_squared_error: 331905442512896.0000 - val_loss: 207046379899259.2812 - val_mean_squared_error: 207046381142016.0000
    Epoch 367/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 331217538281201.5000 - mean_squared_error: 331217475993600.0000 - val_loss: 214114543949141.3438 - val_mean_squared_error: 214114538356736.0000
    Epoch 368/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 330022049352441.8125 - mean_squared_error: 330021998690304.0000 - val_loss: 222108322540202.6562 - val_mean_squared_error: 222108311355392.0000
    Epoch 369/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 329897906157964.3125 - mean_squared_error: 329897880846336.0000 - val_loss: 215897695127931.2812 - val_mean_squared_error: 215897687982080.0000
    Epoch 370/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 328942722445724.9375 - mean_squared_error: 328942653276160.0000 - val_loss: 215401803498078.8125 - val_mean_squared_error: 215401803808768.0000
    Epoch 371/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 329038787209854.3750 - mean_squared_error: 329038786723840.0000 - val_loss: 210379833450799.4062 - val_mean_squared_error: 210379846189056.0000
    Epoch 372/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 326538851132464.6250 - mean_squared_error: 326538813767680.0000 - val_loss: 217651906541188.7188 - val_mean_squared_error: 217651896909824.0000
    Epoch 373/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 326786846379372.3125 - mean_squared_error: 326786848129024.0000 - val_loss: 220646067307709.6250 - val_mean_squared_error: 220646093094912.0000
    Epoch 374/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 325785669844365.5000 - mean_squared_error: 325785684541440.0000 - val_loss: 209807257205570.3750 - val_mean_squared_error: 209807239806976.0000
    Epoch 375/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 325640194866877.2500 - mean_squared_error: 325640226078720.0000 - val_loss: 210632280142279.1250 - val_mean_squared_error: 210632276180992.0000
    Epoch 376/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 323830733969149.3125 - mean_squared_error: 323830736224256.0000 - val_loss: 219346395725824.0000 - val_mean_squared_error: 219346412503040.0000
    Epoch 377/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 324178286809928.0625 - mean_squared_error: 324178293030912.0000 - val_loss: 219559123368163.5625 - val_mean_squared_error: 219559114047488.0000
    Epoch 378/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 323244392286400.1875 - mean_squared_error: 323244406079488.0000 - val_loss: 221070121694852.7188 - val_mean_squared_error: 221070137229312.0000
    Epoch 379/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 322944476406396.0000 - mean_squared_error: 322944463011840.0000 - val_loss: 216956328447544.8750 - val_mean_squared_error: 216956330311680.0000
    Epoch 380/1000
    1726/1726 [==============================] - 0s 84us/sample - loss: 320134984130436.5625 - mean_squared_error: 320134950420480.0000 - val_loss: 209513328459169.1875 - val_mean_squared_error: 209513319759872.0000
    Epoch 381/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 319493295763223.4375 - mean_squared_error: 319493322571776.0000 - val_loss: 211468785685238.5312 - val_mean_squared_error: 211468771393536.0000
    Epoch 382/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 318578230398639.0000 - mean_squared_error: 318578259656704.0000 - val_loss: 215435427835145.4688 - val_mean_squared_error: 215435442126848.0000
    Epoch 383/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 316574057127053.1875 - mean_squared_error: 316574019878912.0000 - val_loss: 211978038814189.0312 - val_mean_squared_error: 211978027008000.0000
    Epoch 384/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 317870329980579.1250 - mean_squared_error: 317870328250368.0000 - val_loss: 214009121205513.4688 - val_mean_squared_error: 214009127108608.0000
    Epoch 385/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 315795433939352.1875 - mean_squared_error: 315795422838784.0000 - val_loss: 214739009327255.7188 - val_mean_squared_error: 214739003113472.0000
    Epoch 386/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 314294568496710.6250 - mean_squared_error: 314294566649856.0000 - val_loss: 220968737599943.1250 - val_mean_squared_error: 220968735735808.0000
    Epoch 387/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 314622683902402.8750 - mean_squared_error: 314622661885952.0000 - val_loss: 212359350602714.0625 - val_mean_squared_error: 212359356350464.0000
    Epoch 388/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 315756476854988.6875 - mean_squared_error: 315756466143232.0000 - val_loss: 211231997029641.4688 - val_mean_squared_error: 211232011321344.0000
    Epoch 389/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 313874867270063.8750 - mean_squared_error: 313874834259968.0000 - val_loss: 207017246574364.4375 - val_mean_squared_error: 207017239117824.0000
    Epoch 390/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 312166087602266.1875 - mean_squared_error: 312166074810368.0000 - val_loss: 205104677815485.6250 - val_mean_squared_error: 205104686825472.0000
    Epoch 391/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 312544091420574.6875 - mean_squared_error: 312544099041280.0000 - val_loss: 217184171740122.0625 - val_mean_squared_error: 217184181682176.0000
    Epoch 392/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 310300107811739.1250 - mean_squared_error: 310300079292416.0000 - val_loss: 215296083735589.9375 - val_mean_squared_error: 215296073793536.0000
    Epoch 393/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 308641172475054.4375 - mean_squared_error: 308641148174336.0000 - val_loss: 210414647106066.9688 - val_mean_squared_error: 210414642135040.0000
    Epoch 394/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 312588089445968.1250 - mean_squared_error: 312588122456064.0000 - val_loss: 202765780424628.1562 - val_mean_squared_error: 202765808697344.0000
    Epoch 395/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 308614183533395.9375 - mean_squared_error: 308614136856576.0000 - val_loss: 216026160445667.5625 - val_mean_squared_error: 216026151124992.0000
    Epoch 396/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 307812711575756.1250 - mean_squared_error: 307812655693824.0000 - val_loss: 230584345427968.0000 - val_mean_squared_error: 230584328650752.0000
    Epoch 397/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 306982386772473.5000 - mean_squared_error: 306982384828416.0000 - val_loss: 213944276333605.9375 - val_mean_squared_error: 213944283168768.0000
    Epoch 398/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 304026528985178.1875 - mean_squared_error: 304026474250240.0000 - val_loss: 202107550433280.0000 - val_mean_squared_error: 202107554627584.0000
    Epoch 399/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 310634422680631.7500 - mean_squared_error: 310634382098432.0000 - val_loss: 201607696118215.1250 - val_mean_squared_error: 201607694254080.0000
    Epoch 400/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 306628258798828.1250 - mean_squared_error: 306628184244224.0000 - val_loss: 215370657528642.3750 - val_mean_squared_error: 215370665295872.0000
    Epoch 401/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 304057719025317.5625 - mean_squared_error: 304057713426432.0000 - val_loss: 220655140985362.9688 - val_mean_squared_error: 220655136014336.0000
    Epoch 402/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 304001922048541.0625 - mean_squared_error: 304001912406016.0000 - val_loss: 207885707975793.7812 - val_mean_squared_error: 207885728481280.0000
    Epoch 403/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 302826960442089.1250 - mean_squared_error: 302826970415104.0000 - val_loss: 213474194586434.3750 - val_mean_squared_error: 213474168799232.0000
    Epoch 404/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 301929615550191.0625 - mean_squared_error: 301929624240128.0000 - val_loss: 220054476573658.0625 - val_mean_squared_error: 220054478127104.0000
    Epoch 405/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 300044285583597.3125 - mean_squared_error: 300044200706048.0000 - val_loss: 207401476511061.3438 - val_mean_squared_error: 207401470918656.0000
    Epoch 406/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 301076086710186.5625 - mean_squared_error: 301076066598912.0000 - val_loss: 215463674453105.7812 - val_mean_squared_error: 215463678181376.0000
    Epoch 407/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 297728903625707.8750 - mean_squared_error: 297728911343616.0000 - val_loss: 220323305672855.7188 - val_mean_squared_error: 220323316236288.0000
    Epoch 408/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 297501659928697.0625 - mean_squared_error: 297501680730112.0000 - val_loss: 207774640320360.2812 - val_mean_squared_error: 207774646534144.0000
    Epoch 409/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 296218270956428.8750 - mean_squared_error: 296218257260544.0000 - val_loss: 217697412254075.2812 - val_mean_squared_error: 217697413496832.0000
    Epoch 410/1000
    1726/1726 [==============================] - ETA: 0s - loss: 298252786639929.9375 - mean_squared_error: 298252830244864.00 - 0s 102us/sample - loss: 296208727607450.2500 - mean_squared_error: 296208761356288.0000 - val_loss: 218847768167006.8125 - val_mean_squared_error: 218847776866304.0000
    Epoch 411/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 294018080440168.1250 - mean_squared_error: 294018059599872.0000 - val_loss: 220136571530505.4688 - val_mean_squared_error: 220136552267776.0000
    Epoch 412/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 294258277994785.5000 - mean_squared_error: 294258275778560.0000 - val_loss: 214728315094812.4375 - val_mean_squared_error: 214728316026880.0000
    Epoch 413/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 293065010423857.8125 - mean_squared_error: 293065013067776.0000 - val_loss: 214333305175457.1875 - val_mean_squared_error: 214333296476160.0000
    Epoch 414/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 295279017959243.6250 - mean_squared_error: 295279035154432.0000 - val_loss: 216125363813186.3750 - val_mean_squared_error: 216125371580416.0000
    Epoch 415/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 291509127377081.1250 - mean_squared_error: 291509094055936.0000 - val_loss: 213069706627451.2812 - val_mean_squared_error: 213069703675904.0000
    Epoch 416/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 290802457605723.9375 - mean_squared_error: 290802404163584.0000 - val_loss: 228367458165646.2188 - val_mean_squared_error: 228367454437376.0000
    Epoch 417/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 290510197001219.5625 - mean_squared_error: 290510178615296.0000 - val_loss: 211527472386806.5312 - val_mean_squared_error: 211527474872320.0000
    Epoch 418/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 289281773313639.8125 - mean_squared_error: 289281817968640.0000 - val_loss: 213843317261501.6250 - val_mean_squared_error: 213843317882880.0000
    Epoch 419/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 289593367953673.8125 - mean_squared_error: 289593337315328.0000 - val_loss: 216804868712523.8438 - val_mean_squared_error: 216804865605632.0000
    Epoch 420/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 289045606888558.3750 - mean_squared_error: 289045594767360.0000 - val_loss: 207497034562370.3750 - val_mean_squared_error: 207497050718208.0000
    Epoch 421/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 289694009709517.0000 - mean_squared_error: 289694000611328.0000 - val_loss: 211766990136206.2188 - val_mean_squared_error: 211767003185152.0000
    Epoch 422/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 288989311426284.7500 - mean_squared_error: 288989323984896.0000 - val_loss: 211838102541046.5312 - val_mean_squared_error: 211838105026560.0000
    Epoch 423/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 286769158939070.1250 - mean_squared_error: 286769194991616.0000 - val_loss: 209490664304488.2812 - val_mean_squared_error: 209490670518272.0000
    Epoch 424/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 285309073805373.6875 - mean_squared_error: 285309073883136.0000 - val_loss: 211955505770344.2812 - val_mean_squared_error: 211955495206912.0000
    Epoch 425/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 285229658435465.3125 - mean_squared_error: 285229684097024.0000 - val_loss: 210220687576026.0625 - val_mean_squared_error: 210220680740864.0000
    Epoch 426/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 284700466822655.4375 - mean_squared_error: 284700497149952.0000 - val_loss: 217408710542525.6250 - val_mean_squared_error: 217408711163904.0000
    Epoch 427/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 282509869180237.4375 - mean_squared_error: 282509828947968.0000 - val_loss: 213964376681130.6562 - val_mean_squared_error: 213964382273536.0000
    Epoch 428/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 282140927838571.0625 - mean_squared_error: 282140897968128.0000 - val_loss: 214670038502968.8750 - val_mean_squared_error: 214670065532928.0000
    Epoch 429/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 282628247554164.3125 - mean_squared_error: 282628309647360.0000 - val_loss: 220407112216424.2812 - val_mean_squared_error: 220407118430208.0000
    Epoch 430/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 281922200075604.5625 - mean_squared_error: 281922257289216.0000 - val_loss: 209700577724188.4375 - val_mean_squared_error: 209700570267648.0000
    Epoch 431/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 279593581739956.0312 - mean_squared_error: 279593630040064.0000 - val_loss: 216981039422577.7812 - val_mean_squared_error: 216981059928064.0000
    Epoch 432/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 281154775514338.6250 - mean_squared_error: 281154833874944.0000 - val_loss: 209536303924413.6250 - val_mean_squared_error: 209536304545792.0000
    Epoch 433/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 279211120227020.6875 - mean_squared_error: 279211143069696.0000 - val_loss: 225222546035446.5312 - val_mean_squared_error: 225222565298176.0000
    Epoch 434/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 280886329191057.3438 - mean_squared_error: 280886297755648.0000 - val_loss: 224664436410747.2812 - val_mean_squared_error: 224664437653504.0000
    Epoch 435/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 278788742577568.4688 - mean_squared_error: 278788726325248.0000 - val_loss: 219878098633917.6250 - val_mean_squared_error: 219878082478080.0000
    Epoch 436/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 278838820973201.3438 - mean_squared_error: 278838839869440.0000 - val_loss: 209940152640398.2188 - val_mean_squared_error: 209940148912128.0000
    Epoch 437/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 280668747887384.6250 - mean_squared_error: 280668697264128.0000 - val_loss: 202922112076762.0625 - val_mean_squared_error: 202922105241600.0000
    Epoch 438/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 279599870493716.1562 - mean_squared_error: 279599887941632.0000 - val_loss: 209809100213816.8750 - val_mean_squared_error: 209809118855168.0000
    Epoch 439/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 276987255678703.0938 - mean_squared_error: 276987306311680.0000 - val_loss: 232242627205650.9688 - val_mean_squared_error: 232242639011840.0000
    Epoch 440/1000
    1726/1726 [==============================] - 0s 96us/sample - loss: 277089509903857.1562 - mean_squared_error: 277089513111552.0000 - val_loss: 207170855035790.2188 - val_mean_squared_error: 207170851307520.0000
    Epoch 441/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 276181808726186.8750 - mean_squared_error: 276181781839872.0000 - val_loss: 221578063404752.5938 - val_mean_squared_error: 221578033889280.0000
    Epoch 442/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 276964499756949.2188 - mean_squared_error: 276964489297920.0000 - val_loss: 210989888132437.3438 - val_mean_squared_error: 210989882540032.0000
    Epoch 443/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 274001573522483.0312 - mean_squared_error: 274001599397888.0000 - val_loss: 207369069764911.4062 - val_mean_squared_error: 207369074114560.0000
    Epoch 444/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 272702139923113.0625 - mean_squared_error: 272702120132608.0000 - val_loss: 216401549410910.8125 - val_mean_squared_error: 216401574887424.0000
    Epoch 445/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 273626112385546.0938 - mean_squared_error: 273626142081024.0000 - val_loss: 228391964707195.2812 - val_mean_squared_error: 228391965949952.0000
    Epoch 446/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 274703651517569.3438 - mean_squared_error: 274703642001408.0000 - val_loss: 220989065837985.1875 - val_mean_squared_error: 220989069721600.0000
    Epoch 447/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 270463111813981.4375 - mean_squared_error: 270463116771328.0000 - val_loss: 225672015108626.9688 - val_mean_squared_error: 225672010137600.0000
    Epoch 448/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 273281012750893.6562 - mean_squared_error: 273281034747904.0000 - val_loss: 221900323587261.6250 - val_mean_squared_error: 221900324208640.0000
    Epoch 449/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 270096617632422.7188 - mean_squared_error: 270096551378944.0000 - val_loss: 222163926894364.4375 - val_mean_squared_error: 222163927826432.0000
    Epoch 450/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 271894493162641.9375 - mean_squared_error: 271894481731584.0000 - val_loss: 217489742010292.1562 - val_mean_squared_error: 217489745117184.0000
    Epoch 451/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 270217100069473.9062 - mean_squared_error: 270217129230336.0000 - val_loss: 218656862556311.7188 - val_mean_squared_error: 218656868925440.0000
    Epoch 452/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 269693955454726.8125 - mean_squared_error: 269693982081024.0000 - val_loss: 216518081157044.1562 - val_mean_squared_error: 216518092652544.0000
    Epoch 453/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 267471168266610.2188 - mean_squared_error: 267471219064832.0000 - val_loss: 212445522889083.2812 - val_mean_squared_error: 212445524131840.0000
    Epoch 454/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 269999571342835.5312 - mean_squared_error: 269999528738816.0000 - val_loss: 214091625340017.7812 - val_mean_squared_error: 214091620679680.0000
    Epoch 455/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 266902027526628.1250 - mean_squared_error: 266902001680384.0000 - val_loss: 219934690668999.1250 - val_mean_squared_error: 219934688804864.0000
    Epoch 456/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 268559493573834.9062 - mean_squared_error: 268559439626240.0000 - val_loss: 214388056687805.6250 - val_mean_squared_error: 214388057309184.0000
    Epoch 457/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 265947018662917.9375 - mean_squared_error: 265947042545664.0000 - val_loss: 209320825440028.4375 - val_mean_squared_error: 209320817983488.0000
    Epoch 458/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 265854807574758.1875 - mean_squared_error: 265854834966528.0000 - val_loss: 218409664414909.6250 - val_mean_squared_error: 218409673424896.0000
    Epoch 459/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 264253854782583.8438 - mean_squared_error: 264253835575296.0000 - val_loss: 213066002124951.7188 - val_mean_squared_error: 213065995911168.0000
    Epoch 460/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 268416886494232.9375 - mean_squared_error: 268416883621888.0000 - val_loss: 217330794976597.3438 - val_mean_squared_error: 217330780995584.0000
    Epoch 461/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 264697619572083.3750 - mean_squared_error: 264697626492928.0000 - val_loss: 216903771954289.7812 - val_mean_squared_error: 216903784071168.0000
    Epoch 462/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 263686916786083.4688 - mean_squared_error: 263686916669440.0000 - val_loss: 227136174555136.0000 - val_mean_squared_error: 227136191332352.0000
    Epoch 463/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 264311993473790.5000 - mean_squared_error: 264311968628736.0000 - val_loss: 223180094284079.4062 - val_mean_squared_error: 223180073467904.0000
    Epoch 464/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 263516243489674.5312 - mean_squared_error: 263516258828288.0000 - val_loss: 223310900955515.2812 - val_mean_squared_error: 223310902198272.0000
    Epoch 465/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 261878960044504.2500 - mean_squared_error: 261878953541632.0000 - val_loss: 216046686437376.0000 - val_mean_squared_error: 216046719991808.0000
    Epoch 466/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 262388247181860.1875 - mean_squared_error: 262388242710528.0000 - val_loss: 217667518983433.4688 - val_mean_squared_error: 217667516497920.0000
    Epoch 467/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 260119826002674.6562 - mean_squared_error: 260119795335168.0000 - val_loss: 209012049634190.2188 - val_mean_squared_error: 209012050100224.0000
    Epoch 468/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 265668162854731.6250 - mean_squared_error: 265668171661312.0000 - val_loss: 224420211098813.6250 - val_mean_squared_error: 224420211720192.0000
    Epoch 469/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 262358186298903.1562 - mean_squared_error: 262358194716672.0000 - val_loss: 213669974444411.2812 - val_mean_squared_error: 213669992464384.0000
    Epoch 470/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 261606656954201.8750 - mean_squared_error: 261606659325952.0000 - val_loss: 217657507024440.8750 - val_mean_squared_error: 217657534054400.0000
    Epoch 471/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 261111941616285.2188 - mean_squared_error: 261111966334976.0000 - val_loss: 218030032367464.2812 - val_mean_squared_error: 218030021804032.0000
    Epoch 472/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 260174539682584.6250 - mean_squared_error: 260174539390976.0000 - val_loss: 223867887438810.0625 - val_mean_squared_error: 223867872215040.0000
    Epoch 473/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 261073058581070.9062 - mean_squared_error: 261073076748288.0000 - val_loss: 213974478429297.7812 - val_mean_squared_error: 213974498934784.0000
    Epoch 474/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 261287351083833.8438 - mean_squared_error: 261287338573824.0000 - val_loss: 230110769359530.6562 - val_mean_squared_error: 230110741397504.0000
    Epoch 475/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 261402095810215.9062 - mean_squared_error: 261402061176832.0000 - val_loss: 218371360477335.7188 - val_mean_squared_error: 218371371040768.0000
    Epoch 476/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 258330986889007.1562 - mean_squared_error: 258330991788032.0000 - val_loss: 223071011000471.7188 - val_mean_squared_error: 223071021563904.0000
    Epoch 477/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 258095521500771.0625 - mean_squared_error: 258095540338688.0000 - val_loss: 229411206832431.4062 - val_mean_squared_error: 229411198599168.0000
    Epoch 478/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 259375860606513.2500 - mean_squared_error: 259375876800512.0000 - val_loss: 222880532121372.4375 - val_mean_squared_error: 222880533053440.0000
    Epoch 479/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 259384593273804.9688 - mean_squared_error: 259384584175616.0000 - val_loss: 213362671149966.2188 - val_mean_squared_error: 213362684198912.0000
    Epoch 480/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 258890332828326.7188 - mean_squared_error: 258890310615040.0000 - val_loss: 224096078392585.4688 - val_mean_squared_error: 224096075907072.0000
    Epoch 481/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 257873309549398.3125 - mean_squared_error: 257873275781120.0000 - val_loss: 222725657290827.8438 - val_mean_squared_error: 222725662572544.0000
    Epoch 482/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 259005312063931.7812 - mean_squared_error: 259005301653504.0000 - val_loss: 223289290037172.1562 - val_mean_squared_error: 223289293144064.0000
    Epoch 483/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 259727932000638.0625 - mean_squared_error: 259727929901056.0000 - val_loss: 215447007842455.7188 - val_mean_squared_error: 215447018405888.0000
    Epoch 484/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 259059895764159.0625 - mean_squared_error: 259059877937152.0000 - val_loss: 214326447643761.7812 - val_mean_squared_error: 214326417817600.0000
    Epoch 485/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 258139580802856.0625 - mean_squared_error: 258139563753472.0000 - val_loss: 215421228096625.7812 - val_mean_squared_error: 215421215047680.0000
    Epoch 486/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 261187420065682.8438 - mean_squared_error: 261187430252544.0000 - val_loss: 219701493580382.8125 - val_mean_squared_error: 219701502279680.0000
    Epoch 487/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 258261246937621.9375 - mean_squared_error: 258261282455552.0000 - val_loss: 218151255830983.1250 - val_mean_squared_error: 218151270744064.0000
    Epoch 488/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 256284828119206.1250 - mean_squared_error: 256284825747456.0000 - val_loss: 210339989737623.7188 - val_mean_squared_error: 210340000301056.0000
    Epoch 489/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 256564732542273.5625 - mean_squared_error: 256564686487552.0000 - val_loss: 226570840164124.4375 - val_mean_squared_error: 226570832707584.0000
    Epoch 490/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 256159675006311.5312 - mean_squared_error: 256159617384448.0000 - val_loss: 216926272220425.4688 - val_mean_squared_error: 216926248763392.0000
    Epoch 491/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 256004789983865.6250 - mean_squared_error: 256004780457984.0000 - val_loss: 224025878792571.2812 - val_mean_squared_error: 224025880035328.0000
    Epoch 492/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 256313293154543.6875 - mean_squared_error: 256313279905792.0000 - val_loss: 225234945019448.8750 - val_mean_squared_error: 225234946883584.0000
    Epoch 493/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 255518421709440.7188 - mean_squared_error: 255518425743360.0000 - val_loss: 221117541874498.3750 - val_mean_squared_error: 221117516087296.0000
    Epoch 494/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 252782480242970.3750 - mean_squared_error: 252782447689728.0000 - val_loss: 229300062747458.3750 - val_mean_squared_error: 229300049543168.0000
    Epoch 495/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 255602365148856.5000 - mean_squared_error: 255602345377792.0000 - val_loss: 228432220083541.3438 - val_mean_squared_error: 228432231268352.0000
    Epoch 496/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 256208784843350.0000 - mean_squared_error: 256208791404544.0000 - val_loss: 220208888788005.9375 - val_mean_squared_error: 220208878845952.0000
    Epoch 497/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 252750393584083.5000 - mean_squared_error: 252750369652736.0000 - val_loss: 220184391256443.2812 - val_mean_squared_error: 220184417665024.0000
    Epoch 498/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 255398574638973.5000 - mean_squared_error: 255398586089472.0000 - val_loss: 232016100924529.7812 - val_mean_squared_error: 232016079486976.0000
    Epoch 499/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 255779540014997.2188 - mean_squared_error: 255779546333184.0000 - val_loss: 217653431869743.4062 - val_mean_squared_error: 217653440413696.0000
    Epoch 500/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 255094647867071.6250 - mean_squared_error: 255094633267200.0000 - val_loss: 218489510137400.8750 - val_mean_squared_error: 218489499418624.0000
    Epoch 501/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 252231797871849.7812 - mean_squared_error: 252231785906176.0000 - val_loss: 216175049072336.5938 - val_mean_squared_error: 216175048916992.0000
    Epoch 502/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 250995162252213.2500 - mean_squared_error: 250995137314816.0000 - val_loss: 218244818636344.8750 - val_mean_squared_error: 218244803723264.0000
    Epoch 503/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 251356336855844.4688 - mean_squared_error: 251356384329728.0000 - val_loss: 234105236832862.8125 - val_mean_squared_error: 234105228754944.0000
    Epoch 504/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 253830869798138.3438 - mean_squared_error: 253830872694784.0000 - val_loss: 220941448526051.5625 - val_mean_squared_error: 220941455982592.0000
    Epoch 505/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 254255481811167.0938 - mean_squared_error: 254255436922880.0000 - val_loss: 237673580418389.3438 - val_mean_squared_error: 237673608380416.0000
    Epoch 506/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 252354426227582.6562 - mean_squared_error: 252354427355136.0000 - val_loss: 222174656234685.6250 - val_mean_squared_error: 222174648467456.0000
    Epoch 507/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 250446706065047.2812 - mean_squared_error: 250446706900992.0000 - val_loss: 226102263706055.1250 - val_mean_squared_error: 226102261841920.0000
    Epoch 508/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 249800953043027.0625 - mean_squared_error: 249800985411584.0000 - val_loss: 222450050507055.4062 - val_mean_squared_error: 222450029690880.0000
    Epoch 509/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 252892938196258.6875 - mean_squared_error: 252892975988736.0000 - val_loss: 228163653516705.1875 - val_mean_squared_error: 228163661594624.0000
    Epoch 510/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 249306459259620.4062 - mean_squared_error: 249306460192768.0000 - val_loss: 224090513017211.2812 - val_mean_squared_error: 224090522648576.0000
    Epoch 511/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 251551209225632.4688 - mean_squared_error: 251551251693568.0000 - val_loss: 223456563851719.1250 - val_mean_squared_error: 223456561987584.0000
    Epoch 512/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 250523878848016.0312 - mean_squared_error: 250523932426240.0000 - val_loss: 225704363445210.0625 - val_mean_squared_error: 225704373387264.0000
    Epoch 513/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 251708814664904.5312 - mean_squared_error: 251708789751808.0000 - val_loss: 241045794309916.4375 - val_mean_squared_error: 241045812019200.0000
    Epoch 514/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 251207803710704.8750 - mean_squared_error: 251207788527616.0000 - val_loss: 222671282955150.2188 - val_mean_squared_error: 222671287615488.0000
    Epoch 515/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 249264194380905.6250 - mean_squared_error: 249264231940096.0000 - val_loss: 218627705153005.0312 - val_mean_squared_error: 218627693346816.0000
    Epoch 516/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 250055696093986.1250 - mean_squared_error: 250055680327680.0000 - val_loss: 221993970900081.7812 - val_mean_squared_error: 221993974628352.0000
    Epoch 517/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 249303086028294.5312 - mean_squared_error: 249303087972352.0000 - val_loss: 220140845370936.8750 - val_mean_squared_error: 220140847235072.0000
    Epoch 518/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 250179451128228.0312 - mean_squared_error: 250179512958976.0000 - val_loss: 234354932915313.7812 - val_mean_squared_error: 234354924060672.0000
    Epoch 519/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 249576361545865.6562 - mean_squared_error: 249576321712128.0000 - val_loss: 216283968467399.1250 - val_mean_squared_error: 216283966603264.0000
    Epoch 520/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 248536763825378.6250 - mean_squared_error: 248536771854336.0000 - val_loss: 225729125217924.7188 - val_mean_squared_error: 225729119780864.0000
    Epoch 521/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 248759442642153.7812 - mean_squared_error: 248759422287872.0000 - val_loss: 222083045178785.1875 - val_mean_squared_error: 222083061645312.0000
    Epoch 522/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 247295151569273.3438 - mean_squared_error: 247295140429824.0000 - val_loss: 226989955525290.6562 - val_mean_squared_error: 226989977894912.0000
    Epoch 523/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 247799698073877.6562 - mean_squared_error: 247799681646592.0000 - val_loss: 230594462089216.0000 - val_mean_squared_error: 230594462089216.0000
    Epoch 524/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 247128645117052.5625 - mean_squared_error: 247128643338240.0000 - val_loss: 234192833163112.2812 - val_mean_squared_error: 234192839376896.0000
    Epoch 525/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 246798169956076.7188 - mean_squared_error: 246798148960256.0000 - val_loss: 222758977152493.0312 - val_mean_squared_error: 222758998900736.0000
    Epoch 526/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 248224225500965.6875 - mean_squared_error: 248224245874688.0000 - val_loss: 225719913438928.5938 - val_mean_squared_error: 225719925866496.0000
    Epoch 527/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 249214215307169.0938 - mean_squared_error: 249214168727552.0000 - val_loss: 219907228230542.2188 - val_mean_squared_error: 219907241279488.0000
    Epoch 528/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 246418371422789.4375 - mean_squared_error: 246418363121664.0000 - val_loss: 222364578358765.0312 - val_mean_squared_error: 222364600107008.0000
    Epoch 529/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 245512095433653.2500 - mean_squared_error: 245512141799424.0000 - val_loss: 225175080805869.0312 - val_mean_squared_error: 225175068999680.0000
    Epoch 530/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 246522975245526.7812 - mean_squared_error: 246522918731776.0000 - val_loss: 230016998392414.8125 - val_mean_squared_error: 230017007091712.0000
    Epoch 531/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 245957607289282.8750 - mean_squared_error: 245957610438656.0000 - val_loss: 224957994194261.3438 - val_mean_squared_error: 224957971824640.0000
    Epoch 532/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 245444909633578.7188 - mean_squared_error: 245444915494912.0000 - val_loss: 228248197636399.4062 - val_mean_squared_error: 228248201986048.0000
    Epoch 533/1000
    1726/1726 [==============================] - 0s 93us/sample - loss: 244240958193474.1562 - mean_squared_error: 244240932143104.0000 - val_loss: 230253957927063.7188 - val_mean_squared_error: 230253951713280.0000
    Epoch 534/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 246519366583980.6562 - mean_squared_error: 246519311630336.0000 - val_loss: 230247671442090.6562 - val_mean_squared_error: 230247677034496.0000
    Epoch 535/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 246073746011796.9062 - mean_squared_error: 246073775882240.0000 - val_loss: 227039987047537.7812 - val_mean_squared_error: 227039957221376.0000
    Epoch 536/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 246473121247587.9688 - mean_squared_error: 246473140731904.0000 - val_loss: 222201469022056.2812 - val_mean_squared_error: 222201492013056.0000
    Epoch 537/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 244397074980455.8438 - mean_squared_error: 244397060915200.0000 - val_loss: 220168751784239.4062 - val_mean_squared_error: 220168747745280.0000
    Epoch 538/1000
    1726/1726 [==============================] - 0s 94us/sample - loss: 243344171032224.7500 - mean_squared_error: 243344139616256.0000 - val_loss: 223028299315427.5625 - val_mean_squared_error: 223028306771968.0000
    Epoch 539/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 245477808772405.6875 - mean_squared_error: 245477731729408.0000 - val_loss: 229846269714128.5938 - val_mean_squared_error: 229846265364480.0000
    Epoch 540/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 244600292333594.0938 - mean_squared_error: 244600266555392.0000 - val_loss: 230175943251285.3438 - val_mean_squared_error: 230175954436096.0000
    Epoch 541/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 243916937351913.1562 - mean_squared_error: 243916964102144.0000 - val_loss: 234453497971901.6250 - val_mean_squared_error: 234453523759104.0000
    Epoch 542/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 244014752992525.3438 - mean_squared_error: 244014724939776.0000 - val_loss: 228390424310215.1250 - val_mean_squared_error: 228390405668864.0000
    Epoch 543/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 243726536499000.6875 - mean_squared_error: 243726542700544.0000 - val_loss: 223018064592289.1875 - val_mean_squared_error: 223018039115776.0000
    Epoch 544/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 242967795799463.6250 - mean_squared_error: 242967776329728.0000 - val_loss: 221729136953116.4375 - val_mean_squared_error: 221729146273792.0000
    Epoch 545/1000
    1726/1726 [==============================] - 0s 89us/sample - loss: 243174152275666.6250 - mean_squared_error: 243174186418176.0000 - val_loss: 224245252697960.2812 - val_mean_squared_error: 224245275688960.0000
    Epoch 546/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 242475263599209.0000 - mean_squared_error: 242475214045184.0000 - val_loss: 243693411427214.2188 - val_mean_squared_error: 243693424476160.0000
    Epoch 547/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 243816423626835.0625 - mean_squared_error: 243816384692224.0000 - val_loss: 222986624089505.1875 - val_mean_squared_error: 222986632167424.0000
    Epoch 548/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 243129665694049.5938 - mean_squared_error: 243129659686912.0000 - val_loss: 225562606999476.1562 - val_mean_squared_error: 225562622689280.0000
    Epoch 549/1000
    1726/1726 [==============================] - 0s 92us/sample - loss: 244199180864933.2188 - mean_squared_error: 244199207206912.0000 - val_loss: 234402700600813.0312 - val_mean_squared_error: 234402705571840.0000
    Epoch 550/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 241709626559909.2188 - mean_squared_error: 241709619347456.0000 - val_loss: 227618110648168.2812 - val_mean_squared_error: 227618116861952.0000
    Epoch 551/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 240808969655055.1250 - mean_squared_error: 240808968060928.0000 - val_loss: 222854240049076.1562 - val_mean_squared_error: 222854243155968.0000
    Epoch 552/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 241534068955181.0938 - mean_squared_error: 241534096113664.0000 - val_loss: 225049023950696.2812 - val_mean_squared_error: 225049021775872.0000
    Epoch 553/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 243797659938887.1875 - mean_squared_error: 243797694873600.0000 - val_loss: 237586156209948.4375 - val_mean_squared_error: 237586165530624.0000
    Epoch 554/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 242061327363217.9375 - mean_squared_error: 242061370458112.0000 - val_loss: 226973977867908.7188 - val_mean_squared_error: 226973972430848.0000
    Epoch 555/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 239796702471425.5000 - mean_squared_error: 239796681179136.0000 - val_loss: 228061904051313.7812 - val_mean_squared_error: 228061891002368.0000
    Epoch 556/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 243543305407814.3125 - mean_squared_error: 243543301947392.0000 - val_loss: 213531555577249.1875 - val_mean_squared_error: 213531563655168.0000
    Epoch 557/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 248289029305739.1250 - mean_squared_error: 248289022705664.0000 - val_loss: 233164617246795.8438 - val_mean_squared_error: 233164630917120.0000
    Epoch 558/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 242149856489144.5000 - mean_squared_error: 242149836718080.0000 - val_loss: 225629715397442.3750 - val_mean_squared_error: 225629697998848.0000
    Epoch 559/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 242838778094716.5625 - mean_squared_error: 242838776315904.0000 - val_loss: 223873678995911.1250 - val_mean_squared_error: 223873677131776.0000
    Epoch 560/1000
    1726/1726 [==============================] - 0s 91us/sample - loss: 238853261979301.5312 - mean_squared_error: 238853298323456.0000 - val_loss: 234179069632208.5938 - val_mean_squared_error: 234179082059776.0000
    Epoch 561/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 242348814480399.4062 - mean_squared_error: 242348797722624.0000 - val_loss: 236895195889664.0000 - val_mean_squared_error: 236895212666880.0000
    Epoch 562/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 242094234633065.2812 - mean_squared_error: 242094237024256.0000 - val_loss: 219257996419375.4062 - val_mean_squared_error: 219257979797504.0000
    Epoch 563/1000
    1726/1726 [==============================] - 0s 84us/sample - loss: 238576662191492.0000 - mean_squared_error: 238576709140480.0000 - val_loss: 229500083668157.6250 - val_mean_squared_error: 229500067512320.0000
    Epoch 564/1000
    1726/1726 [==============================] - 0s 85us/sample - loss: 242156127513100.4688 - mean_squared_error: 242156144951296.0000 - val_loss: 232059242914095.4062 - val_mean_squared_error: 232059247263744.0000
    Epoch 565/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 241798700663467.4375 - mean_squared_error: 241798739918848.0000 - val_loss: 219399126515712.0000 - val_mean_squared_error: 219399143292928.0000
    Epoch 566/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 239821964181200.2500 - mean_squared_error: 239821914112000.0000 - val_loss: 237628080763941.9375 - val_mean_squared_error: 237628091793408.0000
    Epoch 567/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 240670184752831.6250 - mean_squared_error: 240670220484608.0000 - val_loss: 226693699095514.0625 - val_mean_squared_error: 226693709037568.0000
    Epoch 568/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 241347957665084.7812 - mean_squared_error: 241347986456576.0000 - val_loss: 221107024734890.6562 - val_mean_squared_error: 221107030327296.0000
    Epoch 569/1000
    1726/1726 [==============================] - 0s 86us/sample - loss: 237795655054712.1250 - mean_squared_error: 237795662626816.0000 - val_loss: 236418127674785.1875 - val_mean_squared_error: 236418118975488.0000
    Epoch 570/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 239685264679149.3125 - mean_squared_error: 239685230133248.0000 - val_loss: 228994041201246.8125 - val_mean_squared_error: 228994049900544.0000
    Epoch 571/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 239191933189384.5938 - mean_squared_error: 239191912873984.0000 - val_loss: 224848322154647.7188 - val_mean_squared_error: 224848299163648.0000
    Epoch 572/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 239544765419562.7188 - mean_squared_error: 239544788058112.0000 - val_loss: 222666549760758.5312 - val_mean_squared_error: 222666539663360.0000
    Epoch 573/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 236198540794299.7812 - mean_squared_error: 236198555549696.0000 - val_loss: 226564423811072.0000 - val_mean_squared_error: 226564423811072.0000
    Epoch 574/1000
    1726/1726 [==============================] - 0s 90us/sample - loss: 242373615822783.9375 - mean_squared_error: 242373628002304.0000 - val_loss: 221702294805617.7812 - val_mean_squared_error: 221702302728192.0000
    Epoch 575/1000
    1726/1726 [==============================] - 0s 88us/sample - loss: 238877183324629.8750 - mean_squared_error: 238877222633472.0000 - val_loss: 225006730142606.2188 - val_mean_squared_error: 225006709637120.0000
    Epoch 576/1000
    1726/1726 [==============================] - 0s 87us/sample - loss: 239126796843859.9375 - mean_squared_error: 239126783721472.0000 - val_loss: 222569761224931.5625 - val_mean_squared_error: 222569735127040.0000
    Epoch 577/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 237667441900580.7812 - mean_squared_error: 237667417587712.0000 - val_loss: 226606594275555.5625 - val_mean_squared_error: 226606618509312.0000
    Epoch 578/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 237958296283261.7812 - mean_squared_error: 237958284181504.0000 - val_loss: 222703096595645.6250 - val_mean_squared_error: 222703097217024.0000
    Epoch 579/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 239132148853526.2188 - mean_squared_error: 239132135653376.0000 - val_loss: 221812008322237.6250 - val_mean_squared_error: 221811992166400.0000
    Epoch 580/1000
    1726/1726 [==============================] - 0s 95us/sample - loss: 237943815417110.8438 - mean_squared_error: 237943822221312.0000 - val_loss: 223170169318058.6562 - val_mean_squared_error: 223170174910464.0000
    Epoch 581/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 237664248599064.3125 - mean_squared_error: 237664246693888.0000 - val_loss: 232231169299190.5312 - val_mean_squared_error: 232231180173312.0000
    Epoch 582/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 237844464690395.5312 - mean_squared_error: 237844467548160.0000 - val_loss: 241895483870776.8750 - val_mean_squared_error: 241895494123520.0000
    Epoch 583/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 238715205155762.8750 - mean_squared_error: 238715205058560.0000 - val_loss: 228631657685295.4062 - val_mean_squared_error: 228631662034944.0000
    Epoch 584/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 237218943027373.2188 - mean_squared_error: 237218912272384.0000 - val_loss: 231126613632493.0312 - val_mean_squared_error: 231126601826304.0000
    Epoch 585/1000
    1726/1726 [==============================] - 0s 132us/sample - loss: 235522039820530.0312 - mean_squared_error: 235522064646144.0000 - val_loss: 227470139331318.5312 - val_mean_squared_error: 227470141816832.0000
    Epoch 586/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 235408274763030.8438 - mean_squared_error: 235408248012800.0000 - val_loss: 239686919505692.4375 - val_mean_squared_error: 239686941409280.0000
    Epoch 587/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 238132097655628.8438 - mean_squared_error: 238132129693696.0000 - val_loss: 227164783436686.2188 - val_mean_squared_error: 227164796485632.0000
    Epoch 588/1000
    1726/1726 [==============================] - 0s 126us/sample - loss: 238272264539709.1250 - mean_squared_error: 238272269778944.0000 - val_loss: 222152490736146.9688 - val_mean_squared_error: 222152485765120.0000
    Epoch 589/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 238344739303666.0312 - mean_squared_error: 238344764129280.0000 - val_loss: 240923889505014.5312 - val_mean_squared_error: 240923891990528.0000
    Epoch 590/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 238757120862314.7812 - mean_squared_error: 238757097766912.0000 - val_loss: 227501635136777.4688 - val_mean_squared_error: 227501632651264.0000
    Epoch 591/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 236218713962797.3750 - mean_squared_error: 236218704986112.0000 - val_loss: 233253393574570.6562 - val_mean_squared_error: 233253415944192.0000
    Epoch 592/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 235177623269352.2812 - mean_squared_error: 235177611624448.0000 - val_loss: 223712232467721.4688 - val_mean_squared_error: 223712246759424.0000
    Epoch 593/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 236754556434372.6562 - mean_squared_error: 236754569265152.0000 - val_loss: 223422731508242.9688 - val_mean_squared_error: 223422722342912.0000
    Epoch 594/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 236788149319829.5000 - mean_squared_error: 236788140474368.0000 - val_loss: 244862818800981.3438 - val_mean_squared_error: 244862829985792.0000
    Epoch 595/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 237294735315689.1562 - mean_squared_error: 237294728511488.0000 - val_loss: 241570050308930.3750 - val_mean_squared_error: 241570083241984.0000
    Epoch 596/1000
    1726/1726 [==============================] - 0s 135us/sample - loss: 235564673661912.8438 - mean_squared_error: 235564695552000.0000 - val_loss: 224245445325255.1250 - val_mean_squared_error: 224245426683904.0000
    Epoch 597/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 234591361263798.7500 - mean_squared_error: 234591415697408.0000 - val_loss: 225548737989594.0625 - val_mean_squared_error: 225548747931648.0000
    Epoch 598/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 235889832608180.6562 - mean_squared_error: 235889804443648.0000 - val_loss: 225384857522327.7188 - val_mean_squared_error: 225384868085760.0000
    Epoch 599/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 235939260998476.8438 - mean_squared_error: 235939263676416.0000 - val_loss: 236320474335611.2812 - val_mean_squared_error: 236320475578368.0000
    Epoch 600/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 233339760761161.8750 - mean_squared_error: 233339751497728.0000 - val_loss: 236967546390907.2812 - val_mean_squared_error: 236967539245056.0000
    Epoch 601/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 234269129708436.0000 - mean_squared_error: 234269108600832.0000 - val_loss: 228222440881569.1875 - val_mean_squared_error: 228222448959488.0000
    Epoch 602/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 235469955293668.1250 - mean_squared_error: 235469954613248.0000 - val_loss: 228305199159826.9688 - val_mean_squared_error: 228305210966016.0000
    Epoch 603/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 233472809459360.7500 - mean_squared_error: 233472794820608.0000 - val_loss: 224521437674306.3750 - val_mean_squared_error: 224521428664320.0000
    Epoch 604/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 236690935345564.9375 - mean_squared_error: 236690983616512.0000 - val_loss: 246771174614509.0312 - val_mean_squared_error: 246771171196928.0000
    Epoch 605/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 236474157890217.0625 - mean_squared_error: 236474154876928.0000 - val_loss: 233365086958023.1250 - val_mean_squared_error: 233365085093888.0000
    Epoch 606/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 233305967472540.3125 - mean_squared_error: 233305995739136.0000 - val_loss: 232431083187465.4688 - val_mean_squared_error: 232431097479168.0000
    Epoch 607/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 232720301787176.3438 - mean_squared_error: 232720286351360.0000 - val_loss: 237311790940160.0000 - val_mean_squared_error: 237311774162944.0000
    Epoch 608/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 234889411286040.9375 - mean_squared_error: 234889429385216.0000 - val_loss: 228875699224576.0000 - val_mean_squared_error: 228875686641664.0000
    Epoch 609/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 232232924576083.3438 - mean_squared_error: 232232874672128.0000 - val_loss: 229057255265621.3438 - val_mean_squared_error: 229057249673216.0000
    Epoch 610/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 232005915021354.7188 - mean_squared_error: 232005929271296.0000 - val_loss: 230882440516342.5312 - val_mean_squared_error: 230882443001856.0000
    Epoch 611/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 232347414810761.6562 - mean_squared_error: 232347412725760.0000 - val_loss: 233632799750978.3750 - val_mean_squared_error: 233632782352384.0000
    Epoch 612/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 233064317468842.8750 - mean_squared_error: 233064336719872.0000 - val_loss: 220433802902945.1875 - val_mean_squared_error: 220433794203648.0000
    Epoch 613/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 234566968260966.3438 - mean_squared_error: 234566971293696.0000 - val_loss: 222306308602083.5625 - val_mean_squared_error: 222306316058624.0000
    Epoch 614/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 233777890709793.5312 - mean_squared_error: 233777888493568.0000 - val_loss: 226387018732885.3438 - val_mean_squared_error: 226387038306304.0000
    Epoch 615/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 230664857005714.5625 - mean_squared_error: 230664892841984.0000 - val_loss: 223325652478027.8438 - val_mean_squared_error: 223325649371136.0000
    Epoch 616/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 234465142345694.7500 - mean_squared_error: 234465183924224.0000 - val_loss: 224061847590229.3438 - val_mean_squared_error: 224061850386432.0000
    Epoch 617/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 232887007281721.5625 - mean_squared_error: 232886967992320.0000 - val_loss: 244651166143222.5312 - val_mean_squared_error: 244651185405952.0000
    Epoch 618/1000
    1726/1726 [==============================] - 0s 126us/sample - loss: 234280699635268.2188 - mean_squared_error: 234280701657088.0000 - val_loss: 227866848720516.7188 - val_mean_squared_error: 227866839089152.0000
    Epoch 619/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 234917127149670.0625 - mean_squared_error: 234917111791616.0000 - val_loss: 226286247627738.0625 - val_mean_squared_error: 226286257569792.0000
    Epoch 620/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 234020393129231.7188 - mean_squared_error: 234020403150848.0000 - val_loss: 224994557340330.6562 - val_mean_squared_error: 224994562932736.0000
    Epoch 621/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 231250114606683.9688 - mean_squared_error: 231250082136064.0000 - val_loss: 234968485491143.1250 - val_mean_squared_error: 234968500404224.0000
    Epoch 622/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 232188668452759.5938 - mean_squared_error: 232188683485184.0000 - val_loss: 231554562508572.4375 - val_mean_squared_error: 231554538274816.0000
    Epoch 623/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 230644042961660.1250 - mean_squared_error: 230644055539712.0000 - val_loss: 233729089164629.3438 - val_mean_squared_error: 233729083572224.0000
    Epoch 624/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 230275879563752.8438 - mean_squared_error: 230275896311808.0000 - val_loss: 224880195136777.4688 - val_mean_squared_error: 224880192651264.0000
    Epoch 625/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 229637258384617.7812 - mean_squared_error: 229637238030336.0000 - val_loss: 228144290745306.0625 - val_mean_squared_error: 228144283910144.0000
    Epoch 626/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 231746240615885.5625 - mean_squared_error: 231746201190400.0000 - val_loss: 229663868753844.1562 - val_mean_squared_error: 229663863472128.0000
    Epoch 627/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 233045553314321.1875 - mean_squared_error: 233045563015168.0000 - val_loss: 233151844814392.8750 - val_mean_squared_error: 233151846678528.0000
    Epoch 628/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 231636269299064.1250 - mean_squared_error: 231636243316736.0000 - val_loss: 229181360060643.5625 - val_mean_squared_error: 229181367517184.0000
    Epoch 629/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 230024260214622.6250 - mean_squared_error: 230024254849024.0000 - val_loss: 226236748626754.3750 - val_mean_squared_error: 226236764782592.0000
    Epoch 630/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 230299541039628.4688 - mean_squared_error: 230299468300288.0000 - val_loss: 229858697592225.1875 - val_mean_squared_error: 229858680504320.0000
    Epoch 631/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 233071447654419.0000 - mean_squared_error: 233071467036672.0000 - val_loss: 223294681115913.4688 - val_mean_squared_error: 223294678630400.0000
    Epoch 632/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 229591365175355.3438 - mean_squared_error: 229591302012928.0000 - val_loss: 231236169163662.2188 - val_mean_squared_error: 231236173824000.0000
    Epoch 633/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 229646699369250.1250 - mean_squared_error: 229646683602944.0000 - val_loss: 228451215775668.1562 - val_mean_squared_error: 228451223076864.0000
    Epoch 634/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 228489378626331.0000 - mean_squared_error: 228489408020480.0000 - val_loss: 229715281600208.5938 - val_mean_squared_error: 229715285639168.0000
    Epoch 635/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 230026473927448.6250 - mean_squared_error: 230026452664320.0000 - val_loss: 226742217250740.1562 - val_mean_squared_error: 226742211969024.0000
    Epoch 636/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 230537761145154.7188 - mean_squared_error: 230537738321920.0000 - val_loss: 216952241020624.5938 - val_mean_squared_error: 216952236670976.0000
    Epoch 637/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 230575361267681.1562 - mean_squared_error: 230575336062976.0000 - val_loss: 245730395359611.2812 - val_mean_squared_error: 245730413379584.0000
    Epoch 638/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 234251747681575.4375 - mean_squared_error: 234251777736704.0000 - val_loss: 230982439558864.5938 - val_mean_squared_error: 230982451986432.0000
    Epoch 639/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 229892660237773.5625 - mean_squared_error: 229892654366720.0000 - val_loss: 233482610734724.7188 - val_mean_squared_error: 233482609491968.0000
    Epoch 640/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 228632033330480.9375 - mean_squared_error: 228632047910912.0000 - val_loss: 242656425687267.5625 - val_mean_squared_error: 242656424755200.0000
    Epoch 641/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 235758291366168.0312 - mean_squared_error: 235758304624640.0000 - val_loss: 232241699177054.8125 - val_mean_squared_error: 232241699487744.0000
    Epoch 642/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 229497370152015.5000 - mean_squared_error: 229497416712192.0000 - val_loss: 225448613428641.1875 - val_mean_squared_error: 225448621506560.0000
    Epoch 643/1000
    1726/1726 [==============================] - 0s 100us/sample - loss: 226039653645937.3125 - mean_squared_error: 226039666049024.0000 - val_loss: 235831193550241.1875 - val_mean_squared_error: 235831184850944.0000
    Epoch 644/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 230392195691231.6875 - mean_squared_error: 230392212750336.0000 - val_loss: 234083193435173.9375 - val_mean_squared_error: 234083166715904.0000
    Epoch 645/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 230454153337541.5625 - mean_squared_error: 230454137454592.0000 - val_loss: 222559258687715.5625 - val_mean_squared_error: 222559266144256.0000
    Epoch 646/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 229229365790760.3438 - mean_squared_error: 229229434241024.0000 - val_loss: 239376111171204.7188 - val_mean_squared_error: 239376109928448.0000
    Epoch 647/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 227621138082791.0625 - mean_squared_error: 227621136760832.0000 - val_loss: 244259492248993.1875 - val_mean_squared_error: 244259504521216.0000
    Epoch 648/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 228865154127081.7812 - mean_squared_error: 228865167327232.0000 - val_loss: 224966201359777.1875 - val_mean_squared_error: 224966192660480.0000
    Epoch 649/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 225912255613234.1562 - mean_squared_error: 225912276647936.0000 - val_loss: 218162484448824.8750 - val_mean_squared_error: 218162477924352.0000
    Epoch 650/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 230502041713548.9062 - mean_squared_error: 230501986074624.0000 - val_loss: 227429852264675.5625 - val_mean_squared_error: 227429859721216.0000
    Epoch 651/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 226746544553832.1250 - mean_squared_error: 226746557267968.0000 - val_loss: 228800114604259.5625 - val_mean_squared_error: 228800105283584.0000
    Epoch 652/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 226559436044874.1562 - mean_squared_error: 226559407423488.0000 - val_loss: 235277737428157.6250 - val_mean_squared_error: 235277754826752.0000
    Epoch 653/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 226403037860433.2812 - mean_squared_error: 226403026993152.0000 - val_loss: 226796772096834.3750 - val_mean_squared_error: 226796788252672.0000
    Epoch 654/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 225993404596394.8750 - mean_squared_error: 225993411264512.0000 - val_loss: 228338647957504.0000 - val_mean_squared_error: 228338631180288.0000
    Epoch 655/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 228879791227411.5625 - mean_squared_error: 228879729950720.0000 - val_loss: 222242326514043.2812 - val_mean_squared_error: 222242327756800.0000
    Epoch 656/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 228350637173792.0312 - mean_squared_error: 228350660444160.0000 - val_loss: 224218746094781.6250 - val_mean_squared_error: 224218750910464.0000
    Epoch 657/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 228025434870182.4062 - mean_squared_error: 228025434112000.0000 - val_loss: 226585959231943.1250 - val_mean_squared_error: 226585948979200.0000
    Epoch 658/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 227632152509780.5625 - mean_squared_error: 227632125837312.0000 - val_loss: 221735682086836.1562 - val_mean_squared_error: 221735672610816.0000
    Epoch 659/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 225093451362584.0312 - mean_squared_error: 225093431066624.0000 - val_loss: 235675876453641.4688 - val_mean_squared_error: 235675878162432.0000
    Epoch 660/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 225585072081771.6562 - mean_squared_error: 225585037049856.0000 - val_loss: 228810750737825.1875 - val_mean_squared_error: 228810742038528.0000
    Epoch 661/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 226769603116733.2812 - mean_squared_error: 226769575608320.0000 - val_loss: 243743620663675.2812 - val_mean_squared_error: 243743621906432.0000
    Epoch 662/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 226311471655018.7812 - mean_squared_error: 226311490502656.0000 - val_loss: 227871421133255.1250 - val_mean_squared_error: 227871419269120.0000
    Epoch 663/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 224766902410362.2188 - mean_squared_error: 224766896111616.0000 - val_loss: 241096054188714.6562 - val_mean_squared_error: 241096076558336.0000
    Epoch 664/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 228101220359157.3125 - mean_squared_error: 228101200019456.0000 - val_loss: 229600954815222.5312 - val_mean_squared_error: 229600965689344.0000
    Epoch 665/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 227594521785359.4062 - mean_squared_error: 227594544873472.0000 - val_loss: 229213369746014.8125 - val_mean_squared_error: 229213361668096.0000
    Epoch 666/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 223859068298041.8438 - mean_squared_error: 223859030622208.0000 - val_loss: 223629732527521.1875 - val_mean_squared_error: 223629736411136.0000
    Epoch 667/1000
    1726/1726 [==============================] - 0s 146us/sample - loss: 228431280740890.7188 - mean_squared_error: 228431241412608.0000 - val_loss: 220147758205307.2812 - val_mean_squared_error: 220147776225280.0000
    Epoch 668/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 225979327024966.8750 - mean_squared_error: 225979335180288.0000 - val_loss: 236627501738059.8438 - val_mean_squared_error: 236627498631168.0000
    Epoch 669/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 225376261906137.7188 - mean_squared_error: 225376261373952.0000 - val_loss: 237824299470392.8750 - val_mean_squared_error: 237824284557312.0000
    Epoch 670/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 224984585412541.5625 - mean_squared_error: 224984614043648.0000 - val_loss: 233461602242711.7188 - val_mean_squared_error: 233461604417536.0000
    Epoch 671/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 227596583995356.4062 - mean_squared_error: 227596558139392.0000 - val_loss: 232694257432803.5625 - val_mean_squared_error: 232694248112128.0000
    Epoch 672/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 227290987593993.8125 - mean_squared_error: 227290944372736.0000 - val_loss: 237606505730199.7188 - val_mean_squared_error: 237606499516416.0000
    Epoch 673/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 224166968490133.5000 - mean_squared_error: 224166959644672.0000 - val_loss: 222490158927416.8750 - val_mean_squared_error: 222490160791552.0000
    Epoch 674/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 225277874066229.0938 - mean_squared_error: 225277846224896.0000 - val_loss: 230425762211309.0312 - val_mean_squared_error: 230425750405120.0000
    Epoch 675/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 226856716069692.2188 - mean_squared_error: 226856716468224.0000 - val_loss: 223784595707221.3438 - val_mean_squared_error: 223784590114816.0000
    Epoch 676/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 226024505200170.1250 - mean_squared_error: 226024516222976.0000 - val_loss: 235393858132498.9688 - val_mean_squared_error: 235393869938688.0000
    Epoch 677/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 223095906588242.4688 - mean_squared_error: 223095885398016.0000 - val_loss: 233282744381895.1250 - val_mean_squared_error: 233282759294976.0000
    Epoch 678/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 223457236108953.6562 - mean_squared_error: 223457249853440.0000 - val_loss: 236830471875849.4688 - val_mean_squared_error: 236830486167552.0000
    Epoch 679/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 225100106785024.2812 - mean_squared_error: 225100141953024.0000 - val_loss: 231867210899759.4062 - val_mean_squared_error: 231867215249408.0000
    Epoch 680/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 222332907151350.5312 - mean_squared_error: 222332891168768.0000 - val_loss: 227031454901134.2188 - val_mean_squared_error: 227031451172864.0000
    Epoch 681/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 225606887944813.7500 - mean_squared_error: 225606864207872.0000 - val_loss: 227952113949809.7812 - val_mean_squared_error: 227952134455296.0000
    Epoch 682/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 224573153482227.5312 - mean_squared_error: 224573152821248.0000 - val_loss: 219449155863134.8125 - val_mean_squared_error: 219449139396608.0000
    Epoch 683/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 223937069905613.8438 - mean_squared_error: 223937078231040.0000 - val_loss: 223735469533259.8438 - val_mean_squared_error: 223735466426368.0000
    Epoch 684/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 224135184666799.5938 - mean_squared_error: 224135200374784.0000 - val_loss: 228934095276411.2812 - val_mean_squared_error: 228934104907776.0000
    Epoch 685/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 222211737059859.5625 - mean_squared_error: 222211726114816.0000 - val_loss: 236644150950115.5625 - val_mean_squared_error: 236644141629440.0000
    Epoch 686/1000
    1726/1726 [==============================] - 0s 142us/sample - loss: 225619300479616.7188 - mean_squared_error: 225619296124928.0000 - val_loss: 222182194486385.7812 - val_mean_squared_error: 222182198214656.0000
    Epoch 687/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 224486019912539.0938 - mean_squared_error: 224485978406912.0000 - val_loss: 221038642045231.4062 - val_mean_squared_error: 221038629617664.0000
    Epoch 688/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 222152328422828.3438 - mean_squared_error: 222152317992960.0000 - val_loss: 225147885560111.4062 - val_mean_squared_error: 225147889909760.0000
    Epoch 689/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 222595361294129.5312 - mean_squared_error: 222595370713088.0000 - val_loss: 227796824659892.1562 - val_mean_squared_error: 227796810989568.0000
    Epoch 690/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 224107196246312.6250 - mean_squared_error: 224107199201280.0000 - val_loss: 219880387481144.8750 - val_mean_squared_error: 219880397733888.0000
    Epoch 691/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 224203976365270.7812 - mean_squared_error: 224203953405952.0000 - val_loss: 223834041269665.1875 - val_mean_squared_error: 223834049347584.0000
    Epoch 692/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 222549737997806.8125 - mean_squared_error: 222549719908352.0000 - val_loss: 226578652443685.9375 - val_mean_squared_error: 226578667667456.0000
    Epoch 693/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 221992634122961.4062 - mean_squared_error: 221992632451072.0000 - val_loss: 229113954799160.8750 - val_mean_squared_error: 229113956663296.0000
    Epoch 694/1000
    1726/1726 [==============================] - 0s 139us/sample - loss: 222176756970321.5625 - mean_squared_error: 222176779173888.0000 - val_loss: 234763996393927.1250 - val_mean_squared_error: 234764002918400.0000
    Epoch 695/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 221920929448771.3438 - mean_squared_error: 221920926629888.0000 - val_loss: 227217189711720.2812 - val_mean_squared_error: 227217174953984.0000
    Epoch 696/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 222735752602634.6875 - mean_squared_error: 222735745679360.0000 - val_loss: 240099594124705.1875 - val_mean_squared_error: 240099593814016.0000
    Epoch 697/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 222309612810728.8438 - mean_squared_error: 222309637947392.0000 - val_loss: 232870279012048.5938 - val_mean_squared_error: 232870257885184.0000
    Epoch 698/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 221538230673341.5625 - mean_squared_error: 221538238332928.0000 - val_loss: 230389017622755.5625 - val_mean_squared_error: 230389008302080.0000
    Epoch 699/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 220456072519082.0000 - mean_squared_error: 220456107900928.0000 - val_loss: 228464557235313.7812 - val_mean_squared_error: 228464560963584.0000
    Epoch 700/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 225294176074390.0938 - mean_squared_error: 225294204010496.0000 - val_loss: 223827641072450.3750 - val_mean_squared_error: 223827623673856.0000
    Epoch 701/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 221055039416521.7188 - mean_squared_error: 221054970626048.0000 - val_loss: 228028136486532.7188 - val_mean_squared_error: 228028118466560.0000
    Epoch 702/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 221295923077623.0938 - mean_squared_error: 221295925002240.0000 - val_loss: 230318859033713.7812 - val_mean_squared_error: 230318845984768.0000
    Epoch 703/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 222471059007708.6875 - mean_squared_error: 222471068319744.0000 - val_loss: 222621666202965.3438 - val_mean_squared_error: 222621677387776.0000
    Epoch 704/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 221908569442401.3125 - mean_squared_error: 221908628930560.0000 - val_loss: 226557525967454.8125 - val_mean_squared_error: 226557511598080.0000
    Epoch 705/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 222209629886243.3125 - mean_squared_error: 222209612185600.0000 - val_loss: 233941383084866.3750 - val_mean_squared_error: 233941382463488.0000
    Epoch 706/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 220626315109557.5312 - mean_squared_error: 220626262425600.0000 - val_loss: 224725455145339.2812 - val_mean_squared_error: 224725456388096.0000
    Epoch 707/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 221642329118897.9688 - mean_squared_error: 221642290626560.0000 - val_loss: 226326625415623.1250 - val_mean_squared_error: 226326640328704.0000
    Epoch 708/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 219849324016427.5938 - mean_squared_error: 219849326329856.0000 - val_loss: 240332896536955.2812 - val_mean_squared_error: 240332881002496.0000
    Epoch 709/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 219856005936371.2500 - mean_squared_error: 219856020439040.0000 - val_loss: 227967409023051.8438 - val_mean_squared_error: 227967401721856.0000
    Epoch 710/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 219929848082313.3438 - mean_squared_error: 219929789857792.0000 - val_loss: 233664724307512.8750 - val_mean_squared_error: 233664726171648.0000
    Epoch 711/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 223397884170358.6562 - mean_squared_error: 223397875286016.0000 - val_loss: 235573645264668.4375 - val_mean_squared_error: 235573654585344.0000
    Epoch 712/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 219928374545069.8125 - mean_squared_error: 219928363794432.0000 - val_loss: 226226885487881.4688 - val_mean_squared_error: 226226883002368.0000
    Epoch 713/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 219772383810774.7812 - mean_squared_error: 219772436348928.0000 - val_loss: 224284841335314.9688 - val_mean_squared_error: 224284836364288.0000
    Epoch 714/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 219055915377370.9375 - mean_squared_error: 219055898230784.0000 - val_loss: 233600993256334.2188 - val_mean_squared_error: 233600989528064.0000
    Epoch 715/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 222142733721571.5312 - mean_squared_error: 222142721425408.0000 - val_loss: 228556488301074.9688 - val_mean_squared_error: 228556483330048.0000
    Epoch 716/1000
    1726/1726 [==============================] - 0s 160us/sample - loss: 222689935889603.7812 - mean_squared_error: 222689910325248.0000 - val_loss: 220826820372555.8438 - val_mean_squared_error: 220826834042880.0000
    Epoch 717/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 221740733325340.4688 - mean_squared_error: 221740688998400.0000 - val_loss: 220268359358388.1562 - val_mean_squared_error: 220268354076672.0000
    Epoch 718/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 220189966539670.3750 - mean_squared_error: 220189987700736.0000 - val_loss: 222641505882263.7188 - val_mean_squared_error: 222641508057088.0000
    Epoch 719/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 216547246088353.3750 - mean_squared_error: 216547217899520.0000 - val_loss: 227624484126113.1875 - val_mean_squared_error: 227624475426816.0000
    Epoch 720/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 221295044072036.2500 - mean_squared_error: 221295086141440.0000 - val_loss: 229014384507714.3750 - val_mean_squared_error: 229014383886336.0000
    Epoch 721/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 216956812787863.8750 - mean_squared_error: 216956783296512.0000 - val_loss: 217901942672952.8750 - val_mean_squared_error: 217901927759872.0000
    Epoch 722/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 221840964299409.3438 - mean_squared_error: 221840983195648.0000 - val_loss: 226939624498669.0312 - val_mean_squared_error: 226939629469696.0000
    Epoch 723/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 218512657794295.9688 - mean_squared_error: 218512651976704.0000 - val_loss: 235721854569434.0625 - val_mean_squared_error: 235721847734272.0000
    Epoch 724/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 218601948174248.1875 - mean_squared_error: 218601923543040.0000 - val_loss: 224994169444882.9688 - val_mean_squared_error: 224994160279552.0000
    Epoch 725/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 217013338105772.9375 - mean_squared_error: 217013356068864.0000 - val_loss: 233896264801393.7812 - val_mean_squared_error: 233896268529664.0000
    Epoch 726/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 220351021582566.1875 - mean_squared_error: 220351032197120.0000 - val_loss: 231034725441839.4062 - val_mean_squared_error: 231034729791488.0000
    Epoch 727/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 223090744940684.0312 - mean_squared_error: 223090801901568.0000 - val_loss: 214870101522052.7188 - val_mean_squared_error: 214870100279296.0000
    Epoch 728/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 218714454833624.2500 - mean_squared_error: 218714481885184.0000 - val_loss: 227664160067128.8750 - val_mean_squared_error: 227664170319872.0000
    Epoch 729/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 216020873456083.5000 - mean_squared_error: 216020866301952.0000 - val_loss: 237286199471976.2812 - val_mean_squared_error: 237286188908544.0000
    Epoch 730/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 223538548733152.2500 - mean_squared_error: 223538518687744.0000 - val_loss: 233996461442237.6250 - val_mean_squared_error: 233996478840832.0000
    Epoch 731/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 216087954675586.2188 - mean_squared_error: 216087924834304.0000 - val_loss: 219710071708785.7812 - val_mean_squared_error: 219710092214272.0000
    Epoch 732/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 218683209864140.9688 - mean_squared_error: 218683175600128.0000 - val_loss: 231340703336220.4375 - val_mean_squared_error: 231340695879680.0000
    Epoch 733/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 217311513325316.4375 - mean_squared_error: 217311503974400.0000 - val_loss: 229110602462852.7188 - val_mean_squared_error: 229110601220096.0000
    Epoch 734/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 215164197566119.9062 - mean_squared_error: 215164188098560.0000 - val_loss: 213795192128094.8125 - val_mean_squared_error: 213795184050176.0000
    Epoch 735/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 219911472909391.5000 - mean_squared_error: 219911485915136.0000 - val_loss: 230502484420077.0312 - val_mean_squared_error: 230502489391104.0000
    Epoch 736/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 219775186402910.3125 - mean_squared_error: 219775204589568.0000 - val_loss: 226074768955922.9688 - val_mean_squared_error: 226074763984896.0000
    Epoch 737/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 216844550622876.0312 - mean_squared_error: 216844560498688.0000 - val_loss: 232518344905462.5312 - val_mean_squared_error: 232518339002368.0000
    Epoch 738/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 218010372777730.0625 - mean_squared_error: 218010375684096.0000 - val_loss: 218400804025381.9375 - val_mean_squared_error: 218400798277632.0000
    Epoch 739/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 216316235216210.1875 - mean_squared_error: 216316245966848.0000 - val_loss: 229781849554337.1875 - val_mean_squared_error: 229781857632256.0000
    Epoch 740/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 217826814841880.9375 - mean_squared_error: 217826782609408.0000 - val_loss: 236064743892461.0312 - val_mean_squared_error: 236064740474880.0000
    Epoch 741/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 216224118701586.4062 - mean_squared_error: 216224155828224.0000 - val_loss: 218993664701477.9375 - val_mean_squared_error: 218993654759424.0000
    Epoch 742/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 216126902136865.2500 - mean_squared_error: 216126881529856.0000 - val_loss: 228695041075275.8438 - val_mean_squared_error: 228695046356992.0000
    Epoch 743/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 215232260228457.9062 - mean_squared_error: 215232253263872.0000 - val_loss: 217604349658301.6250 - val_mean_squared_error: 217604333502464.0000
    Epoch 744/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 215879675997975.4375 - mean_squared_error: 215879669252096.0000 - val_loss: 234062107892242.9688 - val_mean_squared_error: 234062111309824.0000
    Epoch 745/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 216833220655918.0000 - mean_squared_error: 216833219100672.0000 - val_loss: 224976256193877.3438 - val_mean_squared_error: 224976258990080.0000
    Epoch 746/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 215991752816523.7500 - mean_squared_error: 215991741054976.0000 - val_loss: 218510919573807.4062 - val_mean_squared_error: 218510890369024.0000
    Epoch 747/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 215919337657368.9375 - mean_squared_error: 215919347367936.0000 - val_loss: 224468302056410.0625 - val_mean_squared_error: 224468295221248.0000
    Epoch 748/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 215602309269772.1875 - mean_squared_error: 215602341871616.0000 - val_loss: 219446594852181.3438 - val_mean_squared_error: 219446589259776.0000
    Epoch 749/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 214718237960032.9688 - mean_squared_error: 214718232920064.0000 - val_loss: 231610340537988.7188 - val_mean_squared_error: 231610322518016.0000
    Epoch 750/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 214708414457791.9375 - mean_squared_error: 214708384694272.0000 - val_loss: 232068265328033.1875 - val_mean_squared_error: 232068273405952.0000
    Epoch 751/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 217207661930583.8125 - mean_squared_error: 217207686561792.0000 - val_loss: 235195125795195.2812 - val_mean_squared_error: 235195127037952.0000
    Epoch 752/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 214888426073158.0000 - mean_squared_error: 214888404221952.0000 - val_loss: 216903147702044.4375 - val_mean_squared_error: 216903163314176.0000
    Epoch 753/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 215176523891694.2188 - mean_squared_error: 215176552906752.0000 - val_loss: 221583692160720.5938 - val_mean_squared_error: 221583704588288.0000
    Epoch 754/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 212849809731180.5625 - mean_squared_error: 212849787928576.0000 - val_loss: 227777508491870.8125 - val_mean_squared_error: 227777500413952.0000
    Epoch 755/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 217421640701351.6250 - mean_squared_error: 217421663174656.0000 - val_loss: 216177151040018.9688 - val_mean_squared_error: 216177162846208.0000
    Epoch 756/1000
    1726/1726 [==============================] - 0s 126us/sample - loss: 215226597660470.2812 - mean_squared_error: 215226616119296.0000 - val_loss: 226805136160388.7188 - val_mean_squared_error: 226805160083456.0000
    Epoch 757/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 214441785215870.6562 - mean_squared_error: 214441777954816.0000 - val_loss: 232917024685473.1875 - val_mean_squared_error: 232917032763392.0000
    Epoch 758/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 215265227958492.6875 - mean_squared_error: 215265237270528.0000 - val_loss: 226903825647995.2812 - val_mean_squared_error: 226903810113536.0000
    Epoch 759/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 214387253771908.3125 - mean_squared_error: 214387235225600.0000 - val_loss: 220468452203633.7812 - val_mean_squared_error: 220468472709120.0000
    Epoch 760/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 213962744852247.4375 - mean_squared_error: 213962771660800.0000 - val_loss: 232560880119049.4688 - val_mean_squared_error: 232560886022144.0000
    Epoch 761/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 219992559935440.5312 - mean_squared_error: 219992536645632.0000 - val_loss: 221812694323958.5312 - val_mean_squared_error: 221812696809472.0000
    Epoch 762/1000
    1726/1726 [==============================] - 0s 124us/sample - loss: 216940768141321.4688 - mean_squared_error: 216940794609664.0000 - val_loss: 226306731365717.3438 - val_mean_squared_error: 226306742550528.0000
    Epoch 763/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 214394172244765.3750 - mean_squared_error: 214394214547456.0000 - val_loss: 223030418837048.8750 - val_mean_squared_error: 223030420701184.0000
    Epoch 764/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 214301600043069.7188 - mean_squared_error: 214301570760704.0000 - val_loss: 222734316509108.1562 - val_mean_squared_error: 222734302838784.0000
    Epoch 765/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 213493967372155.1250 - mean_squared_error: 213493949136896.0000 - val_loss: 220610896204534.5312 - val_mean_squared_error: 220610894495744.0000
    Epoch 766/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 212444744535632.0938 - mean_squared_error: 212444735602688.0000 - val_loss: 224025911104246.5312 - val_mean_squared_error: 224025913589760.0000
    Epoch 767/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 213257720764423.0938 - mean_squared_error: 213257759490048.0000 - val_loss: 226243521650991.4062 - val_mean_squared_error: 226243509223424.0000
    Epoch 768/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 214344346751569.2812 - mean_squared_error: 214344352661504.0000 - val_loss: 228629044789248.0000 - val_mean_squared_error: 228629061566464.0000
    Epoch 769/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 213149924118893.4375 - mean_squared_error: 213149915545600.0000 - val_loss: 219242714395079.1250 - val_mean_squared_error: 219242695753728.0000
    Epoch 770/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 214132187288107.3125 - mean_squared_error: 214132204765184.0000 - val_loss: 222402643997733.9375 - val_mean_squared_error: 222402650832896.0000
    Epoch 771/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 212749397072790.3750 - mean_squared_error: 212749426622464.0000 - val_loss: 220209336180432.5938 - val_mean_squared_error: 220209348608000.0000
    Epoch 772/1000
    1726/1726 [==============================] - 0s 138us/sample - loss: 210795685253922.1250 - mean_squared_error: 210795686264832.0000 - val_loss: 221874610328917.3438 - val_mean_squared_error: 221874604736512.0000
    Epoch 773/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 214264348115817.2812 - mean_squared_error: 214264308563968.0000 - val_loss: 220373443450804.1562 - val_mean_squared_error: 220373446557696.0000
    Epoch 774/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 214251868637395.2188 - mean_squared_error: 214251859869696.0000 - val_loss: 239036330914853.9375 - val_mean_squared_error: 239036320972800.0000
    Epoch 775/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 211285376287080.7188 - mean_squared_error: 211285396422656.0000 - val_loss: 238712679155484.4375 - val_mean_squared_error: 238712688476160.0000
    Epoch 776/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 213611586173666.0312 - mean_squared_error: 213611574198272.0000 - val_loss: 235667301276785.7812 - val_mean_squared_error: 235667305005056.0000
    Epoch 777/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 213361871986279.8438 - mean_squared_error: 213361895669760.0000 - val_loss: 221641941411915.8438 - val_mean_squared_error: 221641921527808.0000
    Epoch 778/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 212005056322872.0625 - mean_squared_error: 212005021548544.0000 - val_loss: 219430220600054.5312 - val_mean_squared_error: 219430214696960.0000
    Epoch 779/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 211249037911316.4688 - mean_squared_error: 211249023418368.0000 - val_loss: 211176245874839.7188 - val_mean_squared_error: 211176260632576.0000
    Epoch 780/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 216204021730421.4688 - mean_squared_error: 216204023169024.0000 - val_loss: 216187377063860.1562 - val_mean_squared_error: 216187380170752.0000
    Epoch 781/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 213003050460094.7188 - mean_squared_error: 213003047796736.0000 - val_loss: 234275002996053.3438 - val_mean_squared_error: 234274997403648.0000
    Epoch 782/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 210907980171418.2500 - mean_squared_error: 210907992948736.0000 - val_loss: 218639761757525.3438 - val_mean_squared_error: 218639756165120.0000
    Epoch 783/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 211057221352010.1562 - mean_squared_error: 211057209507840.0000 - val_loss: 237359927258377.4688 - val_mean_squared_error: 237359907995648.0000
    Epoch 784/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 216356570598441.5000 - mean_squared_error: 216356544839680.0000 - val_loss: 231318731241699.5625 - val_mean_squared_error: 231318734503936.0000
    Epoch 785/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 209376851241009.8125 - mean_squared_error: 209376870662144.0000 - val_loss: 221964463816097.1875 - val_mean_squared_error: 221964463505408.0000
    Epoch 786/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 211973326996329.2812 - mean_squared_error: 211973329387520.0000 - val_loss: 241973578393675.8438 - val_mean_squared_error: 241973592064000.0000
    Epoch 787/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 212424495825921.1875 - mean_squared_error: 212424502280192.0000 - val_loss: 230861040245266.9688 - val_mean_squared_error: 230861052051456.0000
    Epoch 788/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 210148993660394.0625 - mean_squared_error: 210148991696896.0000 - val_loss: 222951625729517.0312 - val_mean_squared_error: 222951634894848.0000
    Epoch 789/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 211325300651820.7812 - mean_squared_error: 211325309419520.0000 - val_loss: 214747584348766.8125 - val_mean_squared_error: 214747593048064.0000
    Epoch 790/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 209660284653287.9688 - mean_squared_error: 209660271394816.0000 - val_loss: 225216598823063.7188 - val_mean_squared_error: 225216609386496.0000
    Epoch 791/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 210977193715780.8125 - mean_squared_error: 210977182187520.0000 - val_loss: 221950985808554.6562 - val_mean_squared_error: 221950957846528.0000
    Epoch 792/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 208705747573445.5625 - mean_squared_error: 208705731690496.0000 - val_loss: 222761448374272.0000 - val_mean_squared_error: 222761448374272.0000
    Epoch 793/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 210172715224656.0938 - mean_squared_error: 210172681125888.0000 - val_loss: 221933382780700.4375 - val_mean_squared_error: 221933375324160.0000
    Epoch 794/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 210444397499644.7188 - mean_squared_error: 210444371361792.0000 - val_loss: 231293527047585.1875 - val_mean_squared_error: 231293535125504.0000
    Epoch 795/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 208049497088199.3125 - mean_squared_error: 208049524441088.0000 - val_loss: 221423498952704.0000 - val_mean_squared_error: 221423482175488.0000
    Epoch 796/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 209628564050703.1250 - mean_squared_error: 209628562456576.0000 - val_loss: 216207179149767.1250 - val_mean_squared_error: 216207177285632.0000
    Epoch 797/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 208996380044396.0000 - mean_squared_error: 208996413734912.0000 - val_loss: 226011466034441.4688 - val_mean_squared_error: 226011463548928.0000
    Epoch 798/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 210005382460526.3750 - mean_squared_error: 210005412282368.0000 - val_loss: 235824768497891.5625 - val_mean_squared_error: 235824759177216.0000
    Epoch 799/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 211192777889210.5625 - mean_squared_error: 211192769413120.0000 - val_loss: 230814111265223.1250 - val_mean_squared_error: 230814109401088.0000
    Epoch 800/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 209181602142044.2500 - mean_squared_error: 209181634199552.0000 - val_loss: 231907583716617.4688 - val_mean_squared_error: 231907581231104.0000
    Epoch 801/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 210311846713446.0625 - mean_squared_error: 210311864909824.0000 - val_loss: 215470024629361.7812 - val_mean_squared_error: 215470036746240.0000
    Epoch 802/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 209313498782063.8125 - mean_squared_error: 209313486340096.0000 - val_loss: 222704763132434.9688 - val_mean_squared_error: 222704774938624.0000
    Epoch 803/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 208789565786399.1562 - mean_squared_error: 208789550661632.0000 - val_loss: 221675819116013.0312 - val_mean_squared_error: 221675828281344.0000
    Epoch 804/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 206433769614747.7500 - mean_squared_error: 206433777876992.0000 - val_loss: 228988623092546.3750 - val_mean_squared_error: 228988630859776.0000
    Epoch 805/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 209971980248226.5625 - mean_squared_error: 209971958513664.0000 - val_loss: 212934615707496.2812 - val_mean_squared_error: 212934613532672.0000
    Epoch 806/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 212024835341007.0625 - mean_squared_error: 212024868995072.0000 - val_loss: 223039059724667.2812 - val_mean_squared_error: 223039077744640.0000
    Epoch 807/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 206349410550424.4688 - mean_squared_error: 206349405257728.0000 - val_loss: 225599053015722.6562 - val_mean_squared_error: 225599062802432.0000
    Epoch 808/1000
    1726/1726 [==============================] - 0s 130us/sample - loss: 208890437684626.2500 - mean_squared_error: 208890448838656.0000 - val_loss: 215870822532892.4375 - val_mean_squared_error: 215870827659264.0000
    Epoch 809/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 208479233053512.0938 - mean_squared_error: 208479222497280.0000 - val_loss: 222257585702684.4375 - val_mean_squared_error: 222257578246144.0000
    Epoch 810/1000
    1726/1726 [==============================] - 0s 135us/sample - loss: 210656841300952.8438 - mean_squared_error: 210656821248000.0000 - val_loss: 223707459039080.2812 - val_mean_squared_error: 223707465252864.0000
    Epoch 811/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 209433397088165.8438 - mean_squared_error: 209433376325632.0000 - val_loss: 216720797461769.4688 - val_mean_squared_error: 216720811753472.0000
    Epoch 812/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 210242889886733.0625 - mean_squared_error: 210242876997632.0000 - val_loss: 225095277803140.7188 - val_mean_squared_error: 225095276560384.0000
    Epoch 813/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 206522140499256.0625 - mean_squared_error: 206522109919232.0000 - val_loss: 227140212194114.3750 - val_mean_squared_error: 227140217864192.0000
    Epoch 814/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 207018538564720.7188 - mean_squared_error: 207018497409024.0000 - val_loss: 230167589440474.0625 - val_mean_squared_error: 230167582605312.0000
    Epoch 815/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 206017893573908.4688 - mean_squared_error: 206017887469568.0000 - val_loss: 226719748519556.7188 - val_mean_squared_error: 226719747276800.0000
    Epoch 816/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 205978359682832.3125 - mean_squared_error: 205978343571456.0000 - val_loss: 224973386979252.1562 - val_mean_squared_error: 224973406863360.0000
    Epoch 817/1000
    1726/1726 [==============================] - 0s 137us/sample - loss: 209062306260472.2812 - mean_squared_error: 209062281084928.0000 - val_loss: 238225967769675.8438 - val_mean_squared_error: 238225947885568.0000
    Epoch 818/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 208113783997251.3438 - mean_squared_error: 208113764401152.0000 - val_loss: 224042682727841.1875 - val_mean_squared_error: 224042674028544.0000
    Epoch 819/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 207523452398984.7500 - mean_squared_error: 207523458056192.0000 - val_loss: 215144608466109.6250 - val_mean_squared_error: 215144609087488.0000
    Epoch 820/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 208133398079120.1875 - mean_squared_error: 208133376966656.0000 - val_loss: 223228064383582.8125 - val_mean_squared_error: 223228089860096.0000
    Epoch 821/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 204761107958939.4375 - mean_squared_error: 204761122996224.0000 - val_loss: 234560188948783.4062 - val_mean_squared_error: 234560193298432.0000
    Epoch 822/1000
    1726/1726 [==============================] - 0s 123us/sample - loss: 211857550198995.2188 - mean_squared_error: 211857533042688.0000 - val_loss: 212905309639414.5312 - val_mean_squared_error: 212905303736320.0000
    Epoch 823/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 207861771880469.3438 - mean_squared_error: 207861787394048.0000 - val_loss: 220276171016571.2812 - val_mean_squared_error: 220276189036544.0000
    Epoch 824/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 207184782992014.9688 - mean_squared_error: 207184742842368.0000 - val_loss: 230818227275548.4375 - val_mean_squared_error: 230818219819008.0000
    Epoch 825/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 207234752117037.3750 - mean_squared_error: 207234755723264.0000 - val_loss: 235444541480656.5938 - val_mean_squared_error: 235444553908224.0000
    Epoch 826/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 207400167126790.8125 - mean_squared_error: 207400179073024.0000 - val_loss: 223207216828567.7188 - val_mean_squared_error: 223207202226176.0000
    Epoch 827/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 205846086814229.9375 - mean_squared_error: 205846072000512.0000 - val_loss: 214946826837712.5938 - val_mean_squared_error: 214946822488064.0000
    Epoch 828/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 207724217761032.5938 - mean_squared_error: 207724180668416.0000 - val_loss: 219050613407744.0000 - val_mean_squared_error: 219050613407744.0000
    Epoch 829/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 202673837962730.0625 - mean_squared_error: 202673869553664.0000 - val_loss: 234607781560926.8125 - val_mean_squared_error: 234607790260224.0000
    Epoch 830/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 208005701419748.4062 - mean_squared_error: 208005719130112.0000 - val_loss: 224690097473308.4375 - val_mean_squared_error: 224690106793984.0000
    Epoch 831/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 206107317904280.7812 - mean_squared_error: 206107326808064.0000 - val_loss: 214901088573970.9688 - val_mean_squared_error: 214901071020032.0000
    Epoch 832/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 206063715131287.5938 - mean_squared_error: 206063706046464.0000 - val_loss: 220603512520704.0000 - val_mean_squared_error: 220603512520704.0000
    Epoch 833/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 205407111725492.6562 - mean_squared_error: 205407096143872.0000 - val_loss: 225073570571150.2188 - val_mean_squared_error: 225073566842880.0000
    Epoch 834/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 205727110423923.3750 - mean_squared_error: 205727087984640.0000 - val_loss: 220304970661281.1875 - val_mean_squared_error: 220304978739200.0000
    Epoch 835/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 207785267396418.1562 - mean_squared_error: 207785266511872.0000 - val_loss: 222845071766565.9375 - val_mean_squared_error: 222845066018816.0000
    Epoch 836/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 207024177206220.9688 - mean_squared_error: 207024168108032.0000 - val_loss: 220430733293795.5625 - val_mean_squared_error: 220430723973120.0000
    Epoch 837/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 203472555550579.9688 - mean_squared_error: 203472532144128.0000 - val_loss: 224374728687616.0000 - val_mean_squared_error: 224374762242048.0000
    Epoch 838/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 204315058785264.5938 - mean_squared_error: 204315067154432.0000 - val_loss: 223530286356404.1562 - val_mean_squared_error: 223530297851904.0000
    Epoch 839/1000
    1726/1726 [==============================] - 0s 131us/sample - loss: 205556151987744.6250 - mean_squared_error: 205556195262464.0000 - val_loss: 218298017324145.7812 - val_mean_squared_error: 218298021052416.0000
    Epoch 840/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 202300612064195.5000 - mean_squared_error: 202300626829312.0000 - val_loss: 240597939888431.4062 - val_mean_squared_error: 240597944238080.0000
    Epoch 841/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 204380451590851.2188 - mean_squared_error: 204380397633536.0000 - val_loss: 210717168272573.6250 - val_mean_squared_error: 210717152116736.0000
    Epoch 842/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 208646344939906.8125 - mean_squared_error: 208646340345856.0000 - val_loss: 227957591400144.5938 - val_mean_squared_error: 227957570273280.0000
    Epoch 843/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 203889467744892.0000 - mean_squared_error: 203889445961728.0000 - val_loss: 212189349713844.1562 - val_mean_squared_error: 212189352820736.0000
    Epoch 844/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 205006504341214.4688 - mean_squared_error: 205006489780224.0000 - val_loss: 229498833454876.4375 - val_mean_squared_error: 229498825998336.0000
    Epoch 845/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 204465371792783.8438 - mean_squared_error: 204465374232576.0000 - val_loss: 223766851626477.0312 - val_mean_squared_error: 223766856597504.0000
    Epoch 846/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 203552275048839.5938 - mean_squared_error: 203552223920128.0000 - val_loss: 225411946512384.0000 - val_mean_squared_error: 225411929735168.0000
    Epoch 847/1000
    1726/1726 [==============================] - 0s 150us/sample - loss: 204521992452131.5938 - mean_squared_error: 204521963782144.0000 - val_loss: 214791088912611.5625 - val_mean_squared_error: 214791096369152.0000
    Epoch 848/1000
    1726/1726 [==============================] - 0s 141us/sample - loss: 203535779910718.8750 - mean_squared_error: 203535782248448.0000 - val_loss: 218762825743777.1875 - val_mean_squared_error: 218762833821696.0000
    Epoch 849/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 203238853179465.5625 - mean_squared_error: 203238875856896.0000 - val_loss: 212563007208258.3750 - val_mean_squared_error: 212562998198272.0000
    Epoch 850/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 203176998898691.5625 - mean_squared_error: 203177018261504.0000 - val_loss: 224224004043207.1250 - val_mean_squared_error: 224224002179072.0000
    Epoch 851/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 202800468889794.5938 - mean_squared_error: 202800470425600.0000 - val_loss: 221042067859531.8438 - val_mean_squared_error: 221042068946944.0000
    Epoch 852/1000
    1726/1726 [==============================] - 0s 122us/sample - loss: 202224964520678.7812 - mean_squared_error: 202224944807936.0000 - val_loss: 220967620050944.0000 - val_mean_squared_error: 220967611662336.0000
    Epoch 853/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 204092015652118.8438 - mean_squared_error: 204091980513280.0000 - val_loss: 224858189021790.8125 - val_mean_squared_error: 224858197721088.0000
    Epoch 854/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 203662052708375.7188 - mean_squared_error: 203662114684928.0000 - val_loss: 217760911375777.1875 - val_mean_squared_error: 217760932036608.0000
    Epoch 855/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 200726782324484.4375 - mean_squared_error: 200726806528000.0000 - val_loss: 226211107448301.0312 - val_mean_squared_error: 226211095642112.0000
    Epoch 856/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 203662232903063.0000 - mean_squared_error: 203662232125440.0000 - val_loss: 214207215076541.6250 - val_mean_squared_error: 214207198920704.0000
    Epoch 857/1000
    1726/1726 [==============================] - 0s 133us/sample - loss: 203196171932241.2812 - mean_squared_error: 203196161064960.0000 - val_loss: 224392559139953.7812 - val_mean_squared_error: 224392579645440.0000
    Epoch 858/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 202820564300853.4062 - mean_squared_error: 202820552753152.0000 - val_loss: 229740267535170.3750 - val_mean_squared_error: 229740283691008.0000
    Epoch 859/1000
    1726/1726 [==============================] - 0s 137us/sample - loss: 204197139811559.3750 - mean_squared_error: 204197173657600.0000 - val_loss: 228092493809436.4375 - val_mean_squared_error: 228092492644352.0000
    Epoch 860/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 206257297705209.1875 - mean_squared_error: 206257281564672.0000 - val_loss: 221134734326594.3750 - val_mean_squared_error: 221134746288128.0000
    Epoch 861/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 202775613473451.4375 - mean_squared_error: 202775589814272.0000 - val_loss: 222408945395787.8438 - val_mean_squared_error: 222408959066112.0000
    Epoch 862/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 203344730146002.0000 - mean_squared_error: 203344740089856.0000 - val_loss: 211365766588491.8438 - val_mean_squared_error: 211365759287296.0000
    Epoch 863/1000
    1726/1726 [==============================] - 0s 144us/sample - loss: 201804141757555.0938 - mean_squared_error: 201804138676224.0000 - val_loss: 219390105655220.1562 - val_mean_squared_error: 219390117150720.0000
    Epoch 864/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 202401300292732.5625 - mean_squared_error: 202401290125312.0000 - val_loss: 207704898054826.6562 - val_mean_squared_error: 207704920424448.0000
    Epoch 865/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 201528092779549.6562 - mean_squared_error: 201528086364160.0000 - val_loss: 229078734451749.9375 - val_mean_squared_error: 229078741286912.0000
    Epoch 866/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 202270043976643.5000 - mean_squared_error: 202270025187328.0000 - val_loss: 232605777502511.4062 - val_mean_squared_error: 232605781852160.0000
    Epoch 867/1000
    1726/1726 [==============================] - 0s 130us/sample - loss: 202169993918993.1875 - mean_squared_error: 202169965871104.0000 - val_loss: 214237612906420.1562 - val_mean_squared_error: 214237599236096.0000
    Epoch 868/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 202837189977573.2812 - mean_squared_error: 202837195751424.0000 - val_loss: 212979366134973.6250 - val_mean_squared_error: 212979358367744.0000
    Epoch 869/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 202748256769580.5000 - mean_squared_error: 202748276506624.0000 - val_loss: 222553953359189.3438 - val_mean_squared_error: 222553947766784.0000
    Epoch 870/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 200048639438653.4062 - mean_squared_error: 200048654680064.0000 - val_loss: 240762444840960.0000 - val_mean_squared_error: 240762428063744.0000
    Epoch 871/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 202131417894195.3125 - mean_squared_error: 202131395051520.0000 - val_loss: 211265624628944.5938 - val_mean_squared_error: 211265599307776.0000
    Epoch 872/1000
    1726/1726 [==============================] - 0s 117us/sample - loss: 200160368100808.8125 - mean_squared_error: 200160390938624.0000 - val_loss: 224600536344651.8438 - val_mean_squared_error: 224600550014976.0000
    Epoch 873/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 201087100794365.0625 - mean_squared_error: 201087130796032.0000 - val_loss: 220587161570266.0625 - val_mean_squared_error: 220587171512320.0000
    Epoch 874/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 200082125867522.9375 - mean_squared_error: 200082108448768.0000 - val_loss: 219507456999424.0000 - val_mean_squared_error: 219507473776640.0000
    Epoch 875/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 198517173007717.1250 - mean_squared_error: 198517163294720.0000 - val_loss: 223192161141342.8125 - val_mean_squared_error: 223192169840640.0000
    Epoch 876/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 201692762943122.5625 - mean_squared_error: 201692754739200.0000 - val_loss: 222686474724238.2188 - val_mean_squared_error: 222686454218752.0000
    Epoch 877/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 201226996691413.8750 - mean_squared_error: 201226985668608.0000 - val_loss: 220073545122398.8125 - val_mean_squared_error: 220073553821696.0000
    Epoch 878/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 201722704111020.3438 - mean_squared_error: 201722735624192.0000 - val_loss: 222869744371484.4375 - val_mean_squared_error: 222869745303552.0000
    Epoch 879/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 197946210231458.5625 - mean_squared_error: 197946217857024.0000 - val_loss: 226055839129903.4062 - val_mean_squared_error: 226055839285248.0000
    Epoch 880/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 201625820740739.7188 - mean_squared_error: 201625847201792.0000 - val_loss: 208442443422226.9688 - val_mean_squared_error: 208442463617024.0000
    Epoch 881/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 197673719087616.5938 - mean_squared_error: 197673688760320.0000 - val_loss: 227544687803202.3750 - val_mean_squared_error: 227544682987520.0000
    Epoch 882/1000
    1726/1726 [==============================] - 0s 118us/sample - loss: 201026502428180.7812 - mean_squared_error: 201026531491840.0000 - val_loss: 223737176459643.2812 - val_mean_squared_error: 223737177702400.0000
    Epoch 883/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 199266511970712.1562 - mean_squared_error: 199266484092928.0000 - val_loss: 225226070493563.2812 - val_mean_squared_error: 225226071736320.0000
    Epoch 884/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 199811678448121.4688 - mean_squared_error: 199811693281280.0000 - val_loss: 222116775150174.8125 - val_mean_squared_error: 222116783849472.0000
    Epoch 885/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 198308320269374.8750 - mean_squared_error: 198308320509952.0000 - val_loss: 217915839489517.0312 - val_mean_squared_error: 217915819294720.0000
    Epoch 886/1000
    1726/1726 [==============================] - 0s 128us/sample - loss: 199476550020336.8750 - mean_squared_error: 199476551614464.0000 - val_loss: 221994253627240.2812 - val_mean_squared_error: 221994259841024.0000
    Epoch 887/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 197669220033166.9688 - mean_squared_error: 197669226020864.0000 - val_loss: 237203725163633.7812 - val_mean_squared_error: 237203745669120.0000
    Epoch 888/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 202782621863826.8438 - mean_squared_error: 202782636244992.0000 - val_loss: 222681582612328.2812 - val_mean_squared_error: 222681572048896.0000
    Epoch 889/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 199662508954648.9375 - mean_squared_error: 199662493499392.0000 - val_loss: 220497650617912.8750 - val_mean_squared_error: 220497631510528.0000
    Epoch 890/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 196827673051192.9688 - mean_squared_error: 196827664089088.0000 - val_loss: 227478942398691.5625 - val_mean_squared_error: 227478966632448.0000
    Epoch 891/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 196176779479498.0312 - mean_squared_error: 196176775217152.0000 - val_loss: 208701968312395.8438 - val_mean_squared_error: 208701940039680.0000
    Epoch 892/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 197740945392128.5938 - mean_squared_error: 197740998950912.0000 - val_loss: 216438084206592.0000 - val_mean_squared_error: 216438098886656.0000
    Epoch 893/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 198480288849967.4688 - mean_squared_error: 198480320528384.0000 - val_loss: 212927168643072.0000 - val_mean_squared_error: 212927181225984.0000
    Epoch 894/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 198981997195904.7188 - mean_squared_error: 198981976064000.0000 - val_loss: 212600615202057.4688 - val_mean_squared_error: 212600629493760.0000
    Epoch 895/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 199363039065729.9375 - mean_squared_error: 199363020193792.0000 - val_loss: 219982301301115.2812 - val_mean_squared_error: 219982302543872.0000
    Epoch 896/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 194176585047162.2188 - mean_squared_error: 194176578748416.0000 - val_loss: 207392804554524.4375 - val_mean_squared_error: 207392797097984.0000
    Epoch 897/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 198340267201560.9375 - mean_squared_error: 198340281106432.0000 - val_loss: 216499766340266.6562 - val_mean_squared_error: 216499755155456.0000
    Epoch 898/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 198640947514124.7500 - mean_squared_error: 198640912039936.0000 - val_loss: 228079817146974.8125 - val_mean_squared_error: 228079842623488.0000
    Epoch 899/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 195908786729126.1250 - mean_squared_error: 195908775968768.0000 - val_loss: 212568417860418.3750 - val_mean_squared_error: 212568383684608.0000
    Epoch 900/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 197653977669983.2500 - mean_squared_error: 197653992308736.0000 - val_loss: 217283012533361.7812 - val_mean_squared_error: 217283016261632.0000
    Epoch 901/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 196763681313247.3750 - mean_squared_error: 196763692564480.0000 - val_loss: 225388850499735.7188 - val_mean_squared_error: 225388844285952.0000
    Epoch 902/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 197389073054908.6562 - mean_squared_error: 197389096845312.0000 - val_loss: 215730250900366.2188 - val_mean_squared_error: 215730268143616.0000
    Epoch 903/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 196084253245041.3125 - mean_squared_error: 196084282425344.0000 - val_loss: 210505853955337.4688 - val_mean_squared_error: 210505859858432.0000
    Epoch 904/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 197366706858356.5938 - mean_squared_error: 197366699261952.0000 - val_loss: 228549665877257.4688 - val_mean_squared_error: 228549671780352.0000
    Epoch 905/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 197238285750827.3125 - mean_squared_error: 197238320005120.0000 - val_loss: 223595790821451.8438 - val_mean_squared_error: 223595796103168.0000
    Epoch 906/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 194638945174092.5312 - mean_squared_error: 194638942044160.0000 - val_loss: 210587633873351.1250 - val_mean_squared_error: 210587648786432.0000
    Epoch 907/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 199576563359601.5938 - mean_squared_error: 199576610930688.0000 - val_loss: 220746403758686.8125 - val_mean_squared_error: 220746404069376.0000
    Epoch 908/1000
    1726/1726 [==============================] - 0s 119us/sample - loss: 194126193483310.8750 - mean_squared_error: 194126146437120.0000 - val_loss: 215069497025422.2188 - val_mean_squared_error: 215069514268672.0000
    Epoch 909/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 196159600635804.3125 - mean_squared_error: 196159561793536.0000 - val_loss: 214457073183402.6562 - val_mean_squared_error: 214457078775808.0000
    Epoch 910/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 196863436462239.0312 - mean_squared_error: 196863466668032.0000 - val_loss: 223094194627470.2188 - val_mean_squared_error: 223094174121984.0000
    Epoch 911/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 195841567569025.3438 - mean_squared_error: 195841549664256.0000 - val_loss: 213378233260562.9688 - val_mean_squared_error: 213378236678144.0000
    Epoch 912/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 197242897035714.8750 - mean_squared_error: 197242916962304.0000 - val_loss: 206768861134241.1875 - val_mean_squared_error: 206768869212160.0000
    Epoch 913/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 195369798877051.1250 - mean_squared_error: 195369807904768.0000 - val_loss: 221473097684916.1562 - val_mean_squared_error: 221473092403200.0000
    Epoch 914/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 198293385322429.5625 - mean_squared_error: 198293388787712.0000 - val_loss: 215796423191817.4688 - val_mean_squared_error: 215796437483520.0000
    Epoch 915/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 194513431263571.3438 - mean_squared_error: 194513398136832.0000 - val_loss: 213856208377173.3438 - val_mean_squared_error: 213856219561984.0000
    Epoch 916/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 195808674816427.1562 - mean_squared_error: 195808666320896.0000 - val_loss: 213906329377905.7812 - val_mean_squared_error: 213906333106176.0000
    Epoch 917/1000
    1726/1726 [==============================] - 0s 97us/sample - loss: 195941377462675.4375 - mean_squared_error: 195941357322240.0000 - val_loss: 210907583771079.1250 - val_mean_squared_error: 210907573518336.0000
    Epoch 918/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 196741029131260.4375 - mean_squared_error: 196741009768448.0000 - val_loss: 223728075130652.4375 - val_mean_squared_error: 223728067674112.0000
    Epoch 919/1000
    1726/1726 [==============================] - 0s 115us/sample - loss: 194205951318383.8125 - mean_squared_error: 194205955653632.0000 - val_loss: 222574689376786.9688 - val_mean_squared_error: 222574684405760.0000
    Epoch 920/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 194964349008420.1875 - mean_squared_error: 194964319371264.0000 - val_loss: 231897795143149.0312 - val_mean_squared_error: 231897816891392.0000
    Epoch 921/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 198288522617449.0000 - mean_squared_error: 198288556949504.0000 - val_loss: 229264209060143.4062 - val_mean_squared_error: 229264213409792.0000
    Epoch 922/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 195123000479407.0312 - mean_squared_error: 195122998280192.0000 - val_loss: 219724671925665.1875 - val_mean_squared_error: 219724688392192.0000
    Epoch 923/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 191612963717347.8125 - mean_squared_error: 191612986589184.0000 - val_loss: 232415657158731.8438 - val_mean_squared_error: 232415645663232.0000
    Epoch 924/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 199603252372073.0000 - mean_squared_error: 199603286704128.0000 - val_loss: 237557838754853.9375 - val_mean_squared_error: 237557828812800.0000
    Epoch 925/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 192667474263836.1875 - mean_squared_error: 192667451392000.0000 - val_loss: 210311725410379.8438 - val_mean_squared_error: 210311730692096.0000
    Epoch 926/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 196177359022471.5938 - mean_squared_error: 196177362419712.0000 - val_loss: 230425662790769.7812 - val_mean_squared_error: 230425666519040.0000
    Epoch 927/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 191810959805975.1562 - mean_squared_error: 191810974515200.0000 - val_loss: 210609019076001.1875 - val_mean_squared_error: 210609022959616.0000
    Epoch 928/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 195152569809861.8750 - mean_squared_error: 195152576512000.0000 - val_loss: 221331684084698.0625 - val_mean_squared_error: 221331694026752.0000
    Epoch 929/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 195170093334945.6875 - mean_squared_error: 195170075148288.0000 - val_loss: 214060142894193.7812 - val_mean_squared_error: 214060163399680.0000
    Epoch 930/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 193584925249388.8750 - mean_squared_error: 193584896671744.0000 - val_loss: 223260894909781.3438 - val_mean_squared_error: 223260889317376.0000
    Epoch 931/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 193527265641985.7812 - mean_squared_error: 193527283712000.0000 - val_loss: 207493350099323.2812 - val_mean_squared_error: 207493342953472.0000
    Epoch 932/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 195064283730000.6875 - mean_squared_error: 195064294801408.0000 - val_loss: 213232126389513.4688 - val_mean_squared_error: 213232123904000.0000
    Epoch 933/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 192438373457322.0000 - mean_squared_error: 192438341730304.0000 - val_loss: 209918072426040.8750 - val_mean_squared_error: 209918070095872.0000
    Epoch 934/1000
    1726/1726 [==============================] - 0s 125us/sample - loss: 190911852817952.6250 - mean_squared_error: 190911849955328.0000 - val_loss: 218641478004584.2812 - val_mean_squared_error: 218641467441152.0000
    Epoch 935/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 194710140981518.5312 - mean_squared_error: 194710161326080.0000 - val_loss: 218698946183168.0000 - val_mean_squared_error: 218698962960384.0000
    Epoch 936/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 191357819257401.5625 - mean_squared_error: 191357821911040.0000 - val_loss: 209270776363842.3750 - val_mean_squared_error: 209270788325376.0000
    Epoch 937/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 192228389836446.4062 - mean_squared_error: 192228374872064.0000 - val_loss: 203521811798546.9688 - val_mean_squared_error: 203521823604736.0000
    Epoch 938/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 189595565674026.1250 - mean_squared_error: 189595543142400.0000 - val_loss: 236526129449339.2812 - val_mean_squared_error: 236526130692096.0000
    Epoch 939/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 191002634362889.4688 - mean_squared_error: 191002681802752.0000 - val_loss: 201651369075598.2188 - val_mean_squared_error: 201651365347328.0000
    Epoch 940/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 197028240605392.8438 - mean_squared_error: 197028235706368.0000 - val_loss: 209535125635678.8125 - val_mean_squared_error: 209535130140672.0000
    Epoch 941/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 192369481513527.1875 - mean_squared_error: 192369471258624.0000 - val_loss: 212115365919554.3750 - val_mean_squared_error: 212115365298176.0000
    Epoch 942/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 191779049893503.5625 - mean_squared_error: 191779030695936.0000 - val_loss: 206069233828446.8125 - val_mean_squared_error: 206069225750528.0000
    Epoch 943/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 191143142769266.5000 - mean_squared_error: 191143140655104.0000 - val_loss: 213565132378870.5312 - val_mean_squared_error: 213565134864384.0000
    Epoch 944/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 189607649779759.4688 - mean_squared_error: 189607673069568.0000 - val_loss: 216285805261861.9375 - val_mean_squared_error: 216285812097024.0000
    Epoch 945/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 192630534343227.9062 - mean_squared_error: 192630541516800.0000 - val_loss: 225170480741793.1875 - val_mean_squared_error: 225170505596928.0000
    Epoch 946/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 191137458369513.4375 - mean_squared_error: 191137486733312.0000 - val_loss: 211930275322993.7812 - val_mean_squared_error: 211930262274048.0000
    Epoch 947/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 192661376815051.8125 - mean_squared_error: 192661378039808.0000 - val_loss: 225753833552099.5625 - val_mean_squared_error: 225753815842816.0000
    Epoch 948/1000
    1726/1726 [==============================] - 0s 116us/sample - loss: 192073982775110.8750 - mean_squared_error: 192073957376000.0000 - val_loss: 206992300407618.3750 - val_mean_squared_error: 206992308174848.0000
    Epoch 949/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 192056867208057.9062 - mean_squared_error: 192056861392896.0000 - val_loss: 213339701277127.1250 - val_mean_squared_error: 213339699412992.0000
    Epoch 950/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 192117349516441.0625 - mean_squared_error: 192117292924928.0000 - val_loss: 207147969670409.4688 - val_mean_squared_error: 207147967184896.0000
    Epoch 951/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 190990585395995.0000 - mean_squared_error: 190990618984448.0000 - val_loss: 211079486388451.5625 - val_mean_squared_error: 211079489650688.0000
    Epoch 952/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 191242247577571.5312 - mean_squared_error: 191242226892800.0000 - val_loss: 205964885758710.5312 - val_mean_squared_error: 205964871467008.0000
    Epoch 953/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 188713190345446.7812 - mean_squared_error: 188713162244096.0000 - val_loss: 212097438532190.8125 - val_mean_squared_error: 212097430454272.0000
    Epoch 954/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 193046850265951.8125 - mean_squared_error: 193046784245760.0000 - val_loss: 233789272145313.1875 - val_mean_squared_error: 233789263446016.0000
    Epoch 955/1000
    1726/1726 [==============================] - 0s 114us/sample - loss: 189389186952945.4688 - mean_squared_error: 189389200162816.0000 - val_loss: 206824925929775.4062 - val_mean_squared_error: 206824905113600.0000
    Epoch 956/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 188602520694019.8750 - mean_squared_error: 188602516504576.0000 - val_loss: 224589457168308.1562 - val_mean_squared_error: 224589443497984.0000
    Epoch 957/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 190329081654342.0000 - mean_squared_error: 190329110134784.0000 - val_loss: 214505738294613.3438 - val_mean_squared_error: 214505732702208.0000
    Epoch 958/1000
    1726/1726 [==============================] - 0s 101us/sample - loss: 190854477145003.7500 - mean_squared_error: 190854471876608.0000 - val_loss: 212672592876278.5312 - val_mean_squared_error: 212672603750400.0000
    Epoch 959/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 189958302536823.8438 - mean_squared_error: 189958266552320.0000 - val_loss: 211464909216274.9688 - val_mean_squared_error: 211464895856640.0000
    Epoch 960/1000
    1726/1726 [==============================] - 0s 129us/sample - loss: 191139778378542.0000 - mean_squared_error: 191139768434688.0000 - val_loss: 220856862152628.1562 - val_mean_squared_error: 220856865259520.0000
    Epoch 961/1000
    1726/1726 [==============================] - 0s 121us/sample - loss: 190131059804523.0938 - mean_squared_error: 190131071877120.0000 - val_loss: 219516072099840.0000 - val_mean_squared_error: 219516080488448.0000
    Epoch 962/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 188978088433348.3750 - mean_squared_error: 188978091261952.0000 - val_loss: 221740590820617.4688 - val_mean_squared_error: 221740588335104.0000
    Epoch 963/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 191168036070429.6562 - mean_squared_error: 191168021266432.0000 - val_loss: 212613319904218.0625 - val_mean_squared_error: 212613329846272.0000
    Epoch 964/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 190351914325054.8750 - mean_squared_error: 190351910371328.0000 - val_loss: 213714411075811.5625 - val_mean_squared_error: 213714384977920.0000
    Epoch 965/1000
    1726/1726 [==============================] - 0s 109us/sample - loss: 188663720806744.0938 - mean_squared_error: 188663703011328.0000 - val_loss: 214363900603657.4688 - val_mean_squared_error: 214363898118144.0000
    Epoch 966/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 188820650751655.9062 - mean_squared_error: 188820670644224.0000 - val_loss: 214059739619631.4062 - val_mean_squared_error: 214059743969280.0000
    Epoch 967/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 188237452459177.6875 - mean_squared_error: 188237461061632.0000 - val_loss: 213058426124515.5625 - val_mean_squared_error: 213058429386752.0000
    Epoch 968/1000
    1726/1726 [==============================] - 0s 104us/sample - loss: 189099257832268.8438 - mean_squared_error: 189099239538688.0000 - val_loss: 206081592422855.1250 - val_mean_squared_error: 206081590558720.0000
    Epoch 969/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 190793427776926.0938 - mean_squared_error: 190793419587584.0000 - val_loss: 222058986340352.0000 - val_mean_squared_error: 222059003117568.0000
    Epoch 970/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 190860038468908.2188 - mean_squared_error: 190860058689536.0000 - val_loss: 205966387630231.7188 - val_mean_squared_error: 205966381416448.0000
    Epoch 971/1000
    1726/1726 [==============================] - 0s 98us/sample - loss: 188554141278255.4688 - mean_squared_error: 188554181345280.0000 - val_loss: 217154238079886.2188 - val_mean_squared_error: 217154234351616.0000
    Epoch 972/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 188446474291639.0312 - mean_squared_error: 188446454841344.0000 - val_loss: 204145369342862.2188 - val_mean_squared_error: 204145365614592.0000
    Epoch 973/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 191089410434988.9375 - mean_squared_error: 191089420009472.0000 - val_loss: 221240213460309.3438 - val_mean_squared_error: 221240224645120.0000
    Epoch 974/1000
    1726/1726 [==============================] - 0s 120us/sample - loss: 188525074508528.2812 - mean_squared_error: 188525056098304.0000 - val_loss: 248396877990267.2812 - val_mean_squared_error: 248396866650112.0000
    Epoch 975/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 186603195814291.4375 - mean_squared_error: 186603226005504.0000 - val_loss: 199170749570161.7812 - val_mean_squared_error: 199170753298432.0000
    Epoch 976/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 188391858849813.3438 - mean_squared_error: 188391878557696.0000 - val_loss: 215171233286523.2812 - val_mean_squared_error: 215171217752064.0000
    Epoch 977/1000
    1726/1726 [==============================] - 0s 106us/sample - loss: 186765266894525.2812 - mean_squared_error: 186765260357632.0000 - val_loss: 217275467446499.5625 - val_mean_squared_error: 217275483291648.0000
    Epoch 978/1000
    1726/1726 [==============================] - 0s 126us/sample - loss: 187123404656815.5938 - mean_squared_error: 187123420364800.0000 - val_loss: 206091625819401.4688 - val_mean_squared_error: 206091640111104.0000
    Epoch 979/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 187552515607650.4688 - mean_squared_error: 187552497664000.0000 - val_loss: 232910435589233.7812 - val_mean_squared_error: 232910422540288.0000
    Epoch 980/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 191326758380440.7812 - mean_squared_error: 191326767284224.0000 - val_loss: 220697101423957.3438 - val_mean_squared_error: 220697095831552.0000
    Epoch 981/1000
    1726/1726 [==============================] - 0s 108us/sample - loss: 186508532229586.3438 - mean_squared_error: 186508518621184.0000 - val_loss: 206258757338301.6250 - val_mean_squared_error: 206258741182464.0000
    Epoch 982/1000
    1726/1726 [==============================] - 0s 132us/sample - loss: 188750774862051.8125 - mean_squared_error: 188750810316800.0000 - val_loss: 205619265788434.9688 - val_mean_squared_error: 205619277594624.0000
    Epoch 983/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 187710482595441.3125 - mean_squared_error: 187710522261504.0000 - val_loss: 207233288648931.5625 - val_mean_squared_error: 207233279328256.0000
    Epoch 984/1000
    1726/1726 [==============================] - 0s 131us/sample - loss: 188195758112730.0312 - mean_squared_error: 188195719348224.0000 - val_loss: 208910439202209.1875 - val_mean_squared_error: 208910430502912.0000
    Epoch 985/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 186886209825330.4375 - mean_squared_error: 186886207307776.0000 - val_loss: 212525472226417.7812 - val_mean_squared_error: 212525467566080.0000
    Epoch 986/1000
    1726/1726 [==============================] - 0s 127us/sample - loss: 186276713042646.1562 - mean_squared_error: 186276724604928.0000 - val_loss: 207784949142869.3438 - val_mean_squared_error: 207784947744768.0000
    Epoch 987/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 185533990020714.2188 - mean_squared_error: 185533980475392.0000 - val_loss: 207420942275925.3438 - val_mean_squared_error: 207420932489216.0000
    Epoch 988/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 186043151149124.8125 - mean_squared_error: 186043168980992.0000 - val_loss: 226129861604996.7188 - val_mean_squared_error: 226129843585024.0000
    Epoch 989/1000
    1726/1726 [==============================] - 0s 112us/sample - loss: 187099730090143.0312 - mean_squared_error: 187099730935808.0000 - val_loss: 216793635434344.2812 - val_mean_squared_error: 216793624870912.0000
    Epoch 990/1000
    1726/1726 [==============================] - 0s 110us/sample - loss: 184886993054638.1562 - mean_squared_error: 184887051026432.0000 - val_loss: 205267855814504.2812 - val_mean_squared_error: 205267862028288.0000
    Epoch 991/1000
    1726/1726 [==============================] - 0s 107us/sample - loss: 187145803844023.0312 - mean_squared_error: 187145801170944.0000 - val_loss: 208829276002228.1562 - val_mean_squared_error: 208829262331904.0000
    Epoch 992/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 188936204648220.1875 - mean_squared_error: 188936131444736.0000 - val_loss: 212653801462290.9688 - val_mean_squared_error: 212653813268480.0000
    Epoch 993/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 187053998886480.0938 - mean_squared_error: 187053996244992.0000 - val_loss: 208707479627851.8438 - val_mean_squared_error: 208707493298176.0000
    Epoch 994/1000
    1726/1726 [==============================] - 0s 102us/sample - loss: 186439795353535.9375 - mean_squared_error: 186439799144448.0000 - val_loss: 217936898157833.4688 - val_mean_squared_error: 217936908255232.0000
    Epoch 995/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 186824215222487.9375 - mean_squared_error: 186824215494656.0000 - val_loss: 217777369358639.4062 - val_mean_squared_error: 217777373708288.0000
    Epoch 996/1000
    1726/1726 [==============================] - 0s 111us/sample - loss: 187765393536433.0938 - mean_squared_error: 187765400535040.0000 - val_loss: 201008477343288.8750 - val_mean_squared_error: 201008462430208.0000
    Epoch 997/1000
    1726/1726 [==============================] - 0s 99us/sample - loss: 185885616527006.4062 - mean_squared_error: 185885580591104.0000 - val_loss: 196709824341485.0312 - val_mean_squared_error: 196709837701120.0000
    Epoch 998/1000
    1726/1726 [==============================] - 0s 103us/sample - loss: 186416617957691.6250 - mean_squared_error: 186416629809152.0000 - val_loss: 206653734013914.0625 - val_mean_squared_error: 206653727178752.0000
    Epoch 999/1000
    1726/1726 [==============================] - 0s 105us/sample - loss: 184594426936641.5625 - mean_squared_error: 184594406047744.0000 - val_loss: 196017820882261.3438 - val_mean_squared_error: 196017811095552.0000
    Epoch 1000/1000
    1726/1726 [==============================] - 0s 113us/sample - loss: 184752526623365.5000 - mean_squared_error: 184752548085760.0000 - val_loss: 225808896685700.7188 - val_mean_squared_error: 225808895442944.0000
    

#### Deep Learning Model 2


```python
hist_2 = model_2.fit(X_train, Y_train,
          batch_size=32, epochs=275,
          validation_data=(X_val, Y_val))
```

### Evaluating on Test Set

#### Linear Regression


```python
from sklearn.metrics import mean_squared_error

price_predictions = lin_reg_model.predict(X_test)
lin_reg_mse = mean_squared_error(Y_test, price_predictions)
lin_reg_rmse = np.sqrt(lin_reg_mse)

reduced_num, ten_count = countTens(lin_reg_rmse)
print("RMSE on test set using Linear Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on test set using Linear Regression is: 1.4402431907184545 * 10^17
    

#### Decision Tree Regression


```python
price_predictions = forest_reg_model.predict(X_test)
forest_reg_mse = mean_squared_error(price_predictions, Y_test)
forest_reg_rmse = np.sqrt(forest_reg_mse)
reduced_num, ten_count = countTens(forest_reg_rmse)
print("RMSE on test set using Decision Tree Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on test set using Decision Tree Regression is: 1.3173120002895615 * 10^7
    

#### Random Forest Regression


```python
price_predictions = forest_reg_model.predict(X_test)
forest_reg_mse = mean_squared_error(price_predictions, Y_test)
forest_reg_rmse = np.sqrt(forest_reg_mse)
reduced_num, ten_count = countTens(forest_reg_rmse)
print("RMSE on test set using Random Forest Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on test set using Random Forest Regression is: 1.4274739383490493 * 10^7
    


```python
# evaluating the model tuned by grid-search
price_predictions = grid_search.predict(X_test)
forest_reg_mse = mean_squared_error(price_predictions, Y_test)
forest_reg_rmse = np.sqrt(forest_reg_mse)
reduced_num, ten_count = countTens(forest_reg_rmse)
print("RMSE on test set using grid-search optimized Random Forest Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on test set using grid-search optimized Random Forest Regression is: 1.42468066402298 * 10^7
    

#### Deep Learning Model 1


```python
model_1_mse = model_1.evaluate(X_test, Y_test)[0]
model_1_rmse = np.sqrt(model_1_mse)
reduced_num, ten_count = countTens(model_1_rmse)
print("RMSE on test set using Model 1 is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    216/216 [==============================] - 0s 74us/sample - loss: 172750485673301.3438 - mean_squared_error: 172750480080896.0000
    RMSE on test set using Model 1 is: 1.3143457903965048 * 10^7
    


```python
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
```


    
![png](./images/readme/output_75_0.png)
    


#### Deep Learning Model 2


```python
model_2_mse = model_2.evaluate(X_test, Y_test)[0]
model_2_rmse = np.sqrt(model_2_mse)
reduced_num, ten_count = countTens(model_2_rmse)
print("RMSE on test set using Model 2 is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    216/216 [==============================] - 0s 83us/sample - loss: 141226298135893.3438 - mean_squared_error: 141226292543488.0000
    RMSE on test set using Model 2 is: 1.1883867137253485 * 10^7
    


```python
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss 2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# below figure doesn't show next 25 epochs. RMSE above was displayed after 25 more epoches (making total epochs = 300)
```


    
![png](./images/readme/output_78_0.png)
    


### Making Predictions on Validation Set & Custom Input

#### Linear Regression


```python
price_predictions = lin_reg_model.predict(X_val)
lin_reg_mse = mean_squared_error(Y_val, price_predictions)
lin_reg_rmse = np.sqrt(lin_reg_mse)

reduced_num, ten_count = countTens(lin_reg_rmse)
print("RMSE on validation set using Linear Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on validation set using Linear Regression is: 1.0461315529577755 * 10^17
    

#### Decision Tree Regression


```python
price_predictions = tree_reg_model.predict(X_val)
tree_reg_mse = mean_squared_error(price_predictions, Y_val)
tree_reg_rmse = np.sqrt(tree_reg_mse)
reduced_num, ten_count = countTens(tree_rmse)
print("RMSE on validation set using Decision Tree Regression is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on validation set using Decision Tree Regression is: 2.7435425654245926 * 10^7
    

#### Deep Learning Model 1

##### Predictions on Custom Input


```python
#test_data = np.array([1,7.624,33.669341,72.84489,1,1,0,0,0,1,1,1,1,1])
test_data = np.array([1 ,43.560, 33.669341, 72.84489, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
print(model_1.predict(test_data.reshape(1,14), batch_size=1))
prediction = model_1.predict(test_data.reshape(1,14), batch_size=1)
print(Y.at[1,"price"])
print(prediction - Y.at[1,"price"])
```

##### Predictions on Validation Set


```python
pred_val = model_1.predict(X_val)
model1_val_rmse = np.sqrt(mean_squared_error(Y_val, pred_val))
reduced_num, ten_count = countTens(model1_val_rmse)
print("RMSE on validation set using model 1 is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on validation set using model 1 is: 1.5026939173212719 * 10^7
    

#### Deep Learning Model 2

##### Predictions on Custom Input


```python
#test_data = np.array([1,7.624,33.669341,72.84489,1,1,0,0,0,1,1,1,1,1])
test_data = np.array([1 ,43.560, 33.669341, 72.84489, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
print(model_2.predict(test_data.reshape(1,14), batch_size=1))
prediction = model_2.predict(test_data.reshape(1,14), batch_size=1)
print(Y.at[1,"price"])
print(prediction - Y.at[1,"price"])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-28-d6c66cdca69f> in <module>
          1 #test_data = np.array([1,7.624,33.669341,72.84489,1,1,0,0,0,1,1,1,1,1])
          2 test_data = np.array([1 ,43.560,        33.669341,      72.84489,       1,      1,      0,      0,      0,      1,      1,      1,      1,      1])
    ----> 3 print(model_2.predict(test_data.reshape(1,14), batch_size=1))
          4 prediction = model_2.predict(test_data.reshape(1,14), batch_size=1)
          5 print(Y.at[1,"price"])
    

    ~\.conda\envs\price-prediction\lib\site-packages\keras\engine\training.py in predict(self, x, batch_size, verbose, steps)
       1147                              'argument.')
       1148         # Validate user data.
    -> 1149         x, _, _ = self._standardize_user_data(x)
       1150         if self.stateful:
       1151             if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
    

    ~\.conda\envs\price-prediction\lib\site-packages\keras\engine\training.py in _standardize_user_data(self, x, y, sample_weight, class_weight, check_array_lengths, batch_size)
        749             feed_input_shapes,
        750             check_batch_axis=False,  # Don't enforce the batch size.
    --> 751             exception_prefix='input')
        752 
        753         if y is not None:
    

    ~\.conda\envs\price-prediction\lib\site-packages\keras\engine\training_utils.py in standardize_input_data(data, names, shapes, check_batch_axis, exception_prefix)
        136                             ': expected ' + names[i] + ' to have shape ' +
        137                             str(shape) + ' but got array with shape ' +
    --> 138                             str(data_shape))
        139     return data
        140 
    

    ValueError: Error when checking input: expected dense_5_input to have shape (475,) but got array with shape (14,)


##### Predictions on Training Set


```python
pred_val = model_2.predict(X_val)
model2_val_rmse = np.sqrt(mean_squared_error(Y_val, pred_val))
reduced_num, ten_count = countTens(model2_val_rmse)
print("RMSE on validation set using model 2 is: " + str(reduced_num) + " * 10^" + str(ten_count))
```

    RMSE on validation set using model 2 is: 1.424891094928326 * 10^7
    

### Summary of Results

**Note:** These are some baseline results. DL models were not fine-tuned by trying different hyper-parameters. Secondly, different configurations of DL models were not tried at all except for the given models. This work focused on dataset visualization & analysis.
![Comparison of Results achieved using various ML Models](images/comparison_of_model_results.png)

- Random Forests & Decision Trees performed comparable to Deep Learning models. This is because of the nature of the dataset. 
- Deeper DL models can be used for better results along with hyper-parameter tuning. But, significant improvements are not expected (as validated during some experiments) without improving the dataset.

- **Future work** 
    - Work can be done on modifying the dataset according to the analysis given in this notebook. Suggested modifications can help with achieving better results using deep learning models.
    - Deeper DL models can be tried, because moving from model 1 to model 2 showed some progress. Secondly, different layer configurations can be tried along with hyper-parameter tuning.

## Conclusion

- Dataset needs serious improvements. All the future work on this project depends on the dataset. Investing in the dataset can bring significant improvements in the results.
- Most of suggested dataset improvements are not resource-extensive. Latitude & longitude of plots can be used to extract required attributes from the Google Maps. Cost-effective solution can be achieved through Node.js frameworks & Google Maps APIs.
- Without dataset extension & improvement, hyper-parameter tuning & usage of deeper deep learning models may lead to better results.


```python

```
