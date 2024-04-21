#!/usr/bin/env python
# coding: utf-8

# # UKRI Interview Project
# 
# 
# Product Recommendation based on Customer Demographics

# In[1]:


# Importing the primary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Importing the data

cust_data = pd.read_csv("Dataset/hh_demographic.csv")               # Customer/Household demographic
trans_data = pd.read_csv("Dataset/transaction_data.csv")            # Sales Transaction
product_data = pd.read_csv("Dataset/product.csv")                   # Product details
coup_data = pd.read_csv("Dataset/coupon.csv")                       # Coupon details
coup_red_data = pd.read_csv("Dataset/coupon_redempt.csv")           # Redeemde Coupons
camp_data = pd.read_csv("Dataset/campaign_table.csv")               # Marketing Campaigns
camp_desc_data = pd.read_csv("Dataset/campaign_desc.csv")           # Campaign details


# In[3]:


cust_data.head(5)


# In[4]:


trans_data.head(5)


# In[5]:


product_data.head(5)


# In[6]:


coup_data.head(5)


# In[7]:


coup_red_data.head(5)


# In[8]:


camp_data.head(5)


# In[9]:


camp_desc_data.head(5)


# In[ ]:





# -----

# # Exploratory Data Analysis

# ### Product Analysis

# In[10]:


product_data.info()


# In[11]:


# Total Department

total_departments = product_data['DEPARTMENT'].nunique()
print ("Total Department: ",total_departments)


# In[12]:


# Total Categories

total_categories = product_data['COMMODITY_DESC'].nunique()
print ("Total Categories: ",total_categories)


# In[13]:


# Total Sub-Categories

total_sub_cat = product_data['SUB_COMMODITY_DESC'].nunique()
print ("Total Sub-Categories: ",total_sub_cat)


# In[14]:


# Total Products

total_products = product_data['PRODUCT_ID'].nunique()
print ("Total Products: ",total_products)


# In[15]:


# top 10 Most sold products

# Merge product names with top_products based on Product_ID
top_products_with_names = pd.merge(trans_data[['PRODUCT_ID','QUANTITY']], product_data[['BRAND','DEPARTMENT','COMMODITY_DESC','SUB_COMMODITY_DESC','PRODUCT_ID']], on='PRODUCT_ID', how='left')

# Getting top 10 commmodities preferred by quantity ordered
product_counts = top_products_with_names.groupby('SUB_COMMODITY_DESC')['QUANTITY'].sum()
product_counts = product_counts.sort_values(ascending=False)

#Showing top 10 ordered by product counts, while omitting the first one.
top_products = product_counts.iloc[1:].nlargest(10)

# Bar chart with product names
plt.figure(figsize=(15,10))
plt.bar(x=top_products.index, height=top_products.values)
plt.xticks(rotation=90)
plt.xlabel('Products')
plt.ylabel('Order Quantity')
plt.title('Top 10 most sold products', fontsize=14)
plt.show()


# In[16]:


# Getting top 10 commmodities preferred by quantity ordered
department_counts = top_products_with_names.groupby('DEPARTMENT')['QUANTITY'].sum()
department_counts = department_counts.sort_values(ascending=False)

#Showing top 10 ordered by product counts, while omitting the first one.
top_department = department_counts.iloc[1:].nlargest(10)

# Bar chart with product names
plt.figure(figsize=(15,10))
plt.bar(x=top_department.index, height=top_department.values)
plt.xticks(rotation=90)
plt.xlabel('Department')
plt.ylabel('Order Quantity')
plt.title('Top Selling Departments', fontsize=14)
plt.show()


# In[17]:


# Checking for duplicate rows (products)

product_data[product_data['SUB_COMMODITY_DESC'].duplicated(keep=False)]


# In[ ]:





# In[18]:


# Sales Analysis


# In[19]:


total_revenue = round(trans_data['SALES_VALUE'].sum(),2)
print ("Total Revenue Generated: ",total_revenue)


# In[20]:


total_quantity = trans_data['QUANTITY'].sum()
print ("Total Quantities Sold: ",total_quantity)


# In[21]:


# Discounts used from the coupons

total_discounts = round(trans_data['COUPON_DISC'].sum(),2)
print ("Total Discounts used at Checkout: ",total_discounts)


# In[22]:


total_profit = total_revenue + total_discounts
print ("Overall Revenue: ",total_profit)


# #### Observation
# 
# Some Products have same sub_commodity_desc, commodity_desc, brand, and department, but were manufactured by different companies, hence different manufactures. 

# ----

# ### Campaign and Coupons
# 
# 
# Campaign dataset contains identifying information for the marketing campaigns each household participated in.
# The Campaign description explains the type of campaign and its duration.
# 
# Coupon table holds the data on the coupons asssociated with each campaign, and on what products they were used.
# Coupon Redemption table tells what and how many coupons were used by customers, and when they were used.

# In[23]:


camp_desc_data.head(5)


# In[24]:


camp_desc_data.info()


# In[25]:


camp_data.head(5)


# In[26]:


camp_data.info()


# In[27]:


coup_data.head(5)


# In[28]:


coup_data.info()


# In[29]:


coup_red_data.head(5)


# In[30]:


coup_red_data.info()


# In[ ]:





# In[31]:


# Total Campaigns

total_campaigns = camp_desc_data['CAMPAIGN'].nunique()
print ("Total Campaigns: ",total_campaigns)


# In[32]:


# Total Campaigns

total_coupons = coup_data['COUPON_UPC'].nunique()
print ("Total Coupons: ",total_coupons)


# In[33]:


redeemed_coupons = coup_red_data['COUPON_UPC'].nunique()
print ("Redeemed Coupons: ",redeemed_coupons)


# In[34]:


used_coupons = coup_red_data['COUPON_UPC'].count()
print ("Used Coupons: ",used_coupons)


# In[35]:


# Analysis of campaigns and their respective duration

camp_desc_data["DUR"] = camp_desc_data["END_DAY"] - camp_desc_data["START_DAY"]

fig = plt.figure(figsize=(14,6))
sns.barplot(x="CAMPAIGN",y="DUR",data=camp_desc_data, orient="v", 
            order=camp_desc_data.sort_values(by="CAMPAIGN").CAMPAIGN.values, color='steelblue')
plt.title('Duration of each campaign', fontsize=17)
plt.xlabel('Campaign Number', fontsize=14)
plt.ylabel('Duration', fontsize=14)
plt.show()


# ##### Observation:
# Campaign No:15 lasts the longest with a staggering 160 days figure, where other campaigns are fairly close to each other ranging from 30 to 70 days.
# Average campaign duration is 37 days (median)

# In[36]:


# Most Frequent Campaign

freq_campaigns = pd.DataFrame(list(zip(camp_data['CAMPAIGN'].value_counts().index, camp_data['CAMPAIGN'].value_counts())),columns=["Campaign","Frequency"])
fig = plt.figure(figsize=(15,8))
sns.barplot(x="Campaign",y="Frequency",data = freq_campaigns,orient="v", color='steelblue')
plt.title('Frequency of Each Campaign', fontsize=17)
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Campaign No.', fontsize=14)
plt.show()


# ##### Observation:
# 
# The most frequent campaign is Campaign 18, reaching more than 1000 customers, while the least campaigns are campaign 2 and 27.

# In[ ]:





# In[37]:


# Checking for unique customers from the campaign table

len(camp_data["household_key"].unique())


# In[38]:


njn = camp_data["household_key"].value_counts()
njn


# In[39]:


njn.value_counts()


# In[ ]:





# ## Demographics
# 
# This explains the customer demographic table

# In[40]:


cust_data.head(15)


# In[41]:


def bar_categorical(data, figsize=(20,30)):
    
    #function to plot the histogram of categorical variables in pie graph
    features = data.columns
    
    #plot pie charts of categorical variables
    fig_cat = plt.figure(figsize=figsize)
    count = 1
    
    #calculate dynamic numbers of subplot rows and columns
    cols = 2 #int(np.ceil(np.sqrt(len(features))))
    rows = int(np.ceil(len(features)/cols))
    
    for i in features:
        ax = fig_cat.add_subplot(rows,cols,count)
        data[i].value_counts().plot(kind="bar",ax=ax)
        plt.ylabel("")
        plt.xticks(rotation=0)
        plt.title(i,fontweight="bold", fontsize=8)
        count += 1


# In[42]:


bar_categorical(cust_data.drop("household_key",axis=1))


# ----

# # Feature Engineering

# ## 1. List of products in each basket.

# In[43]:


trans_data.head(5)


# In[44]:


# Counting the total number of baskets in the transaction table

total_baskets = len(trans_data["BASKET_ID"].unique())
total_baskets


# In[45]:


# Counting the total number of baskets in the transaction table

total_baskets = trans_data["WEEK_NO"].unique()
total_baskets


# In[46]:


# what products are in each basket?

items_per_basket = trans_data.groupby("BASKET_ID")["PRODUCT_ID"].apply(list)
items_per_basket


# In[47]:


# How many products are each in basket?

item_count_per_basket = items_per_basket.apply(len)
item_count_per_basket


# In[48]:


# Total sales in each basket

basket_sales = trans_data.groupby('BASKET_ID')['SALES_VALUE'].sum()
basket_sales


# In[49]:


# Calculate total sales for the entire dataset
total_sales = round(basket_sales.sum(),2)
total_sales


# In[50]:


# mean basket sales

print(round(basket_sales.mean(), 2))


# In[51]:


# Average Basket Size

basket_size = int(item_count_per_basket.mean())
basket_size


# In[ ]:





# ## 2. Basket Analysis by Customer

# In[52]:


cust_data.head(5)


# In[53]:


# Checking for unique customers from the transaction table

total_households = len(trans_data["household_key"].unique())
total_households


# In[54]:


fig,ax = plt.subplots(figsize=(15,5))

cust_prof = cust_data.groupby("AGE_DESC")["household_key"].count().reset_index()
sns.barplot(y='household_key', data=cust_prof, x='AGE_DESC')
ax.set_title("Customer Count by Age Description");


# In[55]:


fig,ax = plt.subplots(figsize=(15,5))

cust_prof = cust_data.groupby("MARITAL_STATUS_CODE")["household_key"].count().reset_index()
sns.barplot(y='household_key', data=cust_prof, x='MARITAL_STATUS_CODE')
ax.set_title("Customer Count by Marital Status");


# In[56]:


# customer attributed to a basket.

cust_per_basket = trans_data.groupby("household_key")["BASKET_ID"].unique().apply(list)
cust_per_basket


# In[57]:


# All customer baskets and products in each basket

customer_basket_dict = {}

for customer_id, baskets in cust_per_basket.items():
    basket_list = []
    for basket_id in baskets:
        products_in_basket = trans_data.loc[trans_data['BASKET_ID'] == basket_id, 'PRODUCT_ID'].tolist()
        basket_list.append(products_in_basket)
    customer_basket_dict[customer_id] = basket_list

# Print the dictionary to see the representation
for customer_id, baskets in customer_basket_dict.items():
    print(f"Customer {customer_id}: {baskets}")


# ## 3. Favourite Products

# In[58]:


# Top 20 products shopped frequently by each customer

from collections import Counter

# Dictionary to store the top 20 most bought products for each customer
top_products_per_customer = {}

# Iterate over each customer's shopping history
for customer_id, baskets in customer_basket_dict.items():
    # Flatten the list of products across all baskets
    all_products = [product for basket in baskets for product in basket]
    
    # Count the occurrences of each product
    product_counts = Counter(all_products)
    
    # Extract the top 20 most frequent products
    top_product_ids = [product_id for product_id, _ in product_counts.most_common(20)]
    
    # Store the top products for the current customer
    top_products_per_customer[customer_id] = top_product_ids

# Print the top 10 most bought products for each customer
for customer_id, top_products in top_products_per_customer.items():
    print(f"{customer_id}: {top_products}")


# In[59]:


# Creating a new column in the cust_data for each customer's fav products

# Convert the dictionary to a DataFrame
cust_fave_products = pd.DataFrame(top_products_per_customer.items(), columns=['household_key', 'fave_products'])

# Each household favourite products with their household keys
cust_fave_products


# ----

# # Customer Segmentation

# In[60]:


# Converting the categorical to numerical

# clonning the Cust_data
clust = cust_data.copy()


from sklearn.preprocessing import LabelEncoder


# Initialize a dictionary to store encoders
label_encoding_info = {}

# Iterate through each column in the DataFrame
for col in clust.columns:

    # Skip the 'fave_products' column:
    if col == 'household_key':
        continue

    # Check if the column data type is object (categorical)
    if clust[col].dtype == 'object':
        # Check if the column contains lists
        if clust[col].apply(type).eq(list).any():

            # Flatten lists and convert to strings
            clust[col] = clust[col].apply(lambda x: ','.join(map(str, x)))

        # Initialize LabelEncoder for the column
        label_encoder = LabelEncoder()
        
        # Fit LabelEncoder on the column and transform the data
        clust[col] = label_encoder.fit_transform(clust[col])

        # Store the unique labels and their corresponding encoded values in the dictionary
        unique_labels = list(label_encoder.classes_)
        unique_encoded_values = list(set(clust[col]))


        label_encoding_info[col] = {}
        # Add the unique labels and their encoded values to the dictionary under the column name
        for label, encoded_value in zip(unique_labels, unique_encoded_values):
            label_encoding_info[col][label] = encoded_value
        
        # Store the fitted encoder in the dictionary
        # label_encoding_info[col] = {'encoded_values': clust[col].tolist(), 'labels': list(label_encoder.classes_)}

        
print(label_encoding_info) 


# In[61]:


# Performing the Elbow method to obtain the optimal number of clusters

X = clust.iloc[:,[0,6]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The  Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[62]:


# From the elbow method, the best number of clusters to use is 6.

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to the customer demographics DataFrame
clust['Cluster'] = y_kmeans
print(y_kmeans)


# In[63]:


plt.scatter(X[y_kmeans == 0 ,0],X[y_kmeans == 0 ,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1 ,0],X[y_kmeans == 1 ,1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2 ,0],X[y_kmeans == 2 ,1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3 ,0],X[y_kmeans == 3 ,1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4 ,0],X[y_kmeans == 4 ,1], s=100, c='magenta', label='Cluster 5')
plt.scatter(X[y_kmeans == 5 ,0],X[y_kmeans == 5 ,1], s=100, c='purple', label='Cluster 6')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[64]:


cust_fave_products


# In[65]:


# Merging the cluster labels to the favourite products table.

cust_fave_products = pd.merge(cust_fave_products, clust[['household_key','Cluster']], on='household_key', how='left')
cust_fave_products = cust_fave_products.dropna()
cust_fave_products['Cluster'] = cust_fave_products['Cluster'].astype(int)
cust_fave_products


# In[66]:


# Dropping the household_key column on the clust table as it is no longer needed.

clust = clust.drop(columns='household_key', axis=1)
clust


# In[67]:


corr_matrix = clust.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# -----

# # Building the Recommendation Model

# In[68]:


'''
    After clustering the customers and also assigning the cluster labels,
    The dataset is trained by asssigning the demogrpahics to the X train and test, and the cluster label to the y train and test
'''



# Splitting the data

X = clust.drop(['Cluster'], axis=1)
y = clust['Cluster']

X


# In[69]:


y


# In[ ]:





# In[70]:


# Classifications to be considered

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier()


# Kernel SVM
from sklearn.svm import SVC
ksvm = SVC()


# XGboost
from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[71]:


#classifiers to be compared 
classifiers = [knn, dct, ksvm, xgb]


# In[72]:


# Using Hyperparameter tuning to eradicate the possibility of overfitting in the training modelby using parameters that will give the best model performance
from sklearn.model_selection import GridSearchCV


#Checking the best classifier using cross validation
from sklearn.model_selection import cross_val_score

# Define the hyperparameters grid for each classifier
param_grid_knn = {'n_neighbors': [3, 5, 7]}
param_grid_dct = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20]}
param_grid_ksvm = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
param_grid_xgb = {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01, 0.001]}

# Define the hyperparameter grids for all classifiers
param_grids = [param_grid_knn, param_grid_dct, param_grid_ksvm, param_grid_xgb]

# Initialize a list to store the best classifiers
best_classifiers = []

print ("Hyperparamter Tuning Result:\n\n")

# Perform hyperparameter tuning for each classifier
for classifier, param_grid in zip(classifiers, param_grids):
    grid_search = GridSearchCV(classifier, param_grid, cv=4, scoring='accuracy')
    grid_search.fit(X, y)
    best_classifier = grid_search.best_estimator_
    best_classifiers.append(best_classifier)
    print(f"Best parameters for {type(classifier).__name__}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}\n")

# Select the best classifier based on cross-validation accuracy
best_classifier = max(best_classifiers, key=lambda x: cross_val_score(x, X, y, cv=4).mean())
print(f"The best classifier is: {type(best_classifier).__name__}")


# In[73]:


# Splitting the Model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
display(X_train.shape, X_test.shape)


# In[74]:


# Feature Scaling the X data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[75]:


#import necessary libraries to train the model with the classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[76]:


# Training the K-NN model on the Training set

knn = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)


# Predicting the test results
knn_pred = knn.predict(X_test)


# In[77]:


# Decision Tree Classification

dct = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dct.fit(X_train, y_train)

# Predicting the test results
dct_pred = dct.predict(X_test)


# In[88]:


# Kernel SVM

ksvm = SVC(kernel = 'linear', random_state = 0)
ksvm.fit(X_train, y_train)


# Predicting the test results
ksvm_pred = ksvm.predict(X_test)


# In[79]:


# XGBoost Classifier

# Train the model on the training data
xgb.fit(X_train, y_train)

# Make predictions on the test data
xgb_pred = xgb.predict(X_test)


# ----

# # Model Evaluation

# In[80]:


#check the accuracy of the prediction using different metrics
knn_report = classification_report(y_test, knn_pred)
print('Classification Report KNN:')
print(knn_report)


# In[81]:


#check the accuracy of the prediction using different metrics
dct_report = classification_report(y_test, dct_pred)
print('Classification Report Decision Tree:')
print(dct_report)


# In[82]:


#check the accuracy of the prediction using different metrics
ksvm_report = classification_report(y_test, ksvm_pred)
print('Classification Report Kernel SVM:')
print(ksvm_report)


# In[83]:


#check the accuracy of the prediction using different metrics
xgb_report = classification_report(y_test, xgb_pred)

print('Classification Report for XGBoost')
print(xgb_report)


# In[84]:


cm = confusion_matrix(y_test, xgb_pred)
print("Confusion Matrix:\n\n")
print(cm)
accuracy_score(y_test, xgb_pred)


# In[85]:


# Testing the models with a known demographic

print("KNN Predicted: ", knn.predict(sc.transform([[3,2,8,4,4,0,3]])))
print("Decison Tree Predicted: ", dct.predict(sc.transform([[3,2,8,4,4,0,3]])))
print("Kernel SVM Predicted: ", ksvm.predict(sc.transform([[3,2,8,4,4,0,3]])))
print("XGBoost Classifier Predicted: ", xgb.predict(sc.transform([[3,2,8,4,4,0,3]])))


# In[86]:


# Testing the models with an unknown demographic

print("KNN Predicted: ", knn.predict(sc.transform([[4,0,7,2,0,0,0]])))
print("Decison Tree Predicted: ", dct.predict(sc.transform([[4,0,7,2,0,0,0]])))
print("Kernel SVM Predicted: ", ksvm.predict(sc.transform([[4,0,7,2,0,0,0]])))
print("XGBoost Classifier Predicted: ", xgb.predict(sc.transform([[4,0,7,2,0,0,0]])))


# In[87]:


#Checking the best classifier using cross validation
from sklearn.model_selection import cross_val_score

#classifiers to be compared 
classifier_scores = {}

for classifier in classifiers:
    accuracy_scores = cross_val_score(classifier, X_train, y_train, cv=4)  # 4-fold cross-validation
    mean_accuracy = accuracy_scores.mean()
    classifier_scores[type(classifier).__name__] = mean_accuracy
    
    print(f"Classifier: {type(classifier).__name__}, Mean Accuracy: {mean_accuracy}")
    best_classifier = max(classifier_scores, key=classifier_scores.get)

print(f"The best classifier is: {best_classifier}")


# ----

# # Deployment and Testing

# In[89]:


# The best and recommended classifier to use is the XGBoost Classifier, hence the 'xgb' model is saved.

# Import the joblib library
import joblib

# Save the trained model to a file
joblib.dump(xgb, 'xgb_model.pkl')

# Later, when you want to make predictions:
# Load the saved model from the file
xgb_loaded = joblib.load('xgb_model.pkl')


# In[90]:


import random


# Step 1: Search the fave_products data for households under the predicted clusters
def search_by_cluster(pred_cluster):
    similar_customers = cust_fave_products[
        (cust_fave_products['Cluster'] == pred_cluster)
    ]

    return similar_customers



# Step 2: Select Frequent Products
def select_frequent_products(similar_customers):
    frequent_products = set()
    
    for cluster in similar_customers['Cluster']:
        products_series = cust_fave_products.loc[cust_fave_products['Cluster'] == cluster, 'fave_products'].iloc[0]

        # Sort the products based on their frequency and select the top 10
        top_products = sorted(products_series, key=products_series.count, reverse=True)[:10]
        frequent_products.update(top_products)

    # Randomly select 10 products from the combined frequent products list
    recommended_products = random.sample(frequent_products, min(10, len(frequent_products)))
    
    return recommended_products



# Step 3: Fetch Product Names
def fetch_product_names(frequent_product_ids, product_table):
    product_names = product_table.loc[product_table['PRODUCT_ID'].isin(frequent_product_ids), 'SUB_COMMODITY_DESC'].tolist()
    return product_names





# Step 4: Display Recommended Products
def display_recommended_products(recommended_product_names):
    print("Recommended Products:")
    for product_name in recommended_product_names:
        print("- " + product_name)


# In[99]:


# Example Usage:
customer_demographics = {
    'AGE_DESC': '25-34', 
    'MARITAL_STATUS_CODE': 'A', 
    'INCOME_DESC': 'Under 15K', 
    'HOMEOWNER_DESC': 'Renter', 
    'HH_COMP_DESC': '2 Adults No Kids', 
    'HOUSEHOLD_SIZE_DESC': '1', 
    'KID_CATEGORY_DESC': 'None/Unknown'
}


encoded_demographics = []

# Encode the customer_demographics using the label encoding mappings
for key, value in customer_demographics.items():
    encoded_value = label_encoding_info[key][value]
    encoded_demographics.append(encoded_value)


# Recommending a product
pred_c = xgb.predict(sc.transform([encoded_demographics]))

for i in range(0, len(pred_c)):
    pred_cluster = pred_c[i]


# pred_cluster
similar_customers = search_by_cluster(pred_cluster)
frequent_products = select_frequent_products(similar_customers)
recommended_product_names = fetch_product_names(frequent_products, product_data)
display_recommended_products(recommended_product_names)


# In[ ]:




