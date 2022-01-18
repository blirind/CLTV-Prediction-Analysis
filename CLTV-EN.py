
#####################################################################
#                      CUSTOMER LIFETIME VALUE
#####################################################################
#
# BUSINESS PROBLEM: An e-commerce company wants to predict customers future purchases
# and determine business investment plan according to predicted data.

# DATASET STORY: There is Online Retail II, 2010-2011 sheet file as dataset.
# Products sold are mostly souvenirs and most of the customers are corporates.


# Importing necessary libraries


import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler


# Reading dataset:

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()


# Deleting POSTAGE payments from dataset:

df = df[~(df["Description"] == "POSTAGE")]


# Quick check to the dataset:

def check_df(dataframe):
    print("##################### Shape #####################")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("####################### NA ######################")
    print(dataframe.isnull().sum())
    print("################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Head ######################")
    print(dataframe.head())
check_df(df)


# Setting outlier thresholds:

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

num_cols = ["Quantity", "Price"]


# Outlier analysis:

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print("#####################################################")
        print(str(col_name) + " variable have too much outliers: " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(15))
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    elif (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] < 10) & \
            (dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 0):
        print("#####################################################")
        print(str(col_name) + " variable have less than 10 outlier values: " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print("#####################################################")
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
        print("#####################################################")
        print("Lower threshold: " + str(low) + "   Lowest outlier: " + str(dataframe[col_name].min()) +
              "   Upper threshold: " + str(up) + "   Highest outlier: " + str(dataframe[col_name].max()))
        print("#####################################################")
    else:
        print("#####################################################")
        print(str(col_name) + " variable does not have outlier values")
        print("#####################################################")

    if index:
        print(str(col_name) + " variable's outlier indexes")
        print("#####################################################")
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(df, col)


# Replacing outliers:

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


# Removing NaN values; Removing canceled purchases (Invoices containing "C"):

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]


# Creating TotalPrice variable and setting today's date as follows:

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)





def cltv_create(dataframe, country, expected_purchase=False,
                month_expected=1, expected_profit=False, cltv_prediction=False,
                cltv_month=1):

    if dataframe[dataframe["Country"] == country]["Customer ID"].nunique() < 2:
        print("Datas from: " + str(country) + " aren't enough for CLTV analysis.")

    else:
        one_country = dataframe.loc[dataframe["Country"] == country]
        country_code = one_country.groupby("Customer ID").agg(
            {'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                            lambda date: (today_date - date.min()).days],
            'Invoice': lambda num: num.nunique(),
            'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

        country_code.columns = country_code.columns.droplevel(0)

        country_code.columns = ["recency", "T", "frequency", "monetary"]

        country_code["monetary"] = country_code["monetary"] / country_code["frequency"]
        country_code = country_code[(country_code['frequency'] > 1)]

        country_code["recency"] = country_code["recency"] / 7
        country_code["T"] = country_code["T"] / 7



        if expected_purchase:
            bgf = BetaGeoFitter(penalizer_coef=0.001)
            bgf.fit(country_code['frequency'], country_code['recency'], country_code['T'])

            country_code["expected_purchase"] = bgf.predict(month_expected,
                                                           country_code['frequency'],
                                                           country_code['recency'],
                                                           country_code['T'])

            print("###########################################################")
            print("Expected purchase for: " + str(month_expected) + " months" + " for " + str(country) + " customers")
            print("###########################################################")
            print(country_code)



        if expected_profit:
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(country_code['frequency'], country_code['monetary'])
            country_code["expected_average_profit"] = ggf.conditional_expected_average_profit(country_code['frequency'],
                                                                                       country_code['monetary'])
            print("###########################################################")
            print("Expected profit for " + str(country) + " customers")
            print("###########################################################")
            print(country_code)




        if cltv_prediction:
            bgf = BetaGeoFitter(penalizer_coef=0.001)
            bgf.fit(country_code['frequency'], country_code['recency'], country_code['T'])

            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(country_code['frequency'], country_code['monetary'])

            cltv_x_month = ggf.customer_lifetime_value(bgf,
                                                     country_code['frequency'],
                                                     country_code['recency'],
                                                     country_code['T'],
                                                     country_code['monetary'],
                                                     time=cltv_month,
                                                     freq="W",
                                                     discount_rate=0.01)

            country_final_cltv = country_code.merge(cltv_x_month, on="Customer ID", how="left")

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(country_final_cltv[["clv"]])
            country_final_cltv["scaled_clv"] = scaler.transform(country_final_cltv[["clv"]])

            country_final_cltv["segment"] = pd.qcut(country_final_cltv["scaled_clv"], 4, labels=["D", "C", "B", "A"])

            print("###########################################################")
            print("Customer Lifetime Value analysis for " + str(country) + " for " + str(cltv_month) + " months")
            print("###########################################################")
            return country_final_cltv


cltv_create(df, "Spain", cltv_prediction=True, cltv_month=3)


# 6 month CLTV analysis for Spain customers:

cltv_create(df, "Spain", cltv_prediction=True, cltv_month=6)


# 3 month expected purchase analysis for Belgium customers:

cltv_create(df, "Belgium", expected_purchase=True, month_expected=3)


# Expected profit for Austria customers:

cltv_create(df, "Austria", expected_profit=True)


# Lithuania

cltv_create(df, "Lithuania", cltv_prediction=True, cltv_month=3)

