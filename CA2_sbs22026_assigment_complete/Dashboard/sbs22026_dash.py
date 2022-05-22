import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

def read_data():
    df = pd.DataFrame()
    for i in [2013,2015,2018,2021]:
        print(i)
        if i == 2013 or i == 2015:
            temp = pd.read_excel(f"../data/MS{i}M12TBL1.xls",header=2)
            temp = temp.iloc[[17,18,19,21,22,23,25,26,27]].set_index(["Category"])
        else:
            temp = pd.read_excel(f"../data/MS{i}M12TBL1.xls",header=1)
            temp = temp.iloc[[16,17,18,20,21,22,24,25,26]].set_index(["Category"])
        temp.index = ["Total milk sold for human","Total milk sold for human","Total milk sold for human",
                             "Whole milk sales","Whole milk sales","Whole milk sales",
                             "Skimmed  & semi-","Skimmed  & semi-","Skimmed  & semi-"]
        df = df.append(temp.reset_index())
        
        fertiliser = pd.read_csv("../data/fertiliser_year.csv",index_col=0)
        cattle = pd.read_csv("../data/cattle_year.csv",index_col=0)
        land = pd.read_csv("../data/land_price.csv",index_col=0)
    return df, fertiliser, cattle, land

def preprocess(df):   
    df = df.drop(columns=["Unit","Year.1"])
    months = df.drop(columns=["index","Year"]).columns.tolist()
    df = pd.melt(df, id_vars=["index","Year"], value_vars=months, var_name="Month", value_name='value')
    df["value"] = df["value"].astype("str")\
                             .str.replace(" 4","")\
                             .str.replace(" 1","")\
                             .astype("float")
    df = pd.pivot_table(df, values="value", index=["Year","Month"], columns="index")
    df = df.reset_index()
    month_dic = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                 "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    df["Month"] = df["Month"].apply(lambda x: month_dic[x])
    df["Year"] = df["Year"].astype("int")
    df["datetime"] = pd.to_datetime(df["Month"].astype("str")+"-"+df["Year"].astype("str"))
    df = df.sort_values(by=["datetime"])
    return df

st.set_page_config(page_title='Agriculture Dashboard', layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("## Ireland's Agriculture Dashboard")

df, fertiliser, cattle, land = read_data()
df = preprocess(df)

col1, col2 = st.columns((2,3))
col1.write("### Data")
col1.write(df)

col2.write("### Time series")
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x = df.drop(columns=["Year","Month"])["datetime"],
                               y = df.drop(columns=["Year","Month"])["Skimmed  & semi-"], mode='lines', name="Skimmed"))
fig = fig.add_trace(go.Scatter(x = df.drop(columns=["Year","Month"])["datetime"],
                               y = df.drop(columns=["Year","Month"])["Total milk sold for human"], mode='lines', name="Total"))
fig = fig.add_trace(go.Scatter(x = df.drop(columns=["Year","Month"])["datetime"],
                               y = df.drop(columns=["Year","Month"])["Whole milk sales"], mode='lines', name="Whole"))
col2.plotly_chart(fig)

col1, col2 = st.columns((1,1))
col1.write("### Forecasting")
columns = df.set_index("datetime").drop(columns=["Year","Month"]).columns
column = col1.selectbox("Select time series", columns)
model = ARIMA(df.set_index("datetime")[column],order=(2,1,2))
res = model.fit()

fig = go.Figure()
fig = fig.add_trace(go.Scatter(x = df.drop(columns=["Year","Month"])["datetime"],
                               y = df.drop(columns=["Year","Month"])[column], mode='lines', name=column))
fig = fig.add_trace(go.Scatter(x = res.predict(10).index,
                               y = res.predict(10), mode='lines', name="Prediction"))
fig = fig.add_trace(go.Scatter(x = res.forecast(15).index,
                               y = res.forecast(15), mode='lines', name="Forecast"))
col1.plotly_chart(fig)

col2.write("### Correlations")
cattle = cattle.rename(columns={"YEAR":"Year"})
df = df.merge(fertiliser).merge(cattle,how="left").merge(land,how="left")
df = df.drop(columns=["datetime"])
columns = col2.multiselect("Select columns", df.columns, default=df.columns[0])
plt.figure(figsize=(10,5))
col2.write(sns.heatmap(data=df[columns].corr(), annot=True).figure)