###########################################################################################################
######## version = 1.0
######## status = WIP
###########################################################################################################
import sys
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import pyarrow.parquet as pq
import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
from frontend.Home.load_css import local_css
local_css("style.css")
import io
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image
import pendulum


#### Data Import
@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_parquet(path, engine='pyarrow')

    return df

dataframe = pd.read_excel(r"C:\Users\RafaelOliveira\Brand Delta\Nomad - General\Green Cuisine Campaign\03_FrontEnd_GreenCuisine\FrontEnd_GreenCuisine_Dataset_v2.xlsx",engine='openpyxl')
xls = pd.ExcelFile(r'C:\Users\RafaelOliveira\Brand Delta\Nomad - General\Green Cuisine Campaign\03_FrontEnd_GreenCuisine\FrontEnd_GreenCuisine_Dataset_v2.xlsx')
sheet_to_df_map = {}
for sheet_name in xls.sheet_names:
    sheet_to_df_map[sheet_name] = xls.parse(sheet_name)

final_view_dataframe = sheet_to_df_map['1_ConsumerChannel_RAW_DATA']

final_view_dataframe['WeekCom'] = final_view_dataframe['Created_Time']-pd.to_timedelta(final_view_dataframe['Created_Time'].dt.weekday,unit='D')
final_view_dataframe['WeekCom'] = final_view_dataframe['WeekCom'].dt.strftime('%Y-%m-%d')

spend = sheet_to_df_map['2_Equity_RAW_DATA']
equity = sheet_to_df_map['2_Equity_RAW_DATA']
equity = equity[equity['time_period'].str.contains('weekly')]


def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

final_view_dataframe = filter_rows_by_values(final_view_dataframe,'Sentiment',['uncategorized'])
final_view_dataframe['Created_Time'] = pd.to_datetime(final_view_dataframe['Created_Time'])

### Helper Methods ###

# def add_logo():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"] {
#                 background-image: url(http://placekitten.com/200/200);
#                 background-repeat: no-repeat;
#                 padding-top: 120px;
#                 background-position: 20px 20px;
#             }
#             [data-testid="stSidebarNav"]::before {
#                 content: "My Company Name";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 30px;
#                 position: relative;
#                 top: 100px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_markup_for_logo(
    png_file,
    background_position="5% 20%",
    margin_top="5%",
    image_width="80%",
    image_height="20",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stHeader"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }

            }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )

def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
    


def min_max_scaler(a,b,original_dataframe,count_col):
    original_dataframe_cp = original_dataframe.copy()

    max_val = original_dataframe_cp[count_col].max()
    min_val = original_dataframe_cp[count_col].min()

    original_dataframe_cp['Percentage'] = original_dataframe_cp[count_col].apply(lambda x: ((b-a)*(x-min_val))/(max_val-min_val))+a

    original_dataframe_cp = original_dataframe_cp.loc[:, ~original_dataframe_cp.columns.str.contains('^Unnamed')]

    return original_dataframe_cp

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection


########################
### ANALYSIS METHODS ###
########################
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

def affinity_spend(equity,spend,brand):

    equity_cp = equity.copy()
    spend_cp = spend.copy()

    equity_cp = equity_cp[equity_cp['Brand'].str.contains(brand)]
    equity_cp = equity_cp[equity_cp['time_period'].str.contains('weekly')]

    #spend_by_week = spend_cp.groupby(['Date']).sum()
    list_of_cols = [l for l in list(spend_cp.columns) if l != "Date"]
    # spend_cp['Sum'] = spend_cp[list_of_cols].sum(axis=1)
    # spend_cp['Date'] = spend_cp['Date'].dt.strftime('%Y-%m-%d')

    spend_cp = spend_cp[['time','Media Spend']]

    merged_data = equity_cp.merge(spend_cp, how='left', left_on='time', right_on='time')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    st.dataframe(merged_data)
    fig.add_trace(go.Line(x=merged_data['time'], y=merged_data['Framework - Awareness'],name='Awareness'),secondary_y=False)
    fig.add_trace(go.Line(x=merged_data['time'], y=merged_data['Framework - Saliency'],name='Saliency'),secondary_y=False)
    fig.add_trace(go.Line(x=merged_data['time'], y=merged_data['Framework - Affinity'],name='Affinity'),secondary_y=False)

    fig.add_trace(go.Bar(x=merged_data['time'], y=merged_data['Media Spend_x'],name='Investment'), secondary_y=True)

    fig.update_layout(
        #margin=dict(l=2, r=1, t=55, b=2),
        autosize=True,
        xaxis=dict(title_text="Time"),
        yaxis=dict(title_text="Counts"),
        width=1000,title="{} Equity Build vs Media Spend".format(brand)
        )

    st.plotly_chart(fig)

    
def filled_area_plot(dataframe,x,y,line_group,filled_area_plot_type,brand,type):

    dataframe['Created_Time'] = dataframe['Created_Time'].dt.strftime('%Y-%m-%d')
    dataframe = dataframe[dataframe['Brand'].str.contains(brand)]
    dataframe_grouped = dataframe.groupby(['Brand',type,'Created_Time']).size().reset_index(name='counts')
    if filled_area_plot_type == "AUC":
        fig = px.area(dataframe_grouped, x=x,y=y,color=type,line_group=line_group)
        fig.update_layout(
            #margin=dict(l=2, r=1, t=55, b=2),
            autosize=True,
            xaxis=dict(title_text="Time"),
            yaxis=dict(title_text="Counts"),
            )
        st.plotly_chart(fig)
    elif filled_area_plot_type == "Line":
        fig = px.line(dataframe_grouped, x=x,y=y,color=type,line_group=line_group)
        fig.update_layout(
            #margin=dict(l=20, r=20, t=20, b=20),
            autosize=True,
            xaxis=dict(title_text="Time"),
            yaxis=dict(title_text="Counts"),
        )
        st.plotly_chart(fig)



def sentiment_barplot(dataframe,barplot_type,brand,quick_filter=False):

    #dataframe['Created_Time'] = dataframe['Created_Time'].dt.strftime('%Y-%m-%d')
    dataframe = dataframe[dataframe['Brand'].str.contains(brand)]

    if barplot_type == "Absolute":
        if quick_filter==True:

            final_view_dataframe_grouped = dataframe.groupby(['Sentiment','Brand','View']).size().reset_index(name='counts')

            fig = go.Figure()

            fig.update_layout(
                autosize=True,
                xaxis=dict(title_text="Brands"),
                yaxis=dict(title_text="Count"),
                barmode="stack")

            colors = ['salmon','aqua','aquamarine']

            for r, c in zip(final_view_dataframe_grouped.Sentiment.unique(), colors):
                plot_df = final_view_dataframe_grouped[final_view_dataframe_grouped.Sentiment == r]
                fig.add_trace(
                    go.Bar(
                        x=[
                            plot_df.Brand, 
                            plot_df.View], 
                        y=plot_df.counts,
                        name=r,
                        marker_color=c,
                        text=plot_df.counts),
                )

            #fig = px.bar(final_view_dataframe_grouped, x="Brand", y='counts',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            st.plotly_chart(fig)

        else:

            sentiment = dataframe.groupby(['Sentiment','Brand']).size().reset_index(name='counts')

            fig = px.bar(sentiment, x="Brand", y='counts',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            fig.update_layout(autosize=True)
            st.plotly_chart(fig)

    elif barplot_type == "Normalized":
        if quick_filter==True:

            final_view_dataframe_grouped = dataframe.groupby(['Sentiment','Brand','View']).size().reset_index(name='counts')
            totals_by_brand = final_view_dataframe_grouped.groupby(['Brand','View'])['counts'].sum()

            final_view_dataframe_grouped = (final_view_dataframe_grouped.merge(totals_by_brand,left_on=['Brand','View'],right_on=['Brand','View'],how='left').assign(new=lambda x:round(x['counts_x'].div(x['counts_y'])*100,2)).reindex(columns=[*final_view_dataframe_grouped.columns]+['new']))

            final_view_dataframe_grouped.columns = ['Sentiment','Brand','View','Counts','Percentage']

            fig = go.Figure()

            fig.update_layout(
                autosize=True,
                xaxis=dict(title_text="Brands"),
                yaxis=dict(title_text="Percentage"),
                barmode="stack",
            )

            colors = ['salmon','aqua','aquamarine']

            for r, c in zip(final_view_dataframe_grouped.Sentiment.unique(), colors):
                plot_df = final_view_dataframe_grouped[final_view_dataframe_grouped.Sentiment == r]
                fig.add_trace(
                    go.Bar(
                        x=[
                            plot_df.Brand, 
                            plot_df.View], 
                        y=plot_df.Percentage,
                        name=r,
                        marker_color=c,
                        text=plot_df.Percentage.apply(lambda x: '{0:1.2f}%'.format(x))),
                )

            #fig = px.bar(final_view_dataframe_grouped, x="Brand", y='counts',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            st.plotly_chart(fig)

        else:

            sentiment = dataframe.groupby(['Sentiment','Brand']).size().reset_index(name='counts')
            totals_by_brand = sentiment.groupby(['Brand'])['counts'].sum().reset_index() #we can remove reset_index()

            sentiment = (sentiment.merge(totals_by_brand,left_on='Brand',right_on='Brand',how='left').assign(new=lambda x:round(x['counts_x'].div(x['counts_y'])*100,2)).reindex(columns=[*sentiment.columns]+['new']))

            sentiment.columns = ['Sentiment','Brand','Counts','Percentage']

            fig = px.bar(sentiment, x="Brand", y='Percentage',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            fig.update_layout(autosize=True)
            st.plotly_chart(fig)


####################
### INTRODUCTION ###
####################
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((0.1, 5, .1, 1.3, .1))
with row0_1:
    st.image(Image.open(r"C:\Users\RafaelOliveira\OneDrive - Brand Delta\Documents\Projects\Frontend\Frontend\frontend\images\Picture1.png"))

with row0_2:
    st.text("")
    st.subheader(' ')

#################
### SELECTION ###
#################

#####   FILTERS ##### 

### SEE DATA ###

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first (first 1000 rows)')
    with see_data:
        st.dataframe(data=final_view_dataframe.reset_index(drop=True).head(1000))
st.text('')

#selection = aggrid_interactive_table(df=masked_df)

# df_tt = px.data.gapminder()
# st.dataframe(df_tt)

## SENTIMENT BARPLOT
st.markdown("<h3 <span class = 'highlight  red'> 1 | How is the Campaign Spreading across Consumer Channels? </span> </h3>",unsafe_allow_html=True)

row4_1, row4_spacer2, row4_2 = st.columns((1, 0.05, 1))

with row4_1:
    st.markdown('Who is engaging with the Campaign?')

    filled_area_plot_type = st.selectbox(
        "Select area plot type:",
        ['AUC','Line'])
    type = st.selectbox(
        "Select mix:",
        ['journey_predictions','author_predictions','Message_Type','Gender','Age Range'])
        
    filled_area_plot(
        dataframe=final_view_dataframe,
        x="Created_Time",
        y="counts",
        line_group="Brand",
        filled_area_plot_type=filled_area_plot_type,
        brand='green_cuisine',
        type=type)

with row4_2:
    st.markdown('How is the Campaign Sentiment Trending?')

    barplot_type = st.selectbox(
                    "Select barplot type:",
                    ['Normalized','Absolute'])
    sentiment_barplot(final_view_dataframe,barplot_type,"green_cuisine")

st.markdown("<h3 <span class = 'highlight  red'> 2 | How is the Campaign supporting Equity? </span> </h3>",unsafe_allow_html=True)
row5_spacer1, row5_1, row5_spacer3  = st.columns((.2, 2.3, .2))
with row5_1:
    affinity_spend(equity,spend,'green_cuisine')

    st.markdown('What is the impact on equity score?')
    time_filter=None
    see_quick_date_filters = st.expander('Click to select Quick Date Filters')
    with see_quick_date_filters:
        if st.checkbox("P3M"):
            time_filter="P3M"
        elif st.checkbox("LW"):
            time_filter="LW"
        elif st.checkbox("L4W"):
            time_filter="L4W"
    st.markdown('How is equity trending now?')


st.markdown("<h3 <span class = 'highlight  red'> 3 | How are Efficiency & Financial KPIs tracking? </span> </h3>",unsafe_allow_html=True)
row6_spacer1, row6_1, row6__spacer2, row6__2, row6__spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight  red'> 4 | How well are we Penetrating Target Demand Spaces? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight  red'> 5 | How well are 'Welcome to the Plant Age' Creatives Resonating? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight  red'> 6 | How is Risk Tracking? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))