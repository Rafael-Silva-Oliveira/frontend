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
st.write(sheet_to_df_map.keys())
st.dataframe(sheet_to_df_map['5.2.1_W2tPA_SENT_DELTA_TABLE'].astype(str))

final_view_dataframe = load_data(r"C:\Users\RafaelOliveira\Downloads\English_data_tagged_26_09_2022.parquet")
#equity = pd.read_excel(r"C:\Users\RafaelOliveira\Brand Delta\Food Pilot - General\data\modelled_data\uk\equity_scores_26_09_2022\equity_scores_corrected.xlsx")
final_view_dataframe['WeekCom'] = final_view_dataframe['Created_Time']-pd.to_timedelta(final_view_dataframe['Created_Time'].dt.weekday,unit='D')
final_view_dataframe['WeekCom'] = final_view_dataframe['WeekCom'].dt.strftime('%Y-%m-%d')

spend = pd.read_excel(r"C:\Users\RafaelOliveira\Brand Delta\Nomad - General\Green Cuisine Campaign\02_Data\equity_scores\equity_meatSubs_greenCuisine.xlsx",sheet_name="EQUITY_SPEND")

equity = pd.read_excel(r"C:\Users\RafaelOliveira\Brand Delta\Nomad - General\Green Cuisine Campaign\02_Data\equity_scores\equity_meatSubs_greenCuisine.xlsx",sheet_name="EQUITY_SPEND")

equity = equity[equity['time_period'].str.contains('weeks')]


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

def impact_equity_scores(equity,brands,time_filter):

    equity_res = pd.DataFrame()

    if time_filter == None:
        for brand in brands:
            temp_dict = {}
            for col in equity.columns:
                if col in ['Framework - Awareness','Framework - Saliency','Framework - Affinity']:
                    
                    pre_campaign = equity[equity['Brand'].isin([brand])]
                    pre_campaign_mean = pre_campaign[pre_campaign['Media Spend'].isna()][col].mean()
                    campaign = equity[equity['Brand'].isin([brand])]
                    campaign_mean =campaign[~campaign['Media Spend'].isna()][col].mean()

                    delta = ((campaign_mean-pre_campaign_mean)*100)/(pre_campaign_mean)

                    #st.write("Pre_campaign is {} and campaign is {}".format(pre_campaign_mean, campaign_mean))
                    delta = ((campaign_mean-pre_campaign_mean)*100)/(pre_campaign_mean)
                    temp_dict.update({col:{'Pre-Campaign':[pre_campaign_mean],'Campaign':[campaign_mean],'Delta':[delta]}})

            temp_dataframe = pd.DataFrame.from_dict(temp_dict, orient="index").stack().to_frame()
            temp_dataframe = pd.DataFrame(temp_dataframe[0].values.tolist(), index=temp_dataframe.index)
            #temp_dataframe = temp_dataframe.astype(str)
            temp_dataframe = temp_dataframe.T.rename(index={0: brand})

            equity_res = pd.concat([
                equity_res,
                temp_dataframe],
            axis=0)
    else:
        for brand in brands:
            temp_dict = {}
            for col in equity.columns:
                if col in ['Framework - Awareness','Framework - Saliency','Framework - Affinity']:

                    if time_filter == "P3M":
                        idx_end_date = equity[~equity['Media Spend'].isna()][col].index[0]-1
                        end_date = equity[equity['Media Spend'].isna()]['time'].iloc[idx_end_date]

                        date_time_obj  = datetime.datetime.strptime(end_date, '%Y-%m-%d')

                        start_date = date_time_obj.date() + relativedelta(months=-3)
                        start_date  = start_date.strftime('%Y-%m-%d')
                        
                    elif time_filter == "LW":
                        idx_end_date = equity[~equity['Media Spend'].isna()][col].index[0]-1
                        end_date = equity[equity['Media Spend'].isna()]['time'].iloc[idx_end_date]

                        date_time_obj  = datetime.datetime.strptime(end_date, '%Y-%m-%d')

                        start_date = date_time_obj.date() + relativedelta(weeks=-1)
                        start_date  = start_date.strftime('%Y-%m-%d')
                    elif time_filter == "L4W":

                        idx_end_date = equity[~equity['Media Spend'].isna()][col].index[0]-1
                        end_date = equity[equity['Media Spend'].isna()]['time'].iloc[idx_end_date]

                        date_time_obj  = datetime.datetime.strptime(end_date, '%Y-%m-%d')

                        start_date = date_time_obj.date() + relativedelta(weeks=-4)
                        start_date  = start_date.strftime('%Y-%m-%d')

                    #Select the date range between start_date and end_date of the selected period
                    between_mask = equity['time'].between(start_date,end_date,inclusive="both")
                    #Apply the mask
                    pre_campaign = equity[between_mask]
                    pre_campaign = pre_campaign[pre_campaign['Brand'].isin([brand])]
                    
                    #Select just the Media Spend for the given framework
                    pre_campaign_mean = pre_campaign[col].mean()
                    #Select the investment period for the campaign
                    campaign = equity[~equity['Media Spend'].isna()]
                    campaign = campaign[campaign['Brand'].isin([brand])]
                    campaign_mean = campaign[col].mean()

                    #st.write("Pre_campaign is {} and campaign is {}".format(pre_campaign_mean, campaign_mean))
                    delta = ((campaign_mean-pre_campaign_mean)*100)/(pre_campaign_mean)
                    temp_dict.update({col:{time_filter:[pre_campaign_mean],'Campaign':[campaign_mean],'Delta':[delta]}})

            temp_dataframe = pd.DataFrame.from_dict(temp_dict, orient="index").stack().to_frame()
            temp_dataframe = pd.DataFrame(temp_dataframe[0].values.tolist(), index=temp_dataframe.index)
            #temp_dataframe = temp_dataframe.astype(str)
            temp_dataframe = temp_dataframe.T.rename(index={0: brand})

            equity_res = pd.concat([
                equity_res,
                temp_dataframe],
            axis=0).round(decimals=1).astype(object)

    st.dataframe(equity_res.style.format("{:.3}"))

def affinity_spend(equity,spend,brand):

    equity_cp = equity.copy()
    spend_cp = spend.copy()

    equity_cp = equity_cp[equity_cp['Brand'].str.contains(brand)]
    equity_cp = equity_cp[equity_cp['time_period'].str.contains('weeks')]

    #spend_by_week = spend_cp.groupby(['Date']).sum()
    list_of_cols = [l for l in list(spend_cp.columns) if l != "Date"]
    # spend_cp['Sum'] = spend_cp[list_of_cols].sum(axis=1)
    # spend_cp['Date'] = spend_cp['Date'].dt.strftime('%Y-%m-%d')

    spend_cp = spend_cp[['time','Media Spend']]

    merged_data = equity_cp.merge(spend_cp, how='left', left_on='time', right_on='time')
    fig = make_subplots(specs=[[{"secondary_y": True}]])

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

def sentiment_tracking(dataframe,brands,campaign_id='campaign_id',camp_on='camp',time_filter=['P3M','Jul','Aug']):
    dataframe = dataframe[dataframe['Brand'].isin(brands)]

    final_df = pd.DataFrame()
    for filter in time_filter:
        temp_dict = {}
        sentiment = dataframe.groupby(['Brand','Sentiment','WeekCom']).size().reset_index(name='Total_Counts_sentiment')
        #totals_by_brand = sentiment.groupby(['Brand','Sentiment'])['counts'].sum().reset_index() #we can remove reset_index()

        if filter == 'Jul':
            sentiment = sentiment.loc[(pd.to_datetime(sentiment['WeekCom']).dt.month==7) & (pd.to_datetime(sentiment['WeekCom']).dt.year==2022)]
            totals_by_sentiment = sentiment.groupby(['Brand','Sentiment'])['Total_Counts_sentiment'].sum().reset_index() #we can remove reset_index()
        elif filter == 'Aug':
            sentiment = sentiment.loc[(pd.to_datetime(sentiment['WeekCom']).dt.month==8) & (pd.to_datetime(sentiment['WeekCom']).dt.year==2022)]
            totals_by_sentiment = sentiment.groupby(['Brand','Sentiment'])['Total_Counts_sentiment'].sum().reset_index() #we can remove reset_index()
        elif filter == 'P3M':
            sentiment = sentiment.loc[(pd.to_datetime(sentiment['WeekCom']).dt.month==8) & (pd.to_datetime(sentiment['WeekCom']).dt.year==2022)]
            totals_by_sentiment = sentiment.groupby(['Brand','Sentiment'])['Total_Counts_sentiment'].sum().reset_index() #we can remove reset_index()
        totals_by_brand = totals_by_sentiment.groupby(['Brand'])['Total_Counts_sentiment'].sum()

        sentiment = (totals_by_sentiment.merge(totals_by_brand,left_on=['Brand'],right_on=['Brand'],how='left').assign(new=lambda x:round(x['Total_Counts_sentiment_x'].div(x['Total_Counts_sentiment_y'])*100,2)))

        sentiment.columns = ['Brand','Sentiment','X','Y',filter]
        sentiment = sentiment[['Brand','Sentiment',filter]]

        sentiment.set_index(['Brand','Sentiment'],inplace=True)
        final_df = pd.concat([final_df,sentiment],axis=1)

    st.dataframe(final_df.style.format("{:.3}"))


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

class filters:

    def date_range(dataframe):
        ### DATE RANGE ###

        output = {
            'final_view_dataframe':None,
            'quick_filter':False,
            'curr_view_ref':None,
            'last_view_ref':None,
        }

        st.sidebar.markdown("**First select the data range you want to analyze:** 👇")

        start_date = dataframe['Created_Time'].min()
        end_date = dataframe['Created_Time'].max()
        #df = df.set_index(['Created_Time'])

        try:
            start_date, end_date = st.sidebar.date_input('Start date  - End date :', [start_date,end_date], min_value = start_date, max_value = end_date)
            if start_date < end_date:
                pass
            else:
                st.error('Error: End date must fall after start date.')
        except:
            #st.write("<span class = 'highlight  blue'> No date range selected. Using all the data points untill a subset is selected. </span>",unsafe_allow_html=True)
            st.write(" ")

        mask = (dataframe['Created_Time'] > pd.to_datetime(str(start_date))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date)))
        masked_df_current = dataframe.loc[mask]

        see_quick_date_filters = st.sidebar.expander('Click to select Quick Date Filters')
        with see_quick_date_filters:

            end_date_new = dataframe['Created_Time'].max()

            if st.checkbox("LW vs CW"):
                last_view_ref = "LW"
                current_view_ref = "CW"

                starting_days = datetime.timedelta(14)
                midpoint_days = datetime.timedelta(7)

                end_date_current = end_date_new
                start_date_current = end_date_new-midpoint_days + datetime.timedelta(1)

                start_date_last = end_date_current - starting_days 
                end_date_last = start_date_last + midpoint_days
                
                mask_last = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_last))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_last)))
                masked_df_last = dataframe.loc[mask_last]
                masked_df_last['View'] = last_view_ref

                mask_current = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_current))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_current)))
                masked_df_current = dataframe.loc[mask_current]
                masked_df_current['View'] = current_view_ref

                final_view_dataframe = pd.concat([masked_df_last,masked_df_current],axis=0)
                # st.text("Last week range is {} - {} and current week range is {} - {}".format(start_date_last,end_date_last,start_date_current,end_date_current))

                output.update({
                    'final_view_dataframe':final_view_dataframe,
                    'quick_filter':True,
                    'curr_view_ref':current_view_ref,
                    'last_view_ref':last_view_ref,
                    'start_date_last':start_date_last,
                    'end_date_last':end_date_last,
                    'start_date_current':start_date_current,
                    'end_date_current':end_date_current
                })

                return output

            elif st.checkbox("L4W vs C4W"):
                st.text("Last 4 weeks vs Current 4 weeks")
            elif st.checkbox("LM vs CM"):
                st.text("Last month vs Current Month")
            elif st.checkbox("L6M vs C6M"):
                st.text("Last 6 months vs Current 6 months")
            elif st.checkbox("LY vs CY"):
                st.text("Last year vs Current Year")
        
        output.update({
            'final_view_dataframe':masked_df_current,
            'quick_filter':False,
            'start_date':start_date,
            'end_date':end_date,
        })
        return output

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
    impact_equity_scores(equity,['green_cuisine',"Birds_Eye"],time_filter)
    st.markdown('How is equity trending now?')


st.markdown("<h3 <span class = 'highlight  red'> 3 | How are Efficiency & Financial KPIs tracking? </span> </h3>",unsafe_allow_html=True)
row6_spacer1, row6_1, row6__spacer2, row6__2, row6__spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight  red'> 4 | How well are we Penetrating Target Demand Spaces? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight  red'> 5 | How well are 'Welcome to the Plant Age' Creatives Resonating? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))

with row7_1:
    sentiment_tracking(final_view_dataframe,['green_cuisine','Birds_Eye'])
st.markdown("<h3 <span class = 'highlight  red'> 6 | How is Risk Tracking? </span> </h3>",unsafe_allow_html=True)
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))