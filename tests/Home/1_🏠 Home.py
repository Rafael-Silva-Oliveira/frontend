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


#### Data Import
@st.cache
def load_data(path):
    df = pd.read_parquet(path, engine='pyarrow')

    return df

df = load_data(r"C:\Users\RafaelOliveira\Downloads\English_data_tagged_26_09_2022.parquet")

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]
df = filter_rows_by_values(df,'Sentiment',['uncategorized'])

df['Created_Time'] = pd.to_datetime(df['Created_Time'])

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
    background_position="50% 10%",
    margin_top="10%",
    image_width="80%",
    image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
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

add_logo(r"C:\Users\RafaelOliveira\OneDrive - Brand Delta\Documents\Projects\Frontend\Frontend\frontend\images\brand_delta-3-e1642936725365.png")


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
def filled_area_plot(output,x,y,line_group,color,filled_area_plot_type):

    dataframe = output['final_view_dataframe']
    dataframe['Created_Time'] = dataframe['Created_Time'].dt.strftime('%Y-%m-%d')
    dataframe_grouped = dataframe.groupby(['Brand','author_predictions','Created_Time']).size().reset_index(name='counts')

    if len(dataframe['Brand'].unique())>=2:
        st.write("In order to plot the filled area chart, please select only one brand on the sidebar.")
    elif len(dataframe['Brand'].unique())==1:
        if filled_area_plot_type == "AUC":
            fig = px.area(dataframe_grouped, x=x,y=y,color=color,line_group=line_group)
            fig.update_layout(
                autosize=True,
                width=550,
                height=400,
                xaxis=dict(title_text="Time"),
                yaxis=dict(title_text="Counts"),
                )
            st.plotly_chart(fig)
        elif filled_area_plot_type == "Line":
            fig = px.line(dataframe_grouped, x=x,y=y,color=color,line_group=line_group)
            fig.update_layout(
                autosize=True,
                width=550,
                height=400,
                xaxis=dict(title_text="Time"),
                yaxis=dict(title_text="Counts"),
            )
            st.plotly_chart(fig)
            #     legend=dict(
            #         yanchor="top",
            #         y=0.99,
            #         xanchor="left",
            #         x=0.01
            # )
    # else:
    #     st.write("<span class = 'highlight  blue'> No brand selected. Please select a brand first to display the data. </span>",unsafe_allow_html=True)

def sentiment_barplot(output,barplot_type):


    if len(output['final_view_dataframe']['Brand'].unique())==0:
        # st.write("<span class = 'highlight  blue'> No brand selected. Please select a brand first to display the data. </span>",unsafe_allow_html=True)    
        st.write(" ")
    # fig, ax = plt.subplots()
    # dataframe[sentiment_label].value_counts().plot(kind='bar')
    elif barplot_type == "Absolute":
        if output['quick_filter']==True:

            final_view_dataframe_grouped = output['final_view_dataframe'].groupby(['Sentiment','Brand','View']).size().reset_index(name='counts')

            # fig = go.Figure(
            #     go.bar(
            #         x=[
            #             final_view_dataframe_grouped['Brand'].tolist(),
            #             final_view_dataframe_grouped['View'].tolist()
            #         ],
            #         y=final_view_dataframe_grouped['counts'],
            #         color=final_view_dataframe_grouped['Sentiment'],
            #         text_auto=True, 
            #         color_discrete_sequence=['salmon','aqua','aquamarine']
            #     )
            # )

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

            sentiment = output['final_view_dataframe'].groupby(['Sentiment','Brand']).size().reset_index(name='counts')

            fig = px.bar(sentiment, x="Brand", y='counts',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            fig.update_layout(autosize=True)
            st.plotly_chart(fig)

    elif barplot_type == "Normalized":
        if output['quick_filter']==True:

            final_view_dataframe_grouped = output['final_view_dataframe'].groupby(['Sentiment','Brand','View']).size().reset_index(name='counts')
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

            sentiment = output['final_view_dataframe'].groupby(['Sentiment','Brand']).size().reset_index(name='counts')
            totals_by_brand = sentiment.groupby(['Brand'])['counts'].sum().reset_index() #we can remove reset_index()

            sentiment = (sentiment.merge(totals_by_brand,left_on='Brand',right_on='Brand',how='left').assign(new=lambda x:round(x['counts_x'].div(x['counts_y'])*100,2)).reindex(columns=[*sentiment.columns]+['new']))

            sentiment.columns = ['Sentiment','Brand','Counts','Percentage']

            fig = px.bar(sentiment, x="Brand", y='Percentage',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
            fig.update_layout(autosize=True)
            st.plotly_chart(fig)


####################
### INTRODUCTION ###
####################
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Charts')
with row0_2:
    st.text("")
    st.subheader('Streamlit App')

#################
### SELECTION ###
#################


#st.sidebar.text('Test')

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

            quick_filter=True
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

    def brand_selection(original_dataframe):

        ### BRAND SELECTION ###
        container = st.sidebar.container()

        all = st.sidebar.checkbox("Select all")
        original_dataframe_cp = original_dataframe.copy()
        try:
            if all:
                selected_options = container.multiselect(
                    "Select one or more brands:",
                    list(original_dataframe_cp['Brand'].unique()),
                    default=list(original_dataframe_cp['Brand'].unique()))

                original_dataframe_cp = original_dataframe_cp.loc[original_dataframe_cp['Brand'].isin(selected_options)]

            else:
                selected_options =  container.multiselect("Select one or more brands:",
                    list(original_dataframe_cp['Brand'].unique()))
                    
                original_dataframe_cp = original_dataframe_cp.loc[original_dataframe_cp['Brand'].isin(selected_options)]
        except:
            st.write("No data has been selected! Please, select a brand to get the outputs.")

        return original_dataframe_cp


dataframe_after_brand_selection = filters.brand_selection(df)
output_after_filter=filters.date_range(dataframe_after_brand_selection)
final_view_dataframe = output_after_filter['final_view_dataframe']
quick_filter = output_after_filter['quick_filter']
curr_view_ref= output_after_filter['curr_view_ref']
last_view_ref= output_after_filter['last_view_ref']


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
st.subheader('1 | How is the Campaign Spreading across Consumer Channels?')

row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3 = st.columns((2, 3, .1, 3, 2))

if output_after_filter['quick_filter'] == True:

    start_date_last=output_after_filter['start_date_last']
    end_date_last=output_after_filter['end_date_last']
    start_date_current=output_after_filter['start_date_current']
    end_date_current=output_after_filter['end_date_current']
    try:
        with row5_1:
            st.sidebar.markdown("<span class='bold'>LW </span>: <span class = 'highlight  blue'> {} - {}</span> ".format(start_date_last.strftime("%Y/%m/%d"),end_date_last.strftime("%Y/%m/%d")),unsafe_allow_html=True)
        with row5_2:
            st.sidebar.markdown("<span class='bold'>CW </span>: <span class = 'highlight  blue'> {} - {}</span> ".format(start_date_current.strftime("%Y/%m/%d"),end_date_current.strftime("%Y/%m/%d")),unsafe_allow_html=True)
    except:
        st.write("Please select the brands first.")
else:
    start_date=output_after_filter['start_date']
    end_date=output_after_filter['end_date']
    # try:
    #     with row5_1:
    #         st.markdown("<span class='bold'>Start Date</span>: <span class = 'highlight  blue'> {}</span> ".format(start_date.strftime("%Y/%m/%d")),unsafe_allow_html=True)
    #     with row5_2:
    #         st.markdown("<span class='bold'>End Date</span>: <span class = 'highlight  blue'> {}</span> ".format(end_date.strftime("%Y/%m/%d")),unsafe_allow_html=True)
    # except:
    if dataframe_after_brand_selection.empty==True:
        st.markdown("<span class = 'highlight  blue'> Please select the brands first. </span>",unsafe_allow_html=True)


row4_1, row4_spacer2, row4_2 = st.columns((1, .1, 1.3))
with row4_1:
    st.markdown('Who is engaging with the Campaign?')

    filled_area_plot_type = st.selectbox(
        "Select area plot type:",
        ['AUC','Line'])
        
    filled_area_plot(
        output=output_after_filter,
        x="Created_Time",
        y="counts",
        color="author_predictions",
        line_group="Brand",
        filled_area_plot_type=filled_area_plot_type)


with row4_2:
    st.markdown('How is the Campaign Sentiment Trending?')

    barplot_type = st.selectbox(
                    "Select barplot type:",
                    ['Normalized','Absolute'])
    sentiment_barplot(output_after_filter,barplot_type)

row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))