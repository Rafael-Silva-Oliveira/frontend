###########################################################################################################
######## version = 1.0
######## status = WIP
###########################################################################################################
import base64
import requests
import streamlit.components.v1 as components
import xlsxwriter
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import io
from load_css import local_css
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
local_css(r"style.css")
idx = pd.IndexSlice
buffer = io.BytesIO()

#####################################
###### Styling dataframes ###########
#####################################
cmap_red_green = LinearSegmentedColormap.from_list(
    name='red_green_gradient',
    colors=['#f59e8c', '#FFFFFF', '#ADF2C7']
)
cmap_red_white = LinearSegmentedColormap.from_list(
    name='red_white_gradient',
    colors=['#FFFFFF', '#f9999b', '#f8696b']
)

cmap_white_green = LinearSegmentedColormap.from_list(
    name='white_green_gradient',
    colors=['#FFFFFF', '#D3F1DE', '#ADF2C7', '#4ED981']
)
cmap_green_red = LinearSegmentedColormap.from_list(
    name='green_red_gradient',
    colors=['#f8696b', '#f8797b', '#f8898b', '#f9999b', '#f9aaac', '#fababd', '#fbdbde', '#FFFFFF',
            '#d9eee2', '#c9e7d3', '#b9e1c5', '#a6d9b6', '#96d3a7', '#86cc99', '#74c58a', '#63be7a']
)
cm = sns.diverging_palette(133, 10, as_cmap=True)
th_props = [
    ('font-size', '14px'),
    ('text-align', 'left'),
    ('font-family', 'Sans-serif'),
    ('font-weight', 'bold'),
    # ('background-color', '#bbb9c0'),
    ('color', '#353148'),  # bbb9c0
    ('width', 'auto'),
    # ('border','1px solid #353148'),
    # ('min-width','120px')
    #('padding','12px 35px')
    ('empty-cells', 'hide'),
    ('border-collapse', 'separate'),
    ('border-style', 'none'),

]

td_props = [
    ('font-size', '16px'),
    ('font-family', 'Sans-serif'),
    #('font-weight', 'bold'),
    ('text-align', 'center'),
    ('width', '95vw'),
    ('white-space', 'nowrap'),
    # ('border','1px solid #353148'), TODO: add this to add border
    # ('color', '#353148'), ##bbb9c0

]

cell_hover_props = [  # for column hover use <tr> instead of <td>
    ('background-color', '#eeeeef')
]

headers_props = [
    ('text-align', 'center'),
    ('font-size', '16px'),
    ('width', 'auto'),
    # ('border','1px solid #FFFFFF'),
    # ('border','1px solid #353148'),

]

headers_props_level01 = [
    ('text-align', 'center'),
    ('font-size', '14px'),
    ('width', 'auto'),
    # ('border','1px solid #FFFFFF'),
    # ('border','1px solid #353148'),

]
# dict(selector='th:not(.index_name)',props=headers_props)

# table_props = [
#     ('width','auto')
# ]
# #dict(selector='table',props=table_props),

index_props = [
    ('width', 'auto'),
    ('white-space', 'nowrap'),
    # ('color','#7F7F7F')
    # ('border','1px solid #353148'), TODO: add this to add border

]

table_props = [
    ('border-style', '1px solid #FFFFFF'),
    ('border', '1px solid #353148'),

]


styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props),
    dict(selector="td:hover", props=cell_hover_props),
    dict(selector='th.col_heading.level0', props=headers_props),
    dict(selector='th.col_heading.level1', props=headers_props_level01),
    dict(selector="tr", props=index_props),
]

####################################
####### Helper Functions ###########
####################################


class Tweet(object):
    def __init__(self, s, embed_str=False):
        if not embed_str:
            # Use Twitter's oEmbed API
            # https://dev.twitter.com/web/embedded-tweets
            api = "https://publish.twitter.com/oembed?url={}".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=600)


@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_parquet(path, engine='pyarrow')

    return df


def filter_columns_by_values(df, col, values):
    return df[~df[col].isin(values)]


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


def min_max_scaler(a, b, original_dataframe, count_col):
    original_dataframe_cp = original_dataframe.copy()

    max_val = original_dataframe_cp[count_col].max()
    min_val = original_dataframe_cp[count_col].min()

    original_dataframe_cp['Percentage'] = original_dataframe_cp[count_col].apply(
        lambda x: ((b-a)*(x-min_val))/(max_val-min_val))+a

    original_dataframe_cp = original_dataframe_cp.loc[:, ~
                                                      original_dataframe_cp.columns.str.contains('^Unnamed')]

    return original_dataframe_cp


####################################
####### Reading Data ###########
####################################
file_path = r".\backend\data\FrontEnd_GreenCuisine_Dataset_v18.xlsx"

dataframe = pd.read_excel(file_path, engine='openpyxl')
xls = pd.ExcelFile(file_path)
final_dataframe = {}
for sheet_name in xls.sheet_names:
    final_dataframe[sheet_name] = xls.parse(sheet_name)

final_view_dataframe = final_dataframe['1_ConsumerChannel_RAW_DATA']

final_view_dataframe['WeekCom'] = final_view_dataframe['Created_Time'] - \
    pd.to_timedelta(final_view_dataframe['Created_Time'].dt.weekday, unit='D')
final_view_dataframe['WeekCom'] = final_view_dataframe['WeekCom'].dt.strftime(
    '%Y-%m-%d')

equity_trend = pd.read_excel(file_path, engine='openpyxl', header=[
                             0, 1], index_col=[0, 1], sheet_name='2.3_EquityTrend_TABLE')

spend = final_dataframe['2_Equity_RAW_DATA']
equity = final_dataframe['2_Equity_RAW_DATA']
equity = equity[equity['time_period'].str.contains('weeks')]


final_view_dataframe = filter_columns_by_values(
    final_view_dataframe, 'Sentiment', ['uncategorized'])
final_view_dataframe['Created_Time'] = pd.to_datetime(
    final_view_dataframe['Created_Time'])


#########################
######### PLOTS #########
#########################
def affinity_spend(equity, spend, brand):

    affinity_brand_mapping = {
        'green_cuisine': 'Green Cuisine'
    }
    equity_cp = equity.copy()
    spend_cp = spend.copy()
    equity_cp['Brand'] = equity_cp['Brand'].map(affinity_brand_mapping)
    brand = affinity_brand_mapping[brand]
    equity_cp = equity_cp[equity_cp['Brand'].str.contains(brand)]
    equity_cp = equity_cp[equity_cp['time_period'].str.contains('weeks')]
    spend_cp = spend_cp[['time', 'Media_Spend']]
    merged_data = equity_cp.merge(
        spend_cp, how='left', left_on='time', right_on='time')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'})

    fig.add_trace(
        go.Bar(
            x=merged_data['time'],
            y=merged_data['Media_Spend_x'],
            name='Investment', showlegend=True
        ))

    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=merged_data['time'],
            y=merged_data['Framework - Awareness'], name='Awareness'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=merged_data['time'],
            y=merged_data['Framework - Saliency'],
            name='Saliency'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=merged_data['time'],
            y=merged_data['Framework - Affinity'],
            name='Affinity'),
        secondary_y=True)

    fig.update_layout(
        #margin=dict(l=2, r=1, t=55, b=2),
        autosize=True,
        xaxis=dict(title_text="Time"),
        yaxis=dict(title_text="Investment"),
        width=1000, title="{} Equity Build vs Media Spend".format(brand),
    )

    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Investment', secondary_y=False)
    fig.update_yaxes(title='Metrics', secondary_y=True)

    merged_data['time'] = pd.to_datetime(merged_data['time'])
    for d in merged_data['campaign_id'].unique():
        fig.add_annotation(text=d, x=merged_data['time'][merged_data['campaign_id'] == d].median(
        ), yref='paper', y=1.10, showarrow=False)

    fig.add_vline(x=pd.to_datetime('2022-07-04'), line_width=1,
                  line_dash='dash', line_color='gray')
    fig.add_vline(x=pd.to_datetime('2022-09-05'), line_width=1,
                  line_dash='dash', line_color='gray')

    # fig.add_annotation(x=0, y=1.1, xref='paper', yref='paper', text='             Pre             ', showarcolumn=False, font=dict(color='white'), bgcolor='blue')
    # fig.add_annotation(x=0.31, y=1.1, xref='paper', yref='paper', text='            During            ', showarcolumn=False, font=dict(color='blue'), bgcolor='gray')
    # fig.add_annotation(x=0.94, y=1.1, xref='paper', yref='paper', text='            Post            ', showarcolumn=False, font=dict(color='blue'), bgcolor='green')

    st.plotly_chart(fig)
    return merged_data


def filled_area_plot(dataframe, x, y, line_group, brand, type, filled_area_plot_type='AUC'):

    author_mapping = {
        'consumer ': 'Consumer',
        'deal_sites': 'Deal Sites',
        'influencer': 'Influencer',
        'news_outlet': 'News Outlet',
        'recipe_provider': 'Recipe Provider',
        'reviewer': 'Reviewer',
        'wellness_community': 'Wellness Community'
    }

    #dataframe['WeekCom'] = dataframe['WeekCom'].dt.strftime('%Y-%m-%d')
    dataframe_cp = dataframe.copy()
    dataframe_cp.rename(columns={type: 'Authors'}, inplace=True)
    type = 'Authors'
    dataframe_cp[type] = dataframe_cp[type].map(author_mapping)
    dataframe_cp = dataframe_cp[dataframe_cp['Brand'].str.contains(brand)]

    dataframe_grouped = dataframe_cp.groupby(
        ['Brand', type, 'WeekCom']).size().reset_index(name='counts')

    if filled_area_plot_type == "AUC":
        fig = px.area(dataframe_grouped, x=x, y=y,
                      color=type, line_group=line_group)
        fig.update_layout(
            #margin=dict(l=2, r=1, t=55, b=2),
            autosize=True,
            xaxis=dict(title_text="Time"),
            yaxis=dict(title_text="Ammount"),
        )
        st.plotly_chart(fig)
    elif filled_area_plot_type == "Line":
        fig = px.line(dataframe_grouped, x=x, y=y,
                      color=type, line_group=line_group)
        fig.update_layout(
            #margin=dict(l=20, r=20, t=20, b=20),
            autosize=True,
            xaxis=dict(title_text="Time"),
            yaxis=dict(title_text="Counts"),
        )
        st.plotly_chart(fig)

    return dataframe_grouped


def sentiment_barplot(dataframe, barplot_type, brand, option):

    brand_mapping = {
        'green_cuisine_campaign': 'Green Cuisine Campaign'
    }
    dataframe_cp = dataframe.copy()
    dataframe_cp['Brand'] = dataframe_cp['Brand'].map(brand_mapping)
    brand = brand_mapping[brand]

    # .dt.strftime('%Y-%m-%d')
    dataframe_cp['WeekCom'] = pd.to_datetime(dataframe_cp['WeekCom'])
    dataframe_cp = dataframe_cp[dataframe_cp['Brand'].str.contains(brand)]

    current_week = dataframe_cp['WeekCom'].max()

    mask_current_week = (
        dataframe_cp['WeekCom'] >= pd.to_datetime(str(current_week)))
    current_dataframe = dataframe_cp.loc[mask_current_week]
    current_dataframe['View'] = 'CW ({})'.format(
        current_week.strftime('%Y-%m-%d'))  # 'CW'

    if option == "LW":

        start_date = current_week - relativedelta(weeks=1)
        current_date = start_date + relativedelta(weeks=1)

        mask_last_option = (dataframe_cp['Created_Time'] >= pd.to_datetime(str(start_date))) & (
            dataframe_cp['Created_Time'] < pd.to_datetime(str(current_date)))
        option_dataframe = dataframe_cp.loc[mask_last_option]
        option_dataframe['View'] = '{} ({})'.format(
            option, start_date.strftime('%Y-%m-%d'))  # 'CW'

        final_view_dataframe = pd.concat(
            [option_dataframe, current_dataframe], axis=0)

    elif option == "L4W":
        last_view_ref = "L4W"
        current_view_ref = "CW"

        start_date = current_week - relativedelta(weeks=4)
        current_date = start_date + relativedelta(weeks=1)

        mask_last_option = (dataframe_cp['Created_Time'] >= pd.to_datetime(str(start_date))) & (
            dataframe_cp['Created_Time'] < pd.to_datetime(str(current_date)))
        option_dataframe = dataframe_cp.loc[mask_last_option]
        option_dataframe['View'] = '{} ({})'.format(
            option, start_date.strftime('%Y-%m-%d'))  # 'CW'

        final_view_dataframe = pd.concat(
            [option_dataframe, current_dataframe], axis=0)

    elif option == "P3M":
        last_view_ref = "P3M"
        current_view_ref = "CW"

        start_date = current_week - relativedelta(months=3)
        current_date = start_date + relativedelta(weeks=1)

        mask_last_option = (dataframe_cp['Created_Time'] >= pd.to_datetime(str(start_date))) & (
            dataframe_cp['Created_Time'] < pd.to_datetime(str(current_date)))
        option_dataframe = dataframe_cp.loc[mask_last_option]
        option_dataframe['View'] = '{} ({})'.format(
            option, start_date.strftime('%Y-%m-%d'))  # 'CW'

        final_view_dataframe = pd.concat(
            [option_dataframe, current_dataframe], axis=0)

    if barplot_type == "Absolute":

        final_view_dataframe_grouped = final_view_dataframe.groupby(
            ['Sentiment', 'Brand', 'View']).size().reset_index(name='counts')
        fig = go.Figure()

        fig.update_layout(
            autosize=True,
            xaxis=dict(title_text="Brand"),
            yaxis=dict(title_text="Ammount"),
            barmode="stack")

        colors = ['salmon', 'aqua', 'aquamarine']

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

    if barplot_type == "Absolute Percentage":

        sentiment = dataframe_cp.groupby(
            ['Sentiment', 'Brand']).size().reset_index(name='counts')

        fig = px.bar(sentiment, x="Brand", y='counts', color="Sentiment",
                     text_auto=True, color_discrete_sequence=['salmon', 'aqua', 'aquamarine'])
        fig.update_layout(autosize=True)
        st.plotly_chart(fig)

    if barplot_type == "Normalized":

        final_view_dataframe_grouped = dataframe_cp.groupby(
            ['Sentiment', 'Brand', 'View']).size().reset_index(name='counts')
        totals_by_brand = final_view_dataframe_grouped.groupby(['Brand', 'View'])[
            'counts'].sum()

        final_view_dataframe_grouped = (final_view_dataframe_grouped.merge(totals_by_brand, left_on=['Brand', 'View'], right_on=['Brand', 'View'], how='left').assign(
            new=lambda x: round(x['counts_x'].div(x['counts_y'])*100, 2)).reindex(columns=[*final_view_dataframe_grouped.columns]+['new']))

        final_view_dataframe_grouped.columns = [
            'Sentiment', 'Brand', 'View', 'Counts', 'Percentage']

        fig = go.Figure()

        fig.update_layout(
            autosize=True,
            xaxis=dict(title_text="Brands"),
            yaxis=dict(title_text="Percentage"),
            barmode="stack",
        )

        colors = ['salmon', 'aqua', 'aquamarine']

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

    if barplot_type == "Normalized Percentage":

        sentiment = dataframe_cp.groupby(
            ['Sentiment', 'Brand']).size().reset_index(name='counts')
        totals_by_brand = sentiment.groupby(['Brand'])['counts'].sum(
        ).reset_index()  # we can remove reset_index()

        sentiment = (sentiment.merge(totals_by_brand, left_on='Brand', right_on='Brand', how='left').assign(
            new=lambda x: round(x['counts_x'].div(x['counts_y'])*100, 2)).reindex(columns=[*sentiment.columns]+['new']))

        sentiment.columns = ['Sentiment', 'Brand', 'Counts', 'Percentage']

        fig = px.bar(sentiment, x="Brand", y='Percentage', color="Sentiment",
                     text_auto=True, color_discrete_sequence=['salmon', 'aqua', 'aquamarine'])
        fig.update_layout(autosize=True)
        st.plotly_chart(fig)

    return final_view_dataframe_grouped


############
### LOGO ###
############
column0_spacer1, column0_1, column0_spacer2, column0_2, column0_spacer3 = st.columns(
    (0.1, 5, .1, 1.3, .1))
with column0_1:
    st.image(Image.open(
        r"frontend\images\Picture1.png"))

with column0_2:
    st.text("")
    st.subheader(' ')

###############################
### DATA VIEWER AND FILTERS ###
###############################

column3_spacer1, column3_1, column3_2, column3_spacer2 = st.columns(
    (.2, 2, 5, .2))
with column3_2:
    st.markdown("")
    st.markdown("")

    # see_data = st.expander('You can click here to see the raw data first (first 1000 columns)')
    # with see_data:
    #     st.table(data=final_view_dataframe.reset_index(drop=True).head(1000))
with column3_1:

    option = st.selectbox(
        'Which time filters would you like to apply?',
        ['LW', 'L4W', 'P3M'])
st.text('')


####################
### DATA DISPLAY ###
####################
st.markdown("<h3 <span class = 'highlight  darkbluegrad'> 1 | How is the Campaign Spreading across Consumer Channels? </span> </h3>", unsafe_allow_html=True)

column4_1, column4_2, column4_spacer2, column4_3, column4_4 = st.columns(
    (1, 0.05, 0.05, 1, 0.05))

with column4_1:
    st.markdown('')
    st.markdown("<h5> Who is engaging with the Campaign? </h5>",
                unsafe_allow_html=True)

    # filled_area_plot_type = st.selectbox(
    #     "Select area plot type:",
    #     ['AUC','Line'])
    # type = st.selectbox(
    #     "Select mix:",
    #     ['journey_predictions','author_predictions','Message_Type','Gender','Age Range'])

    author_grouped = filled_area_plot(
        dataframe=final_view_dataframe,
        x="WeekCom",
        y="counts",
        line_group="Brand",
        filled_area_plot_type='AUC',
        brand='green_cuisine',
        type='author_predictions')
with column4_2:
    st.markdown('')

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        author_grouped.to_excel(writer, sheet_name='Author engagement')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Author engagement.xlsx",
        mime="application/vnd.ms-excel",
    )
with column4_3:
    st.markdown('')
    st.markdown("<h5> How is the Campaign Sentiment Trending? </h5>",
                unsafe_allow_html=True)
    # barplot_type = st.selectbox(
    #                 "Select barplot type:",
    #                 ['Normalized','Absolute'])
    res_sentiment = sentiment_barplot(
        final_view_dataframe, 'Absolute', "green_cuisine_campaign", option)

with column4_4:
    st.markdown('')

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        res_sentiment.to_excel(writer, sheet_name='Sentiment')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Sentiment.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h3 <span class = 'highlight  darkbluegrad'> 2 | How is the Campaign supporting Equity? </span> </h3>", unsafe_allow_html=True)
column12_spacer1, column12_1, column12_2, column12_spacer2 = st.columns(
    (1, 5, 0.05, 1))
with column12_1:
    affinity_spend_res = affinity_spend(equity, spend, 'green_cuisine')
with column12_2:
    st.markdown('')

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        affinity_spend_res.to_excel(writer, sheet_name='Affinity with Spend')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Affinity with Spend.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h5> What is the impact on equity score? </h5>",
            unsafe_allow_html=True)
column5_spacer1, column5_1, column5_2, column5_spacer3 = st.columns(
    (.2, 2, .1, .2))
with column5_1:
    equity_impact = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0, 1],
        index_col=[0, 1],
        sheet_name='2.2_ImpactEquity_TABLE')

    st.markdown(
        (equity_impact.style
         .background_gradient(
             cmap=cmap_red_green,
             subset=equity_impact.columns.get_loc_level('Delta', level=1)[0])
         .format("{:.0%}", subset=idx[:, idx[:, 'Delta']])
         .format("{:.1f}", subset=idx[:, idx[:, ['P3M', 'Camp']]])
         .set_properties(**{'text-align': 'center'})
         .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")

with column5_2:

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        equity_impact.to_excel(writer, sheet_name='Equity Impact')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Equity Impact.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h5> How is equity trending now (L4W vs Campaign Average)? </h5>",
            unsafe_allow_html=True)
column6_spacer1, column6_1, column6_2, column6_spacer3 = st.columns(
    (.2, 2, .1, .2))
with column6_1:
    equity_trend = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0, 1],
        index_col=[0, 1],
        sheet_name='2.3_EquityTrend_TABLE')

    st.markdown(
        (equity_trend.style
         .background_gradient(
             cmap=cmap_red_green,
             subset=idx[:, idx[:, 'Vs Camp']])  # equity_impact.columns.get_loc_level('Vs Camp',level=1)[0]
         .format("{:.0%}", subset=idx[:, idx[:, 'Vs Camp']])
         .format("{:.1f}", subset=idx[:, idx[:, ['CM']]])
         .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")
with column6_2:

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        equity_trend.to_excel(writer, sheet_name='Equity trend')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Equity trend.xlsx",
        mime="application/vnd.ms-excel",
    )
    # chosen_cols = [col for col in list(equity_trend.columns) for choice in choices if choice in col]
    # st.markdown(equity_trend.astype(str).style.applymap(lambda x: f"background-color: {'yellow' if isinstance(x,str) else cmap_red_green}"))
st.markdown("<h3 <span class = 'highlight darkbluegrad'> 3 | How are Efficiency & Financial KPIs tracking? </span> </h3>", unsafe_allow_html=True)
column6_spacer1, column6_1, column6__spacer2, column6__2, column6__spacer3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))

st.markdown("<h3 <span class = 'highlight darkbluegrad'> 4 | How well are we Penetrating Target Demand Spaces? </span> </h3>", unsafe_allow_html=True)
st.markdown("<h5> How well is W2tPA helping Green Cuisine Penetrate Target Moments? </h5>",
            unsafe_allow_html=True)

column7_spacer1, column7_1, column7_2, column7_spacer3 = st.columns(
    (0.5, 3.5, .1, 0.5))
with column7_1:
    st.markdown("\n")

    target_demands = pd.read_excel(
        file_path,
        header=0,
        engine='openpyxl',
        sheet_name='4.2_TargetMoments_TABLE')

    st.markdown(
        (target_demands.style
         .background_gradient(
             subset=["J '% v L6M", "A '% v L6M", "S '% v L6M"],
             cmap=cmap_red_green,
             axis=None)
         .format(
             {
                 "J '% v L6M": "{:.0%}",
                 "A '% v L6M": "{:.0%}",
                 "S '% v L6M": "{:.0%}",
                 'Av. L6M': '{:.0f}',
                 'Ratio': '{:.0%}'
             }
         )
            .hide_index()
            .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")

with column7_2:
    st.markdown("\n")

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        target_demands.to_excel(writer, sheet_name='Target Demands')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="target_demands.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h5> Green Cuisine Share of demand space | Consciously Reducing Meat </h5>",
            unsafe_allow_html=True)
column13_spacer1, column13_1, column13_2, column13_spacer3 = st.columns(
    (0.5, 3.5, .1, 0.5))
with column13_1:
    st.markdown("\n")

    target_demands_gc = pd.read_excel(
        file_path,
        header=0,
        index_col=[0],
        engine='openpyxl',
        sheet_name='4.4_GC_Share_Space')

    st.markdown(
        (target_demands_gc.style
         .background_gradient(
             subset=["J '% v L6M", "A '% v L6M", "S '% v L6M"],
             cmap=cmap_red_green,
             axis=None)
         .format(
             {
                 'Av. L6M': '{:.0f}',
                 'July': '{:.0f}',
                 'Aug': '{:.0f}',
                 'Sep': '{:.0f}',
                 "J '% v L6M": "{:.0%}",
                 "A '% v L6M": "{:.0%}",
                 "S '% v L6M": "{:.0%}",
             }
         )
            .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")

with column13_2:
    st.markdown("\n")

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        target_demands_gc.to_excel(
            writer, sheet_name='Target Demands GC Share')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="target_demands_gc_share.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h5> Sentiment of Green Cuisine Share in demand space | Consciously Reducing Meat </h5>",
            unsafe_allow_html=True)
column14_spacer1, column14_1, column14_2, column14_spacer3 = st.columns(
    (0.5, 3.5, .1, 0.5))
with column14_1:
    st.markdown("\n")

    target_demands_sentiment_gc = pd.read_excel(
        file_path,
        header=0,
        index_col=[0, 1],
        engine='openpyxl',
        sheet_name='4.5_GC_Sentiment_Space')

    st.markdown(
        (target_demands_sentiment_gc.style
         .background_gradient(
             cmap=cmap_red_green,
             axis=None)
         .format(
             lambda x: '{:.0%}'.format(float(x))
         )
            .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")

with column14_2:
    st.markdown("\n")

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        target_demands_sentiment_gc.to_excel(
            writer, sheet_name='Target Demands Sentiment GC')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="target_demands_sentiment_gc.xlsx",
        mime="application/vnd.ms-excel",
    )


st.markdown("<h3 <span class = 'highlight darkbluegrad'> 5 | How well are 'Welcome to the Plant Age' Creatives Resonating? </span> </h3>", unsafe_allow_html=True)
st.markdown("<h5> How is it impacting STAR Perceptions? </h5>",
            unsafe_allow_html=True)

column11_spacer1, column11_1, column11_spacer2, column11_2, column11_3 = st.columns(
    (0.5, 0.6, 0.2, 5, 0.5))
with column11_1:
    st.markdown("\n")
    st.markdown("\n")

    st.markdown(
        """
    <style>
        .gc {
            display: flex;
            align-items:center;
            margin:auto;
            max-width: 80%;
            max-height:50%;
            padding-top:2em;
        }
    </style>

    <div >
        <img class="gc" src="https://i.ibb.co/w6Q2j6Z/GC.png" ></div>
    """, unsafe_allow_html=True
    )

with column11_2:
    st.markdown("\n")
    st.markdown("\n")
    star = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=0,
        sheet_name='5.2_W2tPA_STAR_TABLE')

    idx = pd.IndexSlice
    st.markdown((star.style
                 .background_gradient(
                     subset=['C3M v P3M', 'Sep v P3M', 'Aug v P3M'],
                     cmap=cmap_red_green,
                     axis=0)
                 .format({
                     'Jul': '{:.0f}',
                     'P3M': '{:.0f}',
                     'Aug': '{:.0f}',
                     'Sep': '{:.0f}',
                     'C3M': '{:.0f}',
                     'Aug v P3M': lambda x: "+ {:.0f} pts".format(x) if x > 0.444 else "- {:.0f} pts".format(abs(x)) if x < -0.4999 else "{:.0f} pts".format(abs(x)),
                     'C3M v P3M': lambda x: "+ {:.0f} pts".format(x) if x > 0.444 else "- {:.0f} pts".format(abs(x)) if x < -0.4999 else "{:.0f} pts".format(abs(x)),
                     'Sep v P3M': lambda x: "+ {:.0f} pts".format(x) if x > 0.444 else "- {:.0f} pts".format(abs(x)) if x < -0.4999 else "{:.0f} pts".format(abs(x)),

                 }
                 )
                 .set_table_styles(styles, overwrite=False)
                 .hide_index()
                 .set_properties(**{'text-align': 'center'}).to_html()), unsafe_allow_html=True

                )
    st.markdown("\n")
    st.markdown("\n")
with column11_3:
    st.markdown("\n")
    st.markdown("\n")
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        star.to_excel(writer, sheet_name='Star')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Star.xlsx",
        mime="application/vnd.ms-excel",
    )

st.markdown("<h5> How is Sentiment Tracking? </h5>", unsafe_allow_html=True)
column8_spacer1, column8_1, column8_2, column8_spacer2, column8_3, column8_4, column8_spacer3 = st.columns(
    (.2, 3, .1, .4, 3, .1, .2))
with column8_1:

    sentiment = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0],
        index_col=[0, 1],
        sheet_name='5.2_W2tPA_SENT_TABLE')

    st.markdown((sentiment.style
                 .format('{:.0%}', na_rep="-"
                         )
                 .background_gradient(
                     cmap=cmap_red_green,
                     axis=None
                 )
                 .set_properties(**{'text-align': 'center'})
                 .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
                )
    st.markdown("\n")
    st.markdown("\n")
with column8_2:

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        sentiment.to_excel(writer, sheet_name='Sentiment Tracking')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Sentiment Tracking.xlsx",
        mime="application/vnd.ms-excel",
    )

with column8_3:
    sentiment_delta = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0],
        index_col=[0, 1],
        sheet_name='5.2.1_W2tPA_SENT_DELTA_TABLE')

    st.markdown((sentiment_delta.style
                 .format('{:.0%}')
                 .background_gradient(
                     cmap=cmap_red_green,
                     axis=None
                 )
                 .format(lambda x: "🟰 {:.0%}".format(x) if (x > -0.0144) and (x < 0.0144) else "🔺 {:.0%}".format(x) if (x > 0.0144) else "🔻 {:.0%}".format(x))
                 .applymap(lambda v: 'color:transparent; text-shadow: 0 0 0 limegreen;' if (v > 0.0144) else 'color:transparent; text-shadow: 0 0 0 red;' if (v < -0.0144) else None)
                 .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
                )
    st.markdown("\n")
    st.markdown("\n")

with column8_4:

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        sentiment_delta.to_excel(writer, sheet_name='Sentiment Tracking Delta')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Sentiment Tracking Delta.xlsx",
        mime="application/vnd.ms-excel",
    )


st.markdown("<h3 <span class = 'highlight darkbluegrad'> 6 | How is Risk Tracking? </span> </h3>",
            unsafe_allow_html=True)
column9_spacer1, column9_1, column9_spacer2, column9_2, column9_3, column9_spacer3, column9_4, column9_5, column9_spacer4 = st.columns(
    (.1, 1, .1, 3.5, .1, 0.2, 3.5, .1, 0.1)
)

with column9_1:
    st.markdown("\n")

    st.markdown(
        """
    <style>
        .w2tpa {
            display: flex;
            align-items:center;
            margin:auto;
            max-width: 80%;
            max-height:20%;
            padding-top:2em;
        }
    </style>

    <div >
        <img class = "w2tpa" src="https://i.ibb.co/gwQYTtB/Wt2PA.png" ></div>
    """, unsafe_allow_html=True
    )

with column9_2:
    st.markdown("\n")
    st.markdown("<h5> What is the News picking up & Risk? </h5>",
                unsafe_allow_html=True)

    risk_tracking_news = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0],
        index_col=0,
        sheet_name='6.2_RiskTracking_TABLE')

    st.markdown((risk_tracking_news.style
                 .format("{:.0%}", subset=pd.IndexSlice[['% Negative'], :])
                 .format("{:.0f}", subset=pd.IndexSlice[['News', 'Negative News'], :])
                 .background_gradient(
                     subset=pd.IndexSlice[['% Negative'], :],
                     cmap=cmap_red_white,
                     axis=1
                 )
                 .set_properties(**{'text-align': 'center'})
                 .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
                )
    t = Tweet(
        "https://twitter.com/fleurmeston/status/1556245399379353600").component()

with column9_3:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        risk_tracking_news.to_excel(writer, sheet_name='News Risk Tracking')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="News Risk Tracking.xlsx",
        mime="application/vnd.ms-excel",
    )


with column9_4:
    st.markdown("\n")
    st.markdown("<h5> Influencer Pick Up </h5>", unsafe_allow_html=True)

    risk_tracking_inf = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0],
        index_col=[0],
        sheet_name='6.3_RiskTracking_Inf_Table')

    st.markdown((risk_tracking_inf.style
                 .format("{:.0%}", subset=pd.IndexSlice[['% Negative'], :])
                 .format("{:.0f}", subset=pd.IndexSlice[['Influencers', 'Negative Influencers'], :])
                 .background_gradient(
                     subset=pd.IndexSlice[['% Negative'], :],
                     cmap=cmap_green_red,
                     axis=1
                 )
                 .set_properties(**{'text-align': 'center'})
                 .set_table_styles(styles, overwrite=False).to_html()), unsafe_allow_html=True
                )
    t = Tweet(
        "https://twitter.com/Steveredwolf/status/1550564202267410434").component()

with column9_5:
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        risk_tracking_inf.to_excel(
            writer, sheet_name='Influencer Risk Tracking')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Influencer Risk Tracking.xlsx",
        mime="application/vnd.ms-excel",
    )


st.markdown("<h3 <span class = 'highlight darkbluegrad'> 7 | Who is winning in Equity during W2tPA Campaign? </span> </h3>", unsafe_allow_html=True)
column10_spacer1, column10_1, column10_spacer2, column10_2, column10_spacer3, column10_3, ro10_spacer4 = st.columns(
    (0.1, 0.5, .1, 3.5, 0.1, 0.05, 0.2))

with column10_3:
    st.markdown("\n")
    winning_equity = pd.read_excel(
        file_path,
        engine='openpyxl',
        header=[0, 1],
        index_col=0,
        sheet_name='7.2_W2tPA_Equity_TABLE')

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        winning_equity.to_excel(writer, sheet_name='Winning equity')
    writer.save()
    st.download_button(
        label="⭳",
        data=buffer,
        file_name="Winning equity.xlsx",
        mime="application/vnd.ms-excel",
    )
with column10_2:
    st.markdown("\n")
    st.markdown(winning_equity.style
                .format("{:.1f}")
                .format("{:.0%}", subset=idx[:, idx[:, 'Delta']])
                .background_gradient(
                    subset=winning_equity.columns.get_loc_level('Delta', level=1)[
                        0],
                    cmap=cmap_green_red,
                    axis=1
                )
                .set_properties(**{'text-align': 'center'}).set_table_styles(styles, overwrite=False).to_html(), unsafe_allow_html=True)
    st.markdown("\n")
    st.markdown("\n")

with column10_1:
    st.markdown(
        """
    <style>
        .gc {
            display: flex;
            align-items:center;
            margin:auto;
            max-width: 80%;
            max-height:20%;
            padding-top:2em;
        }
    </style>

    <div >
        <img class="gc" src="https://i.ibb.co/w6Q2j6Z/GC.png" ></div>
    """, unsafe_allow_html=True
    )

####################################### DATAFRAME DOWNLOADER #######################################
st.markdown("<h3 <span class = 'highlight darkbluegrad'> Download Tables </span> </h3>",
            unsafe_allow_html=True)
options = st.multiselect(
    'Which Tables would you like to download?',
    ['Impact on equity score', 'Equity trend', 'Financial KPI tracking', 'W2tPA in Target Moments', 'Impact on STAR perceptions', 'Sentiment Tracking', 'News Pick Up', 'Influencer Pick Up', 'Winners in Equity during W2tPAa'])

dataframe_mapping = {
    'Impact on equity score': equity_impact,
    'Equity trend': equity_trend,
    'Financial KPI tracking': None,
    'W2tPA in Target Moments': target_demands,
    'Impact on STAR perceptions': star,
    'Sentiment Tracking': sentiment,
    'News Pick Up': risk_tracking_news,
    'Influencer Pick Up': risk_tracking_inf,
    'Winners in Equity during W2tPAa': winning_equity
}

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:

    list_of_dataframes = [dataframe_mapping[option] for option in options]

    for sheet_name, df in zip(options, list_of_dataframes):
        # Write each dataframe to a different worksheet.
        df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    st.download_button(
        label="Download Excel worksheets",
        data=buffer,
        file_name="data.xlsx",
        mime="application/vnd.ms-excel",
    )
