import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta

file_path = r"C:\Users\RafaelOliveira\Brand Delta\Nomad - General\Green Cuisine Campaign\03_FrontEnd_GreenCuisine\FrontEnd_GreenCuisine_Dataset_v7.xlsx"
temporary_dataframe = pd.read_excel(file_path,engine='openpyxl')
xls = pd.ExcelFile(file_path)
dataframe = {}
sheets = []
for sheet_name in xls.sheet_names:
    dataframe[sheet_name] = xls.parse(sheet_name)
    sheets.append(sheet_name)
raw_dataframe = dataframe['1_ConsumerChannel_RAW_DATA']

raw_dataframe['WeekCom'] = pd.to_datetime(
    pd.to_datetime(
        raw_dataframe['Created_Time']-pd.to_timedelta(raw_dataframe['Created_Time'].dt.weekday,unit='D')).apply(lambda x: x.strftime('%Y-%m-%d')
    )
)


def make_pivots(
    raw_dataframe,
    aggfunc,
    timeframe,
    index='WeekCom',
    week_commencing_col='WeekCom',
    date_col='Created_Time',
    values='Values',
    columns=None,
    market=['uk']
):
    #Subset dataframe based on other columns
    dataframe = raw_dataframe.copy()
    dataframe = dataframe[(dataframe['market'].isin(market))]
    dataframe['Values'] = 1 #create a temporary column just so that we can use it as counts for the categorical groupby such as journey, author, sentiment, etc 

    #Change format of daily dates to remove hours, minutes and seconds
    dataframe[date_col] = pd.to_datetime(
        pd.to_datetime(
            dataframe[date_col]).apply(lambda x: x.strftime('%Y-%m-%d')
        )
    )

    #Create week commencing column on dataframe. Get week commencing, transform the result into datetime, extract just the year, month and day from this newly created datetime and finally convert the strftime to datetime again. 
    dataframe[week_commencing_col] = pd.to_datetime(pd.to_datetime(
        dataframe[date_col]-pd.to_timedelta(dataframe[date_col].dt.weekday,unit='D')).apply(lambda x: x.strftime('%Y-%m-%d')
    ))

    # Get most recent daily date and weekly date from dataframe -> this will be used for the views such as P3M, L4W, LW, etc...
    most_recent_date = dataframe[date_col].max()
    most_recent_weekcom = pd.to_datetime(most_recent_date - timedelta(days=most_recent_date.weekday()))

    #Pass to date string
    # most_recent_date = most_recent_date.strftime('%Y-%m-%d')
    # most_recent_weekcom = most_recent_weekcom.strftime('%Y-%m-%d')

    if timeframe == 'weekly':
        pivoted_dataframe = dataframe.pivot_table(
            values=values,
            index='WeekCom',
            columns=columns,
            aggfunc=aggfunc,
            margins=True,
            margins_name='Total',
            fill_value=0)
    
    elif timeframe == 'daily':
        pivoted_dataframe = dataframe.pivot_table(
            values=values,
            index='Created_Time',
            columns=columns,
            aggfunc=aggfunc,
            margins=True,
            margins_name='Total',
            fill_value=0)

    elif timeframe == 'P3M':
        #Substract 3 months from most latest date
        P3M_date = most_recent_weekcom - relativedelta(months=3) 

        #Subset new dataframe with just those dates
        P3M_dataframe = dataframe[(dataframe[week_commencing_col] <= most_recent_weekcom) & (dataframe[week_commencing_col] >= P3M_date)]

        #Create pivot table based on that new dataframe
        pivoted_dataframe = P3M_dataframe.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            margins=True,
            margins_name='Total',
            fill_value=0)

    elif timeframe == 'L4W':
        #Subtract 4 weeks from most recent date
        L4W_date = most_recent_weekcom - relativedelta(weeks=3) 

        #Subset new dataframe with just those dates
        L4W_dataframe = dataframe[(dataframe[week_commencing_col] <= most_recent_weekcom) & (dataframe[week_commencing_col] >= L4W_date)]

        #Create pivot table based on that new dataframe
        pivoted_dataframe = L4W_dataframe.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            margins=True,
            margins_name='Total',
            fill_value=0)

    elif timeframe == 'LW':
        #Substract 3 months from most recent date
        LW_date = most_recent_date - relativedelta(weeks=2) 

        #Subset new dataframe with just those dates
        LW_dataframe = dataframe[(dataframe[week_commencing_col] <= most_recent_date) & (dataframe[week_commencing_col] >= LW_date)]

        #Create pivot table based on that new dataframe
        pivoted_dataframe = LW_dataframe.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            margins=True,
            margins_name='Total',
            fill_value=0)

    #Change column types based on the aggregate function used 
    if aggfunc == 'count':
        pivoted_dataframe = pivoted_dataframe.astype('int64')

    return pivoted_dataframe

make_pivots(
    raw_dataframe=raw_dataframe,
    index=['WeekCom','Brand'],
    columns=['Sentiment','Gender'],
    aggfunc='count',
    timeframe='LW')



## tests


def sentiment_barplot(dataframe,barplot_type,brand,option,quick_filter=False):
    #.dt.strftime('%Y-%m-%d')
    dataframe['WeekCom'] = pd.to_datetime(dataframe['WeekCom'])
    dataframe = dataframe[dataframe['Brand'].str.contains(brand)]

    start_date = dataframe['WeekCom'].min()
    end_date = dataframe['WeekCom'].max()

    st.write(start_date)
    st.write(end_date)

    mask = (dataframe['WeekCom'] > pd.to_datetime(str(start_date))) & (dataframe['WeekCom'] <= pd.to_datetime(str(end_date)))

    masked_df_current = dataframe.loc[mask]

    """
    Grab current counts of sentiment of current week
    Grab sentiment of the week relative to the current week (L4W, LW, P3M)

    Plot that way
    
    """

    if option == "LW":
        last_view_ref = "LW"
        current_view_ref = "CW"

        starting_days = datetime.timedelta(14)
        midpoint_days = datetime.timedelta(7)

        end_date_current = end_date
        start_date_current = end_date-midpoint_days + datetime.timedelta(1)

        start_date_last = end_date_current - starting_days 
        end_date_last = start_date_last + midpoint_days
        
        mask_last = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_last))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_last)))
        masked_df_last = dataframe.loc[mask_last]
        masked_df_last['View'] = last_view_ref

        mask_current = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_current))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_current)))
        masked_df_current = dataframe.loc[mask_current]
        masked_df_current['View'] = current_view_ref

        final_view_dataframe = pd.concat([masked_df_last,masked_df_current],axis=0)

    elif  option == "L4W":
        last_view_ref = "L4W"
        current_view_ref = "CW"

        starting_days = datetime.timedelta(14)
        midpoint_days = datetime.timedelta(7)

        end_date_current = end_date
        start_date_current = end_date-midpoint_days + datetime.timedelta(1)

        start_date_last = end_date_current - starting_days 
        end_date_last = start_date_last + midpoint_days
        
        mask_last = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_last))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_last)))
        masked_df_last = dataframe.loc[mask_last]
        masked_df_last['View'] = last_view_ref

        mask_current = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_current))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_current)))
        masked_df_current = dataframe.loc[mask_current]
        masked_df_current['View'] = current_view_ref

        final_view_dataframe = pd.concat([masked_df_last,masked_df_current],axis=0)

    elif  option == "P3M":
        last_view_ref = "P3M"
        current_view_ref = "CW"

        starting_days = datetime.timedelta(14)
        midpoint_days = datetime.timedelta(7)

        end_date_current = end_date
        start_date_current = end_date-midpoint_days + datetime.timedelta(1)

        start_date_last = end_date_current - starting_days 
        end_date_last = start_date_last + midpoint_days
        
        mask_last = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_last))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_last)))
        masked_df_last = dataframe.loc[mask_last]
        masked_df_last['View'] = last_view_ref

        mask_current = (dataframe['Created_Time'] > pd.to_datetime(str(start_date_current))) & (dataframe['Created_Time'] <= pd.to_datetime(str(end_date_current)))
        masked_df_current = dataframe.loc[mask_current]
        masked_df_current['View'] = current_view_ref

        final_view_dataframe = pd.concat([masked_df_last,masked_df_current],axis=0)

    if barplot_type == "Absolute":

        final_view_dataframe_grouped = final_view_dataframe.groupby(['Sentiment','Brand','View']).size().reset_index(name='counts')

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

    if barplot_type == "Absolute Percentage":

        sentiment = dataframe.groupby(['Sentiment','Brand']).size().reset_index(name='counts')

        fig = px.bar(sentiment, x="Brand", y='counts',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
        fig.update_layout(autosize=True)
        st.plotly_chart(fig)

    if barplot_type == "Normalized":

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

    if barplot_type == "Normalized Percentage":

        sentiment = dataframe.groupby(['Sentiment','Brand']).size().reset_index(name='counts')
        totals_by_brand = sentiment.groupby(['Brand'])['counts'].sum().reset_index() #we can remove reset_index()

        sentiment = (sentiment.merge(totals_by_brand,left_on='Brand',right_on='Brand',how='left').assign(new=lambda x:round(x['counts_x'].div(x['counts_y'])*100,2)).reindex(columns=[*sentiment.columns]+['new']))

        sentiment.columns = ['Sentiment','Brand','Counts','Percentage']

        fig = px.bar(sentiment, x="Brand", y='Percentage',color="Sentiment", text_auto=True, color_discrete_sequence=['salmon','aqua','aquamarine'])
        fig.update_layout(autosize=True)
        st.plotly_chart(fig)
