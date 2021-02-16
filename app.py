
import calendar
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import st_rerun_patch
import plotly.express as px
import plotly.io as pio
import session_state as state
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from predict import train_and_test_predictive_model


pio.templates.default = "simple_white"
sns.set_style("white")
NUM_SECTIONS = 10

session_state = state.get(sections=[False for i in range(NUM_SECTIONS)],
                          show_workhours=False, show_windspeed=False,
                          show_leakage=False, show_excel=False)



def get_data(df):
    return df[df['_data'] == 'train']


dfs = {}
for name in ['train', 'test']:
    df = pd.read_csv('data/%s.csv' % name)
    df['_data'] = name
    dfs[name] = df

# combine train and test data into one df
df = dfs['train'].append(dfs['test'])

# lowercase column names
df.columns = map(str.lower, df.columns)
# parse datetime colum & add new time related columns
dt = pd.DatetimeIndex(df['datetime']).tz_localize(tz='US/Eastern', ambiguous=True)
df.set_index(dt, inplace=True)

df_toprint = df.copy()

df['date'] = dt.date
df['day'] = dt.day
df['month'] = dt.month
df['year'] = dt.year
df['hour'] = dt.hour
df['dow'] = dt.dayofweek
df['woy'] = dt.weekofyear

# interpolate weather, atemp, humidity
df["weather"] = df["weather"].interpolate(method='time').apply(np.round)
df["temp"] = df["temp"].interpolate(method='time')
df["atemp"] = df["atemp"].interpolate(method='time')
df["humidity"] = df["humidity"].interpolate(method='time').apply(np.round)
df["windspeed"] = df["windspeed"].interpolate(method='time')

# example month
day_start = pd.datetime(2011, 6, 1)
day_end = day_start + pd.offsets.DateOffset(days=19)
rng = pd.date_range(day_start, day_end, freq="H").tz_localize('US/Eastern')


if not session_state.sections[6]:
    st.image('assets/logo_small.png')
    st.title('Bike Sharing')
    st.write('_by Piero Donaggio - Head of Data Science Practice @ FibonacciLab_')
    st.write('''
        ## What is all about?

        With **bike-sharing systems**, people are able to rent a bike from one location
        and return it to a different place on an as-needed basis.
        Currently, there are over 500 bike-sharing programs around the world.

        Today, there exists great interest in these systems due to their important role
        in traffic, environmental and health issues.

        In this example, we are interested in combining historical usage patterns
        with weather data in order to **forecast bike rental demand** in the Capital Bike Share
        program in Washington, D.C.
        ''')

    st.image('assets/bikes.png')

    st.info('''**Hint**

You've got it right, young data scientist. ⛩

This is not dissimilar from what we should do to analyse and predict
demand data for any other kind of business.
    ''')

    if not session_state.sections[0] and st.button('Continue', key='cont-one'):
        session_state.sections[0] = True

    if session_state.sections[0]:

        st.write('''
            ## What data do we have?

            Let's see!
            ''')

    if session_state.sections[0]:
        st.dataframe(df_toprint.iloc[:, 1:].head(50))

    if session_state.sections[0]:
        st.write('''
            This is just a small portion of the dataset. As you can see, **we often start from a
            simple table**, like an Excel table.

            In reality, most of the time we have a lot of tables! But at the end we always try
            to reduce and aggregate them into one, by exploiting related information.

            ❤️ _Data scientists love big tables!_ ❤️

            Each row shows rentals data collected during one hour.
            In total, **we have about two years of data** from 2011 to 2012.

            ### What we want to predict?

            We are interested in predicting the number of total rents (`count` column) or,
            alternatively, the sum of the number of registered user rentals (`registered` column)
            and the number of casual user rentals (`casual` column).

            So, for instance, we can have 15 casual users and 67 registered users who rented
            a bicycle on January 1st at 5PM. And so on!

            Here you can see the trend of rented bikes during the first days of the year.
            We call this type of data _time series_ data, because data are ordered by date and time.
            ''')

        st.line_chart(df.sort_index()['count'].iloc[:600])

        st.write('''
            If you look at the chart, you can see that part of data are missing.
            This is because **we will try to predict the from day 20 of each month up to the end
            of the month**. It is like if we are predicting something in the future, that we do
            not know until it happens. We can only make a **guess**.
            ''')

        if not session_state.sections[1] and st.button('Continue', key='cont-two'):
            session_state.sections[1] = True

    ## =========== SECTION 2 ============= ##
    if session_state.sections[1]:
        st.write('''
            ## What can we use to make predictions?

            Apart from date and number of total rents, we are also provided with other
            useful information.

            Let's take a look.

            > These information are all related to **time**:

            > * season (spring, summer, fall, winter)
            * holiday — whether the day is considered a holiday
            * workingday — whether the day is neither a weekend nor holiday

            > The following information are instead related to **weather**:

            > * weather — can be:
                * 1: Clear, nice weather
                * 2: Cloudy and misty
                * 3: Light snow or rain
                * 4: Heavy weather
            * temp — temperature in Celsius
            * atemp — “feels like” temperature in Celsius
            * humidity — relative humidity
            * windspeed — wind speed
            ''')

        st.header('')
        st.warning('''**Note**

In general, we need to ask a lot of question at this stage. If, for example, these
were data extracted from a MES software, we should ask our client what these columns means: how
and where are measured? who is measuring them? with what instrument? how often? etc...
It is paramount that we precisely understand the data before to start any kind of analysis.
Making data science has a lot to do with making the right questions...
        ''')

        st.header('')
        st.image('assets/bike_winter.jpg',
                 width=500,
                 caption='We can imagine that cycling in winter is a bit less comfortable than in summer...'
                         'Will it be confirmed by data??')

        if not session_state.sections[2] and st.button('Continue', key='cont-three'):
            session_state.sections[2] = True

    ## =========== SECTION 3 ============= ##
    if session_state.sections[2]:
        st.write('''
            ## Time to ride! ⏱

            Most of the time **data scientists use nice visualizations** to understand the data first.
            This is precisely what we are going to do.

            For example, we can easily see the effect of seasonality on rentals:
            ''')

        option = st.radio('', ('Sum', 'Average'))

        if option == 'Sum':
            month_group = get_data(df).resample('1M').agg({'count': 'sum', 'year': 'first', 'month': 'first'})
            month_group['month'] = month_group['month'].apply(lambda x: calendar.month_name[x])
            month_group['year'] = month_group['year'].astype('category')
            st.write(px.bar(month_group, x='month', y='count', color='year', barmode='group', labels={'count':'rentals'}))
        elif option == 'Average':
            dfs = df
            dfs['month'] = df['month'].apply(lambda x: calendar.month_name[x])
            st.write(px.box(dfs, x='month', y='count', color='year', labels={'count':'rentals'}))

        st.write('''
            Data confirms our simple idea: people tend to rent bikes more in summer ☀️ than in winter ❄️.

            There is already something more we can say just by looking at the graph above: **the
            bike sharing business has grown significantly** in a year!

            What about the hours of the day? Are people cycling more in the morning or in the afternoon?
            ''')

        hour_group = df.groupby('hour').agg({'count': 'mean', 'hour': 'first'})
        st.plotly_chart(px.bar(hour_group, y="count", x="hour", labels={'count':'average rentals'}))
        st.markdown('''
                The hourly plot shows a peak at 8 am and one at 5 pm.

                We could stop here, but the _good data scientist_ is always questioning his or her results 🧐.
                This is **one of the most important skills he or she must have**!

                This graph in fact may not tell us all the truth, as it always happens when we try to
                condense a lot of information in a single graph. In other words, **there is always a trade-off
                between _intelligibility and completeness_ when it comes to data**.

                Most of the time it is important not to stop at the evidence, but try to deep dive into
                the data.

                > _Can you guess which information is "hidden" in this graph?_
                ''')

        hour_text = st.empty()
        hour_button = st.empty()

        if not session_state.show_workhours and hour_button.button('Show me!'):
            session_state.show_workhours = True

        if not session_state.show_workhours:

            hour_text.markdown('''
                Click the button to discover it...
                ''')
        else:
            hour_button.markdown('')
            hour_text.markdown('')
            hour_group = df.groupby(['workingday', 'hour']).agg({'count': 'mean', 'hour': 'first', 'workingday': 'first'})
            hour_group['workingday'] = hour_group['workingday'].astype('category').cat.rename_categories(['Holiday', 'Working'])
            st.plotly_chart(px.bar(hour_group, y="count", x="hour", labels={'count':'average rentals'}, color="workingday", barmode='group'))
            st.markdown('''
                ### Of course! 🕵

                There is indeed a peak at 8 am and at 5 pm, but only on standard working days!
                Probably most users use the bikes to get to work or school.

                On holidays instead, people tend to use the bikes in the central hours of the day.

                Sometimes we need to **dig deeper**. Remember the distinction between "casual" and
                "registered" users? Exploring the data we can see for example that casual users
                do not use the bicycles for going to work or school.
                ''')

            dfmelt = df.melt(id_vars=['hour', 'workingday'], value_vars=['casual', 'registered'], value_name='avg rentals')
            hour_group = dfmelt.groupby(['variable', 'workingday', 'hour']).agg({'avg rentals': 'mean', 'hour': 'first', 'workingday': 'first', 'variable': 'first'})
            hour_group['workingday'] = hour_group['workingday'].astype('category').cat.rename_categories(['Holiday', 'Working'])
            st.plotly_chart(px.bar(hour_group, y="avg rentals", x="hour", color="variable",
                                   labels={'count':'average rentals'},
                                   facet_col='workingday', barmode='group'))

            st.markdown('''
                Up to now, you have seen just **one possible way to "slice" data**.

                What about the effect of seasonality? Does it influence when we use the bike during the day?

                A lot of questions can be answered just by carefully inspecting data.

                But **data are tricky and need to be handled with care**.
                You often need acumen, curiosity, domain experience and math skills all at the same time,
                to be a good data scientist.
                ''')

            if not session_state.sections[3] and st.button('Continue', key='cont-four'):
                session_state.sections[3] = True

    if session_state.sections[3]:
        st.markdown('''
            ## Cyclin' in the rain ☔️

            Weather conditions are typically an important factor when it comes to use the bicycle rather than the car,
            and should be related to seasonality as well.

            We can conduct a similar visual data exploration for temperature, humidity or windspeed:
            ''')

        fig = make_subplots(rows=2, cols=2, shared_yaxes=True)
        for ci, var in enumerate(['temp', 'humidity', 'windspeed']):
            dfmelt = get_data(df).melt(id_vars=[var], value_vars=['casual', 'registered'], value_name='count')
            bins = pd.cut(dfmelt[var], bins=20)
            tgroup = dfmelt.groupby(['variable', bins]).agg({'count': 'mean', var: 'mean', 'variable': 'first'})
            casual = tgroup[tgroup.variable == 'casual']
            registered = tgroup[tgroup.variable == 'registered']
            fig.add_trace(go.Bar(name='casual', y=casual['count'],
                                 x=casual[var], legendgroup='group1',
                                 showlegend=True if ci == 0 else False,
                                 marker_color=px.colors.qualitative.D3[0]),
                          row=ci//2+1, col=ci%2+1)
            fig.add_trace(go.Bar(name='registered', y=registered['count'],
                                 x=registered[var], legendgroup='group2',
                                 showlegend=True if ci == 0 else False,
                                 marker_color=px.colors.qualitative.D3[1]),
                          row=ci//2+1, col=ci%2+1)
            fig.update_layout(barmode='stack', legend_title_text=' ')
            fig.update_xaxes(title_text=var, row=ci//2+1, col=ci%2+1)
        fig.update_yaxes(title_text='average rentals', row=1, col=1)
        fig.update_yaxes(title_text='average rentals', row=2, col=1)
        fig.update_layout(height=700)
        st.plotly_chart(fig)

        st.markdown('''
                It can be immediately inferred that:

                * Higher the temperature, more rentals. This makes sense.
                People would avoid biking when the weather is too cold.
                * Higher the humidity, fewer rentals.
                This also makes sense as humidity is roughly inversely proportional to temperature.
                * A medium windspeed is what bikers seem to prefer.
                However this does not seem to be a strong factor.

                Also notice how there are very few points with extremely low humidity (<20%) because this is a
                natural condition that is very rare outdoor (and harmful!).

                But **a good data scientist would become immediately suspicious when looking at these graphs**.

                > _Can you see why?_
            ''')

        windspeed_text = st.empty()
        wind_button = st.empty()
        if not session_state.show_windspeed and wind_button.button('Show me!', key='show-2'):
            session_state.show_windspeed = True

        if session_state.show_windspeed:
            wind_button.markdown('')
            st.markdown('''
                ### Windspeed, indeed! 🕵

                The windspeed attribute looks really suspicious...

                * There are a lot of 0-valued entries.
                * A lot more people seem to prefer cycling with high windspeed.

                #### Zero values

                Let's start with the first point.
                The following is a simple _histogram_ that depicts the frequency of windspeed values in data.
                The shape of this curve is rather "unnatural".
                ''')

            st.plotly_chart(px.histogram(get_data(df), x="windspeed", marginal="rug",
                                         title='Distribution of windspeed values'))

            st.markdown('''
                The small lines on top visualize the original data. This is also strange, because it
                seems that the windspeed is measured only at discrete steps.

                At a high level, we can make two or three simple hypothesis about those zero entries:

                * they can actually be 0, that is, total absence of wind (🤨);
                * values are too low to be measured, for example due to poor accuracy of our measure;
                * all zeros or part of them are nothing but invalid numbers,
                which have been converted to zero at some point.

                Probably a good explanation is that our measure is rather poor, but for the sake of this exercise
                we can consider those values as not valid. However, remember that in general
                **we should deeply investigate any suspicious element in our dataset** before confirming or
                rejecting an hypothesis.

                Even our smallest decisions can greatly affect the outcome of the analysis!
                ''')

#             st.write("Let's see how the distribution of windspeed values may look
#                      like after the missing values are imputed.")
#             windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
#             data = get_data(df)
#             dataWind0 = data[data["windspeed"]==0]
#             dataWindNot0 = data[data["windspeed"]!=0]
#             dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

#             rfModel_wind = RandomForestClassifier()
#             rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])
#             wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])

#             dataWind0["windspeed"] = wind0Values
#             data = dataWindNot0.append(dataWind0)
#             data["windspeed"] = data["windspeed"].astype("float")

#             st.plotly_chart(px.histogram(data, x="windspeed", marginal="rug",
#                                          title='Count of windspeed values after imputing'))

#             st.info('''**Hint**

# Without too much fuss, we have used a well-known machine learning algorithm called
# _Random Forest_ to impute zero values. It uses other information such as the season,
# the humidity or the temperature to infer the windspeed. As you can see, the result
# is plausible, even if we cannot say if it corresponds to the truth.
#                 ''')

            st.markdown('''
                #### High windspeed

                Regarding the second point, let's make it clear: there is nothing wrong about the fact that more people
                want to ride a bike with a lot of wind. 🌬🚴

                But we repeat the concept once again: **a good data scientist should always question the data,
                until he or she find a plausible explanation for what the data tell him/her**.
                And that distribution is definitely something to investigate about.

                Is high windspeed related to nice weather? Or do people really prefer windy days for cycling?

                Leaving aside the reasons, this is an important point that has to do with a common misconception:
                ''')

            st.error('Many people are led to believe that modern AI “understands” the data we fed to it, '
                     'and that is capable to make some sort of human reasoning.\n\n'
                     '**Well...this is definitely not the case, as of today.**\n\n'
                     'Machine learning as we know it is a simple algorithm that learns what we decide it to learn.'
                     )

            st.markdown('''
                If we feed windspeed _as is_ to our "artificial intelligence", without giving to it any notion
                of what windspeed really is - what is plausible and what is not - it will use it as any other attribute,
                treating zeros or high values as such.

                In short, despite our effort and our progress on machine perception, we’re still far from human-level AI.

                However, **while machine learning still needs humans to help it "reasoning", humans do need machine
                learning to achieve previously impossible results in all sciences**.
                ''')

            st.image('assets/burgerking.jpeg',
                     width=500,
                     caption='Burger King’s ‘AI-written’ ads show we’re still very confused '
                             'about artificial intelligence.')

            st.markdown("_Image credits_: [The Verge](https://www.theverge.com/tldr/2018/10/3/17931924/burger-king-ai-ads-confusion-misunderstanding).")

            if not session_state.sections[4] and st.button('Continue', key='cont-five'):
                session_state.sections[4] = True

    if session_state.sections[4]:
        st.markdown('''
            ### Correlations

            Let's close our quick analysis of weather conditions by looking at the `atemp` column,
            which is the "feel like" temperature in Celsius degrees.

            We can understand more using a _scatter plot_:
            ''')

        st.plotly_chart(px.scatter(get_data(df), x='temp', y='atemp', color='humidity'))

        st.markdown('''
            Again, look how much information can be gathered from a simple chart!

            First of all, we can say that "true" and "feel like" temperature are almost
            linearly related but humidity plays a role when temperatures are below ~15°C or above ~24°C.

            Note the data contains few points below 5°C, which means very few people dare to use
            a bicycle when it's freezing in Washington D.C.! ☃️

            Also note the dots to the lower right of the main grouping. These are called _outliers_ as
            they clearly do not follow the general trend. These 24 points all occur on a single day in August.
            Most probably, these data points are unreliable and should be excluded or imputed. For example,
            we could set them equal to the measured temperature.
            ''')

        st.warning('''**Note**

All these operations like inputing erroneous or missing values, removing outliers, checking for
duplicates, converting data types, etc. are sometimes called "data transformation" or "data cleansing"
operations.

More often than not, **the data scientists spend much more time in cleansing and transforming data
then doing any real machine learning analysis**.

And when we have a lot of data (_big data_) this can be really hard. This is the reason why sometimes
we need professionals which also specialize in these kind of operations and help the data scientists
in their day-to-day job.

These people are often called **machine learning engineers** and **data engineers**.
            ''')

        st.markdown('''
            A relationship like the one between "real" and "feel like" temperature, should be
            actually spotted thanks to an old acquaintance of ours from statistics courses:
            the _linear correlation_.

            Apart from being the most beloved term that too many people (ab)use when they
            think about "Advanced Analytics", it is a very useful metric to detect potential
            statistical association, weather causal or not.

            Linear relations among numerical attributes are easier to detect than non-linear ones or
            if attributes are categorical.
            ''')

        corrMatt = get_data(df)[["temp", "atemp", "hour", "casual", "registered", "humidity", "windspeed", "count"]].corr()
        mask = np.array(corrMatt)
        mask[np.tril_indices_from(mask)] = False
        sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
        st.pyplot()

        st.markdown('''
            Looking at the _correlation plot_, we can confirm many of our previous results:

            * temperature and humidity attributes has got positive and negative correlation with `count` respectively;
            * windspeed is not really significant (even after imputing);
            * `atemp` and `temp` are indeed strongly linearly correlated with each other;
            * `casual` and `registered` contain direct information about the bike sharing count which is to predict.

            As a general rule of thumb, when two or more attributes are strongly correlated with each other
            we just need to keep one of those, in order to reduce the "complexity" of the predictive model.

            The correlation plot gives us another useful indication: temperature, humidity and daytime seem
            to be promising features for the bike sharing count prediction.

            Just out of curiosity: if 0.98 is an almost straight line, what does a 0.39 correlation
            value looks like exactly?
            ''')

        st.plotly_chart(px.scatter(get_data(df), x='humidity', y='count', trendline="ols"))

        st.markdown('''
            Okay! Definitely not something we want to deal with with pen and paper 😬.

            And now, a question for you...

            > _Why we cannot use `registered` and `casual` to predict `count`,
            given the very high correlation?_
            ''')

        leakage_button = st.empty()
        if not session_state.show_leakage and leakage_button.button('Show me!', key='show-3'):
            session_state.show_leakage = True

        if session_state.show_leakage:
            leakage_button.markdown('')
            st.markdown('''
                ### Elementary, my dear Watson! 🕵

                In this case the reason is obvious. The value we want to predict is actually the sum
                of `registered` and `casual` attributes.

                We can of course use the data to create our "prediction" model, just to discover
                that our sophisticated machine learning algorithm is actually a simple sum. 😒

                But there are cases (a lot!) when this is not at all so obvious.

                **Whenever the data you are using to train a machine learning algorithm happens
                to have the information you are trying to predict, you are in a bad shape.**

                This is called _Data Leakage_ in jargon and you must always be very careful with it!
                ''')

            st.info('''**Data leakage**

> _“too good to be true” performance is “a dead giveaway” of its existence_

Chapter 13, Doing Data Science: Straight Talk from the Frontline
                ''')

            st.markdown('''
                Can you now appreciate why a good data scientist is always SO suspicious?
                ''')

            if st.checkbox('Click to see how I deal with data leakage'):
                st.image('assets/iceage.gif', caption='Me, trying to fix data leakage in my model.')

            if not session_state.sections[5] and st.button('Continue', key='cont-six'):
                session_state.sections[5] = True

    if session_state.sections[5]:
        st.markdown('''
            ## Back into the future! ⚡️

            Eventually, we are done with that boring _exploratory analysis_ and we are ready
            to build our predictive model.

            At bare minimum, once all the preliminary steps are completed,
            there are four common activities in every machine learning task:

            1. define a metric that tell us how good we are;
            2. select a machine learning algorithm;
            3. split the data into (at least) two sets, one for training the algorithm
            and one for checking how good we are;
            4. do the training, test and celebrate how good we are 🍾.

            Let's start with the first.
            The characteristics of the given problem are:

            * **Continuous quantity**: the target variable (`count`) is a number (ex.: 120 rentals),
            not a set of categories (like "low", "medium", "high").
            * **Small dataset**: Less than 100K samples.
            * **Few important attributes**: the correlation plot indicates that a few
            features contain the information to predict the target variable.

            These characteristics limit the number of possible algorithm and drive the selection of the accuracy metric.

            ### 1. Accuracy metric

            The "classical" metrics in this case is the _Root Mean Square Error_ (RMSE) and the logarithmic
            version of it (RLMSE). For now, it suffices to say that those metrics can be interpreted
            as a measure of the ratio between the true and predicted values: **when the predicted values
            are close to the real values, the RMSE (or the RLMSE) is small**.

            ### 2. Algorithm

            The main idea is the following: **we need an algorithm that is capable of inferring the
            number of rentals from the other attributes like humidity or season**.

            Something that says: given that on 3rd March 2011, a working day, at 3pm, there was a cloudy weather,
            with 75% of relative humidity and 17° Celsius degrees, what could have been the number of
            rented bicycles?

            It is clear that this is possible only if there is some sort of correlation between the
            attributes that we provide and the number of rentals, even if this correlation is deeply hidden
            in the data.

            Luckily, there are several of possible algorithms we can choose from. **You may not be aware of it,
            but selecting the best algorithm is rarely an issue nowadays**, especially if we did our homeworks
            properly before.

            In fact, there are ways to automatically choose the best performing algorithm as well as to
            maximize its performance for this particular task and dataset.

            ''')

        st.info('''**Hint**

Here we have chosen a combination of two algorithms, a so called _Random Forest Regressor_ and
a _Gradient Boosting Regressor_. In a more advanced lesson we will take a look at those two
algorithms and quite a few more.
            ''')

        st.markdown('''
            ### 3. Train, validate and test splits

            You may remember that we only have data from the 1st to the 20th of each month. This is
            because our task is to predict bike rentals for the remaining days (the so called _test set_,
            which is yet to be seen).

            At the same time we also need a way to assess if our algorithm is doing well.
            And despite what some people may think, we cannot see the future!
            ''')

        st.image('assets/mage.jpg',
                 width=500,
                 caption='A typical representation of a data scientist, '
                         'while she trains predictive algorithms.')

        st.markdown('''
            A very common approach is then to use a portion of the data that we know (i.e.
            data that we have collected in the past) to validate the performance of our algorithm.

            That portion is called the _validation set_.

            In other words, we keep aside a portion of the data so that our algorithm cannot use
            it while training. The training is done using the remaining data (called _training set_).
            Once ready, we ask our model to make a prediction, that is, to predict that portion of the data
            that we have kept aside. If the prediction is close to it, we are a little more
            confident that our model should perform well also on unseen data.

            Graphically, this is how we have decided to split the data into train, validation and
            test sets, for each month:
            ''')

        fig = go.Figure()

        a = df.sort_index().loc[rng, :]
        fig.add_trace(go.Scatter(x=a.index, y=a['count'], mode="lines", name="hourly rentals"))
        fig.add_trace(go.Scatter(
            x=["2011-06-5 12:00", "2011-06-17 12:00", "2011-06-25"],
            y=[700, 700, 700],
            text=["Train set", "Validation", "Test set"],
            mode="text",
            showlegend=False
        ))

        # Add shape regions
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0="2011-06-15",
                    y0=0,
                    x1="2011-06-20",
                    y1=1,
                    fillcolor="LightSalmon",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                ),
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0="2011-06-20",
                    y0=0,
                    x1="2011-06-30",
                    y1=1,
                    fillcolor="LightSeaGreen",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
            ]
        )

        st.plotly_chart(fig)

        st.markdown('''
            ### 4. Predict!

            Finally, we are ready to train and test our machine learning algorithm.
            ''')

        if not session_state.sections[6] and st.button('Continue', key='cont-seven'):
            session_state.sections[6] = True
            st_rerun_patch.rerun()

elif session_state.show_excel:

    st.header('Good luck!!')
    st.header('')
    st.image('assets/excel.gif', use_column_width=True)
    st.header('')

    st.markdown('''
        You have chosen to continue to believe on Excel for everything you do, from data analysis
        with clients to tracking expenses of your vacations with the family.

        ### We can only wish you good luck and enjoy your time with Excel!
        ''')

    st.header('')
    st.header('')
    st.header('')
    st.write('_In case you have changed your mind, click here...._')
    if st.button("Show me the truth"):
        session_state.show_excel = False
        session_state.sections[7] = True
        st_rerun_patch.rerun()

elif session_state.sections[6] and not session_state.sections[7]:

    st.image('assets/morpheo.jpg', use_column_width=True)

    st.markdown('''## Now... This is your last chance. After this, there is no turning back.

## You take the blue pill - the story ends, you wake up in your bed and believe whatever you want to believe.

## You take the red pill - you stay in Wonderland, and I show you how deep the rabbit hole goes.

## Remember: all I'm offering is the truth. Nothing more.
        ''')

    st.header('')
    if st.button("Take the blue pill"):
        session_state.show_excel = True
        st_rerun_patch.rerun()

    st.header('')
    if st.button("Take the red pill"):
        session_state.sections[7] = True
        st_rerun_patch.rerun()

elif session_state.sections[7]:

    st.header('Well done!!')
    st.markdown('''

        The model is training right now.

        When it's done, the graph finally shows the results of our predictive model. You can see
        that the model follows pretty well the curve.

        Use the slider on the bottom to navigate the chart across time.
        ''')

    pred, score = train_and_test_predictive_model(df)

    c = pd.concat([df.sort_index(), pred.sort_index()], axis=1, keys=['real', 'pred'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.index, y=c[('real', 'count')], mode='lines', name='real'))
    fig.add_trace(go.Scatter(x=c.index, y=c[('pred', 'count')], mode='lines', name='predicted'))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(xaxis_range=['2011-07-01', '2011-07-20'])

    st.plotly_chart(fig)
    st.markdown(f'''
        As you can see, the line in orange is predicted by the model, and from day 15 to day 20 of each
        month it follows the blue line (real data) with high accuracy.

        The RLMSE score on the validation set is {score:.3f}, which is a pretty good result!

        Where the blue line is missing, from day 20 to end of each month, we are **predicting** what is
        happening, based on time and weather attributes (which would be predicted as well!).

        Therefore, we usually cannot know if we are really doing well in advance.
        Only time will tell us if our **bike-sharing predictive modeling tool** has been worthy
        all this effort...
        ''')

    st.warning('''**Note**

Predictive analysis can be useful in many situations.

For example, imagine that we need to evaluate if we want to expand our bike-sharing business to
a new city. With a predictive model, **we can decide which city can be more profitable according
to the environmental conditions**.

Or imagine that we have a limited number of bicycles to run our business and we need to cover
multiple cities. By knowing in advance what would be the number of rented bicycles **we could preemptively move
them from one city to another to avoid going out of stocks**.

By showing how many users will use the service, we can better schedule demand and **define
the best dates for maintenance of bikes that are not in use**.

You can also understand why having a good metric and high accuracy is so important:
a difference of just one or two points in accuracy can have a great impact on your business.
        ''')

    st.image('assets/ride.jpg', use_column_width=True)
    st.header('')
    st.header('')
    st.subheader('We hope you enjoyed the ride!')
    st.subheader('Feel free to reach us anytime if you need more information and see you next one.')
    st.header('Thank you!')
    st.header('')

    st.image('assets/logo.png')
    st.write('Contact us at: [contact@fibonaccilab.ch](mailto://contact@fibonaccilab.ch) or visit [fibonaccilab.ch](http://www.fibonaccilab.ch)')
    st.header('')
    if st.button("Start again"):
        session_state.sections = [False for i in range(NUM_SECTIONS)]
        session_state.show_workhours = False
        session_state.show_windspeed = False
        session_state.show_leakage = False
        session_state.show_excel = False
        st_rerun_patch.rerun()
