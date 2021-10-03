"""
pclass = 1|2|3
gender = 'male'=1,'female'=0
age = 1-60
sibsp = 1|0
parch = 1|0
fare = 0-100
embarked = 1|0
"""

import business

import streamlit as st

pclass_list = [1,2,3]
gender_list = ['male','female']

gender = 1

pclass = st.selectbox(
    'Passenger Class',
     pclass_list)

gender_str = st.selectbox(
    'Gender',
     gender_list)

age = st.slider('AGE', 1, 60)

y_n = ['yes','no']

sibsp = st.selectbox(
    "siblings or spouse", y_n
)
parch = st.selectbox(
    "parents or children", y_n
)

embarked = st.selectbox(
    "boarding location ", ["massachuests","new-jersey"]
)

fare = st.slider('Fare', 0, 350)

def yes_no(y_n_str):

    if y_n_str == 'yes':

        return 1

    return 0

if gender_str == 'female':

    gender = 0

sibsp = yes_no(sibsp)

parch = yes_no(parch)

embarked = 1 if embarked == 'massachuests' else 0

data = [[
        pclass,
        gender,
        age,
        sibsp,
        parch,
        fare,
        embarked
        ]]

predict = st.button("Predict")

if predict:

    pred = business.predict(data)

    if pred:

        st.write("Congrats, you have survived")

    else:

        st.write("oops, you died")

if __name__ == '__main__':

    print("RUN : streamlit run app.py")