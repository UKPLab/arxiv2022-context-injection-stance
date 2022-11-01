import glob
import json
import plotly.express as px

import numpy
import pandas
import streamlit as st

st.set_page_config(layout="wide")

def get_finger_print(file_path):
    file_path = file_path.replace("token_activations/", "").replace(".csv", "")
    properties = file_path.split("__")
    return {
        "method": properties[0],
        "task": properties[1],
        "source": properties[3],
        "setting": properties[4],
        "variation": properties[5],
        "injection_layer": properties[6],
    }

if "filter_settings" not in st.session_state:

    token_activation_files = glob.glob("token_activations/*.csv")


    finger_prints = pandas.DataFrame([get_finger_print(token_activation_file) for token_activation_file in token_activation_files])

    st.session_state.filter_settings = {
        "task": finger_prints["task"].unique(),
        "source": finger_prints["source"].unique(),
        "setting": finger_prints["setting"].unique(),
        "variation": finger_prints["variation"].unique(),
        "injection_layer": finger_prints["injection_layer"].unique(),
    }
    st.session_state.token_activations = []

    for token_activation_file in token_activation_files:
        finger_print = get_finger_print(token_activation_file)
        finger_print["activations"] = pandas.read_csv(token_activation_file)

        #finger_print["activations"]["self_activations"] = finger_print["activations"]["self_activations"].apply(eval)

        #if "cross_activations" in finger_print["activations"].columns:
        #    finger_print["activations"]["cross_activations"] = finger_print["activations"]["cross_activations"].apply(eval)
        #    finger_print["activations"]["diff_activations"] = finger_print["activations"]["diff_activations"].apply(eval)

        st.session_state.token_activations.append(finger_print)


properties = ["source", "setting", "variation", "injection_layer"]
property_filter_1 = st.sidebar.selectbox("property to filter", properties, key="filter-1")
value_filter_1 = st.sidebar.selectbox("property value", st.session_state.filter_settings[property_filter_1], key="value-1")

properties = [property for property in properties if property not in [property_filter_1]]
property_filter_2 = st.sidebar.selectbox("property to filter", properties, key="filter-2")
value_filter_2 = st.sidebar.selectbox("property value", st.session_state.filter_settings[property_filter_2], key="value-2")

properties = [property for property in properties if property not in [property_filter_1, property_filter_2]]
property_x_axis = st.sidebar.selectbox("property x-axis", properties, key="filter-x-axis")
values_x_axis = st.sidebar.multiselect("property value", st.session_state.filter_settings[property_x_axis], key="value-x-axis")

properties = [property for property in properties if property not in [property_filter_1, property_filter_2, property_x_axis]]
property_y_axis = st.sidebar.selectbox("property y-axis", properties, key="filter-y-axis")
values_y_axis = st.sidebar.multiselect("property value", st.session_state.filter_settings[property_y_axis], key="value-y-axis")

if len(values_y_axis) == 0 or len(values_x_axis) == 0:
    st.header("select x- and y-axis values")
else:
    selected_activations = [
        token_activation
        for token_activation in st.session_state.token_activations
        if token_activation[property_filter_1] == value_filter_1 and token_activation[property_filter_2] == value_filter_2 and token_activation[property_x_axis] in values_x_axis and token_activation[property_y_axis] in values_y_axis
    ]

    overall_activations = [selected_activation for selected_activation in selected_activations if selected_activation["method"] == "overall"]
    topic_activations = [selected_activation for selected_activation in selected_activations if selected_activation["method"] == "topic"]
    accuracy_activations = [selected_activation for selected_activation in selected_activations if selected_activation["method"] == "accuracy"]
    accuracy_detail_activations = [selected_activation for selected_activation in selected_activations if selected_activation["method"] == "accuracy-detail"]

    selected_tokens = st.sidebar.multiselect("tokens", pandas.concat([ele["activations"] for ele in overall_activations])["token"].unique())

    if len(selected_tokens) == 0:
        st.header("select tokens")
    else:


        correlation_plot_samples = []
        box_plot_samples = []

        for overall_activation in overall_activations:
            activations = overall_activation["activations"]

            x_axis_prop = overall_activation[property_x_axis]
            y_axis_prop = overall_activation[property_y_axis]

            token_activations = activations[activations["token"].isin(selected_tokens)]

            for index, row in token_activations.iterrows():
                property_columns = [column for column in ["topic", "result", "label", "prediction"] if column in row]
                activation_columns = [column for column in ["self_activations", "cross_activations", "diff_activations", ] if column in row]

                collected_property_columns = list(zip(*[
                    eval(row[property_column])
                    for property_column in property_columns
                ]))
                collected_activation_columns = list(zip(*[
                    eval(row[activation_column])
                    for activation_column in activation_columns
                ]))

                for collected_property_column, collected_activation_column in zip(collected_property_columns, collected_activation_columns):
                    ele = {
                        "token": row["token"],
                        property_x_axis: x_axis_prop,
                        property_y_axis: y_axis_prop,
                    }

                    for i, column_value in enumerate(collected_property_column):
                        column_name = property_columns[i]
                        ele[column_name] = column_value


                    for i, column_value in enumerate(collected_activation_column):
                        ele_clone = ele.copy()
                        column_name = activation_columns[i]
                        ele_clone["activation"] = column_value
                        ele_clone["type"] = column_name

                        box_plot_samples.append(ele_clone)


                    for i, column_value in enumerate(collected_activation_column):
                        column_name = activation_columns[i]
                        ele[column_name] = column_value

                    correlation_plot_samples.append(ele)

        correlation_plot_frame = pandas.DataFrame(correlation_plot_samples)
        box_plot_frame = pandas.DataFrame(box_plot_samples)

        plot_width = 200 + (300 * len(values_x_axis))
        plot_height = 200 + (200 * len(values_y_axis))


        with st.expander("Box Plot (Overall)"):
            fig = px.box(box_plot_frame, x="token", y="activation", facet_col=property_x_axis, facet_row=property_y_axis, color="type", points="all", width=plot_width, height=plot_height)
            st.plotly_chart(fig)

        if "cross_activations" in correlation_plot_frame.columns:
            with st.expander("Correlation Plot (Overall)"):
                fig = px.scatter(correlation_plot_frame, x="cross_activations", y="self_activations", facet_col=property_x_axis, facet_row=property_y_axis, color="token", trendline="ols", width=plot_width, height=plot_height)
                fig.update_xaxes(
                    scaleanchor = "y",
                    scaleratio = 1,
                )
                st.plotly_chart(fig)


        with st.expander("Box Plot (Topic)"):
            tabs = st.tabs(selected_tokens)
            for tab, selected_token in zip(tabs, selected_tokens):
                token_plot_frame = box_plot_frame[box_plot_frame["token"] == selected_token]
                fig = px.box(token_plot_frame, x="type", y="activation", facet_col=property_x_axis, facet_row=property_y_axis, color="topic", points="all", width=plot_width, height=plot_height)
                tab.plotly_chart(fig)

        if "cross_activations" in correlation_plot_frame.columns:
            with st.expander("Correlation Plot (Topic)"):
                tabs = st.tabs(selected_tokens)
                for tab, selected_token in zip(tabs, selected_tokens):
                    token_plot_frame = correlation_plot_frame[correlation_plot_frame["token"] == selected_token]
                    fig = px.scatter(token_plot_frame, x="cross_activations", y="self_activations", facet_col=property_x_axis, facet_row=property_y_axis, color="topic", trendline="ols", width=plot_width, height=plot_height)
                    fig.update_xaxes(
                        scaleanchor = "y",
                        scaleratio = 1,
                    )
                    tab.plotly_chart(fig)


        result_color_map = {
            "correct": "blue",
            "wrong": "red"
        }

        with st.expander("Box Plot (Result)"):
            tabs = st.tabs(selected_tokens)
            for tab, selected_token in zip(tabs, selected_tokens):
                token_plot_frame = box_plot_frame[box_plot_frame["token"] == selected_token]
                fig = px.box(token_plot_frame, x="type", y="activation", facet_col=property_x_axis, facet_row=property_y_axis, color="result", points="all", width=plot_width, height=plot_height, color_discrete_map=result_color_map)
                tab.plotly_chart(fig)

        if "cross_activations" in correlation_plot_frame.columns:
            with st.expander("Correlation Plot (Result)"):
                tabs = st.tabs(selected_tokens)
                for tab, selected_token in zip(tabs, selected_tokens):
                    token_plot_frame = correlation_plot_frame[correlation_plot_frame["token"] == selected_token]
                    fig = px.scatter(token_plot_frame, x="cross_activations", y="self_activations", facet_col=property_x_axis, facet_row=property_y_axis, color="result", trendline="ols", width=plot_width, height=plot_height, color_discrete_map=result_color_map)
                    fig.update_xaxes(
                        scaleanchor = "y",
                        scaleratio = 1,
                    )
                    tab.plotly_chart(fig)



