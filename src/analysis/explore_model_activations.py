import glob
import json
import plotly.express as px

import numpy
import pandas
import streamlit as st

task_files = glob.glob("normalized_activations/argmin*.json")


if "task_activations" not in st.session_state:

    st.session_state.properties = {
        "tasks": set([task_file.split("__")[0] for task_file in task_files]),
        "settings": set([task_file.split("__")[2] for task_file in task_files]),
        "variations": set([task_file.split("__")[3] for task_file in task_files]),
        "seeds": set([task_file.split("__")[4] for task_file in task_files]),
        "shared_models": set([task_file.split("__")[5] for task_file in task_files]),
        "frozen_models": set([task_file.split("__")[6] for task_file in task_files]),
        "model_names": set([task_file.split("__")[7] for task_file in task_files]),
        "injection_layers": set([task_file.split("__")[8] for task_file in task_files]),
    }

    st.session_state.loaded_model = ()

selected_max_scale = st.sidebar.slider("max_scale", min_value=0.0, max_value=1.0, step=0.01, key="scale")
selected_task = st.sidebar.selectbox("task", st.session_state.properties["tasks"], key="task")
selected_setting = st.sidebar.selectbox("setting", st.session_state.properties["settings"], key="setting")
selected_variation = st.sidebar.selectbox("variation", st.session_state.properties["variations"], key="variations")
selected_seed = st.sidebar.selectbox("seed", st.session_state.properties["seeds"], key="seeds")
selected_injection_layer = st.sidebar.selectbox("injection_layer", st.session_state.properties["injection_layers"], key="injection_layers")
normalize_activation = st.sidebar.radio("normalize activation", ("False", "True"))

if (selected_task, selected_setting, selected_variation, selected_seed, selected_injection_layer) != st.session_state.loaded_model:

    task_files = glob.glob("__".join([selected_task, "test", selected_setting, selected_variation, selected_seed]) + "*"+ selected_injection_layer + "*.json")

    if len(task_files) == 0:
        st.title("No Activations found")
    else:

        activation_frame = pandas.DataFrame(json.load(open(task_files[0], "r"))[0])

        sentences = activation_frame["sentence"]
        selected_sentence = st.sidebar.selectbox("sentences", sentences, key="sentences")

        context = activation_frame[activation_frame["sentence"] == selected_sentence]["context"].values[0]
        label = activation_frame[activation_frame["sentence"] == selected_sentence]["label"].values[0]
        pred = activation_frame[activation_frame["sentence"] == selected_sentence]["pred"].values[0]
        input_tokens = activation_frame[activation_frame["sentence"] == selected_sentence]["input_tokens"].values[0]
        input_self_activations = activation_frame[activation_frame["sentence"] == selected_sentence]["input_self_activations"].values[0]
        input_cross_activations = activation_frame[activation_frame["sentence"] == selected_sentence]["input_cross_activations"].values[0] if "RETRO" in selected_setting else None


        st.title("Explore Model Activations")
        st.header("Sample:")
        st.json({
            "sample": selected_sentence,
            "context": context,
            "label": str(label),
            "prediction": str(pred)
        })

        def get_activation_fig(self_activation, cross_activation = None, tokens= []):

            if cross_activation:
                activations = self_activation + cross_activation
                injection_layer = eval(selected_injection_layer)[0]
                max_layer = len(self_activation)

                y_labels = ["Layer-" + str(ele) for ele in range(injection_layer + 1)] + ["Injection"] + ["Layer-" + str(ele) for ele in range(injection_layer + 1, max_layer)]
            else:
                activations = self_activation
                y_labels = ["Layer-" + str(ele) for ele in range(12)]


            sep_index = tokens.index('[SEP]')
            activations = numpy.array(activations)[:,1:sep_index]
            tokens = tokens[1:sep_index]

            if normalize_activation == "True":
                norms = numpy.linalg.norm(activations, axis=1)
                activations = numpy.divide(activations, norms[:, None])

#            tokens = [str(i) + "-" + ele for i, ele in enumerate(tokens)]

            activation_fig = px.imshow(
                activations,
                labels=dict(x="Tokens", y="Layers", color="Normalized Activation"),
                x=[str(i) + "-" + ele for i, ele in enumerate(tokens)],
                y=y_labels,
                range_color=[0, selected_max_scale],
                color_continuous_scale="rdbu"
            )
            activation_fig.update_xaxes(
                tickangle=45,
                tickmode='array',
                tickvals=list(range(len(tokens))),
                ticktext=tokens
            )

            return activation_fig


        st.header("Input Encoder Activations:")

        input_activation_fig = get_activation_fig(input_self_activations, input_cross_activations, input_tokens)
        st.plotly_chart(input_activation_fig, use_container_width=True)

        if "RETRO" in selected_setting:
            context_tokens = activation_frame[activation_frame["sentence"] == selected_sentence]["context_tokens"].values[0]
            context_self_activations = activation_frame[activation_frame["sentence"] == selected_sentence]["context_self_activations"].values[0]
            context_cross_activations = activation_frame[activation_frame["sentence"] == selected_sentence]["context_cross_activations"].values[0]

            st.header("Context Encoder Activations:")

            cols = st.columns(2)

            for context_self_activation, context_cross_activation, context_token, col in zip(context_self_activations, context_cross_activations, context_tokens, cols):
                context_activation_fig = get_activation_fig(context_self_activation, context_cross_activation, context_token)
                col.plotly_chart(context_activation_fig, use_container_width=True)
