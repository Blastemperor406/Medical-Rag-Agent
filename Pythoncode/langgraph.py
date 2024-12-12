import streamlit as st
import graphviz

def visualize_chain(chain_of_thought):
    dot = graphviz.Digraph()
    for i, step in enumerate(chain_of_thought):
        dot.node(str(i), step)
        if i > 0:
            dot.edge(str(i-1), str(i))
    st.graphviz_chart(dot)