import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plotPredictions(trueLabelsNp, predictionsNp):
    labels = ['PTS', 'REB', 'AST', 'STL', 'BLK']

    for i, label in enumerate(labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trueLabelsNp[:, i], label='Actual', marker='o')
        ax.plot(np.maximum(np.round(predictionsNp[:, i]), 0), label='Predicted', marker='x')
        ax.set_title(f'{label}: Predicted vs Actual')
        ax.set_xlabel('Game')
        ax.set_ylabel(label)
        ax.legend()
        st.pyplot(fig)
