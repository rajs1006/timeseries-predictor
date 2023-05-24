import streamlit as st

progress_bar = st.progress(0)


def stProgressBar(current):
    progress_bar.progress(current + 1)


def progressBar(n, current, total, barLength=50):
    percent = float(current) * 100 / total
    arrow = "-" * int(percent / 100 * barLength - 1) + ">"
    spaces = " " * (barLength - len(arrow))

    print("%s: [%s%s] %d %%" % (n, arrow, spaces, percent), end="\r")
