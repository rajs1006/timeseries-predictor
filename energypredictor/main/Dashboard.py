from energypredictor.main.visualization.Streamlit import Streamlit


def main():

    streamlit = Streamlit()

    trainButton, predictButton = streamlit.input()

    if trainButton:
        streamlit.load()
        streamlit.train()

    if predictButton:
        streamlit.test()


if __name__ == "__main__":
    main()
