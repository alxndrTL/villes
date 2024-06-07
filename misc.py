from IPython.display import display, HTML

def print_colore(text, values):
    assert len(text) == len(values)

    def valeur_en_nuance_rouge(value):
        red_intensity = int(value * 255)
        return f'rgb({red_intensity}, 0, 0)'

    html_text = ''.join([f'<span style="background-color:{valeur_en_nuance_rouge(values[i % len(values)])}">{char}</span>' for i, char in enumerate(text)])
    display(HTML(html_text))

