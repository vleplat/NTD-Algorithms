# How to create a template in plotly, using the common options in the cheatsheet

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd

# We start with an empty template
my_template = go.layout.Template()

# Templates are in fact similar to figures (graph object with data and layout, same as figures)
# We can update them similarly to figures, but we don't have update_layout available

my_template.layout= dict(
            font_size = 16,
            width=600*1.62, # in px
            height=600,
            xaxis=dict(matches=None, showticklabels=True),
            yaxis=dict(matches=None, showticklabels=True),
    # insert layout property here, including axes
    # Basically anything from common_options.py, with ****_foo for xaxes,
    # title_text="test",
    # xaxis_color="#ff0000",
    # colorway=['#ff0000', '#00ff00', '#0000ff'] # this changes the default colors
)


# Finally we can combine themes using + :p
pio.templates["my_template"]=my_template
pio.templates.default = "plotly_white+my_template"

# How to store/load templates:
# Save them in a python file (e.g. my_template.py, this file with only the template)
# and import the file like "import my_template", which will run the code once, creating the template, 
# then run "pio.templates.default = "plotly+my_template""