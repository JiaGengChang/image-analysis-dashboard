import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from flaskwebgui import FlaskUI

# theme = dbc.themes.BOOTSTRAP
# theme = dbc.themes.UNITED
# theme = dbc.themes.QUARTZ
theme = dbc.themes.SUPERHERO
icon_lib = dbc.icons.FONT_AWESOME

app = Dash(
    name=__name__,
    external_stylesheets=[theme, icon_lib],
    use_pages=True
)
OUR_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app.layout = html.Div(children=[
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Process Optical", active='exact', href="/")),
            dbc.NavItem(dbc.NavLink("Process SEM", active='exact', href="/sem")),
            dbc.NavItem(dbc.NavLink("Comparison (optical)", active='exact', href="/compare-optical")),
            dbc.NavItem(dbc.NavLink("Comparison (SEM)", active='exact', href="/compare-sem")),
        ],
        pills=True,
        justified=True,
    ),
    dash.page_container
])

if __name__ == '__main__':
    # app.run_server(debug=True)
    FlaskUI(app=app, server='flask', port=3000).run()
