import glob
import shutil
from datetime import datetime
from pathlib import Path
import dash
import pandas as pd
from dash import html, dcc, Output, Input, State, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import os
import base64

from dashboard.utils import analyse_optical_images

dash.register_page(__name__, path='/', title='Micro-Image (Process Optical)')

output_dir = 'processed_data'
user_upload_dir = 'user_uploads'
analysis_type = 'optical'

def built_in_analysis_filter(idx):
    return html.Div(
        id={
            'type': 'built-in-filter-container',
            'index': idx
        },
        children=[
            dcc.Dropdown(
                options={'scatter': 'Scatter Plot (area>=8000)',
                         'mean_intensity': 'Mean intensity',
                         'attrition': 'Attrition',
                         'eccentricity': 'Eccentricity',
                         'area': 'Area',
                         'solidity': 'Solidity'
                         },
                value='scatter',
                clearable=False,
                id={
                    'type': 'chart-select',
                    'index': idx
                },
                className='filter-dropdown mb-2 mt-3'
            ),
        ]
    )


def new_chart(idx: int, width: int):
    chart = dbc.Col(md=width, className='py-3 px-3', children=
        dbc.Card(children=
            dbc.Row(children=[
                dbc.Col(md=8, className='chart-panel', children=[
                    dbc.Row(children=[
                        dbc.Col(md=9, children=[
                            dbc.Input(
                                className='chart-name-input',
                                placeholder='New Chart',
                                style={'border': 'none'}
                            )
                        ]),
                        dbc.Col(md=3, children=[
                            html.I(
                                className='fa-solid fa-trash-can',
                                title='Remove this chart',
                                id={
                                    'type': 'remove-chart-button',
                                    'index': idx
                                }
                            )
                        ])
                    ]),
                    dcc.Graph(
                        id={
                            'type': 'main-analytics-chart',
                            'index': idx
                        },
                        figure={}
                    )
                ]),
                dbc.Col(md=4, className='filter-panel', children=[
                    html.Div(id={
                        'type': 'filter-tabs-content',
                        'index': idx
                    }, children=[built_in_analysis_filter(idx)])
                ])
            ])
        )
    )

    return chart

@dash.callback(
    Output('refresh-button-spinner', 'children'),
    Input('refresh-button', 'n_clicks'),
    config_prevent_initial_callbacks=False
)
def refresh_button_callback(n_clicks):
    input_filenames_list = None
    output_filenames_list = None
    if n_clicks:
        # delete previous input images
        dir_to_delete_1 = Path(user_upload_dir)/analysis_type
        if dir_to_delete_1.exists():
            input_filenames_list = os.listdir(dir_to_delete_1)
            shutil.rmtree(dir_to_delete_1)
        # delete previous analysis results
        dir_to_delete_2 = Path(output_dir)/analysis_type
        if dir_to_delete_2.exists():
            output_filenames_list = os.listdir(dir_to_delete_2)
            shutil.rmtree(dir_to_delete_2)
        if not input_filenames_list and not output_filenames_list:
            return None
        return [html.Div(f"Previous input files deleted: {input_filenames_list}"),
                html.Div(f"Previous output files deleted: {output_filenames_list}")]

@dash.callback(
    Output('add-chart-spinner', 'children'),
    Input({'type': 'add-chart-button', 'width': ALL}, 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
    config_prevent_initial_callbacks=True,
)
def update_chart_spinner(add_btn_n_clicks, refresh_btn_n_clicks):
    ctx = dash.callback_context
    if ctx.triggered_id == 'refresh-button':
        if refresh_btn_n_clicks:
            return None
    output_csv_filename = Path(output_dir) / analysis_type / f"{analysis_type}.csv"
    if add_btn_n_clicks and not output_csv_filename.exists():
        return "Run analysis first"


@dash.callback(
    Output('charts-container', 'children'),
    Input({'type': 'add-chart-button', 'width': ALL}, 'n_clicks'),
    Input({'type': 'remove-chart-button', 'index': ALL}, 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
    State('charts-container', 'children'),
    config_prevent_initial_callbacks=True
)
# Because of the ALL pattern matcher, when any one button is being clicked,
# all buttons' (of the same type) n_clicks will be returned as an array
def add_chart(add_btn_n_clicks_array, remove_btn_n_clicks_array, refresh_btn_n_clicks, current_content):
    ctx = dash.callback_context
    # begin 7/1/2023
    # FIXME: fix dynamic type of ctx.triggered_id
    # FIXME: ctx.triggered_id for 'refresh-button' is a string
    # FIXME: but it is a dictionary for 'add-chart' and 'remove-chart-btn'
    if ctx.triggered_id == 'refresh-button':
        if refresh_btn_n_clicks:
            # delete all charts
            current_content=[]
    # end 7/1/2023
    elif ctx.triggered_id['type'] == 'add-chart-button':  # add chart
        # begin 7/1/2023
        # if there are is no data to plot, then don't try to plot
        output_csv_filename = Path(output_dir)/analysis_type/f"{analysis_type}.csv"
        if not output_csv_filename.exists():
            return []
        # end 7/1/2023
        width = ctx.triggered_id['width']
        idx = 0
        for i in add_btn_n_clicks_array:
            if i:
                idx = idx + i
        current_content.append(new_chart(idx, width))
    elif ctx.triggered_id['type'] == 'remove-chart-button':  # remove chart
        for idx, val in enumerate(remove_btn_n_clicks_array):
            if val is not None:  # All buttons will have its n_clicks value equal to None, except the one being clicked
                del current_content[idx]
    return current_content


@dash.callback(
    Output({'type': 'main-analytics-chart', 'index': MATCH}, 'figure'),
    Input({'type': 'chart-select', 'index': MATCH}, 'value'),
)
def update_chart(chart_type):
    df = pd.read_csv(f'{output_dir}/{analysis_type}/{analysis_type}.csv')
    fig = visualize_mydata(chart_type, df)
    return fig


def visualize_mydata(chart_type, df):
    fig = go.Figure()
    # plot main diagram of the chart
    if chart_type == 'scatter':
        df1 = df[df["area"] > 8000]  # TODO plot attrited microspheres in orange
        fig.add_trace(go.Scatter(x=df1['fano'],
                                 y=df1['mean_intensity'],
                                 name='mean intensity vs standard_deviation/mean_intensity',
                                 marker=dict(color='royalblue'),
                                 mode='markers'
                                 ))
        fig.update_layout(yaxis={'title': 'mean intensity'}, xaxis={'title': 'fano (stdev/mean intensity)'},
                          title=chart_type)
    elif chart_type == 'mean_intensity':
        fig.add_trace(go.Histogram(x=df['mean_intensity']))
        fig.update_layout(yaxis={'title': 'Number of microspheres'}, title=chart_type)
        # TODO: get homogeneity from cao fan script
    elif chart_type == 'attrition':
        # fig.add_trace(go.Histogram(x=df['eccentricity']))
        fig.add_trace(
            go.Scatter(
                x=df['area'],
                y=df['eccentricity'],
                name='area vs eccentricity',
                marker=dict(color='darkorange'),
                mode='markers'
            )
        )
        fig.update_layout(yaxis={'title': 'eccentricity'}, xaxis={'title': 'area'}, title=chart_type)
    elif chart_type == 'area':
        fig.add_trace(go.Histogram(x=df['area']))
        fig.update_layout(yaxis={'title': 'Number of microspheres'}, title=chart_type)
    elif chart_type == 'eccentricity':
        fig.add_trace(go.Histogram(x=df['eccentricity']))
        fig.update_layout(yaxis={'title': 'Number of microspheres'}, title=chart_type)
    elif chart_type == 'solidity':
        fig.add_trace(go.Histogram(x=df['solidity']))
        fig.update_layout(yaxis={'title': 'Number of microspheres'}, title=chart_type)
    else:
        pass

    return fig

@dash.callback(
    Output('show-labelled-images-spinner-div','children'),
    Input('show-labelled-images-button','n_clicks'),
    Input('refresh-button','n_clicks')
)
def update_show_labelled_images_spinner_div(n_clicks,refresh_btn_n_clicks):
    ctx = dash.callback_context
    results_dir = Path(output_dir) / analysis_type
    if ctx.triggered_id == 'refresh-button':
        return None
    if n_clicks:
        if not results_dir.exists():
            return "Run analysis first"
@dash.callback(
    Output("download-spinner-div","children"),
    Input('download-button', 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
)
def update_download_spinner_div(download_btn_n_clicks, refresh_btn_n_clicks):
    ctx = dash.callback_context
    if ctx.triggered_id == 'refresh-button':
        return None
    csv_file = Path(output_dir) / analysis_type / f"{analysis_type}.csv"
    if download_btn_n_clicks and not csv_file.exists():
        return "Run analysis first"

@dash.callback(
    Output("download", "data"),
    Input('download-button', "n_clicks"),
    prevent_initial_call=True,
)
def generate_csv(download_btn_n_clicks):
    if download_btn_n_clicks:
        csv_file = Path(output_dir)/analysis_type/f"{analysis_type}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file,index_col=0)
            return dcc.send_data_frame(df.to_csv, f"{analysis_type}_results.csv")

@dash.callback(
    Output('loading-run-analysis', 'children'),
    Input('run-button', 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
)
def segment_images(run_btn_n_clicks, refresh_btn_n_clicks):
    ctx = dash.callback_context
    if ctx.triggered_id == 'run-button':
        user_upload_path = Path(user_upload_dir) / analysis_type
        images_been_uploaded = user_upload_path.exists() and len(os.listdir(user_upload_path))
        if run_btn_n_clicks:
            if images_been_uploaded:
                time_taken, input_image_paths = analyse_optical_images(user_upload_dir, analysis_type, output_dir)
                input_image_filenames = [os.path.basename(p) for p in input_image_paths]
                return [html.Div(f"Analysis done!"),
                        html.Div(f"Time taken: {time_taken}."),
                        html.Div(f"Input images: {input_image_filenames}.")]
            return "Upload images first"
    return None


layout = dbc.Col(md=12, className='px-5 py-2', children=[
    html.Div(children=[
        dcc.Upload(
            id='upload-image',
            children=html.Div(
                className='uploader-div',
                children=['Drag and Drop or ', html.A('Select Files',style={'fontWeight':1000})]),
            className='mt-3',
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'font-size': 'small'
            },
            multiple=True
        ),
        dbc.Spinner(html.Div(id='loading-input', style={'textAlign': 'center'}), color='warning'),
        dbc.Card(id='output-image-upload'),
    ]),
    html.Br(),
    dbc.Card(
        className='mx-1 border-0',
        children=[
            dbc.Button("Run analysis", id="run-button", n_clicks=0),
            dbc.Spinner(html.Div(id="loading-run-analysis", style={'textAlign': 'center'}), color='warning'),
        ],
    ),
    dbc.Card(
        className='mx-1 mt-4 border-0',
        children=[
            dbc.Button(
                id={
                    'type': 'add-chart-button',
                    'width': 12
                },
                className='w-100',
                children='Add New Chart',
                n_clicks=0
            ),
            dbc.Spinner(
                html.Div(id="add-chart-spinner",
                         style={'textAlign' : 'center'}),
                color='warning'
            )
        ]
    ),
    dbc.Row(id='charts-container', className='mb-2', children=[]),
    dbc.Card(
        className='mx-1 mt-4 border-0',
        children=[
            dbc.Button("Download csv", id="download-button"),
            dcc.Download(id="download"),
            dbc.Spinner(html.Div(id="download-spinner-div", style={'textAlign':'center'}), color='warning'),
        ]
    ),
    html.Br(),
    dbc.Card(
        className='mx-1 border-0',
        children=[
            dbc.Button('Plot labelled images', id='show-labelled-images-button'),
            dbc.Spinner(html.Div(id='show-labelled-images-spinner-div',style={'textAlign':'center'}),color='warning'),
        ]
    ),
    dbc.Card(id='labelled-images-container'),
    html.Br(),
    dbc.Card(
        className='mx-1 border-0',
        children=[
            # n_clicks=1 does an initial refresh
            dbc.Button("Refresh", id="refresh-button", color='danger', n_clicks=1),
            dbc.Spinner(html.Div(
                id="refresh-button-spinner",
                children="Click here to delete all input and output files",
                style={'textAlign': 'center'}
            ), color='warning'),
        ]
    ),
])


@dash.callback(Output('loading-input', 'children'),
               Input('upload-image', 'contents'),
               Input('refresh-button', 'n_clicks')
               )
def update_input_spinner(contents, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered_id == 'upload-image' and contents:
        return "Images uploaded. Wait for images to display."
    return "No images uploaded."


def parse_contents(contents, filename, date):
    # begin 6/1/2023
    # create a user cache within dash filesystem
    full_filename=Path(user_upload_dir)/analysis_type/filename
    os.makedirs(os.path.dirname(full_filename), exist_ok=True)
    image_base64 = contents.split(",")[1]
    image_bytes = base64.b64decode(image_base64)
    with open(full_filename,'wb') as f:
        f.write(image_bytes)
    # end 6/1/2023
    return dbc.Col([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # html.Img(src=contents, style={'height': '70%', 'width': '70%'}),
        html.Hr(),
    ])


@dash.callback(Output('output-image-upload', 'children'),
               Input('refresh-button', 'n_clicks'),
               Input('upload-image', 'contents'),
               State('upload-image', 'filename'),
               State('upload-image', 'last_modified'),
               config_prevent_initial_callbacks=True)
def update_input_image_container(n_clicks, list_of_contents, list_of_names, list_of_dates):
    ctx = dash.callback_context
    if ctx.triggered_id == 'refresh-button':
        return []
    elif ctx.triggered_id == 'upload-image':
        if list_of_contents is not None:
            # begin 6/1/2023
            # delete previous uploaded images
            input_dir = Path(user_upload_dir)/analysis_type
            print(f"Does input dir exist: {os.path.exists(input_dir)}")
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir)
            # delete previous analysis results
            results_dir = Path(output_dir)/analysis_type
            print(f"Does output dir exist: {os.path.exists(results_dir)}")
            if results_dir.exists():
                shutil.rmtree(output_dir)
            # end 6/1/2023
            cols = [
                parse_contents(content, name, date) for content, name, date in
                zip(list_of_contents, list_of_names, list_of_dates)]
            children = []
            # create rows with 4 columns each
            for i in range(0, len(cols), 4):
                row = []
                for col_idx in range(0, 4):
                    if i + col_idx < len(cols):
                        row.append(cols[i + col_idx])
                    else:
                        row.append(dbc.Col())
                children.append(dbc.Row(children=row))
            return children


@dash.callback(
    Output('labelled-images-container', 'children'),
    Input('show-labelled-images-button', 'n_clicks'),
    Input('refresh-button', 'n_clicks'),
    config_prevent_initial_callbacks=True
)
def update_processed_image(show_btn_n_clicks, refresh_btn_n_clicks):
    ctx = dash.callback_context
    if ctx.triggered_id == 'show-labelled-images-button':
        if show_btn_n_clicks:
            children = []
            image_files = sorted(glob.glob(f'{output_dir}/{analysis_type}/*.png'))
            image_files64 = [base64.b64encode(open(img, 'rb').read()).decode('ascii') for img in image_files]
            cols = [dbc.Col([
                html.H5(image_file),
                html.Img(src='data:image/png;base64,{}'.format(image_file64), style={'height': '90%', 'width': '90%'})
            ]) for image_file, image_file64 in zip(image_files, image_files64)]
            # create rows with 2 columns each
            for i in range(0, len(cols), 2):
                row = []
                for col_idx in range(0, 2):
                    if i + col_idx < len(cols):
                        row.append(cols[i + col_idx])
                    else:
                        row.append(dbc.Col())
                children.append(dbc.Row(children=row))
    elif ctx.triggered_id == 'refresh-button':
        if refresh_btn_n_clicks:
            children = []
    return children
