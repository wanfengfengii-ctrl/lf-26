import os
import io
import pandas as pd
import numpy as np
from flask import Flask, send_file
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from database import init_db, get_all_caves

server = Flask(__name__)

app = dash.Dash(
    __name__,
    server=server,
    use_pages=True,
    pages_folder='pages',
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.config.suppress_callback_exceptions = True

init_db()

sidebar = dbc.Col([
    html.H4("盐穴勘测数据分析", className="text-center mb-4 mt-3"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink(
            [html.I(className="bi bi-upload me-2"), "数据导入"],
            href="/",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-graph-up me-2"), "断面分析"],
            href="/cross-section",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-box-seam me-2"), "三维视图"],
            href="/3d-view",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-arrow-left-right me-2"), "批次对比"],
            href="/batch-compare",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-graph-up-arrow me-2"), "时序演化分析"],
            href="/temporal-analysis",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-check2-circle me-2"), "质量评估中心"],
            href="/quality-assessment",
            active="exact",
            className="mb-2"
        ),
        dbc.NavLink(
            [html.I(className="bi bi-gear me-2"), "数据管理"],
            href="/data-management",
            active="exact",
            className="mb-2"
        ),
    ], vertical=True, pills=True),
    html.Hr(),
    html.Div([
        html.H6("当前盐穴"),
        dcc.Dropdown(
            id='sidebar-cave-selector',
            options=[],
            placeholder='选择盐穴...',
            className='mb-2'
        ),
    ], className='px-2'),
], width=2, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'borderRight': '1px solid #dee2e6'})


content = dbc.Col([
    dash.page_container
], width=10, className='p-4')


app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='selected-cave-store', storage_type='session'),
    dbc.Row([
        sidebar,
        content
    ], className='g-0')
], fluid=True, className='g-0')


@callback(
    Output('sidebar-cave-selector', 'options'),
    Output('sidebar-cave-selector', 'value'),
    Output('selected-cave-store', 'data'),
    Input('url', 'pathname'),
    State('selected-cave-store', 'data')
)
def update_sidebar_cave_selector(pathname, stored_cave_id):
    caves = get_all_caves()
    options = [{'label': c['name'], 'value': c['id']} for c in caves]

    if stored_cave_id:
        valid_ids = [opt['value'] for opt in options]
        if stored_cave_id in valid_ids:
            return options, stored_cave_id, stored_cave_id

    default_value = options[0]['value'] if options else None
    return options, default_value, default_value


@callback(
    Output('sidebar-cave-selector', 'value', allow_duplicate=True),
    Input('selected-cave-store', 'data'),
    State('sidebar-cave-selector', 'value'),
    prevent_initial_call=True
)
def sync_sidebar_from_store(stored_cave_id, current_value):
    if stored_cave_id != current_value:
        return stored_cave_id
    return dash.no_update


@callback(
    Output('selected-cave-store', 'data', allow_duplicate=True),
    Input('sidebar-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_sidebar_to_store(cave_id, stored_cave_id):
    if cave_id != stored_cave_id:
        return cave_id
    return dash.no_update


@server.route('/download/sample')
def download_sample():
    sample_data = generate_sample_csv()
    return send_file(
        io.BytesIO(sample_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_survey.csv'
    )


def generate_sample_csv():
    angles = np.arange(0, 360, 15)
    base_distance = 50.0
    base_depth = 30.0

    data = {
        'batch_name': [],
        'angle': [],
        'distance': [],
        'depth': []
    }

    for batch_name, dist_variation, depth_variation in [
        ('2024-01-勘测', 0.05, 0.03),
        ('2024-06-勘测', 0.08, 0.06),
    ]:
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            distance = base_distance * (1 + dist_variation * np.sin(3 * angle_rad))
            depth = base_depth * (1 + depth_variation * np.cos(2 * angle_rad))

            data['batch_name'].append(batch_name)
            data['angle'].append(round(angle, 1))
            data['distance'].append(round(distance, 2))
            data['depth'].append(round(depth, 2))

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
