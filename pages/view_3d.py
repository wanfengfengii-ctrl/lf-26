import dash
import math
import numpy as np
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    get_volume_estimate, get_batch
)
from analysis import generate_3d_point_cloud, compute_batch_statistics

dash.register_page(__name__, path='/3d-view', name='三维视图')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("三维视图", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='3d-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=6),
            dbc.Col([
                html.H6("选择勘测批次"),
                dcc.Dropdown(
                    id='3d-batch-selector',
                    placeholder='选择一个批次...'
                ),
            ], width=6),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                html.H6("显示模式"),
                dcc.RadioItems(
                    id='3d-view-mode',
                    options=[
                        {'label': ' 点云', 'value': 'scatter'},
                        {'label': ' 曲面', 'value': 'surface'}
                    ],
                    value='scatter',
                    inline=True
                ),
            ], width=6),
            dbc.Col([
                html.H6("颜色映射"),
                dcc.Dropdown(
                    id='3d-colormap',
                    options=[
                        {'label': '深度', 'value': 'depth'},
                        {'label': '距离', 'value': 'distance'},
                        {'label': '角度', 'value': 'angle'}
                    ],
                    value='depth',
                    clearable=False
                ),
            ], width=6),
        ], className='mb-4'),

        dcc.Graph(id='3d-point-cloud-plot', style={'height': '600px'}),

        html.Hr(),

        dbc.Card([
            dbc.CardHeader("三维视图说明"),
            dbc.CardBody([
                html.Ul([
                    html.Li("X、Y 轴表示盐穴平面坐标，Z 轴表示深度"),
                    html.Li("可以用鼠标旋转、缩放三维视图"),
                    html.Li("悬停在点上查看详细信息（角度、距离、深度）"),
                    html.Li("颜色映射可以切换为按深度、距离或角度显示")
                ])
            ])
        ]),
    ], fluid=True)


@callback(
    Output('3d-batch-selector', 'options'),
    Output('3d-batch-selector', 'value'),
    Input('3d-cave-selector', 'value')
)
def update_batch_selector(cave_id):
    if not cave_id:
        return [], None

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]
    return options, options[0]['value'] if options else None


@callback(
    Output('3d-point-cloud-plot', 'figure'),
    Input('3d-batch-selector', 'value'),
    Input('3d-view-mode', 'value'),
    Input('3d-colormap', 'value')
)
def update_3d_plot(batch_id, view_mode, colormap):
    if not batch_id:
        fig = go.Figure()
        fig.update_layout(title='请选择一个勘测批次', scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='深度 (m)'
        ))
        return fig

    measurements = get_measurements_by_batch(batch_id)
    if not measurements:
        fig = go.Figure()
        fig.update_layout(title='该批次没有测量数据')
        return fig

    batch = get_batch(batch_id)
    point_cloud = generate_3d_point_cloud(measurements)

    x = point_cloud['x']
    y = point_cloud['y']
    z = point_cloud['z']
    angles = point_cloud['angles']
    distances = point_cloud['distances']
    depths = point_cloud['depths']

    if colormap == 'depth':
        color_values = depths
        color_title = '深度 (m)'
    elif colormap == 'distance':
        color_values = distances
        color_title = '距离 (m)'
    else:
        color_values = angles
        color_title = '角度 (°)'

    hover_text = [
        f'角度: {a:.1f}°<br>距离: {d:.2f} m<br>深度: {dp:.2f} m'
        for a, d, dp in zip(angles, distances, depths)
    ]

    if view_mode == 'scatter':
        fig = create_scatter_3d(x, y, z, color_values, color_title, hover_text)
    else:
        fig = create_surface_3d(x, y, z, color_values, color_title, hover_text, angles, distances)

    fig.update_layout(
        title=f"三维视图 - {batch['batch_name']}",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='深度 (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def create_scatter_3d(x, y, z, color_values, color_title, hover_text):
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=color_values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(
                title=color_title,
                thickness=20
            )
        ),
        text=hover_text,
        hoverinfo='text'
    )])

    return fig


def create_surface_3d(x, y, z, color_values, color_title, hover_text, angles, distances):
    if len(angles) < 4:
        return create_scatter_3d(x, y, z, color_values, color_title, hover_text)

    sorted_indices = np.argsort(angles)
    sorted_angles = np.array(angles)[sorted_indices]
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_depths = np.array(z)[sorted_indices]
    sorted_x = np.array(x)[sorted_indices]
    sorted_y = np.array(y)[sorted_indices]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=sorted_x,
        y=sorted_y,
        z=sorted_depths,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=color_values,
            colorscale='Viridis',
            colorbar=dict(
                title=color_title,
                thickness=20
            )
        ),
        line=dict(color='blue', width=2),
        text=hover_text,
        hoverinfo='text',
        name='测量点'
    ))

    theta = np.deg2rad(sorted_angles)
    theta = np.append(theta, theta[0] + 2 * np.pi)
    r = np.append(sorted_distances, sorted_distances[0])
    d = np.append(sorted_depths, sorted_depths[0])

    num_radial = 20
    radial_steps = np.linspace(0, 1, num_radial)

    X_surf = []
    Y_surf = []
    Z_surf = []

    for i in range(len(theta)):
        xs = []
        ys = []
        zs = []
        for t in radial_steps:
            xs.append(r[i] * t * np.cos(theta[i]))
            ys.append(r[i] * t * np.sin(theta[i]))
            zs.append(d[i] * t)
        X_surf.append(xs)
        Y_surf.append(ys)
        Z_surf.append(zs)

    fig.add_trace(go.Surface(
        x=np.array(X_surf).T,
        y=np.array(Y_surf).T,
        z=np.array(Z_surf).T,
        colorscale='Viridis',
        opacity=0.4,
        showscale=False,
        name='拟合曲面',
        hoverinfo='skip'
    ))

    return fig
