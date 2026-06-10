import dash
import math
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    get_volume_estimate, get_anomaly_regions, get_batch
)
from analysis import (
    generate_cross_section_data, find_missing_intervals,
    compute_batch_statistics
)

dash.register_page(__name__, path='/cross-section', name='断面分析')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("断面分析", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='cs-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=6),
            dbc.Col([
                html.H6("选择勘测批次"),
                dcc.Dropdown(
                    id='cs-batch-selector',
                    placeholder='选择一个批次...'
                ),
            ], width=6),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                html.H6("视图类型"),
                dcc.RadioItems(
                    id='cs-view-type',
                    options=[
                        {'label': ' 极坐标视图', 'value': 'polar'},
                        {'label': ' 笛卡尔坐标', 'value': 'cartesian'}
                    ],
                    value='polar',
                    inline=True
                ),
            ], width=6),
            dbc.Col([
                html.H6("显示选项"),
                dcc.Checklist(
                    id='cs-display-options',
                    options=[
                        {'label': ' 显示缺失区间', 'value': 'show_missing'},
                        {'label': ' 显示异常区域', 'value': 'show_anomalies'},
                        {'label': ' 显示深度着色', 'value': 'show_depth_color'}
                    ],
                    value=['show_missing', 'show_anomalies'],
                    inline=True
                ),
            ], width=6),
        ], className='mb-4'),

        dcc.Graph(id='cs-cross-section-plot', style={'height': '600px'}),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("批次统计信息"),
                    dbc.CardBody(id='cs-stats-card')
                ]),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("缺失测量区间"),
                    dbc.CardBody(id='cs-missing-intervals-card')
                ]),
            ], width=6),
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader("异常区域"),
            dbc.CardBody(id='cs-anomalies-card')
        ]),
    ], fluid=True)


@callback(
    Output('cs-cave-selector', 'options'),
    Output('cs-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    State('cs-cave-selector', 'value'),
    prevent_initial_call=False
)
def sync_cave_selector(stored_cave_id, current_value):
    caves = get_all_caves()
    options = [{'label': c['name'], 'value': c['id']} for c in caves]

    if stored_cave_id:
        valid_ids = [opt['value'] for opt in options]
        if stored_cave_id in valid_ids:
            return options, stored_cave_id

    return options, current_value


@callback(
    Output('selected-cave-store', 'data', allow_duplicate=True),
    Input('cs-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_cs_to_store(cave_id, stored_cave_id):
    if cave_id != stored_cave_id:
        return cave_id
    return dash.no_update


@callback(
    Output('cs-batch-selector', 'options'),
    Output('cs-batch-selector', 'value'),
    Input('cs-cave-selector', 'value')
)
def update_batch_selector(cave_id):
    if not cave_id:
        return [], None

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]
    return options, options[0]['value'] if options else None


@callback(
    Output('cs-cross-section-plot', 'figure'),
    Output('cs-stats-card', 'children'),
    Output('cs-missing-intervals-card', 'children'),
    Output('cs-anomalies-card', 'children'),
    Input('cs-batch-selector', 'value'),
    Input('cs-view-type', 'value'),
    Input('cs-display-options', 'value')
)
def update_cross_section(batch_id, view_type, display_options):
    if not batch_id:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='请选择一个勘测批次')
        return empty_fig, html.P('请选择批次'), html.P('请选择批次'), html.P('请选择批次')

    measurements = get_measurements_by_batch(batch_id)
    if not measurements:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='该批次没有测量数据')
        return empty_fig, html.P('没有测量数据'), html.P('没有测量数据'), html.P('没有测量数据')

    batch = get_batch(batch_id)
    cs_data = generate_cross_section_data(measurements)
    stats = compute_batch_statistics(batch_id, measurements)
    volume_est = get_volume_estimate(batch_id)
    anomalies = get_anomaly_regions(batch_id)
    missing_intervals = find_missing_intervals([m['angle'] for m in measurements])

    show_missing = 'show_missing' in display_options
    show_anomalies = 'show_anomalies' in display_options
    show_depth_color = 'show_depth_color' in display_options

    if view_type == 'polar':
        fig = create_polar_plot(cs_data, missing_intervals, anomalies,
                                show_missing, show_anomalies, show_depth_color)
    else:
        fig = create_cartesian_plot(cs_data, missing_intervals, anomalies,
                                    show_missing, show_anomalies, show_depth_color)

    fig.update_layout(title=f"断面图 - {batch['batch_name']}")

    stats_content = html.Div([
        html.P([html.Strong("测量点数: "), f"{stats['measurement_count']}"]),
        html.P([html.Strong("最大深度: "), f"{stats['max_depth']:.2f} m"]),
        html.P([html.Strong("最小深度: "), f"{stats['min_depth']:.2f} m"]),
        html.P([html.Strong("平均深度: "), f"{stats['avg_depth']:.2f} m"]),
        html.P([html.Strong("最大距离: "), f"{stats['max_distance']:.2f} m"]),
        html.P([html.Strong("最小距离: "), f"{stats['min_distance']:.2f} m"]),
        html.Hr(),
        html.P([html.Strong("估算容积: "), f"{stats['volume']:.2f} m³"]),
        html.P([html.Strong("计算方法: "), stats['volume_method']]),
    ])

    if missing_intervals:
        missing_df = pd.DataFrame(missing_intervals)
        missing_df['gap_size'] = missing_df['gap_size'].apply(lambda x: f'{x:.1f}°')
        missing_df['start_angle'] = missing_df['start_angle'].apply(lambda x: f'{x:.1f}°')
        missing_df['end_angle'] = missing_df['end_angle'].apply(lambda x: f'{x:.1f}°')
        if 'wraps' in missing_df.columns:
            missing_df = missing_df.drop(columns=['wraps'])
        missing_table = dash_table.DataTable(
            data=missing_df.to_dict('records'),
            columns=[
                {'name': '起始角度', 'id': 'start_angle'},
                {'name': '结束角度', 'id': 'end_angle'},
                {'name': '间隔大小', 'id': 'gap_size'}
            ],
            page_size=5,
            style_table={'overflowX': 'auto'}
        )
    else:
        missing_table = html.P('无缺失区间，数据完整')

    if anomalies:
        anomaly_df = pd.DataFrame(anomalies)
        anomaly_table = dash_table.DataTable(
            data=anomaly_df.to_dict('records'),
            columns=[
                {'name': '起始角度', 'id': 'start_angle'},
                {'name': '结束角度', 'id': 'end_angle'},
                {'name': '类型', 'id': 'anomaly_type'},
                {'name': '描述', 'id': 'description'}
            ],
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )
    else:
        anomaly_table = html.P('未检测到异常区域')

    return fig, stats_content, missing_table, anomaly_table


def create_polar_plot(cs_data, missing_intervals, anomalies, show_missing, show_anomalies, show_depth_color):
    fig = go.Figure()

    angles = cs_data['angles']
    distances = cs_data['distances']
    depths = cs_data['depths']

    if show_depth_color:
        fig.add_trace(go.Scatterpolar(
            r=distances,
            theta=angles,
            mode='markers+lines',
            marker=dict(
                color=depths,
                colorscale='Viridis',
                size=8,
                showscale=True,
                colorbar=dict(title='深度 (m)')
            ),
            line=dict(color='blue', width=1),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.1)',
            name='盐穴断面'
        ))
    else:
        fig.add_trace(go.Scatterpolar(
            r=distances,
            theta=angles,
            mode='markers+lines',
            marker=dict(color='blue', size=8),
            line=dict(color='blue', width=2),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.1)',
            name='盐穴断面'
        ))

    if show_missing and missing_intervals:
        for interval in missing_intervals:
            start_angle = interval['start_angle']
            end_angle = interval['end_angle']

            fig.add_trace(go.Scatterpolar(
                r=[0, max(distances) * 1.1],
                theta=[start_angle, start_angle],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='text',
                text=f'缺失区间起点: {start_angle:.1f}°'
            ))
            fig.add_trace(go.Scatterpolar(
                r=[0, max(distances) * 1.1],
                theta=[end_angle, end_angle],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='缺失区间边界' if interval == missing_intervals[0] else None,
                showlegend=interval == missing_intervals[0],
                hoverinfo='text',
                text=f'缺失区间终点: {end_angle:.1f}°'
            ))

    if show_anomalies and anomalies:
        for i, anomaly in enumerate(anomalies):
            mid_angle = (anomaly['start_angle'] + anomaly['end_angle']) / 2
            fig.add_trace(go.Scatterpolar(
                r=[max(distances) * 0.9],
                theta=[mid_angle],
                mode='markers',
                marker=dict(color='orange', size=15, symbol='circle'),
                name='异常区域' if i == 0 else None,
                showlegend=i == 0,
                hoverinfo='text',
                text=anomaly.get('description', '异常区域')
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(distances) * 1.2],
                title='距离 (m)'
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=90
            )
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='right',
            x=1
        )
    )

    return fig


def create_cartesian_plot(cs_data, missing_intervals, anomalies, show_missing, show_anomalies, show_depth_color):
    fig = go.Figure()

    x = cs_data['x']
    y = cs_data['y']
    depths = cs_data['depths']

    x_closed = x + [x[0]]
    y_closed = y + [y[0]]

    if show_depth_color:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=depths,
                colorscale='Viridis',
                size=10,
                showscale=True,
                colorbar=dict(title='深度 (m)')
            ),
            name='测量点',
            text=[f'角度: {a:.1f}°<br>深度: {d:.2f} m'
                  for a, d in zip(cs_data['angles'], depths)],
            hoverinfo='text'
        ))
        fig.add_trace(go.Scatter(
            x=x_closed,
            y=y_closed,
            mode='lines',
            line=dict(color='blue', width=1),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.1)',
            name='断面轮廓'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x_closed,
            y=y_closed,
            mode='lines+markers',
            marker=dict(color='blue', size=8),
            line=dict(color='blue', width=2),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.1)',
            name='盐穴断面',
            text=[f'角度: {a:.1f}°<br>深度: {d:.2f} m'
                  for a, d in zip(cs_data['angles'], depths)],
            hoverinfo='text'
        ))

    if show_missing and missing_intervals:
        for interval in missing_intervals:
            start_angle_rad = math.radians(interval['start_angle'])
            end_angle_rad = math.radians(interval['end_angle'])
            max_dist = max(max(x), max(y)) * 1.2

            x_start = [0, max_dist * math.cos(start_angle_rad)]
            y_start = [0, max_dist * math.sin(start_angle_rad)]
            x_end = [0, max_dist * math.cos(end_angle_rad)]
            y_end = [0, max_dist * math.sin(end_angle_rad)]

            fig.add_trace(go.Scatter(
                x=x_start,
                y=y_start,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='缺失区间边界' if interval == missing_intervals[0] else None,
                showlegend=interval == missing_intervals[0]
            ))
            fig.add_trace(go.Scatter(
                x=x_end,
                y=y_end,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ))

    if show_anomalies and anomalies:
        for i, anomaly in enumerate(anomalies):
            mid_angle_rad = math.radians((anomaly['start_angle'] + anomaly['end_angle']) / 2)
            max_dist = max(max(x), max(y)) * 0.9
            fig.add_trace(go.Scatter(
                x=[max_dist * math.cos(mid_angle_rad)],
                y=[max_dist * math.sin(mid_angle_rad)],
                mode='markers',
                marker=dict(color='orange', size=15, symbol='circle'),
                name='异常区域' if i == 0 else None,
                showlegend=i == 0,
                text=anomaly.get('description', '异常区域'),
                hoverinfo='text'
            ))

    fig.update_layout(
        xaxis=dict(
            title='X 坐标 (m)',
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='Y 坐标 (m)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='right',
            x=1
        )
    )

    return fig
