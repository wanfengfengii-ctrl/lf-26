import dash
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    get_volume_estimate, get_batch, get_anomaly_regions
)
from analysis import (
    generate_cross_section_data, compare_batches, compute_batch_statistics
)

dash.register_page(__name__, path='/batch-compare', name='批次对比')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("批次对比", className="mb-4"),

        dbc.Alert(
            "注意：不同盐穴的数据不能直接合并比较。请先选择盐穴，然后选择该盐穴下的多个批次进行对比。",
            color="info",
            className="mb-4"
        ),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='cmp-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=12),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                html.H6("选择要对比的批次（可多选）"),
                dcc.Dropdown(
                    id='cmp-batch-selector',
                    multi=True,
                    placeholder='选择多个批次...'
                ),
            ], width=12),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                html.H6("对比视图"),
                dcc.RadioItems(
                    id='cmp-view-type',
                    options=[
                        {'label': ' 断面叠加', 'value': 'cross_section'},
                        {'label': ' 深度对比', 'value': 'depth_compare'},
                        {'label': ' 容积变化', 'value': 'volume_change'}
                    ],
                    value='cross_section',
                    inline=True
                ),
            ], width=12),
        ], className='mb-4'),

        dcc.Graph(id='cmp-plot', style={'height': '500px'}),

        html.Hr(),

        dbc.Card([
            dbc.CardHeader("批次对比数据"),
            dbc.CardBody(id='cmp-stats-table')
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader("容积变化详情"),
            dbc.CardBody(id='cmp-volume-change-table')
        ]),
    ], fluid=True)


@callback(
    Output('cmp-batch-selector', 'options'),
    Output('cmp-batch-selector', 'value'),
    Input('cmp-cave-selector', 'value')
)
def update_batch_selector(cave_id):
    if not cave_id:
        return [], []

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]

    default_values = [opt['value'] for opt in options[:min(3, len(options))]]

    return options, default_values


@callback(
    Output('cmp-plot', 'figure'),
    Output('cmp-stats-table', 'children'),
    Output('cmp-volume-change-table', 'children'),
    Input('cmp-batch-selector', 'value'),
    Input('cmp-view-type', 'value'),
    Input('cmp-cave-selector', 'value')
)
def update_comparison(batch_ids, view_type, cave_id):
    if not batch_ids or len(batch_ids) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='请至少选择2个批次进行对比')
        return empty_fig, html.P('请至少选择2个批次'), html.P('请至少选择2个批次')

    all_stats = []
    all_data = []

    for batch_id in batch_ids:
        batch = get_batch(batch_id)
        measurements = get_measurements_by_batch(batch_id)
        if not measurements:
            continue

        stats = compute_batch_statistics(batch_id, measurements)
        stats['batch_id'] = batch_id
        stats['batch_name'] = batch['batch_name']
        stats['survey_date'] = batch.get('survey_date', '')
        all_stats.append(stats)

        cs_data = generate_cross_section_data(measurements)
        cs_data['batch_name'] = batch['batch_name']
        cs_data['batch_id'] = batch_id
        all_data.append(cs_data)

    if len(all_data) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='有效批次不足2个，无法对比')
        return empty_fig, html.P('有效批次不足2个'), html.P('有效批次不足2个')

    if view_type == 'cross_section':
        fig = create_comparison_cross_section(all_data)
    elif view_type == 'depth_compare':
        fig = create_depth_comparison(all_data)
    else:
        fig = create_volume_change_chart(all_stats)

    stats_table = create_stats_table(all_stats)
    volume_change_table = create_volume_change_table(all_stats)

    return fig, stats_table, volume_change_table


def create_comparison_cross_section(all_data):
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        x = data['x']
        y = data['y']

        x_closed = x + [x[0]]
        y_closed = y + [y[0]]

        fig.add_trace(go.Scatter(
            x=x_closed,
            y=y_closed,
            mode='lines+markers',
            marker=dict(color=color, size=6),
            line=dict(color=color, width=2),
            fill='toself',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
            name=data['batch_name'],
            text=[f'角度: {a:.1f}°<br>深度: {d:.2f} m'
                  for a, d in zip(data['angles'], data['depths'])],
            hoverinfo='text+name'
        ))

    fig.update_layout(
        title='断面叠加对比',
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


def create_depth_comparison(all_data):
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        angles = data['angles']
        depths = data['depths']

        sorted_indices = sorted(range(len(angles)), key=lambda k: angles[k])
        sorted_angles = [angles[i] for i in sorted_indices]
        sorted_depths = [depths[i] for i in sorted_indices]

        fig.add_trace(go.Scatter(
            x=sorted_angles,
            y=sorted_depths,
            mode='lines+markers',
            marker=dict(color=color, size=6),
            line=dict(color=color, width=2),
            name=data['batch_name']
        ))

    fig.update_layout(
        title='深度分布对比',
        xaxis=dict(title='角度 (°)'),
        yaxis=dict(title='深度 (m)', autorange='reversed'),
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


def create_volume_change_chart(all_stats):
    fig = go.Figure()

    batch_names = [s['batch_name'] for s in all_stats]
    volumes = [s['volume'] for s in all_stats]
    max_depths = [s['max_depth'] for s in all_stats]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=batch_names,
        y=volumes,
        name='容积 (m³)',
        yaxis='y',
        marker_color='rgba(55, 83, 109, 0.7)',
        text=[f'{v:.1f} m³' for v in volumes],
        textposition='auto'
    ))

    fig.add_trace(go.Scatter(
        x=batch_names,
        y=max_depths,
        name='最大深度 (m)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='容积与最大深度对比',
        xaxis=dict(title='勘测批次'),
        yaxis=dict(
            title=dict(text='容积 (m³)'),
            side='left'
        ),
        yaxis2=dict(
            title=dict(text='最大深度 (m)'),
            side='right',
            overlaying='y'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='right',
            x=1
        ),
        barmode='group'
    )

    return fig


def create_stats_table(all_stats):
    if not all_stats:
        return html.P('没有数据')

    df = pd.DataFrame(all_stats)
    display_df = pd.DataFrame({
        '批次名称': df['batch_name'],
        '勘测日期': df['survey_date'],
        '测量点数': df['measurement_count'],
        '容积 (m³)': df['volume'].apply(lambda x: f'{x:.2f}'),
        '最大深度 (m)': df['max_depth'].apply(lambda x: f'{x:.2f}'),
        '平均深度 (m)': df['avg_depth'].apply(lambda x: f'{x:.2f}'),
        '最大距离 (m)': df['max_distance'].apply(lambda x: f'{x:.2f}'),
    })

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    )

    return table


def create_volume_change_table(all_stats):
    if len(all_stats) < 2:
        return html.P('需要至少2个批次进行对比')

    comparison = compare_batches(all_stats)

    if not comparison:
        return html.P('没有可比数据')

    df = pd.DataFrame(comparison)

    display_df = pd.DataFrame({
        '批次1': df['batch1_name'],
        '批次2': df['batch2_name'],
        '容积变化 (m³)': df['volume_diff'].apply(lambda x: f'{x:+.2f}'),
        '容积变化率 (%)': df['volume_change_pct'].apply(lambda x: f'{x:+.2f}%'),
        '最大深度变化 (m)': df['max_depth_diff'].apply(lambda x: f'{x:+.2f}'),
    })

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{容积变化 (m³)} contains "+"',
                    'column_id': '容积变化 (m³)'
                },
                'color': 'green'
            },
            {
                'if': {
                    'filter_query': '{容积变化 (m³)} contains "-"',
                    'column_id': '容积变化 (m³)'
                },
                'color': 'red'
            }
        ]
    )

    return table
