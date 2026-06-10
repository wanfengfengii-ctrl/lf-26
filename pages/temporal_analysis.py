import dash
import io
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from flask import send_file

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    get_batch, get_cave
)
from analysis import (
    compute_batch_statistics, calculate_deformation_heatmap,
    compute_cross_section_difference, calculate_volume_trend,
    detect_risk_areas, generate_temporal_analysis_report
)
import json

dash.register_page(__name__, path='/temporal-analysis', name='时序演化分析')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("多时期盐穴形变与容积演化分析", className="mb-4"),

        dbc.Alert(
            "选择盐穴和两个勘测批次，自动生成跨时期形变热力图、断面差值图、容积变化趋势，"
            "并检测新增凹陷/回填/扩容风险区域。支持一键导出完整分析报告。",
            color="info",
            className="mb-4"
        ),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='ta-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=4),
            dbc.Col([
                html.H6("基准批次"),
                dcc.Dropdown(
                    id='ta-base-batch-selector',
                    placeholder='选择基准批次...'
                ),
            ], width=4),
            dbc.Col([
                html.H6("对比批次"),
                dcc.Dropdown(
                    id='ta-compare-batch-selector',
                    placeholder='选择对比批次...'
                ),
            ], width=4),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-play-circle me-2"), "开始分析"],
                    id='ta-analyze-btn',
                    color='primary',
                    n_clicks=0
                ),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "导出分析报告"],
                    id='ta-export-btn',
                    color='success',
                    n_clicks=0,
                    className='ms-2'
                ),
            ], width=12),
        ], className='mb-4'),

        html.Div(id='ta-error-message'),

        dbc.Tabs([
            dbc.Tab(label='形变热力图', tab_id='heatmap-tab'),
            dbc.Tab(label='断面差值图', tab_id='cross-section-tab'),
            dbc.Tab(label='容积趋势', tab_id='volume-trend-tab'),
            dbc.Tab(label='风险分析', tab_id='risk-tab'),
        ], id='ta-tabs', active_tab='heatmap-tab', className='mb-4'),

        html.Div(id='ta-tab-content'),

        dcc.Download(id='ta-report-download'),

        dcc.Store(id='ta-analysis-results'),

    ], fluid=True)


@callback(
    Output('ta-cave-selector', 'options'),
    Output('ta-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    State('ta-cave-selector', 'value'),
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
    Input('ta-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_ta_to_store(cave_id, stored_cave_id):
    if cave_id != stored_cave_id:
        return cave_id
    return dash.no_update


@callback(
    Output('ta-base-batch-selector', 'options'),
    Output('ta-base-batch-selector', 'value'),
    Output('ta-compare-batch-selector', 'options'),
    Output('ta-compare-batch-selector', 'value'),
    Input('ta-cave-selector', 'value')
)
def update_batch_selectors(cave_id):
    if not cave_id:
        return [], None, [], None

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]

    base_value = options[0]['value'] if len(options) > 0 else None
    compare_value = options[1]['value'] if len(options) > 1 else (options[0]['value'] if len(options) > 0 else None)

    return options, base_value, options, compare_value


@callback(
    Output('ta-analysis-results', 'data'),
    Output('ta-error-message', 'children'),
    Input('ta-analyze-btn', 'n_clicks'),
    State('ta-cave-selector', 'value'),
    State('ta-base-batch-selector', 'value'),
    State('ta-compare-batch-selector', 'value'),
    prevent_initial_call=False
)
def run_analysis(n_clicks, cave_id, base_batch_id, compare_batch_id):
    if not cave_id or not base_batch_id or not compare_batch_id:
        return None, ''

    if base_batch_id == compare_batch_id:
        return None, dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "基准批次和对比批次不能相同，请选择两个不同的批次进行分析。"
            ],
            color="danger"
        )

    cave = get_cave(cave_id)
    base_batch = get_batch(base_batch_id)
    compare_batch = get_batch(compare_batch_id)

    base_measurements = get_measurements_by_batch(base_batch_id)
    compare_measurements = get_measurements_by_batch(compare_batch_id)

    if not base_measurements or not compare_measurements:
        return None, dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "所选批次缺少测量数据，请选择有数据的批次。"
            ],
            color="warning"
        )

    base_stats = compute_batch_statistics(base_batch_id, base_measurements)
    base_stats['batch_name'] = base_batch['batch_name']
    base_stats['survey_date'] = base_batch.get('survey_date', '')

    compare_stats = compute_batch_statistics(compare_batch_id, compare_measurements)
    compare_stats['batch_name'] = compare_batch['batch_name']
    compare_stats['survey_date'] = compare_batch.get('survey_date', '')

    all_batches = [base_stats, compare_stats]

    deformation = calculate_deformation_heatmap(base_measurements, compare_measurements)
    cs_diff = compute_cross_section_difference(base_measurements, compare_measurements)

    all_cave_batches = get_batches_by_cave(cave_id)
    all_batch_stats = []
    for batch in all_cave_batches:
        measurements = get_measurements_by_batch(batch['id'])
        if measurements:
            stats = compute_batch_statistics(batch['id'], measurements)
            stats['batch_name'] = batch['batch_name']
            stats['survey_date'] = batch.get('survey_date', '')
            all_batch_stats.append(stats)

    volume_trend = calculate_volume_trend(all_batch_stats)
    risk_areas = detect_risk_areas(base_measurements, compare_measurements)

    results = {
        'cave_name': cave['name'] if cave else '',
        'base_batch': base_batch,
        'compare_batch': compare_batch,
        'base_stats': base_stats,
        'compare_stats': compare_stats,
        'deformation': deformation,
        'cs_diff': cs_diff,
        'volume_trend': volume_trend,
        'risk_areas': risk_areas,
    }

    return results, ''


@callback(
    Output('ta-tab-content', 'children'),
    Input('ta-tabs', 'active_tab'),
    Input('ta-analysis-results', 'data')
)
def render_tab_content(active_tab, results):
    if not results:
        return html.Div([
            dbc.Alert("请选择盐穴和两个不同的批次，然后点击\"开始分析\"按钮。", color="warning")
        ])

    if active_tab == 'heatmap-tab':
        return render_heatmap_tab(results)
    elif active_tab == 'cross-section-tab':
        return render_cross_section_tab(results)
    elif active_tab == 'volume-trend-tab':
        return render_volume_trend_tab(results)
    elif active_tab == 'risk-tab':
        return render_risk_tab(results)

    return html.Div()


def render_heatmap_tab(results):
    deformation = results['deformation']
    base_batch = results['base_batch']
    compare_batch = results['compare_batch']

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("形变类型"),
                dcc.RadioItems(
                    id='ta-heatmap-type',
                    options=[
                        {'label': ' 深度变化', 'value': 'depth'},
                        {'label': ' 径向距离变化', 'value': 'distance'}
                    ],
                    value='depth',
                    inline=True
                ),
            ], width=12),
        ], className='mb-3'),

        dcc.Graph(id='ta-heatmap-figure', style={'height': '550px'}),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"深度变化统计（{base_batch['batch_name']} → {compare_batch['batch_name']}）"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P([html.Strong("最大深度增加: "),
                                            f"{deformation['max_depth_increase']:+.2f} m"]),
                                ], width=6),
                                dbc.Col([
                                    html.P([html.Strong("最大深度减少: "),
                                            f"{deformation['max_depth_decrease']:+.2f} m"]),
                                ], width=6),
                            ]),
                            html.P([html.Strong("平均深度变化: "),
                                    f"{deformation['avg_depth_change']:+.2f} m"]),
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"径向距离变化统计"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.P([html.Strong("最大径向扩张: "),
                                            f"{deformation['max_distance_expansion']:+.2f} m"]),
                                ], width=6),
                                dbc.Col([
                                    html.P([html.Strong("最大径向收缩: "),
                                            f"{deformation['max_distance_contraction']:+.2f} m"]),
                                ], width=6),
                            ]),
                            html.P([html.Strong("平均径向变化: "),
                                    f"{deformation['avg_distance_change']:+.2f} m"]),
                        ])
                    ])
                ])
            ], width=6),
        ]),
    ])


@callback(
    Output('ta-heatmap-figure', 'figure'),
    Input('ta-heatmap-type', 'value'),
    State('ta-analysis-results', 'data')
)
def update_heatmap(heatmap_type, results):
    if not results:
        return go.Figure()

    deformation = results['deformation']
    angles = deformation['angles']

    if heatmap_type == 'depth':
        diff_values = deformation['depth_diff']
        color_title = '深度变化 (m)'
        plot_title = '深度形变热力图'
    else:
        diff_values = deformation['distance_diff']
        color_title = '径向距离变化 (m)'
        plot_title = '径向形变热力图'

    fig = go.Figure()

    diff_array = np.array(diff_values)
    angles_array = np.array(angles)

    max_abs = max(abs(np.min(diff_array)), abs(np.max(diff_array)))
    if max_abs == 0:
        max_abs = 1.0

    num_angles = len(angles_array)
    num_radii = 2

    r_grid = np.array([0, max_abs * 1.1])
    theta_grid = np.deg2rad(angles_array)

    heatmap_data = np.zeros((num_radii, num_angles))
    for i in range(num_angles):
        heatmap_data[:, i] = diff_array[i]

    fig.add_trace(go.Barpolar(
        r=[max_abs * 1.1] * num_angles,
        theta=angles_array,
        width=[360.0 / num_angles] * num_angles,
        marker=dict(
            color=diff_array,
            colorscale='RdBu_r',
            cmin=-max_abs,
            cmax=max_abs,
            colorbar=dict(title=color_title, x=-0.1),
            showscale=True
        ),
        name='形变量',
        hovertemplate='角度: %{theta:.1f}°<br>变化: %{marker.color:.2f} m<extra></extra>'
    ))

    angles_rad = np.deg2rad(angles_array)
    base_r = max_abs * 0.3
    x_base = base_r * np.cos(angles_rad)
    y_base = base_r * np.sin(angles_rad)

    fig.add_trace(go.Scatterpolar(
        r=[base_r] * num_angles,
        theta=angles_array,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='零变化基线',
        showlegend=True
    ))

    fig.update_layout(
        title=plot_title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_abs * 1.2],
                title='形变大小 (m)'
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


def render_cross_section_tab(results):
    cs_diff = results['cs_diff']
    base_batch = results['base_batch']
    compare_batch = results['compare_batch']

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("显示模式"),
                dcc.RadioItems(
                    id='ta-cs-mode',
                    options=[
                        {'label': ' 叠加对比', 'value': 'overlay'},
                        {'label': ' 差值矢量', 'value': 'diff_vector'}
                    ],
                    value='overlay',
                    inline=True
                ),
            ], width=12),
        ], className='mb-3'),

        dcc.Graph(id='ta-cs-figure', style={'height': '550px'}),

        html.Hr(),

        dbc.Card([
            dbc.CardHeader("断面差值说明"),
            dbc.CardBody([
                html.Ul([
                    html.Li([html.Strong("叠加对比: "), "同时显示两个批次的断面轮廓，直观比较形状变化"]),
                    html.Li([html.Strong("差值矢量: "), "用箭头表示每个角度上的断面位移方向和大小"]),
                    html.Li(f"对比基准: {base_batch['batch_name']} → {compare_batch['batch_name']}"),
                ])
            ])
        ]),
    ])


@callback(
    Output('ta-cs-figure', 'figure'),
    Input('ta-cs-mode', 'value'),
    State('ta-analysis-results', 'data')
)
def update_cross_section(mode, results):
    if not results:
        return go.Figure()

    cs_diff = results['cs_diff']
    base_batch = results['base_batch']
    compare_batch = results['compare_batch']

    fig = go.Figure()

    if mode == 'overlay':
        x_base = cs_diff['x_base']
        y_base = cs_diff['y_base']
        x_compare = cs_diff['x_compare']
        y_compare = cs_diff['y_compare']

        x_base_closed = x_base + [x_base[0]]
        y_base_closed = y_base + [y_base[0]]
        x_compare_closed = x_compare + [x_compare[0]]
        y_compare_closed = y_compare + [y_compare[0]]

        fig.add_trace(go.Scatter(
            x=x_base_closed,
            y=y_base_closed,
            mode='lines',
            line=dict(color='blue', width=2),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.15)',
            name=f'基准: {base_batch["batch_name"]}'
        ))

        fig.add_trace(go.Scatter(
            x=x_compare_closed,
            y=y_compare_closed,
            mode='lines',
            line=dict(color='red', width=2),
            fill='toself',
            fillcolor='rgba(255, 100, 100, 0.15)',
            name=f'对比: {compare_batch["batch_name"]}'
        ))

        all_x = x_base + x_compare
        all_y = y_base + y_compare
        max_range = max(max(abs(min(all_x)), abs(max(all_x))),
                        max(abs(min(all_y)), abs(max(all_y)))) * 1.1

        fig.update_layout(
            title='断面叠加对比图',
            xaxis=dict(
                title='X 坐标 (m)',
                scaleanchor='y',
                scaleratio=1,
                range=[-max_range, max_range]
            ),
            yaxis=dict(
                title='Y 坐标 (m)',
                range=[-max_range, max_range]
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

    else:
        x_base = cs_diff['x_base']
        y_base = cs_diff['y_base']
        x_diff = cs_diff['x_diff']
        y_diff = cs_diff['y_diff']
        radial_diff = cs_diff['radial_diff']

        x_base_closed = x_base + [x_base[0]]
        y_base_closed = y_base + [y_base[0]]

        fig.add_trace(go.Scatter(
            x=x_base_closed,
            y=y_base_closed,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'基准: {base_batch["batch_name"]}'
        ))

        step = max(1, len(x_base) // 24)
        for i in range(0, len(x_base), step):
            fig.add_trace(go.Scatter(
                x=[x_base[i], x_base[i] + x_diff[i]],
                y=[y_base[i], y_base[i] + y_diff[i]],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=[0, 6], symbol=['circle', 'arrow-bar-up']),
                showlegend=False,
                hovertemplate=f'角度: {cs_diff["angles"][i]:.1f}°<br>径向变化: {radial_diff[i]:+.2f} m<extra></extra>'
            ))

        all_x = x_base + [x + d for x, d in zip(x_base, x_diff)]
        all_y = y_base + [y + d for y, d in zip(y_base, y_diff)]
        max_range = max(max(abs(min(all_x)), abs(max(all_x))),
                        max(abs(min(all_y)), abs(max(all_y)))) * 1.1

        fig.update_layout(
            title='断面差值矢量图',
            xaxis=dict(
                title='X 坐标 (m)',
                scaleanchor='y',
                scaleratio=1,
                range=[-max_range, max_range]
            ),
            yaxis=dict(
                title='Y 坐标 (m)',
                range=[-max_range, max_range]
            ),
            showlegend=True
        )

    return fig


def render_volume_trend_tab(results):
    volume_trend = results['volume_trend']
    base_stats = results['base_stats']
    compare_stats = results['compare_stats']

    return html.Div([
        dbc.Alert(
            [
                html.I(className="bi bi-info-circle me-2"),
                "下图展示该盐穴所有历史勘测批次的容积变化趋势。"
            ],
            color="info",
            className="mb-3"
        ),

        dcc.Graph(
            figure=create_volume_trend_figure(volume_trend),
            style={'height': '500px'}
        ),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("容积变化总览"),
                    dbc.CardBody([
                        html.Div([
                            html.P([html.Strong("基准容积: "),
                                    f"{volume_trend['volumes'][0]:.2f} m³"]),
                            html.P([html.Strong("对比容积: "),
                                    f"{volume_trend['volumes'][-1]:.2f} m³"]),
                            html.P([html.Strong("容积变化量: "),
                                    html.Span(
                                        f"{volume_trend['total_volume_change']:+.2f} m³",
                                        style={'color': 'green' if volume_trend['total_volume_change'] > 0 else 'red'}
                                    )]),
                            html.P([html.Strong("容积变化率: "),
                                    html.Span(
                                        f"{volume_trend['total_volume_change_pct']:+.2f} %",
                                        style={'color': 'green' if volume_trend['total_volume_change_pct'] > 0 else 'red'}
                                    )]),
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("深度变化总览"),
                    dbc.CardBody([
                        html.Div([
                            html.P([html.Strong("基准最大深度: "),
                                    f"{base_stats['max_depth']:.2f} m"]),
                            html.P([html.Strong("对比最大深度: "),
                                    f"{compare_stats['max_depth']:.2f} m"]),
                            html.P([html.Strong("最大深度变化: "),
                                    f"{compare_stats['max_depth'] - base_stats['max_depth']:+.2f} m"]),
                            html.P([html.Strong("平均深度变化: "),
                                    f"{compare_stats['avg_depth'] - base_stats['avg_depth']:+.2f} m"]),
                        ])
                    ])
                ])
            ], width=6),
        ]),

        html.Hr(),

        dbc.Card([
            dbc.CardHeader("批次容积数据"),
            dbc.CardBody([
                dash_table.DataTable(
                    data=pd.DataFrame({
                        '批次名称': volume_trend['batch_names'],
                        '勘测日期': volume_trend['dates'],
                        '容积 (m³)': [f'{v:.2f}' for v in volume_trend['volumes']],
                        '变化率 (%)': [f'{v:+.2f}' for v in volume_trend['volume_changes']],
                        '最大深度 (m)': [f'{d:.2f}' for d in volume_trend['max_depths']],
                    }).to_dict('records'),
                    columns=[
                        {'name': '批次名称', 'id': '批次名称'},
                        {'name': '勘测日期', 'id': '勘测日期'},
                        {'name': '容积 (m³)', 'id': '容积 (m³)'},
                        {'name': '变化率 (%)', 'id': '变化率 (%)'},
                        {'name': '最大深度 (m)', 'id': '最大深度 (m)'},
                    ],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                )
            ])
        ]),
    ])


def create_volume_trend_figure(volume_trend):
    fig = go.Figure()

    batch_names = volume_trend['batch_names']
    volumes = volume_trend['volumes']
    max_depths = volume_trend['max_depths']
    avg_depths = volume_trend['avg_depths']

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

    fig.add_trace(go.Scatter(
        x=batch_names,
        y=avg_depths,
        name='平均深度 (m)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='容积与深度变化趋势',
        xaxis=dict(title='勘测批次'),
        yaxis=dict(
            title=dict(text='容积 (m³)'),
            side='left'
        ),
        yaxis2=dict(
            title=dict(text='深度 (m)'),
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


def render_risk_tab(results):
    risk_areas = results['risk_areas']
    base_batch = results['base_batch']
    compare_batch = results['compare_batch']

    risk_count = len(risk_areas)
    high_risk = sum(1 for r in risk_areas if r['severity'] == '高')
    medium_risk = sum(1 for r in risk_areas if r['severity'] == '中')
    low_risk = sum(1 for r in risk_areas if r['severity'] == '低')

    new_pits = [r for r in risk_areas if r['risk_type'] == '新增凹陷']
    backfills = [r for r in risk_areas if r['risk_type'] == '回填']
    expansions = [r for r in risk_areas if r['risk_type'] == '扩容风险']

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("风险统计"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H2(f"{risk_count}", className="text-center"),
                                html.P("风险区域总数", className="text-center text-muted")
                            ], width=3),
                            dbc.Col([
                                html.H2(f"{high_risk}", className="text-center text-danger"),
                                html.P("高风险", className="text-center text-muted")
                            ], width=3),
                            dbc.Col([
                                html.H2(f"{medium_risk}", className="text-center text-warning"),
                                html.P("中风险", className="text-center text-muted")
                            ], width=3),
                            dbc.Col([
                                html.H2(f"{low_risk}", className="text-center text-info"),
                                html.P("低风险", className="text-center text-muted")
                            ], width=3),
                        ])
                    ])
                ])
            ], width=12),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-exclamation-triangle-fill text-danger me-2"),
                        "新增凹陷风险"
                    ]),
                    dbc.CardBody([
                        html.P(f"共检测到 {len(new_pits)} 处新增凹陷风险区域"),
                        html.Div(create_risk_detail_list(new_pits, 'depth_increase'))
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-arrow-down-circle-fill text-info me-2"),
                        "回填区域"
                    ]),
                    dbc.CardBody([
                        html.P(f"共检测到 {len(backfills)} 处回填区域"),
                        html.Div(create_risk_detail_list(backfills, 'depth_decrease'))
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-arrows-expand text-warning me-2"),
                        "扩容风险"
                    ]),
                    dbc.CardBody([
                        html.P(f"共检测到 {len(expansions)} 处扩容风险区域"),
                        html.Div(create_risk_detail_list(expansions, 'distance_increase'))
                    ])
                ])
            ], width=4),
        ], className='mb-4'),

        dcc.Graph(
            figure=create_risk_heatmap(results),
            style={'height': '500px'}
        ),

        html.Hr(),

        dbc.Card([
            dbc.CardHeader("风险区域明细表"),
            dbc.CardBody([
                dash_table.DataTable(
                    data=format_risk_table_data(risk_areas),
                    columns=[
                        {'name': '序号', 'id': '序号'},
                        {'name': '风险类型', 'id': '风险类型'},
                        {'name': '严重程度', 'id': '严重程度'},
                        {'name': '起始角度', 'id': '起始角度'},
                        {'name': '结束角度', 'id': '结束角度'},
                        {'name': '最大变化量', 'id': '最大变化量'},
                        {'name': '平均变化量', 'id': '平均变化量'},
                        {'name': '描述', 'id': '描述'},
                    ],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{严重程度} = "高"',
                            },
                            'backgroundColor': 'rgba(255, 200, 200, 0.3)',
                            'color': 'darkred'
                        },
                        {
                            'if': {
                                'filter_query': '{严重程度} = "中"',
                            },
                            'backgroundColor': 'rgba(255, 240, 200, 0.3)',
                        },
                        {
                            'if': {
                                'filter_query': '{严重程度} = "低"',
                            },
                            'backgroundColor': 'rgba(200, 230, 255, 0.3)',
                        }
                    ]
                )
            ])
        ]),
    ])


def create_risk_detail_list(risk_areas, change_type):
    if not risk_areas:
        return html.P("暂无风险区域", className="text-muted fst-italic")

    items = []
    for i, risk in enumerate(risk_areas, 1):
        if change_type == 'depth_increase':
            change_val = risk.get('max_depth_increase', 0)
            change_text = f"最大加深 {change_val:+.2f}m"
        elif change_type == 'depth_decrease':
            change_val = risk.get('max_depth_decrease', 0)
            change_text = f"最大变浅 {change_val:+.2f}m"
        else:
            change_val = risk.get('max_distance_increase', 0)
            change_text = f"最大扩张 {change_val:+.2f}m"

        severity_color = {
            '高': 'danger',
            '中': 'warning',
            '低': 'info'
        }.get(risk['severity'], 'secondary')

        items.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.Strong(f"#{i} "),
                        html.Span(f"{risk['start_angle']:.0f}° - {risk['end_angle']:.0f}°")
                    ], width=8),
                    dbc.Col([
                        dbc.Badge(risk['severity'], color=severity_color, className='float-end')
                    ], width=4),
                ]),
                html.Small([
                    change_text,
                    html.Br(),
                    html.Span(risk['description'], className="text-muted")
                ])
            ], className="mb-2")
        )

    return dbc.ListGroup(items, flush=True)


def format_risk_table_data(risk_areas):
    data = []
    for i, risk in enumerate(risk_areas, 1):
        max_change = ''
        avg_change = ''
        if 'max_depth_increase' in risk:
            max_change = f"+{risk['max_depth_increase']:.2f} m"
            avg_change = f"+{risk['avg_depth_increase']:.2f} m"
        elif 'max_depth_decrease' in risk:
            max_change = f"{risk['max_depth_decrease']:.2f} m"
            avg_change = f"{risk['avg_depth_decrease']:.2f} m"
        elif 'max_distance_increase' in risk:
            max_change = f"+{risk['max_distance_increase']:.2f} m"
            avg_change = f"+{risk['avg_distance_increase']:.2f} m"

        data.append({
            '序号': i,
            '风险类型': risk['risk_type'],
            '严重程度': risk['severity'],
            '起始角度': f"{risk['start_angle']:.1f}°",
            '结束角度': f"{risk['end_angle']:.1f}°",
            '最大变化量': max_change,
            '平均变化量': avg_change,
            '描述': risk['description']
        })
    return data


def create_risk_heatmap(results):
    risk_areas = results['risk_areas']
    deformation = results['deformation']
    angles = deformation['angles']
    depth_diff = deformation['depth_diff']

    fig = go.Figure()

    depth_array = np.array(depth_diff)
    angles_array = np.array(angles)

    max_abs = max(abs(np.min(depth_array)), abs(np.max(depth_array)))
    if max_abs == 0:
        max_abs = 1.0

    num_angles = len(angles_array)

    fig.add_trace(go.Barpolar(
        r=[max_abs * 1.1] * num_angles,
        theta=angles_array,
        width=[360.0 / num_angles] * num_angles,
        marker=dict(
            color=depth_array,
            colorscale='RdBu_r',
            cmin=-max_abs,
            cmax=max_abs,
            colorbar=dict(title='深度变化 (m)'),
            showscale=True
        ),
        name='深度变化',
        hovertemplate='角度: %{theta:.1f}°<br>深度变化: %{marker.color:.2f} m<extra></extra>'
    ))

    risk_colors = {
        '新增凹陷': 'red',
        '回填': 'blue',
        '扩容风险': 'orange'
    }

    for i, risk in enumerate(risk_areas):
        mid_angle = (risk['start_angle'] + risk['end_angle']) / 2
        color = risk_colors.get(risk['risk_type'], 'gray')

        fig.add_trace(go.Scatterpolar(
            r=[max_abs * 0.9],
            theta=[mid_angle],
            mode='markers',
            marker=dict(
                color=color,
                size=15,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            name=f"{risk['risk_type']} ({risk['severity']})",
            showlegend=True,
            legendgroup=risk['risk_type'],
            hovertemplate=f"{risk['risk_type']}<br>严重程度: {risk['severity']}<br>{risk['description']}<extra></extra>"
        ))

    fig.update_layout(
        title='风险区域分布图',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_abs * 1.2],
                title='形变大小 (m)'
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
            y=-0.15,
            xanchor='right',
            x=1
        )
    )

    return fig


@callback(
    Output('ta-report-download', 'data'),
    Input('ta-export-btn', 'n_clicks'),
    State('ta-analysis-results', 'data'),
    State('ta-cave-selector', 'value'),
    prevent_initial_call=True
)
def export_report(n_clicks, results, cave_id):
    if not results or n_clicks == 0:
        return dash.no_update

    cave_name = results['cave_name']
    base_batch = results['base_batch']
    compare_batch = results['compare_batch']
    deformation = results['deformation']
    volume_trend = results['volume_trend']
    risk_areas = results['risk_areas']
    cs_diff = results['cs_diff']

    report_content = generate_temporal_analysis_report(
        cave_name, base_batch, compare_batch,
        deformation, volume_trend, risk_areas, cs_diff
    )

    filename = f"盐穴形变分析报告_{cave_name}_{base_batch['batch_name']}_vs_{compare_batch['batch_name']}.txt"

    return dict(
        content=report_content,
        filename=filename,
        type='text/plain'
    )
