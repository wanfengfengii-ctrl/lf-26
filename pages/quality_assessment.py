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
    evaluate_batch_quality, compute_batch_statistics, generate_quality_report
)

dash.register_page(__name__, path='/quality-assessment', name='质量评估中心')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("勘测数据质量评估与修复建议中心", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='qa-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=6),
            dbc.Col([
                html.H6("选择批次"),
                dcc.Dropdown(
                    id='qa-batch-selector',
                    placeholder='选择一个批次...'
                ),
            ], width=6),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("综合质量评分"),
                    dbc.CardBody([
                        html.Div(id='qa-overall-score', className='text-center')
                    ])
                ]),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("质量等级"),
                    dbc.CardBody([
                        html.Div(id='qa-grade', className='text-center')
                    ])
                ]),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("发现问题"),
                    dbc.CardBody([
                        html.Div(id='qa-issue-count', className='text-center')
                    ])
                ]),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("测量点数"),
                    dbc.CardBody([
                        html.Div(id='qa-measurement-count', className='text-center')
                    ])
                ]),
            ], width=3),
        ], className='mb-4'),

        dbc.Tabs([
            dbc.Tab(label='质量热力图', tab_id='heatmap-tab'),
            dbc.Tab(label='分项评估', tab_id='subscores-tab'),
            dbc.Tab(label='问题明细', tab_id='issues-tab'),
            dbc.Tab(label='修复建议', tab_id='suggestions-tab'),
        ], id='qa-tabs', active_tab='heatmap-tab', className='mb-4'),

        html.Div(id='qa-tab-content'),

        html.Div([
            dbc.Button(
                '导出质量评估报告',
                id='qa-export-btn',
                color='primary',
                className='float-end'
            ),
            dcc.Download(id='qa-download-report'),
        ], className='mb-4'),

        dcc.Store(id='qa-quality-result'),
        dcc.Store(id='qa-cave-name'),
        dcc.Store(id='qa-batch-name'),
    ], fluid=True)


@callback(
    Output('qa-cave-selector', 'options'),
    Output('qa-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    State('qa-cave-selector', 'value'),
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
    Input('qa-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_qa_to_store(cave_id, stored_cave_id):
    if cave_id != stored_cave_id:
        return cave_id
    return dash.no_update


@callback(
    Output('qa-batch-selector', 'options'),
    Output('qa-batch-selector', 'value'),
    Input('qa-cave-selector', 'value')
)
def update_batch_selector(cave_id):
    if not cave_id:
        return [], None

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]

    default_value = options[0]['value'] if options else None

    return options, default_value


@callback(
    Output('qa-overall-score', 'children'),
    Output('qa-grade', 'children'),
    Output('qa-issue-count', 'children'),
    Output('qa-measurement-count', 'children'),
    Output('qa-quality-result', 'data'),
    Output('qa-cave-name', 'data'),
    Output('qa-batch-name', 'data'),
    Input('qa-batch-selector', 'value'),
    Input('qa-cave-selector', 'value')
)
def update_quality_overview(batch_id, cave_id):
    if not batch_id or not cave_id:
        return (
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            {}, '', ''
        )

    measurements = get_measurements_by_batch(batch_id)
    if not measurements:
        return (
            html.H2('0', className='text-muted'),
            html.H2('无数据', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('0', className='text-muted'),
            {}, '', ''
        )

    cave = get_cave(cave_id)
    batch = get_batch(batch_id)
    cave_name = cave['name'] if cave else ''
    batch_name = batch['batch_name'] if batch else ''

    all_batches = get_batches_by_cave(cave_id)
    all_batches_stats = []
    for b in all_batches:
        b_measurements = get_measurements_by_batch(b['id'])
        if b_measurements:
            stats = compute_batch_statistics(b['id'], b_measurements)
            stats['batch_name'] = b['batch_name']
            all_batches_stats.append(stats)

    quality_result = evaluate_batch_quality(batch_id, measurements, all_batches_stats)

    score_color = 'success' if quality_result['overall_score'] >= 80 else (
        'warning' if quality_result['overall_score'] >= 60 else 'danger'
    )
    grade_color = 'success' if quality_result['grade'] in ['优秀', '良好'] else (
        'warning' if quality_result['grade'] == '合格' else 'danger'
    )

    overall_score_el = html.Div([
        html.H2(f"{quality_result['overall_score']:.1f}", className=f'text-{score_color}'),
        html.Small('满分 100 分', className='text-muted')
    ])

    grade_el = html.Div([
        html.H2(quality_result['grade'], className=f'text-{grade_color}'),
        html.Small('质量等级', className='text-muted')
    ])

    issue_count_el = html.Div([
        html.H2(f"{quality_result.get('issue_count', 0)}", className='text-danger'),
        html.Small('项问题', className='text-muted')
    ])

    measurement_count_el = html.Div([
        html.H2(f"{len(measurements)}", className='text-info'),
        html.Small('个测量点', className='text-muted')
    ])

    return (
        overall_score_el,
        grade_el,
        issue_count_el,
        measurement_count_el,
        quality_result,
        cave_name,
        batch_name
    )


@callback(
    Output('qa-tab-content', 'children'),
    Input('qa-tabs', 'active_tab'),
    Input('qa-quality-result', 'data'),
    Input('qa-batch-selector', 'value')
)
def render_tab_content(active_tab, quality_result, batch_id):
    if not quality_result or not quality_result.get('heatmap_data'):
        return html.P('请选择一个有数据的批次以查看质量评估结果')

    if active_tab == 'heatmap-tab':
        return render_heatmap_tab(quality_result)
    elif active_tab == 'subscores-tab':
        return render_subscores_tab(quality_result)
    elif active_tab == 'issues-tab':
        return render_issues_tab(quality_result)
    elif active_tab == 'suggestions-tab':
        return render_suggestions_tab(quality_result)
    else:
        return html.P('未知标签页')


def render_heatmap_tab(quality_result):
    heatmap_data = quality_result.get('heatmap_data', {})
    angles = heatmap_data.get('angles', [])
    quality_scores = heatmap_data.get('quality_scores', [])
    distance_deviations = heatmap_data.get('distance_deviations', [])
    depth_deviations = heatmap_data.get('depth_deviations', [])

    if not angles:
        return html.P('暂无数据')

    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=quality_scores,
        theta=angles,
        width=[360 / len(angles)] * len(angles),
        marker=dict(
            color=quality_scores,
            colorscale=[
                [0, 'red'],
                [0.4, 'orange'],
                [0.6, 'yellow'],
                [0.8, 'lightgreen'],
                [1, 'green']
            ],
            cmin=0,
            cmax=100,
            colorbar=dict(title='质量分数')
        ),
        name='质量评分',
        hovertemplate='角度: %{theta}°<br>质量评分: %{r:.1f}<extra></extra>'
    ))

    fig.update_layout(
        title='数据质量热力图（极坐标）',
        polar=dict(
            radialaxis=dict(range=[0, 100], title='质量评分'),
            angularaxis=dict(direction='clockwise', rotation=90)
        ),
        height=500
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=angles,
        y=distance_deviations,
        mode='lines+markers',
        name='距离偏差 (Z分数)',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    fig2.add_trace(go.Scatter(
        x=angles,
        y=depth_deviations,
        mode='lines+markers',
        name='深度偏差 (Z分数)',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))

    fig2.add_hline(y=1, line_dash='dash', line_color='orange', annotation_text='1σ')
    fig2.add_hline(y=2, line_dash='dash', line_color='red', annotation_text='2σ')
    fig2.add_hline(y=3, line_dash='dash', line_color='darkred', annotation_text='3σ')

    fig2.update_layout(
        title='各角度数据偏差分布',
        xaxis=dict(title='角度 (°)'),
        yaxis=dict(title='Z 分数'),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='right', x=1)
    )

    return html.Div([
        dcc.Graph(figure=fig),
        html.Hr(),
        dcc.Graph(figure=fig2),
        dbc.Alert(
            "热力图颜色越绿表示数据质量越好，越红表示数据质量越差。"
            "Z分数越高表示该点与平均值偏差越大，越可能是异常点。",
            color="info",
            className="mt-3"
        )
    ])


def render_subscores_tab(quality_result):
    subscores = [
        {'name': '角度覆盖率', 'key': 'angle_coverage', 'weight': 0.2},
        {'name': '异常跳点检测', 'key': 'outlier_detection', 'weight': 0.2},
        {'name': '缺失区间检测', 'key': 'missing_intervals', 'weight': 0.15},
        {'name': '重复趋势检测', 'key': 'repetitive_patterns', 'weight': 0.15},
        {'name': '数据波动性', 'key': 'volatility', 'weight': 0.15},
        {'name': '批次间一致性', 'key': 'batch_consistency', 'weight': 0.15},
    ]

    scores_data = []
    for item in subscores:
        section = quality_result.get(item['key'], {})
        score = section.get('score', 0)
        scores_data.append({
            '评估项目': item['name'],
            '权重': f'{item["weight"] * 100:.0f}%',
            '得分': f'{score:.1f}',
            '加权得分': f'{score * item["weight"]:.2f}',
            '状态': '优秀' if score >= 90 else ('良好' if score >= 75 else ('合格' if score >= 60 else '较差'))
        })

    scores_table = dash_table.DataTable(
        data=scores_data,
        columns=[
            {'name': '评估项目', 'id': '评估项目'},
            {'name': '权重', 'id': '权重'},
            {'name': '得分', 'id': '得分'},
            {'name': '加权得分', 'id': '加权得分'},
            {'name': '状态', 'id': '状态'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{状态} = "优秀"'},
                'backgroundColor': '#d4edda',
                'color': '#155724'
            },
            {
                'if': {'filter_query': '{状态} = "良好"'},
                'backgroundColor': '#d1ecf1',
                'color': '#0c5460'
            },
            {
                'if': {'filter_query': '{状态} = "合格"'},
                'backgroundColor': '#fff3cd',
                'color': '#856404'
            },
            {
                'if': {'filter_query': '{状态} = "较差"'},
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            }
        ]
    )

    fig = go.Figure()

    categories = [s['name'] for s in subscores]
    scores = [quality_result.get(s['key'], {}).get('score', 0) for s in subscores]

    fig.add_trace(go.Bar(
        x=categories,
        y=scores,
        name='得分',
        marker_color=[
            'green' if s >= 90 else ('lightgreen' if s >= 75 else ('orange' if s >= 60 else 'red'))
            for s in scores
        ],
        text=[f'{s:.1f}' for s in scores],
        textposition='auto'
    ))

    fig.update_layout(
        title='分项质量评分',
        xaxis=dict(title='评估项目'),
        yaxis=dict(title='得分', range=[0, 100]),
        height=400
    )

    detail_cards = []

    angle_cov = quality_result.get('angle_coverage', {})
    detail_cards.append(dbc.Card([
        dbc.CardHeader('角度覆盖率详情'),
        dbc.CardBody([
            html.P(f"覆盖比例: {angle_cov.get('coverage_ratio', 0) * 100:.1f}%"),
            html.P(f"覆盖角度: {angle_cov.get('covered_degrees', 0):.1f}°"),
            html.P(f"缺失区间数: {angle_cov.get('gap_count', 0)}"),
        ])
    ], className='mb-3'))

    outliers = quality_result.get('outlier_detection', {})
    detail_cards.append(dbc.Card([
        dbc.CardHeader('异常跳点检测详情'),
        dbc.CardBody([
            html.P(f"异常点总数: {outliers.get('outlier_count', 0)}"),
            html.P(f"距离异常点: {len(outliers.get('distance_outliers', []))}"),
            html.P(f"深度异常点: {len(outliers.get('depth_outliers', []))}"),
        ])
    ], className='mb-3'))

    volatility = quality_result.get('volatility', {})
    detail_cards.append(dbc.Card([
        dbc.CardHeader('数据波动性详情'),
        dbc.CardBody([
            html.P(f"距离均值: {volatility.get('distance_mean', 0):.2f} m"),
            html.P(f"距离标准差: {volatility.get('distance_std', 0):.2f} m"),
            html.P(f"距离变异系数: {volatility.get('distance_cv', 0) * 100:.2f}%"),
            html.P(f"距离波动等级: {volatility.get('distance_volatility_level', '-')}"),
            html.Hr(),
            html.P(f"深度均值: {volatility.get('depth_mean', 0):.2f} m"),
            html.P(f"深度标准差: {volatility.get('depth_std', 0):.2f} m"),
            html.P(f"深度变异系数: {volatility.get('depth_cv', 0) * 100:.2f}%"),
            html.P(f"深度波动等级: {volatility.get('depth_volatility_level', '-')}"),
        ])
    ], className='mb-3'))

    return html.Div([
        dcc.Graph(figure=fig),
        html.Hr(),
        html.H5('评分明细', className='mb-3'),
        scores_table,
        html.Hr(),
        html.H5('详细信息', className='mb-3'),
        dbc.Row([
            dbc.Col(detail_cards[0], width=4),
            dbc.Col(detail_cards[1], width=4),
            dbc.Col(detail_cards[2], width=4),
        ])
    ])


def render_issues_tab(quality_result):
    all_issues = quality_result.get('all_issues', [])

    if not all_issues:
        return dbc.Alert('未检测到明显数据质量问题，数据质量良好！', color='success')

    issues_data = []
    for i, issue in enumerate(all_issues, 1):
        issues_data.append({
            '序号': i,
            '问题描述': issue,
            '严重程度': '高' if ('异常' in issue and '波动' not in issue) or ('不足' in issue) else '中'
        })

    issues_table = dash_table.DataTable(
        data=issues_data,
        columns=[
            {'name': '序号', 'id': '序号'},
            {'name': '问题描述', 'id': '问题描述'},
            {'name': '严重程度', 'id': '严重程度'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{严重程度} = "高"'},
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            },
            {
                'if': {'filter_query': '{严重程度} = "中"'},
                'backgroundColor': '#fff3cd',
                'color': '#856404'
            }
        ],
        page_size=15
    )

    outlier_data = []
    outliers = quality_result.get('outlier_detection', {})
    for ot in outliers.get('distance_outliers', []):
        outlier_data.append({
            '类型': '距离异常',
            '角度': f'{ot["angle"]:.1f}°',
            '数值': f'{ot["distance"]:.2f} m',
            'Z分数': f'{ot["z_score"]:.2f}'
        })
    for ot in outliers.get('depth_outliers', []):
        outlier_data.append({
            '类型': '深度异常',
            '角度': f'{ot["angle"]:.1f}°',
            '数值': f'{ot["depth"]:.2f} m',
            'Z分数': f'{ot["z_score"]:.2f}'
        })

    outlier_table = dash_table.DataTable(
        data=outlier_data,
        columns=[
            {'name': '类型', 'id': '类型'},
            {'name': '角度', 'id': '角度'},
            {'name': '数值', 'id': '数值'},
            {'name': 'Z分数', 'id': 'Z分数'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        page_size=10
    ) if outlier_data else html.P('无异常跳点')

    missing_intervals = quality_result.get('missing_intervals', {})
    missing_data = []
    for i, m in enumerate(missing_intervals.get('missing_intervals', []), 1):
        missing_data.append({
            '序号': i,
            '起始角度': f'{m["start_angle"]:.1f}°',
            '结束角度': f'{m["end_angle"]:.1f}°',
            '间隔大小': f'{m["gap_size"]:.1f}°',
            '超出标准': f'{m["gap_size"] - m["expected_interval"]:.1f}°'
        })

    missing_table = dash_table.DataTable(
        data=missing_data,
        columns=[
            {'name': '序号', 'id': '序号'},
            {'name': '起始角度', 'id': '起始角度'},
            {'name': '结束角度', 'id': '结束角度'},
            {'name': '间隔大小', 'id': '间隔大小'},
            {'name': '超出标准', 'id': '超出标准'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        page_size=10
    ) if missing_data else html.P('无缺失区间')

    return html.Div([
        dbc.Card([
            dbc.CardHeader('问题汇总'),
            dbc.CardBody([
                html.H6(f'共发现 {len(all_issues)} 项数据质量问题', className='mb-3'),
                issues_table
            ])
        ], className='mb-4'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('异常跳点明细'),
                    dbc.CardBody(outlier_table)
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('缺失区间明细'),
                    dbc.CardBody(missing_table)
                ])
            ], width=6),
        ])
    ])


def render_suggestions_tab(quality_result):
    suggestions = quality_result.get('repair_suggestions', [])

    if not suggestions:
        return dbc.Alert('暂无修复建议，数据质量良好！', color='success')

    suggestion_cards = []
    for i, suggestion in enumerate(suggestions, 1):
        severity_color = {
            '高': 'danger',
            '中': 'warning',
            '低': 'info'
        }.get(suggestion.get('severity', '低'), 'secondary')

        suggestion_cards.append(
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.Strong(f"{i}. {suggestion['category']}"), width=8),
                        dbc.Col(
                            dbc.Badge(suggestion['severity'], color=severity_color, className='float-end'),
                            width=4
                        ),
                    ])
                ]),
                dbc.CardBody([
                    html.P(suggestion['description']),
                    dbc.Badge(f"建议操作: {suggestion['action']}", color='primary', pill=True),
                ])
            ], className='mb-3')
        )

    action_summary = {}
    for s in suggestions:
        action = s.get('action', '其他')
        if action not in action_summary:
            action_summary[action] = {'count': 0, 'max_severity': '低'}
        action_summary[action]['count'] += 1
        severity_order = {'高': 3, '中': 2, '低': 1}
        if severity_order.get(s.get('severity', '低'), 1) > severity_order.get(action_summary[action]['max_severity'], 1):
            action_summary[action]['max_severity'] = s.get('severity', '低')

    summary_items = []
    for action, info in action_summary.items():
        color = {'高': 'danger', '中': 'warning', '低': 'info'}.get(info['max_severity'], 'secondary')
        summary_items.append(
            dbc.ListGroupItem([
                html.Strong(action),
                html.Span(f" ({info['count']}项建议)", className='text-muted'),
                dbc.Badge(info['max_severity'], color=color, className='float-end')
            ])
        )

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5('修复建议汇总', className='mb-3'),
                dbc.ListGroup(summary_items),
            ], width=4),
            dbc.Col([
                html.H5('详细建议', className='mb-3'),
                html.Div(suggestion_cards),
            ], width=8),
        ])
    ])


@callback(
    Output('qa-download-report', 'data'),
    Input('qa-export-btn', 'n_clicks'),
    State('qa-quality-result', 'data'),
    State('qa-cave-name', 'data'),
    State('qa-batch-name', 'data'),
    prevent_initial_call=True
)
def export_report(n_clicks, quality_result, cave_name, batch_name):
    if not quality_result or not cave_name or not batch_name:
        return dash.no_update

    report_content = generate_quality_report(cave_name, batch_name, quality_result)

    filename = f"{cave_name}_{batch_name}_质量评估报告.txt"

    return dcc.send_string(report_content, filename)
