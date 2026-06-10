import dash
import numpy as np
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    get_batch, get_cave
)
from analysis import (
    simulate_resurvey_plans, generate_resurvey_report,
    generate_cross_section_data
)

dash.register_page(__name__, path='/resurvey-simulation', name='勘测方案模拟与补测优化')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("勘测方案模拟与补测优化", className="mb-4"),

        dbc.Alert([
            html.Strong("功能说明："),
            "基于当前测点分布、缺失区间、异常点和历史批次差异，自动模拟不同补测方案对容积估算精度和质量评分的提升效果，推荐最优补测路径。"
        ], color="info", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='rs-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=4),
            dbc.Col([
                html.H6("选择批次"),
                dcc.Dropdown(
                    id='rs-batch-selector',
                    placeholder='选择一个批次...'
                ),
            ], width=4),
            dbc.Col([
                html.H6("对比批次（可选，用于批次差异分析）"),
                dcc.Dropdown(
                    id='rs-compare-batch-selector',
                    multi=True,
                    placeholder='选择对比批次...'
                ),
            ], width=4),
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("当前质量评分"),
                    dbc.CardBody([html.Div(id='rs-current-quality', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("容积估算精度"),
                    dbc.CardBody([html.Div(id='rs-current-accuracy', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("当前容积"),
                    dbc.CardBody([html.Div(id='rs-current-volume', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("预计误差"),
                    dbc.CardBody([html.Div(id='rs-current-error', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("缺失区间"),
                    dbc.CardBody([html.Div(id='rs-current-missing', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("异常区域"),
                    dbc.CardBody([html.Div(id='rs-current-anomaly', className='text-center')])
                ]),
            ], width=2),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("最大间隔"),
                    dbc.CardBody([html.Div(id='rs-current-max-gap', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("测量点数"),
                    dbc.CardBody([html.Div(id='rs-current-points', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("置信度"),
                    dbc.CardBody([html.Div(id='rs-current-confidence', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("问题总数"),
                    dbc.CardBody([html.Div(id='rs-current-issues', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("覆盖率"),
                    dbc.CardBody([html.Div(id='rs-current-coverage', className='text-center')])
                ]),
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("平均间隔"),
                    dbc.CardBody([html.Div(id='rs-current-avg-gap', className='text-center')])
                ]),
            ], width=2),
        ], className='mb-4'),

        dbc.Tabs([
            dbc.Tab(label='方案对比图', tab_id='comparison-tab'),
            dbc.Tab(label='预计收益表', tab_id='benefit-tab'),
            dbc.Tab(label='推荐补测路径', tab_id='path-tab'),
        ], id='rs-tabs', active_tab='comparison-tab', className='mb-4'),

        html.Div(id='rs-tab-content'),

        html.Div([
            dbc.Button(
                '导出补测建议报告',
                id='rs-export-btn',
                color='primary',
                className='float-end'
            ),
            dcc.Download(id='rs-download-report'),
        ], className='mb-4'),

        dcc.Store(id='rs-simulation-result'),
        dcc.Store(id='rs-cave-name'),
        dcc.Store(id='rs-batch-name'),
    ], fluid=True)


@callback(
    Output('rs-cave-selector', 'options'),
    Output('rs-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    State('rs-cave-selector', 'value'),
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
    Input('rs-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_rs_to_store(cave_id, stored_cave_id):
    if cave_id != stored_cave_id:
        return cave_id
    return dash.no_update


@callback(
    Output('rs-batch-selector', 'options'),
    Output('rs-batch-selector', 'value'),
    Input('rs-cave-selector', 'value')
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
    Output('rs-compare-batch-selector', 'options'),
    Input('rs-cave-selector', 'value'),
    Input('rs-batch-selector', 'value')
)
def update_compare_batch_selector(cave_id, current_batch_id):
    if not cave_id:
        return []

    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches if b['id'] != current_batch_id]

    return options


@callback(
    Output('rs-current-quality', 'children'),
    Output('rs-current-accuracy', 'children'),
    Output('rs-current-volume', 'children'),
    Output('rs-current-error', 'children'),
    Output('rs-current-missing', 'children'),
    Output('rs-current-anomaly', 'children'),
    Output('rs-current-max-gap', 'children'),
    Output('rs-current-points', 'children'),
    Output('rs-current-confidence', 'children'),
    Output('rs-current-issues', 'children'),
    Output('rs-current-coverage', 'children'),
    Output('rs-current-avg-gap', 'children'),
    Output('rs-simulation-result', 'data'),
    Output('rs-cave-name', 'data'),
    Output('rs-batch-name', 'data'),
    Input('rs-batch-selector', 'value'),
    Input('rs-cave-selector', 'value'),
    Input('rs-compare-batch-selector', 'value')
)
def run_simulation(batch_id, cave_id, compare_batch_ids):
    if not batch_id or not cave_id:
        return (
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            html.H2('--', className='text-muted'),
            {}, '', ''
        )

    measurements = get_measurements_by_batch(batch_id)
    if not measurements:
        return (
            html.Div([html.H2('无数据', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('0', className='text-muted'), html.Small('个测量点', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            html.Div([html.H2('--', className='text-muted')]),
            {}, '', ''
        )

    cave = get_cave(cave_id)
    batch = get_batch(batch_id)
    cave_name = cave['name'] if cave else ''
    batch_name = batch['batch_name'] if batch else ''

    all_batches_measurements = [measurements]
    if compare_batch_ids:
        for cb_id in compare_batch_ids:
            cb_measurements = get_measurements_by_batch(cb_id)
            if cb_measurements:
                all_batches_measurements.append(cb_measurements)

    simulation_result = simulate_resurvey_plans(measurements, all_batches_measurements)

    cs = simulation_result.get('current_status', {})

    quality_score = cs.get('quality_score', 0)
    quality_color = 'success' if quality_score >= 80 else ('warning' if quality_score >= 60 else 'danger')

    quality_el = html.Div([
        html.H2(f"{quality_score:.1f}", className=f'text-{quality_color}'),
        html.Small(f"{cs.get('grade', '-')}", className='text-muted')
    ])

    accuracy_score = cs.get('accuracy_score', 0)
    accuracy_color = 'success' if accuracy_score >= 80 else ('warning' if accuracy_score >= 60 else 'danger')

    accuracy_el = html.Div([
        html.H2(f"{accuracy_score:.1f}", className=f'text-{accuracy_color}'),
        html.Small(f"{cs.get('error_level', '-')}", className='text-muted')
    ])

    volume_el = html.Div([
        html.H2(f"{cs.get('volume', 0):.1f}", className='text-info'),
        html.Small('m³', className='text-muted')
    ])

    error_pct = cs.get('estimated_error_pct', 0)
    error_color = 'success' if error_pct <= 3 else ('warning' if error_pct <= 6 else 'danger')
    error_el = html.Div([
        html.H2(f"±{error_pct:.1f}%", className=f'text-{error_color}'),
        html.Small('容积误差', className='text-muted')
    ])

    missing_count = cs.get('missing_count', 0)
    missing_color = 'danger' if missing_count > 3 else ('warning' if missing_count > 0 else 'success')
    missing_el = html.Div([
        html.H2(f"{missing_count}", className=f'text-{missing_color}'),
        html.Small('个区间', className='text-muted')
    ])

    anomaly_count = cs.get('anomaly_count', 0)
    anomaly_color = 'danger' if anomaly_count > 2 else ('warning' if anomaly_count > 0 else 'success')
    anomaly_el = html.Div([
        html.H2(f"{anomaly_count}", className=f'text-{anomaly_color}'),
        html.Small('个区域', className='text-muted')
    ])

    max_gap = cs.get('max_gap', 0)
    gap_color = 'danger' if max_gap > 30 else ('warning' if max_gap > 15 else 'success')
    gap_el = html.Div([
        html.H2(f"{max_gap:.1f}°", className=f'text-{gap_color}'),
        html.Small('最大间隔', className='text-muted')
    ])

    points_el = html.Div([
        html.H2(f"{cs.get('point_count', 0)}", className='text-info'),
        html.Small('个测量点', className='text-muted')
    ])

    confidence = cs.get('confidence', '-')
    conf_color = 'success' if confidence == '高' else ('warning' if confidence == '中等' else 'danger')
    confidence_el = html.Div([
        html.H2(f"{confidence}", className=f'text-{conf_color}'),
        html.Small('置信度', className='text-muted')
    ])

    issue_count = cs.get('issue_count', 0)
    issue_color = 'danger' if issue_count > 3 else ('warning' if issue_count > 0 else 'success')
    issues_el = html.Div([
        html.H2(f"{issue_count}", className=f'text-{issue_color}'),
        html.Small('个问题', className='text-muted')
    ])

    coverage = cs.get('coverage_ratio', 0) * 100
    cov_color = 'success' if coverage >= 90 else ('warning' if coverage >= 70 else 'danger')
    coverage_el = html.Div([
        html.H2(f"{coverage:.0f}%", className=f'text-{cov_color}'),
        html.Small('覆盖率', className='text-muted')
    ])

    avg_gap = cs.get('avg_gap', 0)
    avg_gap_el = html.Div([
        html.H2(f"{avg_gap:.1f}°", className='text-info'),
        html.Small('平均间隔', className='text-muted')
    ])

    return (
        quality_el,
        accuracy_el,
        volume_el,
        error_el,
        missing_el,
        anomaly_el,
        gap_el,
        points_el,
        confidence_el,
        issues_el,
        coverage_el,
        avg_gap_el,
        simulation_result,
        cave_name,
        batch_name
    )


@callback(
    Output('rs-tab-content', 'children'),
    Input('rs-tabs', 'active_tab'),
    Input('rs-simulation-result', 'data'),
    Input('rs-batch-selector', 'value')
)
def render_tab_content(active_tab, simulation_result, batch_id):
    if not simulation_result or not simulation_result.get('plans'):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className='bi bi-inbox', style={'fontSize': '3rem', 'color': '#6c757d'}),
                        html.H5('当前批次暂无测量数据', className='mt-3 mb-2 text-muted'),
                        html.P('无法进行补测方案模拟，请先为该批次导入测量数据。', className='text-muted'),
                        html.Hr(className='my-3'),
                        html.H6('你可以执行以下操作：', className='mb-3'),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.I(className='bi bi-upload me-2'),
                                html.A('前往数据导入页面', href='/', className='text-decoration-none'),
                                ' — 导入 CSV 格式的勘测数据'
                            ]),
                            dbc.ListGroupItem([
                                html.I(className='bi bi-plus-circle me-2'),
                                html.A('前往数据管理页面', href='/data-management', className='text-decoration-none'),
                                ' — 手动添加测量记录'
                            ]),
                        ]),
                    ], className='text-center p-4')
                ], width=8, className='mx-auto')
            ])
        ])

    if active_tab == 'comparison-tab':
        return render_comparison_tab(simulation_result, batch_id)
    elif active_tab == 'benefit-tab':
        return render_benefit_tab(simulation_result)
    elif active_tab == 'path-tab':
        return render_path_tab(simulation_result, batch_id)
    else:
        return html.P('未知标签页')


def render_comparison_tab(simulation_result, batch_id):
    chart_data = simulation_result.get('comparison_chart_data', {})
    plan_names = chart_data.get('plan_names', [])
    quality_scores = chart_data.get('quality_scores', [])
    current_quality = chart_data.get('current_quality', 0)
    accuracy_scores = chart_data.get('accuracy_scores', [])
    current_accuracy = chart_data.get('current_accuracy', 0)
    error_reductions = chart_data.get('error_reductions_pct', [])
    estimated_errors = chart_data.get('estimated_errors_pct', [])
    current_error = chart_data.get('current_error_pct', 0)

    if not plan_names:
        return html.P('暂无模拟数据')

    fig_quality = go.Figure()

    fig_quality.add_trace(go.Bar(
        x=plan_names,
        y=quality_scores,
        name='预计质量评分',
        marker_color=[
            'green' if s >= 90 else ('lightgreen' if s >= 75 else ('orange' if s >= 60 else 'red'))
            for s in quality_scores
        ],
        text=[f'{s:.1f}' for s in quality_scores],
        textposition='auto'
    ))

    fig_quality.add_hline(
        y=current_quality,
        line_dash='dash',
        line_color='blue',
        annotation_text=f'当前评分: {current_quality:.1f}'
    )

    fig_quality.update_layout(
        title='各补测方案预计质量评分对比',
        xaxis=dict(title='补测方案'),
        yaxis=dict(title='质量评分', range=[0, 100]),
        height=450
    )

    fig_accuracy = go.Figure()

    fig_accuracy.add_trace(go.Bar(
        x=plan_names,
        y=accuracy_scores,
        name='预计容积精度',
        marker_color=[
            '#2E8B57' if s >= 90 else ('#90EE90' if s >= 75 else ('#FFA500' if s >= 60 else '#DC143C'))
            for s in accuracy_scores
        ],
        text=[f'{s:.1f}' for s in accuracy_scores],
        textposition='auto'
    ))

    fig_accuracy.add_hline(
        y=current_accuracy,
        line_dash='dash',
        line_color='darkblue',
        annotation_text=f'当前精度: {current_accuracy:.1f}'
    )

    fig_accuracy.update_layout(
        title='各补测方案预计容积精度对比',
        xaxis=dict(title='补测方案'),
        yaxis=dict(title='精度评分', range=[0, 100]),
        height=450
    )

    fig_multi = go.Figure()

    num_sup = chart_data.get('num_supplementary', [])

    fig_multi.add_trace(go.Scatter(
        x=num_sup,
        y=quality_scores,
        mode='lines+markers',
        name='质量评分',
        marker=dict(size=10),
        line=dict(width=2)
    ))

    fig_multi.add_trace(go.Scatter(
        x=num_sup,
        y=accuracy_scores,
        mode='lines+markers',
        name='精度评分',
        marker=dict(size=10, symbol='square'),
        line=dict(width=2, color='green')
    ))

    fig_multi.add_trace(go.Scatter(
        x=num_sup,
        y=[current_quality] * len(num_sup),
        mode='lines',
        name='当前质量',
        line=dict(dash='dash', color='blue', width=1)
    ))

    fig_multi.add_trace(go.Scatter(
        x=num_sup,
        y=[current_accuracy] * len(num_sup),
        mode='lines',
        name='当前精度',
        line=dict(dash='dash', color='green', width=1)
    ))

    fig_multi.update_layout(
        title='质量/精度 vs 补测点数（边际效益曲线）',
        xaxis=dict(title='补测角度数量'),
        yaxis=dict(title='评分'),
        height=400
    )

    fig_gap_vol = go.Figure()

    gap_red = chart_data.get('gap_reductions_pct', [])
    vol_chg = chart_data.get('volume_changes_pct', [])

    fig_gap_vol.add_trace(go.Bar(
        x=plan_names,
        y=gap_red,
        name='间隔缩减率 (%)',
        marker_color='rgba(55, 83, 109, 0.7)',
        yaxis='y'
    ))

    fig_gap_vol.add_trace(go.Scatter(
        x=plan_names,
        y=vol_chg,
        name='容积变化率 (%)',
        marker=dict(size=8, color='red'),
        line=dict(color='red', width=2),
        yaxis='y2'
    ))

    fig_gap_vol.update_layout(
        title='间隔缩减率 & 容积变化率对比',
        xaxis=dict(title='补测方案'),
        yaxis=dict(title='间隔缩减率 (%)', side='left'),
        yaxis2=dict(title='容积变化率 (%)', side='right', overlaying='y'),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='right', x=1)
    )

    fig_error = go.Figure()

    fig_error.add_trace(go.Bar(
        x=plan_names,
        y=estimated_errors,
        name='预计误差率 (%)',
        marker_color='rgba(220, 20, 60, 0.7)',
        text=[f'±{e:.1f}%' for e in estimated_errors],
        textposition='auto'
    ))

    fig_error.add_hline(
        y=current_error,
        line_dash='dash',
        line_color='darkred',
        annotation_text=f'当前误差: ±{current_error:.1f}%'
    )

    fig_error.update_layout(
        title='各方案预计容积误差率对比',
        xaxis=dict(title='补测方案'),
        yaxis=dict(title='预计误差率 (%)'),
        height=400
    )

    fig_radar = go.Figure()

    if simulation_result.get('plans'):
        plan = simulation_result['plans'][0]
        categories = ['质量评分', '间隔缩减', '覆盖提升', '成本效率', '容积精度']
        for p in simulation_result['plans'][:4]:
            max_q = 100
            max_gap = max(plan['gap_reduction_pct'] for plan in simulation_result['plans']) or 1
            max_cov = max(plan['coverage_improvement'] for plan in simulation_result['plans']) or 0.01
            max_eff = max(plan['cost_efficiency'] for plan in simulation_result['plans']) or 0.01
            max_vol = max(abs(plan['volume_change_pct']) for plan in simulation_result['plans']) or 1

            values = [
                p['projected_quality_score'] / max_q * 100,
                p['gap_reduction_pct'] / max_gap * 100 if max_gap > 0 else 0,
                p['coverage_improvement'] / max_cov * 100 if max_cov > 0 else 0,
                p['cost_efficiency'] / max_eff * 100 if max_eff > 0 else 0,
                (1 - abs(p['volume_change_pct']) / max_vol) * 100 if max_vol > 0 else 100
            ]
            values.append(values[0])

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=p['plan_name'],
                opacity=0.6
            ))

        fig_radar.update_layout(
            title='方案多维对比雷达图',
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=450,
            showlegend=True
        )

    measurements = get_measurements_by_batch(batch_id) if batch_id else []
    fig_polar = go.Figure()

    if measurements:
        cs_data = generate_cross_section_data(measurements)
        angles = cs_data['angles']
        distances = cs_data['distances']

        fig_polar.add_trace(go.Scatterpolar(
            r=distances,
            theta=angles,
            mode='lines+markers',
            name='当前测点',
            line=dict(color='blue', width=2),
            marker=dict(size=5)
        ))

    recommended = simulation_result.get('recommended_plan')
    if recommended and recommended.get('supplementary_details'):
        sup_angles = [s['angle'] for s in recommended['supplementary_details']]
        if measurements:
            import pandas as pd
            df = pd.DataFrame(measurements).sort_values('angle')
            m_angles = df['angle'].values
            m_distances = df['distance'].values
            m_angles_ext = np.concatenate([m_angles - 360, m_angles, m_angles + 360])
            m_distances_ext = np.concatenate([m_distances, m_distances, m_distances])
            from scipy.interpolate import interp1d
            valid = np.diff(m_angles_ext) != 0
            valid_idx = np.where(valid)[0]
            if len(valid_idx) >= 2:
                f_dist = interp1d(m_angles_ext[valid_idx], m_distances_ext[valid_idx],
                                  kind='linear', fill_value='extrapolate')
                sup_distances = [float(f_dist(a % 360)) for a in sup_angles]
            else:
                sup_distances = [np.mean([d['distance'] for d in measurements])] * len(sup_angles)
        else:
            sup_distances = [50] * len(sup_angles)

        priority_colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        priorities = [s['priority'] for s in recommended['supplementary_details']]
        marker_colors = [priority_colors.get(p, 'gray') for p in priorities]

        fig_polar.add_trace(go.Scatterpolar(
            r=sup_distances,
            theta=sup_angles,
            mode='markers',
            name='推荐补测点',
            marker=dict(
                size=12,
                color=marker_colors,
                symbol='diamond',
                line=dict(color='white', width=2)
            )
        ))

    fig_polar.update_layout(
        title='当前测点分布 & 推荐补测位置',
        polar=dict(
            radialaxis=dict(showgrid=True),
            angularaxis=dict(direction='clockwise', rotation=90, showgrid=True)
        ),
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5)
    )

    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_quality)], width=6),
            dbc.Col([dcc.Graph(figure=fig_accuracy)], width=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_multi)], width=6),
            dbc.Col([dcc.Graph(figure=fig_error)], width=6),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_gap_vol)], width=6),
            dbc.Col([dcc.Graph(figure=fig_radar)], width=6),
        ]),
        html.Hr(),
        dcc.Graph(figure=fig_polar),
        dbc.Alert([
            html.Strong('图表解读：'),
            "第一行左：各方案预计质量评分柱状图，蓝色虚线为当前评分基准；",
            "第一行右：各方案预计容积精度柱状图，深蓝虚线为当前精度基准；",
            "第二行左：质量/精度随补测点数的变化曲线，用于评估边际效益；",
            "第二行右：各方案预计容积误差率，数值越低越好；",
            "第三行左：间隔缩减率与容积变化率双轴对比；",
            "第三行右：方案多维雷达图，综合比较各方案表现；",
            "下方极坐标图：蓝色线条为当前测点分布，红色/橙色/绿色菱形为推荐补测位置（红=高优先级，橙=中，绿=低）。"
        ], color="info", className="mt-3")
    ])


def render_benefit_tab(simulation_result):
    plans = simulation_result.get('plans', [])
    if not plans:
        return html.P('暂无数据')

    table_data = []
    for i, plan in enumerate(plans, 1):
        is_recommended = simulation_result.get('recommended_plan', {}).get('plan_name') == plan['plan_name']
        table_data.append({
            '方案': f"{'⭐ ' if is_recommended else ''}方案{i}",
            '补测点数': plan['num_supplementary'],
            '预计质量评分': f"{plan['projected_quality_score']:.1f}",
            '质量提升': f"{plan['quality_improvement']:+.1f}",
            '预计等级': plan['projected_grade'],
            '容积精度': f"{plan['accuracy_score']:.1f}",
            '精度提升': f"{plan['accuracy_improvement']:+.1f}",
            '误差率(%)': f"±{plan['estimated_error_pct']:.1f}",
            '误差缩减(%)': f"{plan['error_reduction_pct']:.1f}",
            '容积变化(%)': f"{plan['volume_change_pct']:+.2f}",
            '间隔缩减(%)': f"{plan['gap_reduction_pct']:.1f}",
            '覆盖提升': f"{plan['coverage_improvement']:.3f}",
            '成本效率': f"{plan['cost_efficiency']:.2f}",
            '补测后总点数': plan['total_points_after'],
        })

    benefit_table = dash_table.DataTable(
        data=table_data,
        columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{方案} contains "⭐"'},
                'backgroundColor': '#d4edda',
                'fontWeight': 'bold'
            },
        ],
        page_size=10
    )

    recommended = simulation_result.get('recommended_plan')
    best_quality = simulation_result.get('best_quality_plan')

    rec_card = html.Div('暂无推荐')
    if recommended:
        rec_card = dbc.Card([
            dbc.CardHeader([
                html.Strong('⭐ 推荐方案（最优性价比）'),
                dbc.Badge('推荐', color='success', className='float-end')
            ]),
            dbc.CardBody([
                html.P(f"方案名称: {recommended['plan_name']}"),
                html.P(f"补测角度数: {recommended['num_supplementary']} 个"),
                html.P(f"预计质量评分: {recommended['projected_quality_score']:.1f} (提升 {recommended['quality_improvement']:+.1f})"),
                html.P(f"预计容积精度: {recommended['accuracy_score']:.1f} (提升 {recommended['accuracy_improvement']:+.1f})"),
                html.P(f"预计误差率: ±{recommended['estimated_error_pct']:.1f}% (缩减 {recommended['error_reduction_pct']:.1f}%)"),
                html.P(f"预计容积变化: {recommended['volume_change_pct']:+.2f}%"),
                html.P(f"间隔缩减: {recommended['max_gap_before']:.1f}° → {recommended['max_gap_after']:.1f}°"),
                html.P(f"成本效率: {recommended['cost_efficiency']:.2f} 分/点"),
                html.Hr(),
                html.H6('优先级分布'),
                dbc.Progress([
                    dbc.Progress(value=recommended['high_priority_count'], color='danger', label=f'高{recommended["high_priority_count"]}'),
                    dbc.Progress(value=recommended['medium_priority_count'], color='warning', label=f'中{recommended["medium_priority_count"]}'),
                    dbc.Progress(value=recommended['low_priority_count'], color='success', label=f'低{recommended["low_priority_count"]}'),
                ], multi=True, className='mb-2'),
            ])
        ], className='mb-3')

    bq_card = html.Div('暂无')
    if best_quality and best_quality != recommended:
        bq_card = dbc.Card([
            dbc.CardHeader([
                html.Strong('最佳质量方案'),
                dbc.Badge('最高质量', color='primary', className='float-end')
            ]),
            dbc.CardBody([
                html.P(f"方案名称: {best_quality['plan_name']}"),
                html.P(f"补测角度数: {best_quality['num_supplementary']} 个"),
                html.P(f"预计质量评分: {best_quality['projected_quality_score']:.1f} (提升 {best_quality['quality_improvement']:+.1f})"),
                html.P(f"预计容积精度: {best_quality['accuracy_score']:.1f}"),
                html.P(f"成本效率: {best_quality['cost_efficiency']:.2f} 分/点"),
            ])
        ], className='mb-3')

    cs = simulation_result.get('current_status', {})
    fig_efficiency = go.Figure()

    supp_counts = [p['num_supplementary'] for p in plans]
    quality_improvements = [p['quality_improvement'] for p in plans]
    cost_efficiencies = [p['cost_efficiency'] for p in plans]

    fig_efficiency.add_trace(go.Bar(
        x=[p['plan_name'] for p in plans],
        y=quality_improvements,
        name='质量提升量',
        marker_color='rgba(55, 83, 109, 0.7)',
        yaxis='y'
    ))

    fig_efficiency.add_trace(go.Scatter(
        x=[p['plan_name'] for p in plans],
        y=cost_efficiencies,
        name='成本效率 (分/点)',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=2),
        yaxis='y2'
    ))

    fig_efficiency.update_layout(
        title='质量提升量 & 成本效率对比',
        xaxis=dict(title='补测方案'),
        yaxis=dict(title='质量提升量 (分)', side='left'),
        yaxis2=dict(title='成本效率 (分/点)', side='right', overlaying='y'),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='right', x=1)
    )

    return html.Div([
        dbc.Row([
            dbc.Col([rec_card], width=6),
            dbc.Col([bq_card], width=6),
        ], className='mb-4'),
        html.H5('预计收益对比表', className='mb-3'),
        benefit_table,
        html.Hr(),
        dcc.Graph(figure=fig_efficiency),
    ])


def render_path_tab(simulation_result, batch_id):
    recommended = simulation_result.get('recommended_plan')
    path_data = simulation_result.get('path_data', {})

    if not recommended or not path_data or not path_data.get('angles'):
        return dbc.Alert('暂无推荐补测路径数据', color='warning')

    path_table_data = []
    for i, (angle, priority, reason, source, weight) in enumerate(zip(
        path_data['angles'],
        path_data['priorities'],
        path_data['reasons'],
        path_data['sources'],
        path_data['weights']
    ), 1):
        priority_color = {'high': '高', 'medium': '中', 'low': '低'}.get(priority, '-')
        source_label = {
            'missing_interval': '缺失区间',
            'anomaly': '异常区域',
            'batch_diff': '批次差异',
            'gap_filling': '间隔加密',
            'uniform_density': '均匀加密',
            'regular': '常规加密'
        }.get(source, source)

        path_table_data.append({
            '序号': i,
            '角度 (°)': f'{angle:.1f}',
            '优先级': priority_color,
            '来源': source_label,
            '原因': reason,
            '权重': f'{weight:.1f}',
        })

    path_table = dash_table.DataTable(
        data=path_table_data,
        columns=[
            {'name': '序号', 'id': '序号'},
            {'name': '角度 (°)', 'id': '角度 (°)'},
            {'name': '优先级', 'id': '优先级'},
            {'name': '来源', 'id': '来源'},
            {'name': '原因', 'id': '原因'},
            {'name': '权重', 'id': '权重'},
        ],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{优先级} = "高"'},
                'backgroundColor': '#f8d7da',
                'color': '#721c24'
            },
            {
                'if': {'filter_query': '{优先级} = "中"'},
                'backgroundColor': '#fff3cd',
                'color': '#856404'
            },
            {
                'if': {'filter_query': '{优先级} = "低"'},
                'backgroundColor': '#d4edda',
                'color': '#155724'
            },
        ],
        page_size=20
    )

    measurements = get_measurements_by_batch(batch_id) if batch_id else []

    fig_path = go.Figure()

    if measurements:
        cs_data = generate_cross_section_data(measurements)
        angles = cs_data['angles']
        distances = cs_data['distances']

        x_closed = cs_data['x'] + [cs_data['x'][0]]
        y_closed = cs_data['y'] + [cs_data['y'][0]]

        fig_path.add_trace(go.Scatter(
            x=x_closed,
            y=y_closed,
            mode='lines+markers',
            name='当前测点',
            line=dict(color='blue', width=2),
            marker=dict(size=5),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.05)'
        ))

    if measurements and path_data.get('angles'):
        import pandas as pd
        df = pd.DataFrame(measurements).sort_values('angle')
        m_angles = df['angle'].values
        m_distances = df['distance'].values
        m_angles_ext = np.concatenate([m_angles - 360, m_angles, m_angles + 360])
        m_distances_ext = np.concatenate([m_distances, m_distances, m_distances])
        from scipy.interpolate import interp1d
        valid = np.diff(m_angles_ext) != 0
        valid_idx = np.where(valid)[0]

        sup_x = []
        sup_y = []
        sup_colors = []
        sup_sizes = []
        sup_texts = []

        if len(valid_idx) >= 2:
            f_dist = interp1d(m_angles_ext[valid_idx], m_distances_ext[valid_idx],
                              kind='linear', fill_value='extrapolate')

            for angle, priority, reason, weight in zip(
                path_data['angles'],
                path_data['priorities'],
                path_data['reasons'],
                path_data['weights']
            ):
                try:
                    r = float(f_dist(angle % 360))
                    angle_rad = np.deg2rad(angle)
                    sup_x.append(r * np.cos(angle_rad))
                    sup_y.append(r * np.sin(angle_rad))

                    if priority == 'high':
                        sup_colors.append('red')
                        sup_sizes.append(16)
                    elif priority == 'medium':
                        sup_colors.append('orange')
                        sup_sizes.append(13)
                    else:
                        sup_colors.append('green')
                        sup_sizes.append(10)

                    sup_texts.append(f'{angle:.0f}° [{priority}] {reason}')
                except Exception:
                    continue

        fig_path.add_trace(go.Scatter(
            x=sup_x,
            y=sup_y,
            mode='markers',
            name='推荐补测点',
            marker=dict(
                size=sup_sizes,
                color=sup_colors,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            text=sup_texts,
            hovertemplate='%{text}<extra></extra>'
        ))

        for sx, sy in zip(sup_x, sup_y):
            fig_path.add_trace(go.Scatter(
                x=[0, sx],
                y=[0, sy],
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig_path.update_layout(
        title='推荐补测路径可视化',
        xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1),
        yaxis=dict(title='Y (m)'),
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=-0.12, xanchor='right', x=1)
    )

    fig_gantt = go.Figure()

    if path_data.get('angles'):
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_indices = sorted(range(len(path_data['priorities'])),
                                key=lambda i: (priority_order.get(path_data['priorities'][i], 3),
                                               -path_data['weights'][i]))

        for rank, idx in enumerate(sorted_indices):
            angle = path_data['angles'][idx]
            priority = path_data['priorities'][idx]
            reason = path_data['reasons'][idx]

            color = {'high': '#d73027', 'medium': '#fc8d59', 'low': '#91cf60'}.get(priority, 'gray')

            fig_gantt.add_trace(go.Bar(
                x=[1],
                y=[f'#{rank + 1}: {angle:.1f}°'],
                orientation='h',
                marker_color=color,
                name=f'{priority}',
                showlegend=rank == 0 or (rank > 0 and priority != sorted_indices[:rank].__class__),
                hovertemplate=f'角度: {angle:.1f}°<br>优先级: {priority}<br>原因: {reason}<extra></extra>'
            ))

        priority_shown = set()
        for trace in fig_gantt.data:
            p_name = trace.name
            if p_name in priority_shown:
                trace.showlegend = False
            priority_shown.add(p_name)

    fig_gantt.update_layout(
        title='推荐补测顺序（从高优先级到低优先级）',
        xaxis=dict(title='补测步骤', range=[0, 2], tickvals=[0.5], ticktext=['执行']),
        yaxis=dict(title='', autorange='reversed'),
        height=max(300, len(path_data.get('angles', [])) * 35),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
    )

    source_summary = {}
    for source, priority in zip(path_data.get('sources', []), path_data.get('priorities', [])):
        source_label = {
            'missing_interval': '缺失区间',
            'anomaly': '异常区域',
            'batch_diff': '批次差异',
            'gap_filling': '间隔加密',
            'uniform_density': '均匀加密',
            'regular': '常规加密'
        }.get(source, source)
        if source_label not in source_summary:
            source_summary[source_label] = {'high': 0, 'medium': 0, 'low': 0, 'total': 0}
        source_summary[source_label][priority] = source_summary[source_label].get(priority, 0) + 1
        source_summary[source_label]['total'] += 1

    source_cards = []
    for source_label, counts in source_summary.items():
        icon = {
            '缺失区间': 'bi-exclamation-triangle',
            '异常区域': 'bi-bug',
            '批次差异': 'bi-arrow-left-right',
            '间隔加密': 'bi-rulers',
            '均匀加密': 'bi-grid-3x3',
            '常规加密': 'bi-plus-circle'
        }.get(source_label, 'bi-info-circle')

        source_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.I(className=f'bi {icon} me-2', style={'fontSize': '1.2rem'}),
                        html.Strong(source_label),
                        html.Hr(),
                        html.P(f"高优先级: {counts.get('high', 0)}", className='text-danger mb-1'),
                        html.P(f"中优先级: {counts.get('medium', 0)}", className='text-warning mb-1'),
                        html.P(f"低优先级: {counts.get('low', 0)}", className='text-success mb-1'),
                        html.Hr(),
                        html.P(f"合计: {counts['total']} 点", className='fw-bold mb-0'),
                    ])
                ])
            ], width=3)
        )

    return html.Div([
        dbc.Row(source_cards, className='mb-4'),
        html.Hr(),
        html.H5('推荐补测路径明细', className='mb-3'),
        path_table,
        html.Hr(),
        dcc.Graph(figure=fig_path),
        html.Hr(),
        dcc.Graph(figure=fig_gantt),
        dbc.Alert([
            html.Strong('路径说明：'),
            "上方卡片汇总了补测来源分布；表格列出每个补测点的角度、优先级和原因；",
            "中间极坐标图展示了补测点在盐穴截面上的位置；",
            "下方甘特图按优先级排序展示建议的补测执行顺序——红色为高优先级，应优先执行。"
        ], color="info", className="mt-3")
    ])


@callback(
    Output('rs-download-report', 'data'),
    Input('rs-export-btn', 'n_clicks'),
    State('rs-simulation-result', 'data'),
    State('rs-cave-name', 'data'),
    State('rs-batch-name', 'data'),
    prevent_initial_call=True
)
def export_report(n_clicks, simulation_result, cave_name, batch_name):
    if not simulation_result or not cave_name or not batch_name:
        return dash.no_update

    report_content = generate_resurvey_report(cave_name, batch_name, simulation_result)

    filename = f"{cave_name}_{batch_name}_补测建议报告.txt"

    return dcc.send_string(report_content, filename)
