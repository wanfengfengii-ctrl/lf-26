import dash
import pandas as pd
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from database import (
    get_all_caves, get_batches_by_cave, get_measurements_by_batch,
    add_measurement, update_measurement, delete_measurement,
    delete_batch, delete_cave, check_angle_duplicate,
    get_volume_estimate, save_volume_estimate, get_batch,
    save_anomaly_regions
)
from analysis import (
    calculate_volume_conical, detect_anomalies, compute_batch_statistics
)

dash.register_page(__name__, path='/data-management', name='数据管理')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("数据管理", className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label='测量记录管理', tab_id='measurements-tab'),
            dbc.Tab(label='批次管理', tab_id='batches-tab'),
            dbc.Tab(label='盐穴管理', tab_id='caves-tab'),
        ], id='mgmt-tabs', active_tab='measurements-tab', className='mb-4'),

        html.Div(id='mgmt-tab-content'),

        dbc.Modal([
            dbc.ModalHeader("编辑测量记录"),
            dbc.ModalBody([
                dbc.Label("角度 (°)"),
                dbc.Input(id='mgmt-edit-angle', type='number', min=0, max=360, step=0.1),
                dbc.Label("距离 (m)", className='mt-2'),
                dbc.Input(id='mgmt-edit-distance', type='number', min=0, step=0.01),
                dbc.Label("深度 (m)", className='mt-2'),
                dbc.Input(id='mgmt-edit-depth', type='number', min=0, step=0.01),
                html.Div(id='mgmt-edit-error', className='text-danger mt-2'),
            ]),
            dbc.ModalFooter([
                dbc.Button('取消', id='mgmt-edit-cancel', color='secondary'),
                dbc.Button('保存', id='mgmt-edit-save', color='primary'),
            ]),
        ], id='mgmt-edit-modal', is_open=False),

        dbc.Modal([
            dbc.ModalHeader("确认删除"),
            dbc.ModalBody(id='mgmt-delete-confirm-body'),
            dbc.ModalFooter([
                dbc.Button('取消', id='mgmt-delete-cancel', color='secondary'),
                dbc.Button('确认删除', id='mgmt-delete-confirm', color='danger'),
            ]),
        ], id='mgmt-delete-modal', is_open=False),

        dcc.Store(id='mgmt-editing-id'),
        dcc.Store(id='mgmt-delete-id'),
        dcc.Store(id='mgmt-delete-type'),
    ], fluid=True)


@callback(
    Output('mgmt-tab-content', 'children'),
    Input('mgmt-tabs', 'active_tab')
)
def render_tab_content(active_tab):
    if active_tab == 'measurements-tab':
        return render_measurements_tab()
    elif active_tab == 'batches-tab':
        return render_batches_tab()
    else:
        return render_caves_tab()


def render_measurements_tab():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='mgmt-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=6),
            dbc.Col([
                html.H6("选择批次"),
                dcc.Dropdown(
                    id='mgmt-batch-selector',
                    placeholder='选择一个批次...'
                ),
            ], width=6),
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader("添加测量记录"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("角度 (°)"),
                        dbc.Input(id='mgmt-angle', type='number', min=0, max=360, step=0.1, placeholder='0-360'),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("距离 (m)"),
                        dbc.Input(id='mgmt-distance', type='number', min=0, step=0.01, placeholder='>= 0'),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("深度 (m)"),
                        dbc.Input(id='mgmt-depth', type='number', min=0, step=0.01, placeholder='>= 0'),
                    ], width=4),
                ]),
                dbc.Button('添加记录', id='mgmt-add-btn', color='primary', n_clicks=0, className='mt-3'),
                html.Div(id='mgmt-add-result', className='mt-2'),
            ])
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col("测量记录列表", width=6),
                    dbc.Col([
                        dbc.Button(
                            '重新计算容积',
                            id='mgmt-recalc-btn',
                            color='warning',
                            size='sm',
                            className='float-end'
                        ),
                    ], width=6),
                ])
            ]),
            dbc.CardBody([
                html.Div(id='mgmt-result-message', className='mb-3'),
                dash_table.DataTable(
                    id='mgmt-measurements-table',
                    columns=[
                        {'name': 'ID', 'id': 'id'},
                        {'name': '角度 (°)', 'id': 'angle'},
                        {'name': '距离 (m)', 'id': 'distance'},
                        {'name': '深度 (m)', 'id': 'depth'},
                        {'name': '操作', 'id': 'actions'},
                    ],
                    page_size=15,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    row_deletable=False,
                    editable=False
                ),
            ])
        ]),
    ])


def render_batches_tab():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='mgmt-batch-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=12),
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader("批次列表"),
            dbc.CardBody([
                dash_table.DataTable(
                    id='mgmt-batches-table',
                    columns=[
                        {'name': 'ID', 'id': 'id'},
                        {'name': '批次名称', 'id': 'batch_name'},
                        {'name': '勘测日期', 'id': 'survey_date'},
                        {'name': '备注', 'id': 'notes'},
                        {'name': '测量点数', 'id': 'measurement_count'},
                        {'name': '操作', 'id': 'actions'},
                    ],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),
            ])
        ]),
    ])


def render_caves_tab():
    return html.Div([
        dbc.Card([
            dbc.CardHeader("盐穴列表"),
            dbc.CardBody([
                dash_table.DataTable(
                    id='mgmt-caves-table',
                    columns=[
                        {'name': 'ID', 'id': 'id'},
                        {'name': '盐穴名称', 'id': 'name'},
                        {'name': '描述', 'id': 'description'},
                        {'name': '创建时间', 'id': 'created_at'},
                        {'name': '批次数', 'id': 'batch_count'},
                    ],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),
            ])
        ]),
    ])


@callback(
    Output('mgmt-batch-selector', 'options'),
    Output('mgmt-batch-selector', 'value'),
    Input('mgmt-cave-selector', 'value')
)
def update_batch_selector(cave_id):
    if not cave_id:
        return [], None
    batches = get_batches_by_cave(cave_id)
    options = [{'label': f"{b['batch_name']} ({b['survey_date'] or '未知日期'})", 'value': b['id']}
               for b in batches]
    return options, options[0]['value'] if options else None


@callback(
    Output('mgmt-measurements-table', 'data'),
    Output('mgmt-add-result', 'children'),
    Output('mgmt-result-message', 'children'),
    Output('mgmt-angle', 'value'),
    Output('mgmt-distance', 'value'),
    Output('mgmt-depth', 'value'),
    Input('mgmt-add-btn', 'n_clicks'),
    Input('mgmt-batch-selector', 'value'),
    Input('mgmt-recalc-btn', 'n_clicks'),
    Input('mgmt-edit-save', 'n_clicks'),
    Input('mgmt-delete-confirm', 'n_clicks'),
    State('mgmt-angle', 'value'),
    State('mgmt-distance', 'value'),
    State('mgmt-depth', 'value'),
    State('mgmt-editing-id', 'data'),
    State('mgmt-edit-angle', 'value'),
    State('mgmt-edit-distance', 'value'),
    State('mgmt-edit-depth', 'value'),
    State('mgmt-delete-id', 'data'),
    State('mgmt-delete-type', 'data'),
    prevent_initial_call=False
)
def update_measurements_table(add_clicks, batch_id, recalc_clicks, edit_save_clicks,
                              delete_confirm_clicks, angle, distance, depth,
                              editing_id, edit_angle, edit_distance, edit_depth,
                              delete_id, delete_type):
    if not batch_id:
        return [], '', '', None, None, None

    triggered = ctx.triggered_id
    add_result = ''
    result_msg = ''

    if triggered == 'mgmt-add-btn' and add_clicks and add_clicks > 0:
        if angle is None or distance is None or depth is None:
            add_result = dbc.Alert('请填写所有字段', color='warning')
        elif angle < 0 or angle > 360:
            add_result = dbc.Alert('角度必须在 0-360 度之间', color='danger')
        elif distance < 0:
            add_result = dbc.Alert('距离不能为负数', color='danger')
        elif depth < 0:
            add_result = dbc.Alert('深度不能为负数', color='danger')
        elif check_angle_duplicate(batch_id, angle):
            add_result = dbc.Alert('该角度已存在于此批次中', color='danger')
        else:
            try:
                add_measurement(batch_id, angle, distance, depth)
                add_result = dbc.Alert('添加成功', color='success')
                angle = distance = depth = None
                recalculate_volume(batch_id)
                result_msg = dbc.Alert('容积已自动重新计算', color='info')
            except Exception as e:
                add_result = dbc.Alert(f'添加失败: {str(e)}', color='danger')

    if triggered == 'mgmt-recalc-btn' and recalc_clicks and recalc_clicks > 0:
        try:
            recalculate_volume(batch_id)
            result_msg = dbc.Alert('容积和异常区域已重新计算', color='success')
        except Exception as e:
            result_msg = dbc.Alert(f'计算失败: {str(e)}', color='danger')

    if triggered == 'mgmt-delete-confirm' and delete_confirm_clicks and delete_confirm_clicks > 0:
        if delete_type == 'measurement' and delete_id:
            try:
                delete_measurement(delete_id)
                recalculate_volume(batch_id)
                result_msg = dbc.Alert('删除成功，容积已重新计算', color='success')
            except Exception as e:
                result_msg = dbc.Alert(f'删除失败: {str(e)}', color='danger')

    measurements = get_measurements_by_batch(batch_id)
    table_data = []
    for m in measurements:
        table_data.append({
            'id': m['id'],
            'angle': m['angle'],
            'distance': m['distance'],
            'depth': m['depth'],
            'actions': html.Div([
                dbc.Button('编辑', size='sm', color='link',
                           id={'type': 'edit-btn', 'index': m['id']}),
                dbc.Button('删除', size='sm', color='link', className='text-danger',
                           id={'type': 'delete-btn', 'index': m['id']}),
            ])
        })

    return table_data, add_result, result_msg, angle, distance, depth


@callback(
    Output('mgmt-batches-table', 'data'),
    Input('mgmt-batch-cave-selector', 'value')
)
def update_batches_table(cave_id):
    if not cave_id:
        return []

    batches = get_batches_by_cave(cave_id)
    table_data = []
    for b in batches:
        measurements = get_measurements_by_batch(b['id'])
        table_data.append({
            'id': b['id'],
            'batch_name': b['batch_name'],
            'survey_date': b.get('survey_date', ''),
            'notes': b.get('notes', ''),
            'measurement_count': len(measurements),
            'actions': html.Div([
                dbc.Button('删除', size='sm', color='link', className='text-danger',
                           id={'type': 'delete-batch-btn', 'index': b['id']}),
            ])
        })

    return table_data


@callback(
    Output('mgmt-caves-table', 'data'),
    Input('mgmt-tabs', 'active_tab')
)
def update_caves_table(active_tab):
    caves = get_all_caves()
    table_data = []
    for c in caves:
        batches = get_batches_by_cave(c['id'])
        table_data.append({
            'id': c['id'],
            'name': c['name'],
            'description': c.get('description', ''),
            'created_at': c['created_at'],
            'batch_count': len(batches),
        })
    return table_data


def recalculate_volume(batch_id):
    measurements = get_measurements_by_batch(batch_id)
    if not measurements:
        return

    vol_result = calculate_volume_conical(measurements)
    save_volume_estimate(
        batch_id,
        vol_result['volume'],
        vol_result['max_depth'],
        vol_result['max_distance'],
        vol_result['method']
    )

    anomalies = detect_anomalies(measurements)
    save_anomaly_regions(batch_id, anomalies)


@callback(
    Output('mgmt-edit-modal', 'is_open', allow_duplicate=True),
    Output('mgmt-editing-id', 'data', allow_duplicate=True),
    Output('mgmt-edit-angle', 'value'),
    Output('mgmt-edit-distance', 'value'),
    Output('mgmt-edit-depth', 'value'),
    Output('mgmt-edit-error', 'children'),
    Input({'type': 'edit-btn', 'index': dash.ALL}, 'n_clicks'),
    Input('mgmt-edit-cancel', 'n_clicks'),
    Input('mgmt-edit-save', 'n_clicks'),
    State('mgmt-measurements-table', 'data'),
    State('mgmt-editing-id', 'data'),
    State('mgmt-edit-angle', 'value'),
    State('mgmt-edit-distance', 'value'),
    State('mgmt-edit-depth', 'value'),
    State('mgmt-batch-selector', 'value'),
    prevent_initial_call=True
)
def handle_edit_modal(edit_clicks, cancel_clicks, save_clicks, table_data, editing_id,
                      edit_angle, edit_distance, edit_depth, batch_id):
    triggered = ctx.triggered_id
    error_msg = ''

    if isinstance(triggered, dict) and triggered.get('type') == 'edit-btn':
        meas_id = triggered['index']
        for row in table_data:
            if row['id'] == meas_id:
                return True, meas_id, row['angle'], row['distance'], row['depth'], ''

    if triggered == 'mgmt-edit-cancel':
        return False, None, None, None, None, ''

    if triggered == 'mgmt-edit-save' and editing_id:
        if edit_angle is None or edit_distance is None or edit_depth is None:
            error_msg = '请填写所有字段'
            return True, editing_id, edit_angle, edit_distance, edit_depth, error_msg
        if edit_angle < 0 or edit_angle > 360:
            error_msg = '角度必须在 0-360 度之间'
            return True, editing_id, edit_angle, edit_distance, edit_depth, error_msg
        if edit_distance < 0 or edit_depth < 0:
            error_msg = '距离和深度不能为负数'
            return True, editing_id, edit_angle, edit_distance, edit_depth, error_msg
        if check_angle_duplicate(batch_id, edit_angle, exclude_id=editing_id):
            error_msg = '该角度已存在于此批次中'
            return True, editing_id, edit_angle, edit_distance, edit_depth, error_msg

        try:
            update_measurement(editing_id, edit_angle, edit_distance, edit_depth)
            recalculate_volume(batch_id)
            return False, None, None, None, None, ''
        except Exception as e:
            error_msg = f'保存失败: {str(e)}'
            return True, editing_id, edit_angle, edit_distance, edit_depth, error_msg

    return False, None, None, None, None, ''


@callback(
    Output('mgmt-delete-modal', 'is_open', allow_duplicate=True),
    Output('mgmt-delete-confirm-body', 'children'),
    Output('mgmt-delete-id', 'data', allow_duplicate=True),
    Output('mgmt-delete-type', 'data', allow_duplicate=True),
    Input({'type': 'delete-btn', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'delete-batch-btn', 'index': dash.ALL}, 'n_clicks'),
    Input('mgmt-delete-cancel', 'n_clicks'),
    State('mgmt-measurements-table', 'data'),
    State('mgmt-batches-table', 'data'),
    prevent_initial_call=True
)
def handle_delete_modal(delete_clicks, delete_batch_clicks, cancel_clicks,
                        measurements_data, batches_data):
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get('type') == 'delete-btn':
        meas_id = triggered['index']
        meas_info = None
        for row in measurements_data:
            if row['id'] == meas_id:
                meas_info = row
                break
        body = html.Div([
            html.P(f"确定要删除这条测量记录吗？"),
            html.P(f"ID: {meas_id}"),
            html.P(f"角度: {meas_info['angle']}°") if meas_info else '',
            html.P("此操作不可撤销，删除后容积将自动重新计算。"),
        ])
        return True, body, meas_id, 'measurement'

    if isinstance(triggered, dict) and triggered.get('type') == 'delete-batch-btn':
        batch_id = triggered['index']
        batch_info = None
        for row in batches_data:
            if row['id'] == batch_id:
                batch_info = row
                break
        body = html.Div([
            html.P(f"确定要删除这个批次吗？"),
            html.P(f"批次名称: {batch_info['batch_name']}") if batch_info else '',
            html.P(f"测量点数: {batch_info['measurement_count']}") if batch_info else '',
            html.P("此操作不可撤销，批次下的所有测量数据都将被删除。"),
        ])
        return True, body, batch_id, 'batch'

    if triggered == 'mgmt-delete-cancel':
        return False, '', None, None

    return False, '', None, None


@callback(
    Output('mgmt-delete-modal', 'is_open', allow_duplicate=True),
    Output('mgmt-batches-table', 'data', allow_duplicate=True),
    Output('mgmt-result-message', 'children', allow_duplicate=True),
    Input('mgmt-delete-confirm', 'n_clicks'),
    State('mgmt-delete-id', 'data'),
    State('mgmt-delete-type', 'data'),
    State('mgmt-batch-cave-selector', 'value'),
    prevent_initial_call=True
)
def confirm_delete(confirm_clicks, delete_id, delete_type, cave_id):
    if not confirm_clicks or confirm_clicks == 0:
        return False, dash.no_update, dash.no_update

    result_msg = ''

    if delete_type == 'batch' and delete_id:
        try:
            delete_batch(delete_id)
            result_msg = dbc.Alert('批次删除成功', color='success')
        except Exception as e:
            result_msg = dbc.Alert(f'删除失败: {str(e)}', color='danger')

    batches = get_batches_by_cave(cave_id) if cave_id else []
    table_data = []
    for b in batches:
        measurements = get_measurements_by_batch(b['id'])
        table_data.append({
            'id': b['id'],
            'batch_name': b['batch_name'],
            'survey_date': b.get('survey_date', ''),
            'notes': b.get('notes', ''),
            'measurement_count': len(measurements),
            'actions': html.Div([
                dbc.Button('删除', size='sm', color='link', className='text-danger',
                           id={'type': 'delete-batch-btn', 'index': b['id']}),
            ])
        })

    return False, table_data, result_msg


@callback(
    Output('mgmt-cave-selector', 'options'),
    Output('mgmt-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    Input('mgmt-tabs', 'active_tab'),
    State('mgmt-cave-selector', 'value'),
    prevent_initial_call=False
)
def sync_mgmt_cave_selector(stored_cave_id, active_tab, current_value):
    if active_tab != 'measurements-tab':
        return dash.no_update, dash.no_update

    caves = get_all_caves()
    options = [{'label': c['name'], 'value': c['id']} for c in caves]

    if stored_cave_id:
        valid_ids = [opt['value'] for opt in options]
        if stored_cave_id in valid_ids:
            return options, stored_cave_id

    return options, current_value


@callback(
    Output('mgmt-batch-cave-selector', 'options'),
    Output('mgmt-batch-cave-selector', 'value'),
    Input('selected-cave-store', 'data'),
    Input('mgmt-tabs', 'active_tab'),
    State('mgmt-batch-cave-selector', 'value'),
    prevent_initial_call=False
)
def sync_mgmt_batch_cave_selector(stored_cave_id, active_tab, current_value):
    if active_tab != 'batches-tab':
        return dash.no_update, dash.no_update

    caves = get_all_caves()
    options = [{'label': c['name'], 'value': c['id']} for c in caves]

    if stored_cave_id:
        valid_ids = [opt['value'] for opt in options]
        if stored_cave_id in valid_ids:
            return options, stored_cave_id

    return options, current_value


@callback(
    Output('selected-cave-store', 'data', allow_duplicate=True),
    Input('mgmt-cave-selector', 'value'),
    Input('mgmt-batch-cave-selector', 'value'),
    State('selected-cave-store', 'data'),
    prevent_initial_call=True
)
def sync_mgmt_to_store(mgmt_cave_id, mgmt_batch_cave_id, stored_cave_id):
    triggered = ctx.triggered_id

    if triggered == 'mgmt-cave-selector' and mgmt_cave_id is not None:
        if mgmt_cave_id != stored_cave_id:
            return mgmt_cave_id
    elif triggered == 'mgmt-batch-cave-selector' and mgmt_batch_cave_id is not None:
        if mgmt_batch_cave_id != stored_cave_id:
            return mgmt_batch_cave_id

    return dash.no_update
