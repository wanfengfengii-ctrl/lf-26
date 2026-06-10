import dash
import base64
import io
import pandas as pd
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import date

from database import (
    get_all_caves, create_cave, create_batch, add_measurement,
    check_angle_duplicate, get_batch_by_name, get_batches_by_cave,
    save_volume_estimate, save_anomaly_regions
)
from analysis import (
    validate_measurements, compute_batch_statistics, calculate_volume_conical,
    detect_anomalies
)

dash.register_page(__name__, path='/', name='数据导入')


def layout():
    caves = get_all_caves()
    cave_options = [{'label': c['name'], 'value': c['id']} for c in caves]

    return dbc.Container([
        html.H4("数据导入", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H6("选择盐穴"),
                dcc.Dropdown(
                    id='import-cave-selector',
                    options=cave_options,
                    value=cave_options[0]['value'] if cave_options else None,
                    placeholder='选择一个盐穴...'
                ),
            ], width=6),
            dbc.Col([
                html.H6("新建盐穴"),
                dbc.InputGroup([
                    dbc.Input(id='new-cave-name', type='text', placeholder='输入盐穴名称'),
                    dbc.Button('创建', id='create-cave-btn', color='primary', n_clicks=0),
                ]),
                html.Div(id='create-cave-result', className='mt-2'),
            ], width=6),
        ], className='mb-4'),

        html.Hr(),

        html.H6("上传 CSV 文件", className='mb-3'),
        dcc.Upload(
            id='upload-csv',
            children=html.Div([
                '拖拽 CSV 文件到这里 或 ',
                html.A('点击选择文件')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0'
            },
            multiple=False
        ),

        dbc.Row([
            dbc.Col([
                html.H6("勘测日期"),
                dcc.DatePickerSingle(
                    id='survey-date',
                    date=date.today(),
                    display_format='YYYY-MM-DD'
                ),
            ], width=6),
            dbc.Col([
                html.H6("备注"),
                dbc.Input(
                    id='import-notes',
                    type='text',
                    placeholder='可选备注信息'
                ),
            ], width=6),
        ], className='mb-4'),

        dbc.Button('导入数据', id='import-data-btn', color='success', n_clicks=0, disabled=True),

        html.Hr(),

        html.H6("导入结果", className='mb-3'),
        html.Div(id='import-result-summary', className='mb-3'),

        dbc.Card([
            dbc.CardHeader("失败数据行"),
            dbc.CardBody([
                html.Div(id='import-errors-container')
            ])
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader("成功导入的批次"),
            dbc.CardBody([
                html.Div(id='import-success-container')
            ])
        ]),

        html.Hr(),

        html.H6("示例数据", className='mb-3'),
        html.P("点击下载示例 CSV 文件，查看正确的数据格式："),
        html.A(
            '下载示例 CSV',
            id='download-sample-link',
            href='/download/sample',
            download='sample_survey.csv',
            className='btn btn-outline-primary'
        ),

        html.H6("CSV 文件格式要求：", className='mt-4'),
        html.Ul([
            html.Li("必需列: batch_name（批次名称）、angle（测量角度，0-360度）、distance（距离，米）、depth（深度，米）"),
            html.Li("同一批次内测量角度不能重复"),
            html.Li("距离和深度不能为负数"),
            html.Li("同一 CSV 文件可包含多个批次的数据"),
        ]),
    ], fluid=True)


@callback(
    Output('import-cave-selector', 'options'),
    Output('import-cave-selector', 'value'),
    Output('create-cave-result', 'children'),
    Input('create-cave-btn', 'n_clicks'),
    State('new-cave-name', 'value'),
    State('import-cave-selector', 'value'),
    prevent_initial_call=True
)
def create_new_cave(n_clicks, cave_name, current_value):
    if not cave_name or not cave_name.strip():
        caves = get_all_caves()
        options = [{'label': c['name'], 'value': c['id']} for c in caves]
        return options, current_value, dbc.Alert('请输入盐穴名称', color='warning')

    try:
        cave_id = create_cave(cave_name.strip())
        caves = get_all_caves()
        options = [{'label': c['name'], 'value': c['id']} for c in caves]
        return options, cave_id, dbc.Alert(f'盐穴 "{cave_name}" 创建成功', color='success')
    except Exception as e:
        caves = get_all_caves()
        options = [{'label': c['name'], 'value': c['id']} for c in caves]
        return options, current_value, dbc.Alert(f'创建失败: {str(e)}', color='danger')


@callback(
    Output('import-data-btn', 'disabled'),
    Output('upload-csv', 'contents'),
    Input('upload-csv', 'filename'),
    State('import-cave-selector', 'value')
)
def enable_import_button(filename, cave_id):
    if filename and cave_id:
        return False, None
    return True, None


@callback(
    Output('import-result-summary', 'children'),
    Output('import-errors-container', 'children'),
    Output('import-success-container', 'children'),
    Input('import-data-btn', 'n_clicks'),
    State('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('import-cave-selector', 'value'),
    State('survey-date', 'date'),
    State('import-notes', 'value'),
    prevent_initial_call=True
)
def process_import(n_clicks, contents, filename, cave_id, survey_date, notes):
    if not contents or not cave_id:
        return '', '', ''

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return dbc.Alert('请上传 CSV 文件', color='danger'), '', ''
    except Exception as e:
        return dbc.Alert(f'文件解析失败: {str(e)}', color='danger'), '', ''

    valid_df, errors = validate_measurements(df)

    success_batches = []
    failed_count = len(errors)
    success_count = len(valid_df)

    if not valid_df.empty:
        for batch_name in valid_df['batch_name'].unique():
            batch_data = valid_df[valid_df['batch_name'] == batch_name]

            existing_batch = get_batch_by_name(cave_id, batch_name)
            if existing_batch:
                for _, row in batch_data.iterrows():
                    errors.append({
                        'row': '-',
                        'batch_name': batch_name,
                        'angle': row['angle'],
                        'reason': f'批次 "{batch_name}" 已存在于该盐穴中'
                    })
                    failed_count += 1
                    success_count -= 1
                continue

            try:
                batch_id = create_batch(cave_id, batch_name, survey_date, notes or '')

                measurements = []
                for _, row in batch_data.iterrows():
                    add_measurement(batch_id, row['angle'], row['distance'], row['depth'])
                    measurements.append({
                        'angle': row['angle'],
                        'distance': row['distance'],
                        'depth': row['depth']
                    })

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

                success_batches.append({
                    'batch_name': batch_name,
                    'measurement_count': len(batch_data),
                    'volume': vol_result['volume'],
                    'max_depth': vol_result['max_depth']
                })
            except Exception as e:
                for _, row in batch_data.iterrows():
                    errors.append({
                        'row': '-',
                        'batch_name': batch_name,
                        'angle': row['angle'],
                        'reason': f'数据库错误: {str(e)}'
                    })
                    failed_count += 1
                    success_count -= 1

    summary = dbc.Alert(
        f'导入完成: 成功 {success_count} 条，失败 {failed_count} 条，共 {len(success_batches)} 个批次',
        color='success' if success_count > 0 else 'warning'
    )

    if errors:
        error_df = pd.DataFrame(errors)
        error_table = dash_table.DataTable(
            data=error_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in error_df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'}
        )
    else:
        error_table = html.P('没有失败的数据行')

    if success_batches:
        success_df = pd.DataFrame(success_batches)
        success_df['volume'] = success_df['volume'].apply(lambda x: f'{x:.2f} m³')
        success_df['max_depth'] = success_df['max_depth'].apply(lambda x: f'{x:.2f} m')
        success_table = dash_table.DataTable(
            data=success_df.to_dict('records'),
            columns=[
                {'name': '批次名称', 'id': 'batch_name'},
                {'name': '测量点数', 'id': 'measurement_count'},
                {'name': '估算容积', 'id': 'volume'},
                {'name': '最大深度', 'id': 'max_depth'}
            ],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    else:
        success_table = html.P('没有成功导入的批次')

    return summary, error_table, success_table
