import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.integrate import simpson


def validate_measurements(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    errors = []
    valid_rows = []

    required_columns = ['batch_name', 'angle', 'distance', 'depth']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        for idx in range(len(df)):
            errors.append({
                'row': idx + 2,
                'batch_name': df.iloc[idx].get('batch_name', '') if 'batch_name' in df.columns else '',
                'reason': f'缺少必需列: {", ".join(missing_cols)}'
            })
        return pd.DataFrame(), errors

    for idx, row in df.iterrows():
        row_errors = []
        batch_name = str(row.get('batch_name', '')).strip()
        angle = row.get('angle')
        distance = row.get('distance')
        depth = row.get('depth')

        if not batch_name:
            row_errors.append('批次名称不能为空')

        try:
            angle = float(angle)
            if angle < 0 or angle > 360:
                row_errors.append('角度必须在 0-360 度之间')
        except (ValueError, TypeError):
            row_errors.append('角度必须是数字')
            angle = None

        try:
            distance = float(distance)
            if distance < 0:
                row_errors.append('距离不能为负数')
        except (ValueError, TypeError):
            row_errors.append('距离必须是数字')
            distance = None

        try:
            depth = float(depth)
            if depth < 0:
                row_errors.append('深度不能为负数')
        except (ValueError, TypeError):
            row_errors.append('深度必须是数字')
            depth = None

        if row_errors:
            errors.append({
                'row': idx + 2,
                'batch_name': batch_name,
                'angle': angle if angle is not None else '',
                'reason': '; '.join(row_errors)
            })
        else:
            valid_rows.append({
                'batch_name': batch_name,
                'angle': angle,
                'distance': distance,
                'depth': depth
            })

    valid_df = pd.DataFrame(valid_rows)
    if not valid_df.empty:
        for batch_name in valid_df['batch_name'].unique():
            batch_data = valid_df[valid_df['batch_name'] == batch_name]
            angle_counts = batch_data['angle'].value_counts()
            duplicate_angles = angle_counts[angle_counts > 1].index.tolist()
            for dup_angle in duplicate_angles:
                dup_rows = valid_df[(valid_df['batch_name'] == batch_name) & (valid_df['angle'] == dup_angle)]
                for idx in dup_rows.index[1:]:
                    errors.append({
                        'row': idx + 2,
                        'batch_name': batch_name,
                        'angle': dup_angle,
                        'reason': '同一批次内角度重复'
                    })
                    valid_df = valid_df.drop(idx)

    return valid_df, errors


def find_missing_intervals(angles: List[float], threshold: float = 15.0) -> List[Dict]:
    if len(angles) < 2:
        return []

    sorted_angles = sorted(angles)
    missing_intervals = []

    for i in range(len(sorted_angles) - 1):
        gap = sorted_angles[i + 1] - sorted_angles[i]
        if gap > threshold:
            missing_intervals.append({
                'start_angle': sorted_angles[i],
                'end_angle': sorted_angles[i + 1],
                'gap_size': gap
            })

    first_angle = sorted_angles[0]
    last_angle = sorted_angles[-1]
    wrap_gap = (360 - last_angle) + first_angle
    if wrap_gap > threshold:
        missing_intervals.append({
            'start_angle': last_angle,
            'end_angle': first_angle,
            'gap_size': wrap_gap,
            'wraps': True
        })

    return missing_intervals


def calculate_volume_cylindrical(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {'volume': 0, 'max_depth': 0, 'max_distance': 0}

    df = pd.DataFrame(measurements)
    df = df.sort_values('angle')

    angles = df['angle'].values
    distances = df['distance'].values
    depths = df['depth'].values

    max_depth = float(np.max(depths))
    max_distance = float(np.max(distances))

    angles_rad = np.deg2rad(angles)

    area = 0.0
    for i in range(len(angles_rad)):
        j = (i + 1) % len(angles_rad)
        dtheta = angles_rad[j] - angles_rad[i]
        if dtheta < 0:
            dtheta += 2 * math.pi
        r1 = distances[i]
        r2 = distances[j]
        area += 0.5 * (r1 ** 2 + r2 ** 2) * dtheta / 2
        area += 0.5 * r1 * r2 * dtheta
    area = area / 2

    avg_depth = float(np.mean(depths))
    volume = area * avg_depth

    return {
        'volume': volume,
        'max_depth': max_depth,
        'max_distance': max_distance,
        'method': '圆柱近似法',
        'avg_depth': avg_depth,
        'cross_section_area': area
    }


def calculate_volume_conical(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {'volume': 0, 'max_depth': 0, 'max_distance': 0}

    df = pd.DataFrame(measurements)
    df = df.sort_values('angle')

    angles = df['angle'].values
    distances = df['distance'].values
    depths = df['depth'].values

    max_depth = float(np.max(depths))
    max_distance = float(np.max(distances))

    angles_rad = np.deg2rad(angles)
    volume = 0.0

    for i in range(len(angles_rad)):
        j = (i + 1) % len(angles_rad)
        dtheta = angles_rad[j] - angles_rad[i]
        if dtheta < 0:
            dtheta += 2 * math.pi

        r1, h1 = distances[i], depths[i]
        r2, h2 = distances[j], depths[j]

        sector_volume = (dtheta / 3) * (
            r1 * h1 + r2 * h2 + math.sqrt(r1 * r2 * h1 * h2)
        )
        volume += sector_volume

    return {
        'volume': volume,
        'max_depth': max_depth,
        'max_distance': max_distance,
        'method': '扇形锥台法'
    }


def detect_anomalies(measurements: List[Dict], depth_threshold: float = None) -> List[Dict]:
    if len(measurements) < 3:
        return []

    df = pd.DataFrame(measurements)
    df = df.sort_values('angle')

    depths = df['depth'].values
    angles = df['angle'].values

    if depth_threshold is None:
        depth_threshold = np.mean(depths) + 2 * np.std(depths)

    anomalies = []
    in_anomaly = False
    start_idx = 0

    for i in range(len(depths)):
        is_anomalous = depths[i] > depth_threshold

        if is_anomalous and not in_anomaly:
            in_anomaly = True
            start_idx = i
        elif not is_anomalous and in_anomaly:
            in_anomaly = False
            anomaly_region = {
                'start_angle': float(angles[start_idx]),
                'end_angle': float(angles[i - 1]),
                'anomaly_type': '深度异常',
                'description': f'深度超过阈值 {depth_threshold:.2f}m 的异常凹陷区域，共 {i - start_idx} 个测量点'
            }
            anomalies.append(anomaly_region)

    if in_anomaly:
        anomaly_region = {
            'start_angle': float(angles[start_idx]),
            'end_angle': float(angles[-1]),
            'anomaly_type': '深度异常',
            'description': f'深度超过阈值 {depth_threshold:.2f}m 的异常凹陷区域，共 {len(depths) - start_idx} 个测量点'
        }
        anomalies.append(anomaly_region)

    return anomalies


def compute_batch_statistics(batch_id: int, measurements: List[Dict]) -> Dict:
    if not measurements:
        return {}

    df = pd.DataFrame(measurements)

    volume_result = calculate_volume_conical(measurements)

    anomalies = detect_anomalies(measurements)

    missing_intervals = find_missing_intervals(df['angle'].tolist())

    stats = {
        'batch_id': batch_id,
        'measurement_count': len(measurements),
        'max_depth': float(df['depth'].max()),
        'min_depth': float(df['depth'].min()),
        'avg_depth': float(df['depth'].mean()),
        'std_depth': float(df['depth'].std()),
        'max_distance': float(df['distance'].max()),
        'min_distance': float(df['distance'].min()),
        'avg_distance': float(df['distance'].mean()),
        'volume': volume_result['volume'],
        'volume_method': volume_result['method'],
        'anomaly_count': len(anomalies),
        'anomalies': anomalies,
        'missing_intervals': missing_intervals,
        'missing_interval_count': len(missing_intervals)
    }

    return stats


def compare_batches(batches_data: List[Dict]) -> List[Dict]:
    comparison = []

    for i in range(len(batches_data)):
        for j in range(i + 1, len(batches_data)):
            batch1 = batches_data[i]
            batch2 = batches_data[j]

            vol_diff = batch2['volume'] - batch1['volume']
            vol_change_pct = (vol_diff / batch1['volume'] * 100) if batch1['volume'] > 0 else 0

            depth_diff = batch2['max_depth'] - batch1['max_depth']

            comparison.append({
                'batch1_id': batch1['batch_id'],
                'batch1_name': batch1['batch_name'],
                'batch2_id': batch2['batch_id'],
                'batch2_name': batch2['batch_name'],
                'volume_diff': vol_diff,
                'volume_change_pct': vol_change_pct,
                'max_depth_diff': depth_diff,
            })

    return comparison


def generate_cross_section_data(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {'x': [], 'y': [], 'angles': []}

    df = pd.DataFrame(measurements)
    df = df.sort_values('angle')

    angles = df['angle'].values
    distances = df['distance'].values

    angles_rad = np.deg2rad(angles)
    x = distances * np.cos(angles_rad)
    y = distances * np.sin(angles_rad)

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'angles': angles.tolist(),
        'distances': distances.tolist(),
        'depths': df['depth'].values.tolist()
    }


def generate_3d_point_cloud(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {'x': [], 'y': [], 'z': []}

    df = pd.DataFrame(measurements)

    angles_rad = np.deg2rad(df['angle'].values)
    x = df['distance'].values * np.cos(angles_rad)
    y = df['distance'].values * np.sin(angles_rad)
    z = df['depth'].values

    return {
        'x': x.tolist(),
        'y': y.tolist(),
        'z': z.tolist(),
        'angles': df['angle'].values.tolist(),
        'distances': df['distance'].values.tolist(),
        'depths': df['depth'].values.tolist()
    }


def interpolate_to_common_angles(measurements1: List[Dict], measurements2: List[Dict],
                                 num_points: int = 360) -> Dict:
    if not measurements1 or not measurements2:
        return {'angles': [], 'distances1': [], 'distances2': [], 'depths1': [], 'depths2': []}

    df1 = pd.DataFrame(measurements1).sort_values('angle')
    df2 = pd.DataFrame(measurements2).sort_values('angle')

    angles1 = df1['angle'].values
    distances1 = df1['distance'].values
    depths1 = df1['depth'].values

    angles2 = df2['angle'].values
    distances2 = df2['distance'].values
    depths2 = df2['depth'].values

    common_angles = np.linspace(0, 360, num_points, endpoint=False)

    angles1_ext = np.concatenate([angles1 - 360, angles1, angles1 + 360])
    distances1_ext = np.concatenate([distances1, distances1, distances1])
    depths1_ext = np.concatenate([depths1, depths1, depths1])

    angles2_ext = np.concatenate([angles2 - 360, angles2, angles2 + 360])
    distances2_ext = np.concatenate([distances2, distances2, distances2])
    depths2_ext = np.concatenate([depths2, depths2, depths2])

    from scipy.interpolate import interp1d

    f_dist1 = interp1d(angles1_ext, distances1_ext, kind='linear')
    f_depth1 = interp1d(angles1_ext, depths1_ext, kind='linear')
    f_dist2 = interp1d(angles2_ext, distances2_ext, kind='linear')
    f_depth2 = interp1d(angles2_ext, depths2_ext, kind='linear')

    interp_distances1 = f_dist1(common_angles)
    interp_depths1 = f_depth1(common_angles)
    interp_distances2 = f_dist2(common_angles)
    interp_depths2 = f_depth2(common_angles)

    return {
        'angles': common_angles.tolist(),
        'distances1': interp_distances1.tolist(),
        'distances2': interp_distances2.tolist(),
        'depths1': interp_depths1.tolist(),
        'depths2': interp_depths2.tolist()
    }


def calculate_deformation_heatmap(measurements_base: List[Dict], measurements_compare: List[Dict]) -> Dict:
    interp = interpolate_to_common_angles(measurements_base, measurements_compare)

    if not interp['angles']:
        return {
            'angles': [],
            'distance_diff': [],
            'depth_diff': [],
            'max_distance_expansion': 0,
            'max_distance_contraction': 0,
            'max_depth_increase': 0,
            'max_depth_decrease': 0,
            'avg_distance_change': 0,
            'avg_depth_change': 0
        }

    distances1 = np.array(interp['distances1'])
    distances2 = np.array(interp['distances2'])
    depths1 = np.array(interp['depths1'])
    depths2 = np.array(interp['depths2'])

    distance_diff = distances2 - distances1
    depth_diff = depths2 - depths1

    return {
        'angles': interp['angles'],
        'distance_diff': distance_diff.tolist(),
        'depth_diff': depth_diff.tolist(),
        'max_distance_expansion': float(np.max(distance_diff)),
        'max_distance_contraction': float(np.min(distance_diff)),
        'max_depth_increase': float(np.max(depth_diff)),
        'max_depth_decrease': float(np.min(depth_diff)),
        'avg_distance_change': float(np.mean(distance_diff)),
        'avg_depth_change': float(np.mean(depth_diff))
    }


def compute_cross_section_difference(measurements_base: List[Dict], measurements_compare: List[Dict]) -> Dict:
    interp = interpolate_to_common_angles(measurements_base, measurements_compare)

    if not interp['angles']:
        return {
            'angles': [],
            'x_base': [], 'y_base': [],
            'x_compare': [], 'y_compare': [],
            'x_diff': [], 'y_diff': [],
            'radial_diff': []
        }

    angles = np.array(interp['angles'])
    angles_rad = np.deg2rad(angles)

    distances1 = np.array(interp['distances1'])
    distances2 = np.array(interp['distances2'])

    x_base = distances1 * np.cos(angles_rad)
    y_base = distances1 * np.sin(angles_rad)
    x_compare = distances2 * np.cos(angles_rad)
    y_compare = distances2 * np.sin(angles_rad)

    radial_diff = distances2 - distances1

    x_diff = x_compare - x_base
    y_diff = y_compare - y_base

    return {
        'angles': angles.tolist(),
        'x_base': x_base.tolist(),
        'y_base': y_base.tolist(),
        'x_compare': x_compare.tolist(),
        'y_compare': y_compare.tolist(),
        'x_diff': x_diff.tolist(),
        'y_diff': y_diff.tolist(),
        'radial_diff': radial_diff.tolist()
    }


def calculate_volume_trend(batches_stats: List[Dict]) -> Dict:
    if not batches_stats:
        return {'dates': [], 'volumes': [], 'max_depths': [], 'avg_depths': [], 'volume_changes': []}

    sorted_batches = sorted(batches_stats, key=lambda x: x.get('survey_date', ''))

    dates = [b.get('survey_date', b['batch_name']) for b in sorted_batches]
    volumes = [b['volume'] for b in sorted_batches]
    max_depths = [b['max_depth'] for b in sorted_batches]
    avg_depths = [b['avg_depth'] for b in sorted_batches]

    volume_changes = [0.0]
    for i in range(1, len(volumes)):
        if volumes[i - 1] > 0:
            change = ((volumes[i] - volumes[i - 1]) / volumes[i - 1]) * 100
        else:
            change = 0.0
        volume_changes.append(change)

    total_volume_change = volumes[-1] - volumes[0] if len(volumes) >= 2 else 0
    total_volume_change_pct = (total_volume_change / volumes[0] * 100) if len(volumes) >= 2 and volumes[0] > 0 else 0

    return {
        'dates': dates,
        'volumes': volumes,
        'max_depths': max_depths,
        'avg_depths': avg_depths,
        'volume_changes': volume_changes,
        'total_volume_change': total_volume_change,
        'total_volume_change_pct': total_volume_change_pct,
        'batch_names': [b['batch_name'] for b in sorted_batches]
    }


def detect_risk_areas(measurements_base: List[Dict], measurements_compare: List[Dict],
                      depth_threshold_pct: float = 10.0,
                      distance_threshold_pct: float = 5.0) -> List[Dict]:
    interp = interpolate_to_common_angles(measurements_base, measurements_compare)

    if not interp['angles']:
        return []

    angles = np.array(interp['angles'])
    depths1 = np.array(interp['depths1'])
    depths2 = np.array(interp['depths2'])
    distances1 = np.array(interp['distances1'])
    distances2 = np.array(interp['distances2'])

    depth_change_pct = np.where(depths1 != 0, (depths2 - depths1) / depths1 * 100, 0)
    distance_change_pct = np.where(distances1 != 0, (distances2 - distances1) / distances1 * 100, 0)

    risks = []

    in_new_pit = False
    pit_start_idx = 0

    for i in range(len(angles)):
        is_new_pit = depth_change_pct[i] > depth_threshold_pct

        if is_new_pit and not in_new_pit:
            in_new_pit = True
            pit_start_idx = i
        elif not is_new_pit and in_new_pit:
            in_new_pit = False
            pit_angles = angles[pit_start_idx:i]
            pit_depth_changes = (depths2 - depths1)[pit_start_idx:i]
            max_depth_increase = float(np.max(pit_depth_changes))
            avg_depth_increase = float(np.mean(pit_depth_changes))

            risks.append({
                'risk_type': '新增凹陷',
                'severity': '高' if max_depth_increase > np.mean(depths1) * 0.15 else '中',
                'start_angle': float(pit_angles[0]),
                'end_angle': float(pit_angles[-1]),
                'max_depth_increase': max_depth_increase,
                'avg_depth_increase': avg_depth_increase,
                'description': f'该区域深度增加 {max_depth_increase:.2f}m（+{np.mean(pit_depth_changes)/np.mean(depths1)*100:.1f}%），存在新增凹陷风险'
            })

    if in_new_pit:
        pit_angles = angles[pit_start_idx:]
        pit_depth_changes = (depths2 - depths1)[pit_start_idx:]
        max_depth_increase = float(np.max(pit_depth_changes))
        avg_depth_increase = float(np.mean(pit_depth_changes))
        risks.append({
            'risk_type': '新增凹陷',
            'severity': '高' if max_depth_increase > np.mean(depths1) * 0.15 else '中',
            'start_angle': float(pit_angles[0]),
            'end_angle': float(pit_angles[-1]),
            'max_depth_increase': max_depth_increase,
            'avg_depth_increase': avg_depth_increase,
            'description': f'该区域深度增加 {max_depth_increase:.2f}m（+{np.mean(pit_depth_changes)/np.mean(depths1)*100:.1f}%），存在新增凹陷风险'
        })

    in_backfill = False
    backfill_start_idx = 0

    for i in range(len(angles)):
        is_backfill = depth_change_pct[i] < -depth_threshold_pct

        if is_backfill and not in_backfill:
            in_backfill = True
            backfill_start_idx = i
        elif not is_backfill and in_backfill:
            in_backfill = False
            backfill_angles = angles[backfill_start_idx:i]
            backfill_depth_changes = (depths2 - depths1)[backfill_start_idx:i]
            max_depth_decrease = float(np.min(backfill_depth_changes))
            avg_depth_decrease = float(np.mean(backfill_depth_changes))

            risks.append({
                'risk_type': '回填',
                'severity': '中' if abs(max_depth_decrease) > np.mean(depths1) * 0.1 else '低',
                'start_angle': float(backfill_angles[0]),
                'end_angle': float(backfill_angles[-1]),
                'max_depth_decrease': max_depth_decrease,
                'avg_depth_decrease': avg_depth_decrease,
                'description': f'该区域深度减少 {abs(max_depth_decrease):.2f}m（{np.mean(backfill_depth_changes)/np.mean(depths1)*100:.1f}%），存在回填现象'
            })

    if in_backfill:
        backfill_angles = angles[backfill_start_idx:]
        backfill_depth_changes = (depths2 - depths1)[backfill_start_idx:]
        max_depth_decrease = float(np.min(backfill_depth_changes))
        avg_depth_decrease = float(np.mean(backfill_depth_changes))
        risks.append({
            'risk_type': '回填',
            'severity': '中' if abs(max_depth_decrease) > np.mean(depths1) * 0.1 else '低',
            'start_angle': float(backfill_angles[0]),
            'end_angle': float(backfill_angles[-1]),
            'max_depth_decrease': max_depth_decrease,
            'avg_depth_decrease': avg_depth_decrease,
            'description': f'该区域深度减少 {abs(max_depth_decrease):.2f}m（{np.mean(backfill_depth_changes)/np.mean(depths1)*100:.1f}%），存在回填现象'
        })

    in_expansion = False
    expansion_start_idx = 0

    for i in range(len(angles)):
        is_expansion = distance_change_pct[i] > distance_threshold_pct

        if is_expansion and not in_expansion:
            in_expansion = True
            expansion_start_idx = i
        elif not is_expansion and in_expansion:
            in_expansion = False
            exp_angles = angles[expansion_start_idx:i]
            exp_dist_changes = (distances2 - distances1)[expansion_start_idx:i]
            max_dist_increase = float(np.max(exp_dist_changes))
            avg_dist_increase = float(np.mean(exp_dist_changes))

            risks.append({
                'risk_type': '扩容风险',
                'severity': '高' if max_dist_increase > np.mean(distances1) * 0.1 else '中',
                'start_angle': float(exp_angles[0]),
                'end_angle': float(exp_angles[-1]),
                'max_distance_increase': max_dist_increase,
                'avg_distance_increase': avg_dist_increase,
                'description': f'该区域半径增加 {max_dist_increase:.2f}m（+{np.mean(exp_dist_changes)/np.mean(distances1)*100:.1f}%），存在扩容风险'
            })

    if in_expansion:
        exp_angles = angles[expansion_start_idx:]
        exp_dist_changes = (distances2 - distances1)[expansion_start_idx:]
        max_dist_increase = float(np.max(exp_dist_changes))
        avg_dist_increase = float(np.mean(exp_dist_changes))
        risks.append({
            'risk_type': '扩容风险',
            'severity': '高' if max_dist_increase > np.mean(distances1) * 0.1 else '中',
            'start_angle': float(exp_angles[0]),
            'end_angle': float(exp_angles[-1]),
            'max_distance_increase': max_dist_increase,
            'avg_distance_increase': avg_dist_increase,
            'description': f'该区域半径增加 {max_dist_increase:.2f}m（+{np.mean(exp_dist_changes)/np.mean(distances1)*100:.1f}%），存在扩容风险'
        })

    return risks


def generate_temporal_analysis_report(cave_name: str, base_batch: Dict, compare_batch: Dict,
                                      deformation_data: Dict, volume_trend: Dict,
                                      risk_areas: List[Dict], cs_diff_data: Dict) -> str:
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("盐穴多时期形变与容积演化分析报告")
    report_lines.append("=" * 60)
    report_lines.append("")

    report_lines.append(f"盐穴名称: {cave_name}")
    report_lines.append(f"基准批次: {base_batch['batch_name']} ({base_batch.get('survey_date', '未知日期')})")
    report_lines.append(f"对比批次: {compare_batch['batch_name']} ({compare_batch.get('survey_date', '未知日期')})")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("一、容积变化分析")
    report_lines.append("-" * 60)
    report_lines.append("")

    if volume_trend['volumes']:
        report_lines.append(f"基准容积: {volume_trend['volumes'][0]:.2f} m³")
        report_lines.append(f"对比容积: {volume_trend['volumes'][-1]:.2f} m³")
        report_lines.append(f"容积变化量: {volume_trend['total_volume_change']:+.2f} m³")
        report_lines.append(f"容积变化率: {volume_trend['total_volume_change_pct']:+.2f} %")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("二、断面形变分析")
    report_lines.append("-" * 60)
    report_lines.append("")

    report_lines.append(f"最大径向扩张: {deformation_data['max_distance_expansion']:+.2f} m")
    report_lines.append(f"最大径向收缩: {deformation_data['max_distance_contraction']:+.2f} m")
    report_lines.append(f"平均径向变化: {deformation_data['avg_distance_change']:+.2f} m")
    report_lines.append("")
    report_lines.append(f"最大深度增加: {deformation_data['max_depth_increase']:+.2f} m")
    report_lines.append(f"最大深度减少: {deformation_data['max_depth_decrease']:+.2f} m")
    report_lines.append(f"平均深度变化: {deformation_data['avg_depth_change']:+.2f} m")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("三、风险区域分析")
    report_lines.append("-" * 60)
    report_lines.append("")

    if risk_areas:
        for i, risk in enumerate(risk_areas, 1):
            report_lines.append(f"{i}. {risk['risk_type']} - 严重程度: {risk['severity']}")
            report_lines.append(f"   角度范围: {risk['start_angle']:.1f}° - {risk['end_angle']:.1f}°")
            report_lines.append(f"   描述: {risk['description']}")
            report_lines.append("")
    else:
        report_lines.append("未检测到显著风险区域。")
        report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("四、容积变化趋势")
    report_lines.append("-" * 60)
    report_lines.append("")

    for i, (name, vol, change) in enumerate(zip(
        volume_trend.get('batch_names', []),
        volume_trend['volumes'],
        volume_trend['volume_changes']
    )):
        if i == 0:
            report_lines.append(f"{i + 1}. {name}: {vol:.2f} m³ (基准)")
        else:
            report_lines.append(f"{i + 1}. {name}: {vol:.2f} m³ ({change:+.2f}%)")
    report_lines.append("")

    report_lines.append("=" * 60)
    report_lines.append("报告生成完毕")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)
