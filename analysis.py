import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.signal import find_peaks


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


def evaluate_angle_coverage(measurements: List[Dict], expected_interval: float = 15.0) -> Dict:
    if not measurements:
        return {
            'score': 0,
            'coverage_ratio': 0,
            'covered_degrees': 0,
            'gaps': [],
            'gap_count': 0,
            'issues': []
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values

    covered_degrees = 0
    gaps = []
    issues = []

    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        if j == 0:
            gap = (360 - angles[i]) + angles[j]
            wraps = True
        else:
            gap = angles[j] - angles[i]
            wraps = False

        if gap > expected_interval * 1.5:
            gaps.append({
                'start_angle': float(angles[i]),
                'end_angle': float(angles[j]),
                'gap_size': float(gap),
                'wraps': wraps
            })

        covered_degrees += min(gap, expected_interval * 2)

    coverage_ratio = min(covered_degrees / 360.0, 1.0)

    if coverage_ratio >= 0.95:
        score = 100
    elif coverage_ratio >= 0.85:
        score = 80
    elif coverage_ratio >= 0.7:
        score = 60
    else:
        score = 40

    if coverage_ratio < 0.9:
        issues.append(f'角度覆盖率不足，仅覆盖 {coverage_ratio * 100:.1f}% 的圆周范围')

    return {
        'score': score,
        'coverage_ratio': float(coverage_ratio),
        'covered_degrees': float(min(covered_degrees, 360)),
        'gaps': gaps,
        'gap_count': len(gaps),
        'issues': issues
    }


def detect_outlier_points(measurements: List[Dict], z_threshold: float = 3.0) -> Dict:
    if len(measurements) < 5:
        return {
            'score': 100,
            'outliers': [],
            'outlier_count': 0,
            'distance_outliers': [],
            'depth_outliers': [],
            'issues': []
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    distances = df['distance'].values
    depths = df['depth'].values

    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    depth_mean = np.mean(depths)
    depth_std = np.std(depths)

    dist_zscores = np.abs((distances - dist_mean) / dist_std) if dist_std > 0 else np.zeros_like(distances)
    depth_zscores = np.abs((depths - depth_mean) / depth_std) if depth_std > 0 else np.zeros_like(depths)

    dist_outlier_indices = np.where(dist_zscores > z_threshold)[0]
    depth_outlier_indices = np.where(depth_zscores > z_threshold)[0]

    distance_outliers = []
    for idx in dist_outlier_indices:
        distance_outliers.append({
            'angle': float(df.iloc[idx]['angle']),
            'distance': float(distances[idx]),
            'z_score': float(dist_zscores[idx]),
            'type': 'distance_outlier'
        })

    depth_outliers = []
    for idx in depth_outlier_indices:
        depth_outliers.append({
            'angle': float(df.iloc[idx]['angle']),
            'depth': float(depths[idx]),
            'z_score': float(depth_zscores[idx]),
            'type': 'depth_outlier'
        })

    all_outlier_angles = set(dist_outlier_indices) | set(depth_outlier_indices)
    outlier_count = len(all_outlier_angles)
    outlier_ratio = outlier_count / len(measurements)

    if outlier_ratio == 0:
        score = 100
    elif outlier_ratio < 0.02:
        score = 85
    elif outlier_ratio < 0.05:
        score = 70
    elif outlier_ratio < 0.1:
        score = 55
    else:
        score = 40

    issues = []
    if outlier_count > 0:
        issues.append(f'检测到 {outlier_count} 个异常跳点（占比 {outlier_ratio * 100:.1f}%）')

    return {
        'score': score,
        'outliers': distance_outliers + depth_outliers,
        'outlier_count': outlier_count,
        'distance_outliers': distance_outliers,
        'depth_outliers': depth_outliers,
        'issues': issues
    }


def detect_missing_intervals_quality(measurements: List[Dict], threshold: float = 15.0) -> Dict:
    if not measurements:
        return {
            'score': 0,
            'missing_intervals': [],
            'missing_count': 0,
            'total_missing_degrees': 0,
            'max_gap_size': 0,
            'avg_gap_size': 0,
            'issues': []
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values

    missing_intervals = []
    total_missing = 0

    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        if j == 0:
            gap = (360 - angles[i]) + angles[j]
            wraps = True
        else:
            gap = angles[j] - angles[i]
            wraps = False

        if gap > threshold:
            missing_intervals.append({
                'start_angle': float(angles[i]),
                'end_angle': float(angles[j]),
                'gap_size': float(gap),
                'expected_interval': float(threshold),
                'wraps': wraps
            })
            total_missing += gap - threshold

    max_gap = max([m['gap_size'] for m in missing_intervals]) if missing_intervals else 0
    avg_gap = np.mean([m['gap_size'] for m in missing_intervals]) if missing_intervals else 0

    if len(missing_intervals) == 0:
        score = 100
    elif len(missing_intervals) == 1 and max_gap < threshold * 2:
        score = 80
    elif len(missing_intervals) <= 3:
        score = 60
    else:
        score = 40

    issues = []
    if missing_intervals:
        issues.append(f'存在 {len(missing_intervals)} 个缺失区间，最大间隔 {max_gap:.1f}°')

    return {
        'score': score,
        'missing_intervals': missing_intervals,
        'missing_count': len(missing_intervals),
        'total_missing_degrees': float(total_missing),
        'max_gap_size': float(max_gap),
        'avg_gap_size': float(avg_gap),
        'issues': issues
    }


def detect_repetitive_patterns(measurements: List[Dict]) -> Dict:
    if len(measurements) < 10:
        return {
            'score': 100,
            'patterns': [],
            'pattern_count': 0,
            'autocorrelation_peaks': [],
            'issues': []
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    distances = df['distance'].values
    depths = df['depth'].values

    def compute_autocorrelation(data):
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return np.zeros(n)
        result = []
        for lag in range(1, n // 2):
            cov = np.mean((data[:n - lag] - mean) * (data[lag:] - mean))
            result.append(cov / var)
        return np.array(result)

    dist_autocorr = compute_autocorrelation(distances)
    depth_autocorr = compute_autocorrelation(depths)

    dist_peaks, dist_props = find_peaks(dist_autocorr, height=0.5, distance=3)
    depth_peaks, depth_props = find_peaks(depth_autocorr, height=0.5, distance=3)

    patterns = []
    for peak in dist_peaks:
        patterns.append({
            'type': 'distance_pattern',
            'periodicity': int(peak + 1),
            'correlation_strength': float(dist_autocorr[peak])
        })

    for peak in depth_peaks:
        patterns.append({
            'type': 'depth_pattern',
            'periodicity': int(peak + 1),
            'correlation_strength': float(depth_autocorr[peak])
        })

    strong_patterns = [p for p in patterns if p['correlation_strength'] > 0.7]

    if not strong_patterns:
        score = 100
    elif len(strong_patterns) == 1:
        score = 75
    else:
        score = 50

    issues = []
    if strong_patterns:
        issues.append(f'检测到 {len(strong_patterns)} 个强重复趋势模式，可能存在设备周期性误差')

    return {
        'score': score,
        'patterns': patterns,
        'pattern_count': len(patterns),
        'strong_pattern_count': len(strong_patterns),
        'autocorrelation_peaks': patterns,
        'issues': issues
    }


def evaluate_volatility(measurements: List[Dict]) -> Dict:
    if len(measurements) < 5:
        return {
            'score': 100,
            'distance_std': 0,
            'depth_std': 0,
            'distance_cv': 0,
            'depth_cv': 0,
            'distance_volatility_level': '正常',
            'depth_volatility_level': '正常',
            'issues': []
        }

    df = pd.DataFrame(measurements)
    distances = df['distance'].values
    depths = df['depth'].values

    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    dist_cv = dist_std / dist_mean if dist_mean > 0 else 0

    depth_mean = np.mean(depths)
    depth_std = np.std(depths)
    depth_cv = depth_std / depth_mean if depth_mean > 0 else 0

    def classify_volatility(cv):
        if cv < 0.05:
            return '极低波动'
        elif cv < 0.1:
            return '正常'
        elif cv < 0.2:
            return '中等波动'
        elif cv < 0.35:
            return '高波动'
        else:
            return '异常波动'

    dist_level = classify_volatility(dist_cv)
    depth_level = classify_volatility(depth_cv)

    dist_score = 100 if dist_cv < 0.15 else (80 if dist_cv < 0.25 else (60 if dist_cv < 0.4 else 40))
    depth_score = 100 if depth_cv < 0.15 else (80 if depth_cv < 0.25 else (60 if depth_cv < 0.4 else 40))

    avg_score = (dist_score + depth_score) / 2

    issues = []
    if dist_cv >= 0.25:
        issues.append(f'距离数据波动异常，变异系数 {dist_cv * 100:.1f}%')
    if depth_cv >= 0.25:
        issues.append(f'深度数据波动异常，变异系数 {depth_cv * 100:.1f}%')

    return {
        'score': float(avg_score),
        'distance_std': float(dist_std),
        'depth_std': float(depth_std),
        'distance_cv': float(dist_cv),
        'depth_cv': float(depth_cv),
        'distance_volatility_level': dist_level,
        'depth_volatility_level': depth_level,
        'distance_mean': float(dist_mean),
        'depth_mean': float(depth_mean),
        'issues': issues
    }


def evaluate_batch_consistency(batches_data: List[Dict]) -> Dict:
    if len(batches_data) < 2:
        return {
            'score': 100,
            'consistency_score': 100,
            'batch_pairs': [],
            'volume_variation_cv': 0,
            'max_depth_variation_cv': 0,
            'issues': []
        }

    volumes = [b.get('volume', 0) for b in batches_data]
    max_depths = [b.get('max_depth', 0) for b in batches_data]

    volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
    depth_cv = np.std(max_depths) / np.mean(max_depths) if np.mean(max_depths) > 0 else 0

    batch_pairs = []
    for i in range(len(batches_data)):
        for j in range(i + 1, len(batches_data)):
            vol_diff_pct = abs((volumes[j] - volumes[i]) / volumes[i] * 100) if volumes[i] > 0 else 0
            depth_diff_pct = abs((max_depths[j] - max_depths[i]) / max_depths[i] * 100) if max_depths[i] > 0 else 0
            pair_score = max(0, 100 - vol_diff_pct - depth_diff_pct)

            batch_pairs.append({
                'batch1_name': batches_data[i].get('batch_name', f'批次{i + 1}'),
                'batch2_name': batches_data[j].get('batch_name', f'批次{j + 1}'),
                'volume_diff_pct': float(vol_diff_pct),
                'depth_diff_pct': float(depth_diff_pct),
                'pair_consistency_score': float(pair_score)
            })

    avg_pair_score = np.mean([p['pair_consistency_score'] for p in batch_pairs]) if batch_pairs else 100

    volume_consistency_score = max(0, 100 - volume_cv * 200)
    depth_consistency_score = max(0, 100 - depth_cv * 200)

    overall_score = (volume_consistency_score + depth_consistency_score + avg_pair_score) / 3

    issues = []
    if volume_cv > 0.1:
        issues.append(f'批次间容积变异度过高（CV={volume_cv * 100:.1f}%），可能存在系统误差')
    if depth_cv > 0.1:
        issues.append(f'批次间最大深度变异度过高（CV={depth_cv * 100:.1f}%），建议校准设备')

    return {
        'score': float(overall_score),
        'consistency_score': float(overall_score),
        'batch_pairs': batch_pairs,
        'volume_variation_cv': float(volume_cv),
        'max_depth_variation_cv': float(depth_cv),
        'volume_consistency_score': float(volume_consistency_score),
        'depth_consistency_score': float(depth_consistency_score),
        'issues': issues
    }


def generate_quality_heatmap_data(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {
            'angles': [],
            'quality_scores': [],
            'distance_deviations': [],
            'depth_deviations': [],
            'overall_quality': 0
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values
    distances = df['distance'].values
    depths = df['depth'].values

    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    depth_mean = np.mean(depths)
    depth_std = np.std(depths)

    distance_deviations = []
    depth_deviations = []
    quality_scores = []

    for i in range(len(angles)):
        if dist_std > 0:
            dist_z = abs((distances[i] - dist_mean) / dist_std)
        else:
            dist_z = 0

        if depth_std > 0:
            depth_z = abs((depths[i] - depth_mean) / depth_std)
        else:
            depth_z = 0

        distance_deviations.append(float(dist_z))
        depth_deviations.append(float(depth_z))

        max_z = max(dist_z, depth_z)
        if max_z < 1:
            quality = 100
        elif max_z < 2:
            quality = 80
        elif max_z < 3:
            quality = 60
        else:
            quality = 40

        quality_scores.append(float(quality))

    overall_quality = float(np.mean(quality_scores))

    return {
        'angles': angles.tolist(),
        'quality_scores': quality_scores,
        'distance_deviations': distance_deviations,
        'depth_deviations': depth_deviations,
        'overall_quality': overall_quality,
        'distance_mean': float(dist_mean),
        'distance_std': float(dist_std),
        'depth_mean': float(depth_mean),
        'depth_std': float(depth_std)
    }


def generate_repair_suggestions(quality_result: Dict) -> List[Dict]:
    suggestions = []

    angle_cov = quality_result.get('angle_coverage', {})
    if angle_cov.get('score', 100) < 80:
        gaps = angle_cov.get('gaps', [])
        for gap in gaps[:3]:
            suggestions.append({
                'type': '补测',
                'severity': '高',
                'category': '角度覆盖',
                'description': f'在 {gap["start_angle"]:.1f}° - {gap["end_angle"]:.1f}° 区间存在 {gap["gap_size"]:.1f}° 的测量空白，建议补充测量',
                'action': '补测角区'
            })

    outliers = quality_result.get('outlier_detection', {})
    if outliers.get('outlier_count', 0) > 0:
        suggestions.append({
            'type': '数据修正',
            'severity': '中',
            'category': '异常点',
            'description': f'检测到 {outliers["outlier_count"]} 个异常跳点，建议复核或剔除这些数据点',
            'action': '剔除异常点'
        })

    missing = quality_result.get('missing_intervals', {})
    if missing.get('missing_count', 0) > 0:
        suggestions.append({
            'type': '补测',
            'severity': '中',
            'category': '缺失区间',
            'description': f'存在 {missing["missing_count"]} 个数据缺失区间，建议加密测量点',
            'action': '补测角区'
        })

    patterns = quality_result.get('repetitive_patterns', {})
    if patterns.get('strong_pattern_count', 0) > 0:
        suggestions.append({
            'type': '设备校准',
            'severity': '中',
            'category': '重复趋势',
            'description': '检测到周期性重复模式，可能存在仪器系统误差，建议校准设备',
            'action': '重新校准批次'
        })

    volatility = quality_result.get('volatility', {})
    if volatility.get('score', 100) < 70:
        suggestions.append({
            'type': '设备检查',
            'severity': '高',
            'category': '波动异常',
            'description': f'数据波动异常（距离CV={volatility.get("distance_cv", 0) * 100:.1f}%，'
                         f'深度CV={volatility.get("depth_cv", 0) * 100:.1f}%），建议检查测量设备',
            'action': '重新校准批次'
        })

    consistency = quality_result.get('batch_consistency', {})
    if consistency.get('score', 100) < 70:
        suggestions.append({
            'type': '批次校准',
            'severity': '高',
            'category': '批次一致性',
            'description': '批次间数据一致性较差，建议统一校准基准或复核测量流程',
            'action': '重新校准批次'
        })

    if not suggestions:
        suggestions.append({
            'type': '正常',
            'severity': '低',
            'category': '数据质量',
            'description': '数据质量良好，暂无修复建议',
            'action': '保持现状'
        })

    return suggestions


def evaluate_batch_quality(batch_id: int, measurements: List[Dict],
                           all_batches_stats: List[Dict] = None) -> Dict:
    if not measurements:
        return {
            'batch_id': batch_id,
            'overall_score': 0,
            'grade': '无数据',
            'angle_coverage': {},
            'outlier_detection': {},
            'missing_intervals': {},
            'repetitive_patterns': {},
            'volatility': {},
            'batch_consistency': {},
            'heatmap_data': {},
            'repair_suggestions': [],
            'all_issues': []
        }

    angle_cov = evaluate_angle_coverage(measurements)
    outliers = detect_outlier_points(measurements)
    missing = detect_missing_intervals_quality(measurements)
    patterns = detect_repetitive_patterns(measurements)
    volatility = evaluate_volatility(measurements)

    if all_batches_stats and len(all_batches_stats) >= 2:
        consistency = evaluate_batch_consistency(all_batches_stats)
    else:
        consistency = {
            'score': 85,
            'consistency_score': 85,
            'issues': ['批次数量不足，无法评估批次间一致性']
        }

    weights = {
        'angle_coverage': 0.2,
        'outlier_detection': 0.2,
        'missing_intervals': 0.15,
        'repetitive_patterns': 0.15,
        'volatility': 0.15,
        'batch_consistency': 0.15
    }

    overall_score = (
        angle_cov['score'] * weights['angle_coverage'] +
        outliers['score'] * weights['outlier_detection'] +
        missing['score'] * weights['missing_intervals'] +
        patterns['score'] * weights['repetitive_patterns'] +
        volatility['score'] * weights['volatility'] +
        consistency['score'] * weights['batch_consistency']
    )

    if overall_score >= 90:
        grade = '优秀'
    elif overall_score >= 75:
        grade = '良好'
    elif overall_score >= 60:
        grade = '合格'
    elif overall_score >= 45:
        grade = '较差'
    else:
        grade = '不合格'

    heatmap_data = generate_quality_heatmap_data(measurements)

    quality_result = {
        'angle_coverage': angle_cov,
        'outlier_detection': outliers,
        'missing_intervals': missing,
        'repetitive_patterns': patterns,
        'volatility': volatility,
        'batch_consistency': consistency
    }

    repair_suggestions = generate_repair_suggestions(quality_result)

    all_issues = []
    for key, value in quality_result.items():
        all_issues.extend(value.get('issues', []))

    return {
        'batch_id': batch_id,
        'overall_score': float(overall_score),
        'grade': grade,
        'angle_coverage': angle_cov,
        'outlier_detection': outliers,
        'missing_intervals': missing,
        'repetitive_patterns': patterns,
        'volatility': volatility,
        'batch_consistency': consistency,
        'heatmap_data': heatmap_data,
        'repair_suggestions': repair_suggestions,
        'all_issues': all_issues,
        'issue_count': len(all_issues)
    }


def generate_quality_report(cave_name: str, batch_name: str, quality_result: Dict) -> str:
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("勘测数据质量评估报告")
    report_lines.append("=" * 60)
    report_lines.append("")

    report_lines.append(f"盐穴名称: {cave_name}")
    report_lines.append(f"勘测批次: {batch_name}")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("一、综合质量评分")
    report_lines.append("-" * 60)
    report_lines.append("")
    report_lines.append(f"综合质量评分: {quality_result['overall_score']:.1f} / 100")
    report_lines.append(f"质量等级: {quality_result['grade']}")
    report_lines.append(f"发现问题数: {quality_result.get('issue_count', 0)} 项")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("二、分项评估")
    report_lines.append("-" * 60)
    report_lines.append("")

    report_lines.append(f"1. 角度覆盖率: {quality_result['angle_coverage']['score']} 分")
    report_lines.append(f"   覆盖比例: {quality_result['angle_coverage'].get('coverage_ratio', 0) * 100:.1f}%")
    report_lines.append(f"   缺失区间数: {quality_result['angle_coverage'].get('gap_count', 0)}")
    report_lines.append("")

    report_lines.append(f"2. 异常跳点检测: {quality_result['outlier_detection']['score']} 分")
    report_lines.append(f"   异常点数量: {quality_result['outlier_detection'].get('outlier_count', 0)}")
    report_lines.append("")

    report_lines.append(f"3. 缺失区间检测: {quality_result['missing_intervals']['score']} 分")
    report_lines.append(f"   缺失区间数: {quality_result['missing_intervals'].get('missing_count', 0)}")
    report_lines.append(f"   最大间隔: {quality_result['missing_intervals'].get('max_gap_size', 0):.1f}°")
    report_lines.append("")

    report_lines.append(f"4. 重复趋势检测: {quality_result['repetitive_patterns']['score']} 分")
    report_lines.append(f"   检测模式数: {quality_result['repetitive_patterns'].get('pattern_count', 0)}")
    report_lines.append("")

    report_lines.append(f"5. 数据波动性: {quality_result['volatility']['score']} 分")
    report_lines.append(f"   距离变异系数: {quality_result['volatility'].get('distance_cv', 0) * 100:.2f}%")
    report_lines.append(f"   深度变异系数: {quality_result['volatility'].get('depth_cv', 0) * 100:.2f}%")
    report_lines.append(f"   距离波动等级: {quality_result['volatility'].get('distance_volatility_level', '-')}")
    report_lines.append(f"   深度波动等级: {quality_result['volatility'].get('depth_volatility_level', '-')}")
    report_lines.append("")

    report_lines.append(f"6. 批次间一致性: {quality_result['batch_consistency']['score']:.1f} 分")
    report_lines.append(f"   容积变异系数: {quality_result['batch_consistency'].get('volume_variation_cv', 0) * 100:.2f}%")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("三、问题明细")
    report_lines.append("-" * 60)
    report_lines.append("")

    all_issues = quality_result.get('all_issues', [])
    if all_issues:
        for i, issue in enumerate(all_issues, 1):
            report_lines.append(f"{i}. {issue}")
    else:
        report_lines.append("未发现明显数据质量问题。")
    report_lines.append("")

    report_lines.append("-" * 60)
    report_lines.append("四、修复建议")
    report_lines.append("-" * 60)
    report_lines.append("")

    suggestions = quality_result.get('repair_suggestions', [])
    for i, suggestion in enumerate(suggestions, 1):
        report_lines.append(f"{i}. [{suggestion['severity']}] {suggestion['category']} - {suggestion['action']}")
        report_lines.append(f"   说明: {suggestion['description']}")
        report_lines.append("")

    report_lines.append("=" * 60)
    report_lines.append("报告生成完毕")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)


def _generate_supplementary_angles(missing_intervals: List[Dict], anomaly_regions: List[Dict],
                                   batch_diff_angles: List[float],
                                   num_points: int,
                                   existing_measurements: List[Dict] = None) -> List[Dict]:
    candidates = []

    for interval in missing_intervals:
        start = interval['start_angle']
        end = interval['end_angle']
        gap = interval['gap_size']
        wraps = interval.get('wraps', False)

        if wraps:
            angular_span = (360 - start) + end
        else:
            angular_span = end - start

        n_in_gap = max(1, int(angular_span / 15.0))
        if wraps:
            angles_in_gap = [(start + (360 - start + end) * k / (n_in_gap + 1)) % 360
                             for k in range(1, n_in_gap + 1)]
        else:
            angles_in_gap = [(start + angular_span * k / (n_in_gap + 1))
                             for k in range(1, n_in_gap + 1)]

        for angle in angles_in_gap:
            candidates.append({
                'angle': angle,
                'priority': 'high',
                'reason': f'缺失区间补测 ({start:.0f}°-{end:.0f}°, 间隔{gap:.1f}°)',
                'source': 'missing_interval',
                'weight': 3.0
            })

    for region in anomaly_regions:
        start = region['start_angle']
        end = region['end_angle']
        mid_angle = (start + end) / 2.0

        candidates.append({
            'angle': mid_angle % 360,
            'priority': 'high',
            'reason': f'异常区域验证 ({region["anomaly_type"]}: {start:.0f}°-{end:.0f}°)',
            'source': 'anomaly',
            'weight': 2.5
        })

        if end - start > 20:
            q1 = (start + mid_angle) / 2.0
            q3 = (mid_angle + end) / 2.0
            candidates.append({
                'angle': q1 % 360,
                'priority': 'medium',
                'reason': f'异常区域加密 ({region["anomaly_type"]}: {start:.0f}°-{mid_angle:.0f}°)',
                'source': 'anomaly',
                'weight': 2.0
            })
            candidates.append({
                'angle': q3 % 360,
                'priority': 'medium',
                'reason': f'异常区域加密 ({region["anomaly_type"]}: {mid_angle:.0f}°-{end:.0f}°)',
                'source': 'anomaly',
                'weight': 2.0
            })

    for angle in batch_diff_angles:
        candidates.append({
            'angle': angle % 360,
            'priority': 'medium',
            'reason': f'批次差异验证 ({angle:.0f}°)',
            'source': 'batch_diff',
            'weight': 1.5
        })

    if existing_measurements and len(existing_measurements) > 0:
        df = pd.DataFrame(existing_measurements).sort_values('angle')
        angles = df['angle'].values
        distances = df['distance'].values
        depths = df['depth'].values

        for i in range(len(angles)):
            j = (i + 1) % len(angles)
            if j == 0:
                gap = (360 - angles[i]) + angles[j]
                mid_angle = (angles[i] + angles[j] + 360) / 2 % 360
            else:
                gap = angles[j] - angles[i]
                mid_angle = (angles[i] + angles[j]) / 2

            if gap > 10:
                dist_var = abs(distances[j] - distances[i]) / max(distances[i], distances[j], 1)
                depth_var = abs(depths[j] - depths[i]) / max(depths[i], depths[j], 1)
                variability = dist_var + depth_var

                priority = 'low'
                weight = 1.0

                if gap > 30:
                    priority = 'medium'
                    weight = 1.8
                elif gap > 20:
                    priority = 'low'
                    weight = 1.3

                if variability > 0.3:
                    weight += 0.5
                    if priority == 'low':
                        priority = 'medium'

                candidates.append({
                    'angle': float(mid_angle),
                    'priority': priority,
                    'reason': f'大间隔加密 ({angles[i]:.0f}°-{angles[j]:.0f}°, 间隔{gap:.1f}°)',
                    'source': 'gap_filling',
                    'weight': weight
                })

    candidates.sort(key=lambda x: x['weight'], reverse=True)

    seen = set()
    unique = []
    for c in candidates:
        rounded = round(c['angle'], 1)
        if rounded not in seen:
            is_duplicate_of_existing = False
            if existing_measurements:
                for m in existing_measurements:
                    if abs(c['angle'] - m['angle']) < 2.0 or \
                       abs(c['angle'] - m['angle'] + 360) < 2.0 or \
                       abs(c['angle'] - m['angle'] - 360) < 2.0:
                        is_duplicate_of_existing = True
                        break
            if not is_duplicate_of_existing:
                seen.add(rounded)
                unique.append(c)

    if len(unique) < num_points:
        if existing_measurements and len(existing_measurements) > 0:
            df = pd.DataFrame(existing_measurements).sort_values('angle')
            existing_angles = set(round(a, 1) for a in df['angle'].values)
            
            for step in [5, 10, 15, 20, 30, 45, 60]:
                if len(unique) >= num_points:
                    break
                for angle in range(0, 360, step):
                    if len(unique) >= num_points:
                        break
                    angle_float = float(angle)
                    rounded = round(angle_float, 1)
                    if rounded not in seen and rounded not in existing_angles:
                        unique.append({
                            'angle': angle_float,
                            'priority': 'low',
                            'reason': f'均匀加密补测 ({angle}°)',
                            'source': 'uniform_density',
                            'weight': 0.8
                        })
                        seen.add(rounded)

    high_count = sum(1 for c in unique if c['priority'] == 'high')
    med_count = sum(1 for c in unique if c['priority'] == 'medium')
    low_count = sum(1 for c in unique if c['priority'] == 'low')

    return unique[:num_points]


def _simulate_volume_with_supplements(measurements: List[Dict],
                                      supplement_angles: List[Dict]) -> Dict:
    if not measurements:
        return {'volume': 0, 'quality_score': 0}

    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values
    distances = df['distance'].values
    depths = df['depth'].values

    angles_ext = np.concatenate([angles - 360, angles, angles + 360])
    distances_ext = np.concatenate([distances, distances, distances])
    depths_ext = np.concatenate([depths, depths, depths])

    from scipy.interpolate import interp1d

    valid = np.diff(angles_ext) != 0
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return calculate_volume_conical(measurements)

    angles_ext_clean = angles_ext[valid_idx]
    distances_ext_clean = distances_ext[valid_idx]
    depths_ext_clean = depths_ext[valid_idx]

    f_dist = interp1d(angles_ext_clean, distances_ext_clean, kind='linear', fill_value='extrapolate')
    f_depth = interp1d(angles_ext_clean, depths_ext_clean, kind='linear', fill_value='extrapolate')

    all_angles = list(angles)
    all_distances = list(distances)
    all_depths = list(depths)

    for sup in supplement_angles:
        sa = sup['angle']
        if sa < 0:
            sa += 360
        if sa >= 360:
            sa -= 360

        is_close = any(abs(sa - a) < 1.0 or abs(sa - a + 360) < 1.0 or abs(sa - a - 360) < 1.0
                       for a in all_angles)
        if is_close:
            continue

        try:
            sd = float(f_dist(sa))
            sh = float(f_depth(sa))
            sd = max(0, sd)
            sh = max(0, sh)
            all_angles.append(sa)
            all_distances.append(sd)
            all_depths.append(sh)
        except Exception:
            continue

    enhanced_measurements = [
        {'angle': a, 'distance': d, 'depth': h}
        for a, d, h in zip(all_angles, all_distances, all_depths)
    ]

    volume_result = calculate_volume_conical(enhanced_measurements)

    new_count = len(all_angles) - len(measurements)
    max_gap_before = _compute_max_gap(measurements)
    max_gap_after = _compute_max_gap(enhanced_measurements)

    coverage_before = len(measurements) / 24.0
    coverage_after = len(all_angles) / 24.0

    return {
        'volume': volume_result['volume'],
        'max_depth': volume_result['max_depth'],
        'max_distance': volume_result['max_distance'],
        'point_count': len(all_angles),
        'added_points': new_count,
        'max_gap_before': max_gap_before,
        'max_gap_after': max_gap_after,
        'coverage_ratio_before': min(coverage_before, 1.0),
        'coverage_ratio_after': min(coverage_after, 1.0),
        'enhanced_measurements': enhanced_measurements
    }


def _compute_max_gap(measurements: List[Dict]) -> float:
    if len(measurements) < 2:
        return 360.0
    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values
    max_gap = 0
    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        if j == 0:
            gap = (360 - angles[i]) + angles[j]
        else:
            gap = angles[j] - angles[i]
        max_gap = max(max_gap, gap)
    return max_gap


def estimate_volume_accuracy(measurements: List[Dict]) -> Dict:
    if not measurements or len(measurements) < 3:
        return {
            'accuracy_score': 0,
            'estimated_error_pct': 100.0,
            'error_level': '无法评估',
            'confidence': '低'
        }

    df = pd.DataFrame(measurements).sort_values('angle')
    angles = df['angle'].values
    distances = df['distance'].values
    depths = df['depth'].values

    n_points = len(angles)
    avg_gap = 360.0 / n_points
    max_gap = _compute_max_gap(measurements)

    dist_std = np.std(distances)
    dist_mean = np.mean(distances)
    dist_cv = dist_std / dist_mean if dist_mean > 0 else 0

    depth_std = np.std(depths)
    depth_mean = np.mean(depths)
    depth_cv = depth_std / depth_mean if depth_mean > 0 else 0

    gap_factor = max(0, 1 - max_gap / 60.0)
    density_factor = min(1.0, n_points / 36.0)
    smoothness_factor = max(0, 1 - (dist_cv + depth_cv))

    accuracy_score = (gap_factor * 0.4 + density_factor * 0.3 + smoothness_factor * 0.3) * 100

    if accuracy_score >= 90:
        estimated_error_pct = 1.0
        error_level = '极高精度'
        confidence = '高'
    elif accuracy_score >= 75:
        estimated_error_pct = 3.0
        error_level = '高精度'
        confidence = '较高'
    elif accuracy_score >= 60:
        estimated_error_pct = 6.0
        error_level = '中等精度'
        confidence = '中等'
    elif accuracy_score >= 45:
        estimated_error_pct = 10.0
        error_level = '低精度'
        confidence = '较低'
    else:
        estimated_error_pct = 20.0
        error_level = '极低精度'
        confidence = '低'

    return {
        'accuracy_score': float(accuracy_score),
        'estimated_error_pct': float(estimated_error_pct),
        'error_level': error_level,
        'confidence': confidence,
        'max_gap': float(max_gap),
        'avg_gap': float(avg_gap),
        'point_count': n_points,
        'distance_cv': float(dist_cv),
        'depth_cv': float(depth_cv),
        'gap_factor': float(gap_factor),
        'density_factor': float(density_factor),
        'smoothness_factor': float(smoothness_factor)
    }


def simulate_resurvey_plans(measurements: List[Dict],
                            all_batches_measurements: List[List[Dict]] = None,
                            plan_sizes: List[int] = None) -> Dict:
    if not measurements:
        return {
            'current_status': {},
            'plans': [],
            'recommended_plan': None,
            'comparison_chart_data': {},
            'path_data': {}
        }

    if plan_sizes is None:
        plan_sizes = [3, 6, 9, 12, 18, 24]

    current_quality = evaluate_batch_quality(0, measurements)
    current_volume = calculate_volume_conical(measurements)
    current_missing = detect_missing_intervals_quality(measurements)
    current_anomalies = detect_anomalies(measurements)

    batch_diff_angles = []
    if all_batches_measurements and len(all_batches_measurements) >= 2:
        ref = all_batches_measurements[0]
        for other in all_batches_measurements[1:]:
            interp = interpolate_to_common_angles(ref, other, num_points=360)
            if interp['angles']:
                dist_diff = np.abs(np.array(interp['distances1']) - np.array(interp['distances2']))
                depth_diff = np.abs(np.array(interp['depths1']) - np.array(interp['depths2']))
                combined = dist_diff / (np.mean(np.array(interp['distances1'])) + 1e-6) + \
                           depth_diff / (np.mean(np.array(interp['depths1'])) + 1e-6)
                threshold = np.mean(combined) + np.std(combined)
                significant = np.where(combined > threshold)[0]
                for idx in significant[::10]:
                    batch_diff_angles.append(float(interp['angles'][idx]))

    anomaly_regions = get_anomaly_regions_from_measurements(measurements)

    actual_max_gap = _compute_max_gap(measurements)
    current_accuracy = estimate_volume_accuracy(measurements)

    current_status = {
        'point_count': len(measurements),
        'volume': current_volume['volume'],
        'quality_score': current_quality['overall_score'],
        'grade': current_quality['grade'],
        'missing_count': current_missing.get('missing_count', 0),
        'max_gap': actual_max_gap,
        'anomaly_count': len(current_anomalies),
        'issue_count': current_quality.get('issue_count', 0),
        'avg_gap': 360.0 / len(measurements) if measurements else 0,
        'coverage_ratio': current_quality.get('angle_coverage', {}).get('coverage_ratio', 0),
        'accuracy_score': current_accuracy['accuracy_score'],
        'estimated_error_pct': current_accuracy['estimated_error_pct'],
        'error_level': current_accuracy['error_level'],
        'confidence': current_accuracy['confidence']
    }

    plans = []
    for size in plan_sizes:
        sup_angles = _generate_supplementary_angles(
            current_missing.get('missing_intervals', []),
            anomaly_regions,
            batch_diff_angles,
            size,
            measurements
        )

        sim_result = _simulate_volume_with_supplements(measurements, sup_angles)

        enhanced_quality = evaluate_batch_quality(0, sim_result['enhanced_measurements'])
        enhanced_accuracy = estimate_volume_accuracy(sim_result['enhanced_measurements'])

        volume_change = sim_result['volume'] - current_volume['volume']
        volume_change_pct = (volume_change / current_volume['volume'] * 100) if current_volume['volume'] > 0 else 0

        quality_improvement = enhanced_quality['overall_score'] - current_quality['overall_score']
        accuracy_improvement = enhanced_accuracy['accuracy_score'] - current_accuracy['accuracy_score']
        error_reduction_pct = ((current_accuracy['estimated_error_pct'] - enhanced_accuracy['estimated_error_pct']) 
                               / current_accuracy['estimated_error_pct'] * 100) if current_accuracy['estimated_error_pct'] > 0 else 0

        gap_reduction = sim_result['max_gap_before'] - sim_result['max_gap_after']
        gap_reduction_pct = (gap_reduction / sim_result['max_gap_before'] * 100) if sim_result['max_gap_before'] > 0 else 0

        high_priority = sum(1 for s in sup_angles if s['priority'] == 'high')
        medium_priority = sum(1 for s in sup_angles if s['priority'] == 'medium')
        low_priority = sum(1 for s in sup_angles if s['priority'] == 'low')

        cost_efficiency = quality_improvement / size if size > 0 else 0

        plan = {
            'plan_name': f'方案{len(plans) + 1}：补测{size}个角度',
            'num_supplementary': size,
            'supplementary_details': sup_angles,
            'high_priority_count': high_priority,
            'medium_priority_count': medium_priority,
            'low_priority_count': low_priority,
            'projected_volume': sim_result['volume'],
            'volume_change': volume_change,
            'volume_change_pct': volume_change_pct,
            'projected_quality_score': enhanced_quality['overall_score'],
            'projected_grade': enhanced_quality['grade'],
            'quality_improvement': quality_improvement,
            'max_gap_before': sim_result['max_gap_before'],
            'max_gap_after': sim_result['max_gap_after'],
            'gap_reduction': gap_reduction,
            'gap_reduction_pct': gap_reduction_pct,
            'coverage_improvement': sim_result['coverage_ratio_after'] - sim_result['coverage_ratio_before'],
            'cost_efficiency': cost_efficiency,
            'total_points_after': sim_result['point_count'],
            'enhanced_missing_count': enhanced_quality.get('missing_intervals', {}).get('missing_count', 0),
            'enhanced_anomaly_count': enhanced_quality.get('outlier_detection', {}).get('outlier_count', 0),
            'enhanced_issue_count': enhanced_quality.get('issue_count', 0),
            'accuracy_score': enhanced_accuracy['accuracy_score'],
            'accuracy_improvement': accuracy_improvement,
            'estimated_error_pct': enhanced_accuracy['estimated_error_pct'],
            'error_reduction_pct': error_reduction_pct,
            'error_level': enhanced_accuracy['error_level'],
            'confidence': enhanced_accuracy['confidence']
        }
        plans.append(plan)

    if plans:
        valid_plans = [p for p in plans if p['quality_improvement'] > 0]
        if not valid_plans:
            valid_plans = plans

        best_quality_plan = max(valid_plans, key=lambda p: p['projected_quality_score'])
        best_efficiency_plan = max(valid_plans, key=lambda p: p['cost_efficiency'])

        quality_threshold = best_quality_plan['projected_quality_score'] * 0.9
        candidate_plans = [p for p in valid_plans if p['projected_quality_score'] >= quality_threshold]
        if candidate_plans:
            recommended = min(candidate_plans, key=lambda p: p['num_supplementary'])
        else:
            recommended = best_efficiency_plan
    else:
        best_quality_plan = None
        best_efficiency_plan = None
        recommended = None

    comparison_chart_data = {
        'plan_names': [p['plan_name'] for p in plans],
        'quality_scores': [p['projected_quality_score'] for p in plans],
        'volume_changes_pct': [p['volume_change_pct'] for p in plans],
        'gap_reductions_pct': [p['gap_reduction_pct'] for p in plans],
        'cost_efficiencies': [p['cost_efficiency'] for p in plans],
        'num_supplementary': [p['num_supplementary'] for p in plans],
        'current_quality': current_quality['overall_score'],
        'accuracy_scores': [p['accuracy_score'] for p in plans],
        'current_accuracy': current_accuracy['accuracy_score'],
        'error_reductions_pct': [p['error_reduction_pct'] for p in plans],
        'estimated_errors_pct': [p['estimated_error_pct'] for p in plans],
        'current_error_pct': current_accuracy['estimated_error_pct']
    }

    if recommended:
        path_angles = sorted(recommended['supplementary_details'], key=lambda x: x['angle'])
        path_data = {
            'angles': [p['angle'] for p in path_angles],
            'priorities': [p['priority'] for p in path_angles],
            'reasons': [p['reason'] for p in path_angles],
            'sources': [p['source'] for p in path_angles],
            'weights': [p['weight'] for p in path_angles]
        }
    else:
        path_data = {}

    return {
        'current_status': current_status,
        'plans': plans,
        'recommended_plan': recommended,
        'best_quality_plan': best_quality_plan,
        'best_efficiency_plan': best_efficiency_plan,
        'comparison_chart_data': comparison_chart_data,
        'path_data': path_data
    }


def get_anomaly_regions_from_measurements(measurements: List[Dict]) -> List[Dict]:
    if len(measurements) < 3:
        return []

    df = pd.DataFrame(measurements).sort_values('angle')
    distances = df['distance'].values
    depths = df['depth'].values
    angles = df['angle'].values

    regions = []

    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    if dist_std > 0:
        dist_zscores = np.abs((distances - dist_mean) / dist_std)
        in_anomaly = False
        start_idx = 0
        for i in range(len(dist_zscores)):
            if dist_zscores[i] > 2.0 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif dist_zscores[i] <= 2.0 and in_anomaly:
                in_anomaly = False
                regions.append({
                    'start_angle': float(angles[start_idx]),
                    'end_angle': float(angles[i - 1]),
                    'anomaly_type': '距离异常',
                    'description': f'距离偏差超过2σ的异常区域'
                })
        if in_anomaly:
            regions.append({
                'start_angle': float(angles[start_idx]),
                'end_angle': float(angles[-1]),
                'anomaly_type': '距离异常',
                'description': f'距离偏差超过2σ的异常区域'
            })

    depth_mean = np.mean(depths)
    depth_std = np.std(depths)
    if depth_std > 0:
        depth_zscores = np.abs((depths - depth_mean) / depth_std)
        in_anomaly = False
        start_idx = 0
        for i in range(len(depth_zscores)):
            if depth_zscores[i] > 2.0 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif depth_zscores[i] <= 2.0 and in_anomaly:
                in_anomaly = False
                regions.append({
                    'start_angle': float(angles[start_idx]),
                    'end_angle': float(angles[i - 1]),
                    'anomaly_type': '深度异常',
                    'description': f'深度偏差超过2σ的异常区域'
                })
        if in_anomaly:
            regions.append({
                'start_angle': float(angles[start_idx]),
                'end_angle': float(angles[-1]),
                'anomaly_type': '深度异常',
                'description': f'深度偏差超过2σ的异常区域'
            })

    return regions


def generate_resurvey_report(cave_name: str, batch_name: str, simulation_result: Dict) -> str:
    lines = []

    lines.append("=" * 60)
    lines.append("勘测方案模拟与补测优化报告")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"盐穴名称: {cave_name}")
    lines.append(f"勘测批次: {batch_name}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("一、当前数据状况")
    lines.append("-" * 60)
    lines.append("")

    cs = simulation_result.get('current_status', {})
    lines.append(f"当前测量点数: {cs.get('point_count', 0)}")
    lines.append(f"当前容积估算: {cs.get('volume', 0):.2f} m³")
    lines.append(f"当前质量评分: {cs.get('quality_score', 0):.1f} 分")
    lines.append(f"当前质量等级: {cs.get('grade', '-')}")
    lines.append(f"容积估算精度: {cs.get('accuracy_score', 0):.1f} 分 ({cs.get('error_level', '-')})")
    lines.append(f"预计容积误差: ±{cs.get('estimated_error_pct', 0):.1f}%")
    lines.append(f"置信度: {cs.get('confidence', '-')}")
    lines.append(f"缺失区间数: {cs.get('missing_count', 0)}")
    lines.append(f"最大间隔: {cs.get('max_gap', 0):.1f}°")
    lines.append(f"异常区域数: {cs.get('anomaly_count', 0)}")
    lines.append(f"问题总数: {cs.get('issue_count', 0)} 项")
    lines.append("")

    lines.append("-" * 60)
    lines.append("二、补测方案对比")
    lines.append("-" * 60)
    lines.append("")

    for i, plan in enumerate(simulation_result.get('plans', []), 1):
        lines.append(f"方案{i}：补测 {plan['num_supplementary']} 个角度")
        lines.append(f"  预计容积: {plan['projected_volume']:.2f} m³ (变化 {plan['volume_change_pct']:+.2f}%)")
        lines.append(f"  预计质量评分: {plan['projected_quality_score']:.1f} 分 (提升 {plan['quality_improvement']:+.1f})")
        lines.append(f"  预计质量等级: {plan['projected_grade']}")
        lines.append(f"  预计容积精度: {plan['accuracy_score']:.1f} 分 (提升 {plan['accuracy_improvement']:+.1f})")
        lines.append(f"  预计误差率: ±{plan['estimated_error_pct']:.1f}% (缩减 {plan['error_reduction_pct']:.1f}%)")
        lines.append(f"  最大间隔缩减: {plan['max_gap_before']:.1f}° → {plan['max_gap_after']:.1f}° (缩减 {plan['gap_reduction_pct']:.1f}%)")
        lines.append(f"  高/中/低优先级点数: {plan['high_priority_count']}/{plan['medium_priority_count']}/{plan['low_priority_count']}")
        lines.append(f"  成本效率: {plan['cost_efficiency']:.2f} 分/点")
        lines.append("")

    lines.append("-" * 60)
    lines.append("三、推荐方案")
    lines.append("-" * 60)
    lines.append("")

    recommended = simulation_result.get('recommended_plan')
    if recommended:
        lines.append(f"推荐方案: {recommended['plan_name']}")
        lines.append(f"  补测角度数: {recommended['num_supplementary']}")
        lines.append(f"  预计质量提升: {recommended['quality_improvement']:+.1f} 分")
        lines.append(f"  预计容积变化: {recommended['volume_change_pct']:+.2f}%")
        lines.append(f"  预计精度提升: {recommended['accuracy_improvement']:+.1f} 分")
        lines.append(f"  预计误差缩减: {recommended['error_reduction_pct']:.1f}%")
        lines.append(f"  成本效率: {recommended['cost_efficiency']:.2f} 分/点")
        lines.append("")
        lines.append("  推荐补测路径（按优先级排序）:")
        sorted_details = sorted(recommended['supplementary_details'], 
                               key=lambda x: (0 if x['priority'] == 'high' else (1 if x['priority'] == 'medium' else 2), 
                                              -x['weight']))
        for j, detail in enumerate(sorted_details, 1):
            priority_label = {'high': '高', 'medium': '中', 'low': '低'}.get(detail['priority'], detail['priority'])
            source_label = {
                'missing_interval': '缺失区间',
                'anomaly': '异常区域',
                'batch_diff': '批次差异',
                'gap_filling': '大间隔加密',
                'uniform_density': '均匀加密',
                'regular': '常规加密'
            }.get(detail['source'], detail['source'])
            lines.append(f"    {j}. 角度 {detail['angle']:.1f}° [优先级: {priority_label}] - {detail['reason']} (来源: {source_label})")
        lines.append("")
    else:
        lines.append("暂无推荐方案。")
        lines.append("")

    lines.append("-" * 60)
    lines.append("四、补测建议总结")
    lines.append("-" * 60)
    lines.append("")

    if recommended:
        quality_improvement = recommended['accuracy_improvement']
        if quality_improvement > 15:
            lines.append("当前容积估算精度存在较大提升空间，强烈建议执行补测方案。")
        elif quality_improvement > 5:
            lines.append("当前容积估算精度有一定提升空间，建议执行补测方案。")
        else:
            lines.append("当前数据质量较好，补测收益有限，可根据实际需求决定是否补测。")
        lines.append("")
        lines.append(f"最优性价比方案: 补测 {recommended['num_supplementary']} 个角度")
        lines.append(f"预期质量评分从 {cs.get('quality_score', 0):.1f} 提升至 {recommended['projected_quality_score']:.1f}")
        lines.append(f"预期精度评分从 {cs.get('accuracy_score', 0):.1f} 提升至 {recommended['accuracy_score']:.1f}")
        lines.append(f"预期容积误差从 ±{cs.get('estimated_error_pct', 0):.1f}% 降至 ±{recommended['estimated_error_pct']:.1f}%")
        lines.append(f"预期最大间隔从 {cs.get('max_gap', 0):.1f}° 缩减至 {recommended['max_gap_after']:.1f}°")
    else:
        lines.append("无需补测，数据质量良好。")

    lines.append("")
    lines.append("=" * 60)
    lines.append("报告生成完毕")
    lines.append("=" * 60)

    return "\n".join(lines)
