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
