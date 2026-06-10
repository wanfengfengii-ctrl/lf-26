import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (
    get_cave_by_name, create_cave, create_batch, add_measurement,
    save_volume_estimate, save_anomaly_regions, get_batches_by_cave
)
from analysis import calculate_volume_conical, detect_anomalies


def generate_sample_data():
    angles = np.arange(0, 360, 10)
    base_distance = 50.0
    base_depth = 30.0

    caves_data = [
        {
            'cave_name': '测试盐穴A',
            'batches': [
                {
                    'batch_name': '2024-01-勘测',
                    'survey_date': '2024-01-15',
                    'dist_factor': 0.05,
                    'depth_factor': 0.03,
                    'has_gap': False,
                    'has_anomaly': False
                },
                {
                    'batch_name': '2024-06-勘测',
                    'survey_date': '2024-06-20',
                    'dist_factor': 0.08,
                    'depth_factor': 0.06,
                    'has_gap': True,
                    'has_anomaly': True
                },
                {
                    'batch_name': '2025-01-勘测',
                    'survey_date': '2025-01-10',
                    'dist_factor': 0.12,
                    'depth_factor': 0.10,
                    'has_gap': False,
                    'has_anomaly': True
                }
            ]
        },
        {
            'cave_name': '测试盐穴B',
            'batches': [
                {
                    'batch_name': '2024-03-勘测',
                    'survey_date': '2024-03-10',
                    'dist_factor': 0.03,
                    'depth_factor': 0.02,
                    'has_gap': False,
                    'has_anomaly': False
                },
                {
                    'batch_name': '2024-09-勘测',
                    'survey_date': '2024-09-15',
                    'dist_factor': 0.06,
                    'depth_factor': 0.05,
                    'has_gap': True,
                    'has_anomaly': False
                }
            ]
        }
    ]

    for cave_data in caves_data:
        cave_name = cave_data['cave_name']

        cave = get_cave_by_name(cave_name)
        if not cave:
            cave_id = create_cave(cave_name, f'示例盐穴 - {cave_name}')
        else:
            cave_id = cave['id']

        print(f'盐穴: {cave_name} (ID: {cave_id})')

        for batch_data in cave_data['batches']:
            batch_name = batch_data['batch_name']

            existing_batches = get_batches_by_cave(cave_id)
            if any(b['batch_name'] == batch_name for b in existing_batches):
                print(f'  批次 {batch_name} 已存在，跳过')
                continue

            batch_id = create_batch(
                cave_id,
                batch_name,
                batch_data['survey_date'],
                '示例勘测数据'
            )

            measurements = []
            for angle in angles:
                angle_rad = np.deg2rad(angle)

                if batch_data['has_gap'] and 90 < angle < 130:
                    continue

                distance = base_distance * (
                    1 + batch_data['dist_factor'] * np.sin(3 * angle_rad)
                )
                depth = base_depth * (
                    1 + batch_data['depth_factor'] * np.cos(2 * angle_rad)
                )

                if batch_data['has_anomaly'] and 200 < angle < 220:
                    depth *= 1.3

                add_measurement(batch_id, float(angle), float(distance), float(depth))
                measurements.append({
                    'angle': float(angle),
                    'distance': float(distance),
                    'depth': float(depth)
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

            print(f'  批次: {batch_name} (ID: {batch_id}) - {len(measurements)} 个测量点, '
                  f'容积: {vol_result["volume"]:.2f} m³')

    print('\n示例数据生成完成!')


if __name__ == '__main__':
    generate_sample_data()
