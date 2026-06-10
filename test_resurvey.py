from analysis import simulate_resurvey_plans, generate_resurvey_report
from database import get_all_caves, get_batches_by_cave, get_measurements_by_batch

caves = get_all_caves()
print(f'盐穴数量: {len(caves)}')

if caves:
    cave = caves[0]
    print(f'盐穴名称: {cave["name"]}')
    batches = get_batches_by_cave(cave['id'])
    print(f'批次数量: {len(batches)}')
    
    if batches:
        batch = batches[0]
        print(f'批次名称: {batch["batch_name"]}')
        measurements = get_measurements_by_batch(batch['id'])
        print(f'测量点数: {len(measurements)}')
        
        if measurements:
            all_batches_data = []
            for b in batches:
                m = get_measurements_by_batch(b['id'])
                if m:
                    all_batches_data.append(m)
            
            print(f'用于对比的批次数量: {len(all_batches_data)}')
            
            result = simulate_resurvey_plans(measurements, all_batches_data)
            print(f'\n方案数量: {len(result.get("plans", []))}')
            print(f'推荐方案: {result.get("recommended_plan", {}).get("plan_name", "无")}')
            
            cs = result.get('current_status', {})
            print(f'当前质量评分: {cs.get("quality_score", 0):.1f}')
            print(f'当前容积: {cs.get("volume", 0):.2f} m³')
            print(f'缺失区间: {cs.get("missing_count", 0)} 个')
            print(f'最大间隔: {cs.get("max_gap", 0):.1f}°')
            print(f'异常区域: {cs.get("anomaly_count", 0)} 个')
            
            if result.get('plans'):
                print('\n各方案对比:')
                for p in result['plans']:
                    print(f'  {p["plan_name"]}:')
                    print(f'    预计质量: {p["projected_quality_score"]:.1f} (提升{p["quality_improvement"]:+.1f})')
                    print(f'    容积变化: {p["volume_change_pct"]:+.2f}%')
                    print(f'    间隔缩减: {p["gap_reduction_pct"]:.1f}%')
                    print(f'    成本效率: {p["cost_efficiency"]:.2f} 分/点')
                    print(f'    高/中/低优先级: {p["high_priority_count"]}/{p["medium_priority_count"]}/{p["low_priority_count"]}')
            
            path = result.get('path_data', {})
            if path.get('angles'):
                print(f'\n推荐路径点数: {len(path["angles"])}')
                for i, (angle, pri, reason) in enumerate(zip(
                    path['angles'][:5], path['priorities'][:5], path['reasons'][:5]
                )):
                    print(f'  {i+1}. {angle:.1f}° [{pri}] - {reason}')
            
            report = generate_resurvey_report(cave['name'], batch['batch_name'], result)
            print('\n=== 报告预览（前30行）===')
            lines = report.split('\n')
            for line in lines[:30]:
                print(line)
            
            print(f'\n报告总长度: {len(lines)} 行')
    else:
        print('没有批次数据')
else:
    print('没有盐穴数据')
