from collections import defaultdict
import config.settings as cfg

class Reporter:
    """
    [工具] 调试报告生成器
    功能：负责格式化并打印车辆离场时的详细数据报告。
    """
    @staticmethod
    def print_exit_report(tid, record, kinematics_estimator):
        """
        打印离场车辆的详细分析报告
        :param tid: 车辆追踪 ID
        :param record: 车辆档案字典 (来自 VehicleRegistry)
        :param kinematics_estimator: 用于获取运动历史 (可能为 None)
        """
        # 1. 检查配置开关
        if not cfg.DEBUG_MODE:
            return

        # 2. 噪点过滤 (存活时间过短)
        life_span = record.get('last_seen_frame', 0) - record.get('first_frame', 0)
        if life_span < cfg.MIN_SURVIVAL_FRAMES:
            return 

        yolo_name = record.get('class_name', 'Unknown')
        history = record.get('plate_history', [])
        final_plate = "Unknown"
        vote_info = "无识别记录"
        
        # 3. 面积加权投票逻辑 (Area-Weighted Voting)
        if history:
            scores = defaultdict(float)
            total_weight = 0.0
            
            for entry in history:
                w = entry['area'] # 权重 = 像素面积
                scores[entry['color']] += w
                total_weight += w
                
            if scores:
                winner = max(scores, key=scores.get)
                # 计算胜出者的权重占比
                confidence = scores[winner] / total_weight if total_weight > 0 else 0
                final_plate = winner
                vote_info = f"得分 {int(scores[winner]/1000)}k (占比 {confidence:.1%})"

        # 4. 最终归类逻辑 (Type Classification)
        final_type = "Calculating..."
        
        # 情况 A: 成功识别到车牌 (精确)
        if final_plate != "Unknown":
            if final_plate == "Green": final_type = "LDV-electric"
            elif final_plate == "Yellow": final_type = "HDV-diesel"
            elif final_plate == "Blue": final_type = "LDV-gasoline"
            else: final_type = f"{final_plate}?"
            
            # 特殊逻辑: 大车 + 绿牌 = 重型电动车
            if record.get('class_id') in [5, 7] and final_plate == "Green":
                final_type = "HDV-electric (Large EV)"
                
        # 情况 B: 无车牌信息 (兜底策略)
        else:
            # 区分是 OCR 没开还是没识别出来，仅用于显示后缀
            suffix = "(Default)" if not cfg.ENABLE_OCR else "(Fallback)"
            class_id = record.get('class_id')
            
            if class_id == 2: # Car
                final_type = f"LDV-gasoline {suffix}"
            
            elif class_id == 5: # Bus -> 核心修改：默认电动
                final_type = f"HDV-electric {suffix}"
                
            elif class_id == 7: # Truck -> 核心修改：默认柴油
                final_type = f"HDV-diesel {suffix}"
                
            else:
                final_type = f"Unknown {suffix}"

        # 5. 运动统计
        speed_info = "N/A (Motion Off)"
        if kinematics_estimator and tid in kinematics_estimator.trackers:
            hist = kinematics_estimator.trackers[tid]['speed_history']
            if hist: 
                speed_info = f"Max: {max(hist):.1f} m/s"

        # 6. 排放与工况统计
        op_stats = record.get('op_mode_stats', {})
        total_emission = record.get('total_emission_mg', 0.0)
        
        # 格式化 OpMode 时间 (帧数 -> 秒)
        op_summary = []
        for mode in sorted(op_stats.keys()):
            seconds = op_stats[mode] / cfg.FPS
            mode_name = {
                0: "Braking", 1: "Idling", 11: "Coast", 
                21: "Cruise(L)", 33: "Cruise(H)"
            }.get(mode, str(mode))
            op_summary.append(f"{mode_name}: {seconds:.1f}s")
        
        op_str = " | ".join(op_summary) if op_summary else "无有效工况数据"

        # 7. 执行打印
        print("-" * 70)
        print(f"[离场] ID: {tid} | 存活: {life_span/cfg.FPS:.1f}s | 类型: {final_type}")
        print(f"       车牌: {final_plate} [{vote_info}]")
        print(f"       速度: {speed_info}")
        print(f"       统计: {op_str}")
        print(f"       排放: 总计 {total_emission:.2f} mg (PM2.5 Brake Wear)")
        print("-" * 70 + "\n")
