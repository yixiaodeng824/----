"""
饮食推荐服务 — 基于用户目标（增重/减脂/维持）与当日营养缺口生成推荐建议
"""


def get_professional_recommendation(goal, cal_gap, protein_gap, intake_carbs, intake_fat):
    gap_text = f"营养缺口分析：\n"
    gap_text += f"- 热量缺口：{cal_gap:.0f}大卡\n" if cal_gap > 0 else "- 热量已达标或超标\n"
    gap_text += f"- 蛋白质缺口：{protein_gap:.1f}g\n" if protein_gap > 0 else "- 蛋白质已达标\n"

    if goal == 'gain':
        supplement = (
            "补充建议：\n"
            "- 早餐：鸡蛋3个+燕麦粥+牛奶\n"
            "- 午餐：鸡胸肉150g+糙米饭+蔬菜\n"
            "- 加餐：蛋白棒或坚果\n"
            "- 晚餐：鱼肉200g+红薯+绿叶菜\n"
            "建议多摄入高蛋白食物，如鸡胸肉、牛肉、鱼、蛋、奶制品。"
        )
        notice = "注意事项：保证蛋白质摄入，适量增加碳水，避免高糖高油食物。"
    elif goal == 'lose':
        supplement = (
            "补充建议：\n"
            "- 早餐：2个鸡蛋+蔬菜沙拉\n"
            "- 午餐：鸡胸肉120g+藜麦+大量蔬菜\n"
            "- 晚餐：清蒸鱼150g+西兰花+豆腐\n"
            "建议多吃蔬菜、瘦肉，控制主食和油脂摄入。"
        )
        notice = "注意事项：控制总热量，优先补充蛋白质，减少油脂和精制碳水摄入。"
    else:
        supplement = (
            "补充建议：\n"
            "- 早餐：全麦面包+鸡蛋+水果\n"
            "- 午餐：鱼肉/鸡肉+杂粮饭+多种蔬菜\n"
            "- 晚餐：豆腐+蔬菜+少量主食\n"
            "保持饮食多样化，均衡营养。"
        )
        notice = "注意事项：保证食物多样性，适量运动，控制油盐摄入。"

    replace_text = "食材替换推荐：如不喜欢鸡胸肉，可用牛肉、鱼肉、豆腐等替代；主食可用糙米、红薯、玉米等替换。"

    return f"{gap_text}\n{supplement}\n\n{replace_text}\n\n{notice}"
