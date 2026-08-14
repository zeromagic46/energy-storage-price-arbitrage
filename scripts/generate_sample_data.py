# -*- coding: utf-8 -*-
"""
生成示例节点电价 Excel (宽表, 96时点/15分钟颗粒度), 写到 ../data/sample_prices.xlsx
用途: 让工具开箱即跑通 (双击 4-Run-Backtest.bat 即可看到结果), 也方便你改参数后重新造数据。

宽表格式: 每行一天, 列 = [类型, 日期, 00:00, 00:15, ..., 23:45]
价格单位: 元/MWh (与代码内部一致)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# 可复现
rng = np.random.default_rng(20260501)

# 24 小时基准电价 (元/MWh): 夜间低价、午后小高峰、晚高峰(17-21点)最高
HOURLY_BASE = [
    240, 225, 215, 210, 215, 225, 260, 330, 420, 480, 440, 400,
    380, 370, 390, 420, 520, 800, 1050, 1150, 980, 760, 560, 380,
]


def hourly_to_96(base24):
    """把 24 点基准线性插值成 96 个 15 分钟点"""
    xs = np.arange(24)
    xq = np.arange(0, 24, 0.25)
    return np.interp(xq, xs, base24)


def build_day(day_offset: int) -> list[float]:
    """生成某一天的 96 点价格 (带日间波动 + 噪声), 单位 元/MWh"""
    base = hourly_to_96(HOURLY_BASE)
    # 日间整体涨跌 (0.90 ~ 1.15)
    day_factor = 0.90 + 0.25 * rng.random()
    # 峰段抬升幅度随机 (让不同天的晚高峰高低不同, 套利空间有差异)
    peak_boost = rng.uniform(0.0, 0.25)
    prices = base * day_factor
    # 给 17:00-21:00 (索引 68-84) 叠加晚高峰扰动
    for i in range(68, 85):
        prices[i] *= (1.0 + peak_boost)
    # 轻微噪声 ±15 元/MWh, 并保证非负
    noise = rng.normal(0, 12, size=len(prices))
    prices = np.clip(prices + noise, 120, None)
    return [round(float(p), 1) for p in prices]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_prices.xlsx")

    n_days = 7
    start = pd.Timestamp("2026-05-01")
    time_cols = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]

    rows = []
    for d in range(n_days):
        day_date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        prices = build_day(d)
        row = {"类型": "日前价格", "日期": day_date}
        for t, p in zip(time_cols, prices):
            row[t] = p
        rows.append(row)

    df = pd.DataFrame(rows, columns=["类型", "日期", *time_cols])
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"示例数据已生成: {out_path}")
    print(f"  天数={n_days}, 时点={len(time_cols)}/天, 单位=元/MWh")
    print(f"  首日价格区间: {df.iloc[0, 2:].min():.0f} ~ {df.iloc[0, 2:].max():.0f} 元/MWh")


if __name__ == "__main__":
    main()
