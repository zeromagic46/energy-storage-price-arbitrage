# -*- coding: utf-8 -*-
"""
200MW/800MWh 储能电站: 峰谷分析 + MILP最优充放策略 + 逐日收益测算
====================================================================
流程:
  1. 峰谷分析层: 用分位数法识别每天的"峰段"/"谷段", 输出峰谷价格、
     价差、持续时长 (不是简单取全天最高/最低点, 因为节点价格常有毛刺尖峰,
     用价格高于85分位/低于15分位的时间段作为"峰段/谷段"更贴近实际调度窗口)
  2. 优化层: 复用 battery_arbitrage.py 的 MILP 引擎, 按电站真实参数
     (200MW/800MWh, 4小时时长) 逐日求解最优充放计划
  3. 年度330次循环约束: 换算为 330/365 ≈ 0.904 次/日的近似日循环上限
     (注: 这是年度预算的日均近似, 严格实现需要跨日联合优化, 见文末说明)
  4. 输出: 逐日峰谷特征 + 逐日收益 + 按月汇总
"""

from __future__ import annotations
import os as _os
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import replace as dc_replace

from battery_arbitrage import (
    BatteryConfig, RiskConfig, load_price_data_xlsx_wide,
    run_risk_checks, _CJK_PROP
)

# 改成你本地Excel文件的实际路径, 例如: "C:/Users/你的用户名/Desktop/全省平均节点电价.xlsx"
# 默认指向 本脚本所在目录/../data/sample_prices.xlsx (已随包提供示例, 双击 4-Run-Backtest.bat 即可跑通)
# 路径按脚本位置解析, 不依赖命令行当前目录, 从任意位置运行都能找到
# 如需重新生成/自定义示例, 运行: python scripts/generate_sample_data.py
XLSX_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "sample_prices.xlsx")
# 如果价格数据不在第一个sheet (比如前面还有一页汇总), 改成对应的sheet名称; 否则留 0
SHEET_NAME = 0
# 如果同一张表里混有"日前价格"/"实时价格"等多种类型, 指定要用哪种; 只有一种类型则留 None
PRICE_TYPE = None

# ------------------------- 电站真实参数 -------------------------
CAPACITY_MWH = 800.0
POWER_MW = 200.0          # 4小时时长电站 (充/放电时长 = 容量/功率, 自动算)
ANNUAL_CYCLES = 330       # 全年最多启用完整充放的天数预算 (按数据实际跨度自动折算)

PEAK_PCTL = 0.85   # 高于85分位视为"峰段"
VALLEY_PCTL = 0.15  # 低于15分位视为"谷段"
MIN_BLOCK_HOURS = 1.0  # 充放电最短持续时长(小时), 会按数据颗粒度自动换算成对应的时段数

# 单日96时点价格曲线图要看哪一天: 填 "2026-04-02" 这种日期字符串指定;
# 留 None 则自动选全期价差最大的一天
TARGET_DATE = None


# 单日启用门槛: 最优4小时峰值窗口均价 - 最优4小时谷值窗口均价 < 此值(元/MWh)时,
# 当天不参与充放电 (你口头说的"1.5元"按150元/MWh理解, 疑似漏了个0——
# 1.5元/MWh连效率损耗都盖不住, 等于没有门槛; 要改直接改这个常量)
MIN_SPREAD_4H = 150.0

# 第二循环规则(仅每日2次循环的系统生效): 第二个循环的价差(放电均价-充电均价)
# 低于此值时, 当日放弃第二循环
SECOND_CYCLE_MIN_SPREAD = 200.0
# 放弃第二循环后的处理: "skip"=当日只做一充一放; "charge_only"=第二循环只做低价充电
# (把便宜电留到次日, 成本记当日、收益随SOC跨天传递在次日体现)
SECOND_CYCLE_FALLBACK = "skip"

# 电池损耗成本(元/MWh放电) 与 往返效率(0~1): 直接决定"当日净收益是否>0"这个启用/待机判断,
# 换电芯/换供应商报价、或者拿去测算别的项目时都应该重新核实, 不要长期沿用下面这组默认值。
# 60元/MWh 是经验值, 没有配套公式反推; 更严谨的估算方式是按LCOS口径:
#   损耗成本 ≈ 电池Capex(元/MWh产能) / (循环寿命(次) × 单次放电深度)
# 网页版侧边栏 / 命令行 --degradation_cost、--round_trip_eff 均可覆盖这两个默认值。
DEGRADATION_COST_PER_MWH = 60.0
ROUND_TRIP_EFF = 0.87   # 往返效率; 单向效率 = sqrt(往返效率), 充放各拆一半

# 年度循环次数硬上限(可选, 次/年): 默认 None = 不设上限(现状行为不变, 只要过门槛
# +当日盈利就启用, 不做跨天比较)。如果电池质保或年度预算要求"全年最多N次完整循环",
# 填一个数字开启硬约束: 会把所有过门槛的候选日按"假设启用"的估算净收益从高到低排序,
# 只保留收益最高的N天真正启用, 其余即使自己盈利也会被让位为待机
# (daily表的idle_reason会标注"超出年度循环预算上限")。
ANNUAL_CYCLE_CAP = None


import os as _os
_OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "output")
if not _os.path.isdir(_OUT_DIR):
    _OUT_DIR = "."

def _out(name):
    """结果文件统一输出目录: 若存在 ../output 文件夹则归档到那里, 否则当前目录"""
    return _os.path.join(_OUT_DIR, name)


# ------------------------- 1. 峰谷分析层 -------------------------

def _longest_run(mask: pd.Series, max_gap: int = 1, values: pd.Series = None,
                 prefer_high: bool = True) -> tuple[int, int, int]:
    """返回最长"近似连续"True区间的 (起始index, 结束index(含), 长度)。
    max_gap: 容差——两段True之间最多允许 max_gap 个连续False点而不被切断
    (例如85分位峰段中间某一个15分钟点略低于阈值8块钱, 不至于把整段高价窗口拦腰斩断)。
    区间两端必须是True(容差点只能出现在中间), 长度按区间总跨度计(含容差点)。
    values/prefer_high: 多段长度打平时的平手判定——峰段选均价更高的那段
    (prefer_high=True), 谷段选均价更低的那段(prefer_high=False)。"""
    vals = list(mask)
    true_idx = [i for i, v in enumerate(vals) if v]
    if not true_idx:
        return 0, -1, 0

    def seg_score(s, e):
        if values is None:
            return 0.0
        avg = float(values.iloc[s:e + 1].mean())
        return avg if prefer_high else -avg

    segments = []
    cur_start, prev_true = true_idx[0], true_idx[0]
    for i in true_idx[1:]:
        if i - prev_true - 1 <= max_gap:   # 间隔的False点数量在容差内, 视为同一段
            prev_true = i
        else:                               # 断开, 另起一段
            segments.append((cur_start, prev_true))
            cur_start, prev_true = i, i
    segments.append((cur_start, prev_true))

    best_start, best_end = max(
        segments, key=lambda se: (se[1] - se[0], seg_score(se[0], se[1])))
    return best_start, best_end, best_end - best_start + 1


def best_4h_windows(day_df: pd.DataFrame, dt_hours: float, window_h: float = 4.0) -> dict:
    """固定时长滑动窗口: 找出当天均价最高的连续 window_h 小时(峰值窗口)和
    均价最低的连续 window_h 小时(谷值窗口), 返回两个窗口的均价/起止时间/价差。
    用于"这一天值不值得做一次完整充放"的门槛判断(对应4小时系统的实际充放时长)。"""
    prices = day_df["price"].reset_index(drop=True)
    n = len(prices)
    w = round(window_h / dt_hours)
    if w < 1 or w > n:
        return {"peak4h_avg": float("nan"), "valley4h_avg": float("nan"),
                "spread_4h": float("nan"), "peak4h_start": None, "peak4h_end": None,
                "valley4h_start": None, "valley4h_end": None}
    roll = prices.rolling(w).mean()
    peak_end = int(roll.idxmax()); peak_start = peak_end - w + 1
    valley_end = int(roll.idxmin()); valley_start = valley_end - w + 1
    peak_avg = float(roll.iloc[peak_end]); valley_avg = float(roll.iloc[valley_end])
    return {
        "peak4h_avg": peak_avg,
        "valley4h_avg": valley_avg,
        "spread_4h": peak_avg - valley_avg,
        "peak4h_start": day_df.iloc[peak_start]["time"],
        "peak4h_end": day_df.iloc[peak_end]["time"],
        "valley4h_start": day_df.iloc[valley_start]["time"],
        "valley4h_end": day_df.iloc[valley_end]["time"],
    }


def analyze_peak_valley(day_df: pd.DataFrame, dt_hours: float, window_h: float = 4.0) -> dict:
    """
    window_h: 最优峰/谷滑动窗口的时长(小时), 默认4.0。
    与 run_pipeline / 网页版的口径保持一致——均传电站实际时长 duration_h,
    否则4h/2h系统会混用4h窗口, 导致 CLI 与网页"峰谷价差"数字对不上。
    """
    prices = day_df["price"]
    peak_th = prices.quantile(PEAK_PCTL)
    valley_th = prices.quantile(VALLEY_PCTL)

    # 最长连续"峰段"/"谷段" (允许中间1个点略过阈值线的容差; 长度打平时峰段取均价更高段, 谷段取更低段)
    p_start, p_end, p_len = _longest_run(prices >= peak_th, values=prices, prefer_high=True)
    v_start, v_end, v_len = _longest_run(prices <= valley_th, values=prices, prefer_high=False)

    peak_window = day_df.iloc[p_start:p_end + 1] if p_len > 0 else day_df.iloc[0:0]
    valley_window = day_df.iloc[v_start:v_end + 1] if v_len > 0 else day_df.iloc[0:0]

    stats = {
        "peak_price_max": prices.max(),
        "peak_time_max": day_df.loc[prices.idxmax(), "time"],
        "valley_price_min": prices.min(),
        "valley_time_min": day_df.loc[prices.idxmin(), "time"],
        "peak_window_start": day_df.iloc[p_start]["time"] if p_len > 0 else None,
        "peak_window_end": day_df.iloc[p_end]["time"] if p_len > 0 else None,
        "peak_window_avg_price": peak_window["price"].mean() if p_len > 0 else float("nan"),
        "peak_duration_h": p_len * dt_hours,
        "valley_window_start": day_df.iloc[v_start]["time"] if v_len > 0 else None,
        "valley_window_end": day_df.iloc[v_end]["time"] if v_len > 0 else None,
        "valley_window_avg_price": valley_window["price"].mean() if v_len > 0 else float("nan"),
        "valley_duration_h": v_len * dt_hours,
        "spread_max_min": prices.max() - prices.min(),
        "spread_window_avg": (peak_window["price"].mean() - valley_window["price"].mean()
                               if p_len > 0 and v_len > 0 else float("nan")),
    }
    # 最优N小时峰/谷滑动窗口(固定时长)及其价差 —— 用于启用日门槛判断
    # 注意: 必须显式传 window_h=duration_h, 与 run_pipeline/网页版口径保持一致
    stats.update(best_4h_windows(day_df, dt_hours, window_h=window_h))
    return stats


def extract_action_windows(df_out: pd.DataFrame, date: str, dt_hours: float) -> list[dict]:
    """把逐时段的 charge/discharge/待机 动作压缩成连续时间窗口,
    输出: [{date, action, start_time, end_time, start_idx, end_idx,
            duration_h, energy_mwh, avg_price}, ...]
    end_idx 为独占边界(时段索引, 末尾块 = 总时段数), 供绘图直接使用,
    避免 "23:45 + dt -> 00:00" 被误解析成当天首格导致色块跨图错位。
    """
    def kind(row):
        if row["charge_mwh"] > 1e-3:
            return "充电"
        if row["discharge_mwh"] > 1e-3:
            return "放电"
        return "待机"

    df_out = df_out.reset_index(drop=True)
    kinds = df_out.apply(kind, axis=1)

    windows = []
    n = len(df_out)
    i = 0
    while i < n:
        k = kinds.iloc[i]
        j = i
        while j + 1 < n and kinds.iloc[j + 1] == k:
            j += 1
        if k != "待机":
            seg = df_out.iloc[i:j + 1]
            energy_col = "charge_mwh" if k == "充电" else "discharge_mwh"
            price_col = "buy_price" if k == "充电" else "sell_price"
            # 独占边界(时段数): 绘图直接用它当右边界, 彻底绕开时间字符串跨午夜解析
            end_idx = j + 1
            # 收盘时刻字符串: 用"距零点分钟数"表示, 跨午夜则记为 24:00(而非 00:00)
            end_minutes = int(round(end_idx * dt_hours * 60))
            if end_minutes >= 1440:
                end_str = "24:00"
            else:
                end_str = f"{end_minutes // 60:02d}:{end_minutes % 60:02d}"
            windows.append({
                "date": date,
                "action": k,
                "start_time": df_out.iloc[i]["time"],
                "end_time": end_str,
                "start_idx": i,
                "end_idx": end_idx,
                "duration_h": (j - i + 1) * dt_hours,
                "energy_mwh": seg[energy_col].sum(),
                "avg_price": seg[price_col].mean(),
            })
        i = j + 1
    return windows


def fig_schedule_heatmap(all_schedule_df: pd.DataFrame, dates: list[str], time_cols_ordered: list[str]):
    """热力图: 行=日期, 列=时段, 颜色=充电/放电/待机。返回 fig 对象。"""
    action_code = {"待机": 0, "充电": 1, "放电": 2}

    def kind(row):
        if row["charge_mwh"] > 1e-3:
            return "充电"
        if row["discharge_mwh"] > 1e-3:
            return "放电"
        return "待机"

    pivot = all_schedule_df.copy()
    pivot["action_kind"] = pivot.apply(kind, axis=1)
    grid = pivot.pivot(index="date", columns="time", values="action_kind").reindex(
        index=dates, columns=time_cols_ordered
    )
    grid_num = grid.replace(action_code).astype(float).values

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#e8e8e8", "#2ca02c", "#d62728"])  # 待机灰 / 充电绿 / 放电红

    fig_h = min(24, max(6, len(dates) * 0.18))  # 封顶24英寸, 避免天数太多时图片过高渲染失败
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.imshow(grid_num, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    y_tick_step = max(1, len(dates) // 60)  # 天数多时Y轴标签按间隔抽样, 避免挤成一团
    y_ticks = range(0, len(dates), y_tick_step)
    ax.set_yticks(list(y_ticks))
    ax.set_yticklabels([dates[i] for i in y_ticks], fontsize=6, fontproperties=_CJK_PROP)
    tick_step = 8  # 每2小时(15分钟颗粒度*8)标一个刻度
    ax.set_xticks(range(0, len(time_cols_ordered), tick_step))
    ax.set_xticklabels([time_cols_ordered[i] for i in range(0, len(time_cols_ordered), tick_step)],
                        rotation=90, fontsize=8, fontproperties=_CJK_PROP)
    ax.set_xlabel("时段", fontproperties=_CJK_PROP)
    legend_elems = [
        Patch(facecolor="#2ca02c", label="充电"),
        Patch(facecolor="#d62728", label="放电"),
        Patch(facecolor="#e8e8e8", label="待机"),
    ]
    fig.suptitle("逐日充放电策略时间分布", fontproperties=_CJK_PROP, fontsize=13, y=1.0)
    fig.legend(handles=legend_elems, prop=_CJK_PROP, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def solve_schedule_fixed_duration(df: pd.DataFrame, bat: BatteryConfig, risk: RiskConfig,
                                   dt_hours: float, duration_h: float | None = None,
                                   min_block_slots: int = 4,
                                   cycles_per_day: int = 1,
                                   n_charge_blocks: int | None = None,
                                   n_discharge_blocks: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    严格版调度: 当天充电总时长和放电总时长都必须**精确等于**要求值。

    每日1次循环(4h系统): 充/放电各允许拆成多段(每段至少min_block_slots),
    MILP在硬约束下搜索收益最优组合。

    每日多次循环(2h系统, cycles_per_day>=2): 自动启用**严格交替结构约束**——
    充电/放电各恰好 cycles_per_day 个连续块, 每块恰好 duration_h 小时,
    且顺序必须为 充1->放1->充2->放2 (不允许连续两个充电块)。
    n_charge_blocks/n_discharge_blocks 可单独指定块数(用于"第二循环只做低价充电"
    这类非对称结构: 充2块+放1块)。
    """
    n = len(df)
    cap = bat.capacity_mwh
    if duration_h is None:
        duration_h = cap / bat.max_power_mw
    w = round(duration_h / dt_hours)          # 单个循环内单相(充或放)的时段数
    if n_charge_blocks is None:
        n_charge_blocks = cycles_per_day
    if n_discharge_blocks is None:
        n_discharge_blocks = cycles_per_day
    # 始终启用严格交替结构约束: 既保证"每日2次循环"的 充1->放1->充2->放2,
    # 也保证"每日1次循环(4h系统)"必然是 充1->放1(先充后放, 不会"先放后充"
    # 去蹭前日残电)。每块恰好 duration_h 小时连续。
    strict_alternation = True

    req_c_slots = n_charge_blocks * w
    req_d_slots = n_discharge_blocks * w
    if req_c_slots < 1 or req_c_slots + req_d_slots > n:
        raise ValueError(f"充放电总时段数({req_c_slots}+{req_d_slots})超出当天可用时段数({n})")

    prob = pulp.LpProblem("battery_arbitrage_fixed", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", range(n), lowBound=0)
    is_c = pulp.LpVariable.dicts("is_c", range(n), cat="Binary")
    is_d = pulp.LpVariable.dicts("is_d", range(n), cat="Binary")
    soc = pulp.LpVariable.dicts("soc", range(n), lowBound=0)

    max_e = bat.max_power_mw * dt_hours

    revenue = pulp.lpSum(discharge[t] * bat.discharge_eff * df.loc[t, "sell_price"] for t in range(n))
    cost = pulp.lpSum(charge[t] * df.loc[t, "buy_price"] for t in range(n))
    degradation = pulp.lpSum(discharge[t] * bat.degradation_cost_per_mwh for t in range(n))
    prob += revenue - cost - degradation

    for t in range(n):
        prob += is_c[t] + is_d[t] <= 1
        prob += charge[t] <= max_e * is_c[t]
        prob += discharge[t] <= max_e * is_d[t]
        prob += charge[t] >= 0.3 * max_e * is_c[t]
        prob += discharge[t] >= 0.3 * max_e * is_d[t]

        prev_soc = bat.soc_init * cap if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + charge[t] * bat.charge_eff - discharge[t]
        prob += soc[t] >= max(bat.soc_min, risk.hard_soc_min) * cap
        prob += soc[t] <= min(bat.soc_max, risk.hard_soc_max) * cap

    # 硬约束: 充电/放电总时长恰好等于要求值
    prob += pulp.lpSum(is_c[t] for t in range(n)) == req_c_slots
    prob += pulp.lpSum(is_d[t] for t in range(n)) == req_d_slots

    if strict_alternation:
        # ---- 严格交替结构: 块起点计数 + 充放交替顺序 ----
        sc = pulp.LpVariable.dicts("sc", range(n), cat="Binary")   # 充电块起点
        sd = pulp.LpVariable.dicts("sd", range(n), cat="Binary")   # 放电块起点
        prob += sc[0] == is_c[0]
        prob += sd[0] == is_d[0]
        for t in range(1, n):
            prob += sc[t] >= is_c[t] - is_c[t - 1]
            prob += sc[t] <= is_c[t]
            prob += sc[t] <= 1 - is_c[t - 1]
            prob += sd[t] >= is_d[t] - is_d[t - 1]
            prob += sd[t] <= is_d[t]
            prob += sd[t] <= 1 - is_d[t - 1]
        # 块数恰好等于要求 (配合总时段数=块数×w, 逼出"每块恰好w个时段")
        prob += pulp.lpSum(sc[t] for t in range(n)) == n_charge_blocks
        prob += pulp.lpSum(sd[t] for t in range(n)) == n_discharge_blocks
        # 交替顺序: 任意时刻 已开始的放电块数 <= 已开始的充电块数 <= 放电块数+1
        # => 必须先充后放, 且放完一次才能开始下一次充电 (充1->放1->充2->放2)
        # 用增量累计变量实现(O(n)约束, 避免O(n^2)展开拖慢建模与求解)
        cum_c = pulp.LpVariable.dicts("cum_c", range(n), lowBound=0)
        cum_d = pulp.LpVariable.dicts("cum_d", range(n), lowBound=0)
        prob += cum_c[0] == sc[0]
        prob += cum_d[0] == sd[0]
        for t in range(1, n):
            prob += cum_c[t] == cum_c[t - 1] + sc[t]
            prob += cum_d[t] == cum_d[t - 1] + sd[t]
        for t in range(n):
            prob += cum_d[t] <= cum_c[t]
            prob += cum_c[t] <= cum_d[t] + 1
        # 每块最短时长=w (与总时段数联合, 每块恰好w)
        eff_min_block = w
    else:
        eff_min_block = min_block_slots

    # 最小连续时长约束(标准机组组合 min-up-time 形式)
    for state_var in (is_c, is_d):
        # t=0 起始的块同样要满足最短时长(否则开头会漏出超短块)
        first_end = min(eff_min_block - 1, n - 1)
        prob += pulp.lpSum(state_var[k] for k in range(0, first_end + 1)) >= \
                (first_end + 1) * state_var[0]
        for t in range(1, n):
            start_event = state_var[t] - state_var[t - 1]
            window_end = min(t + eff_min_block - 1, n - 1)
            prob += pulp.lpSum(state_var[k] for k in range(t, window_end + 1)) >= \
                    (window_end - t + 1) * start_event

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    df_out = df.copy()
    df_out["charge_mwh"] = [charge[t].value() for t in range(n)]
    df_out["discharge_mwh"] = [discharge[t].value() for t in range(n)]
    df_out["soc_mwh"] = [soc[t].value() for t in range(n)]
    df_out["soc_pct"] = df_out["soc_mwh"] / cap * 100

    def action(row):
        if row["charge_mwh"] > 1e-3:
            return f"充电 {row['charge_mwh']:.1f} MWh"
        if row["discharge_mwh"] > 1e-3:
            return f"放电 {row['discharge_mwh']:.1f} MWh"
        return "待机"
    df_out["action"] = df_out.apply(action, axis=1)

    summary = {
        "status": pulp.LpStatus[status],
        "total_revenue": sum(df_out["discharge_mwh"] * bat.discharge_eff * df_out["sell_price"]),
        "total_cost": sum(df_out["charge_mwh"] * df_out["buy_price"]),
        "total_degradation": sum(df_out["discharge_mwh"] * bat.degradation_cost_per_mwh),
        "total_charge_mwh": df_out["charge_mwh"].sum(),
        "total_discharge_mwh": df_out["discharge_mwh"].sum(),
        "cycles_used": df_out["discharge_mwh"].sum() / cap,
        "end_soc_pct": float(df_out["soc_pct"].iloc[-1]),
    }
    summary["net_profit"] = summary["total_revenue"] - summary["total_cost"] - summary["total_degradation"]
    return df_out, summary


def solve_day_with_cycle_rule(day_df: pd.DataFrame, bat: BatteryConfig, risk: RiskConfig,
                               dt_hours: float, duration_h: float,
                               min_block_slots: int, cycles_per_day: int,
                               date_str: str = "") -> tuple[pd.DataFrame, dict]:
    """
    带"第二循环价差规则"的单日求解(仅每日>=2次循环时生效):
      1. 先按 严格交替(充1->放1->充2->放2) 求全循环方案;
      2. 从解里取第二循环的充/放窗口均价, 算第二循环价差;
      3. 若价差 < SECOND_CYCLE_MIN_SPREAD:
           SECOND_CYCLE_FALLBACK="skip"        -> 重解为当日一充一放;
           SECOND_CYCLE_FALLBACK="charge_only" -> 重解为 充-放-充 (第二循环只做
             低价充电, 电留到次日, 成本记当日、收益经SOC跨天传递在次日体现)。
    返回的 summary 额外带 n_cycles_final / cycle2_spread / cycle_rule 字段。
    """
    df_out, summary = solve_schedule_fixed_duration(
        day_df, bat, risk, dt_hours=dt_hours, duration_h=duration_h,
        min_block_slots=min_block_slots, cycles_per_day=cycles_per_day)
    summary["n_cycles_final"] = cycles_per_day
    summary["cycle2_spread"] = float("nan")
    summary["cycle_rule"] = ""
    if cycles_per_day < 2 or summary["status"] != "Optimal":
        return df_out, summary

    wins = extract_action_windows(df_out, date_str or "day", dt_hours)
    cw = [x for x in wins if x["action"] == "充电"]
    dw = [x for x in wins if x["action"] == "放电"]
    if len(cw) < 2 or len(dw) < 2:
        return df_out, summary
    sp2 = dw[1]["avg_price"] - cw[1]["avg_price"]
    summary["cycle2_spread"] = sp2
    if sp2 >= SECOND_CYCLE_MIN_SPREAD:
        return df_out, summary

    # 第二循环价差不足 -> 按回退模式重解
    if SECOND_CYCLE_FALLBACK == "charge_only":
        try:
            df2, s2 = solve_schedule_fixed_duration(
                day_df, bat, risk, dt_hours=dt_hours, duration_h=duration_h,
                min_block_slots=min_block_slots, cycles_per_day=2,
                n_charge_blocks=2, n_discharge_blocks=1)
            if s2["status"] == "Optimal":
                s2["n_cycles_final"] = 1
                s2["cycle2_spread"] = sp2
                s2["cycle_rule"] = (f"第二循环价差{sp2:.0f}<{SECOND_CYCLE_MIN_SPREAD:.0f}, "
                                     f"改为只做低价充电留到次日")
                return df2, s2
        except Exception:
            pass
    # skip 或 charge_only 失败兜底: 一充一放
    df1, s1 = solve_schedule_fixed_duration(
        day_df, bat, risk, dt_hours=dt_hours, duration_h=duration_h,
        min_block_slots=min_block_slots, cycles_per_day=2,
        n_charge_blocks=1, n_discharge_blocks=1)
    s1["n_cycles_final"] = 1
    s1["cycle2_spread"] = sp2
    s1["cycle_rule"] = f"第二循环价差{sp2:.0f}<{SECOND_CYCLE_MIN_SPREAD:.0f}, 当日仅一充一放"
    return df1, s1


def make_idle_schedule(df: pd.DataFrame, bat: BatteryConfig) -> tuple[pd.DataFrame, dict]:
    """当天不参与充放电的"待机"日程: SOC全天保持不变(不消耗年度循环预算)。
    用于全年330天配额用完、或当天价差不足以覆盖损耗成本的日子。"""
    df_out = df.copy()
    cap = bat.capacity_mwh
    df_out["charge_mwh"] = 0.0
    df_out["discharge_mwh"] = 0.0
    df_out["soc_mwh"] = bat.soc_init * cap
    df_out["soc_pct"] = df_out["soc_mwh"] / cap * 100
    df_out["action"] = "待机"
    summary = {
        "status": "Idle", "total_revenue": 0.0, "total_cost": 0.0, "total_degradation": 0.0,
        "total_charge_mwh": 0.0, "total_discharge_mwh": 0.0, "cycles_used": 0.0,
        "end_soc_pct": bat.soc_init * 100, "net_profit": 0.0,
        "n_cycles_final": 0, "cycle2_spread": float("nan"), "cycle_rule": "",
    }
    return df_out, summary


def generate_html_tags(windows_df: pd.DataFrame, out_html: str | None = None,
                        title_text: str = "储能电站充放电时段") -> str:
    """
    生成充放电时段的标签HTML。返回HTML字符串;
    如果传了 out_html 路径, 同时把内容写入该文件。
    """
    import json

    dates = sorted(windows_df["date"].unique()) if len(windows_df) else []
    data = []
    for d in dates:
        day = windows_df[windows_df["date"] == d]
        windows = []
        for _, r in day.iterrows():
            windows.append({
                "action": r["action"],
                "start": r["start_time"],
                "end": r["end_time"],
                "energy": round(float(r["energy_mwh"]), 0),
                "price": round(float(r["avg_price"]), 0),
            })
        data.append({"date": d, "windows": windows})

    data_json = json.dumps(data, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; background:#fafafa; color:#222; padding:24px; }
  h1 { font-size:18px; font-weight:500; margin-bottom:16px; }
  .legend { display:flex; gap:16px; align-items:center; margin-bottom:14px; font-size:13px; color:#555; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
  input#monthFilter { margin-left:auto; width:200px; padding:6px 10px; border:1px solid #ccc; border-radius:6px; font-size:13px; }
  .toolbar { display:flex; align-items:center; margin-bottom:14px; }
  #list { display:flex; flex-direction:column; gap:6px; }
  .row { display:flex; align-items:center; gap:10px; padding:8px 10px; border:1px solid #e3e3e3; border-radius:8px; background:#fff; flex-wrap:wrap; }
  .date { font-size:13px; font-weight:500; min-width:84px; color:#111; }
  .tags { display:flex; gap:6px; flex-wrap:wrap; flex:1; }
  .tag { font-size:12px; padding:3px 8px; border-radius:12px; white-space:nowrap; cursor:default; }
  .tag.charge { background:#EAF3DE; color:#173404; }
  .tag.discharge { background:#FCEBEB; color:#501313; }
  .empty { font-size:12px; color:#999; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="legend">
  <span><span class="dot" style="background:#97C459;"></span>充电</span>
  <span><span class="dot" style="background:#F09595;"></span>放电</span>
</div>
<div class="toolbar">
  <input id="monthFilter" type="text" placeholder="筛选日期关键字, 如 2026-05" />
</div>
<div id="list"></div>

<script>
const DATA = __DATA_JSON__;
function fmt(n){ return Math.round(n).toLocaleString(); }
function render(filterText){
  const list = document.getElementById('list');
  list.innerHTML = '';
  const filtered = DATA.filter(d => !filterText || d.date.includes(filterText.trim()));
  filtered.forEach(day => {
    const row = document.createElement('div');
    row.className = 'row';
    const dateEl = document.createElement('span');
    dateEl.className = 'date';
    dateEl.textContent = day.date;
    row.appendChild(dateEl);
    const tagsWrap = document.createElement('div');
    tagsWrap.className = 'tags';
    if(day.windows.length === 0){
      const empty = document.createElement('span');
      empty.className = 'empty';
      empty.textContent = '无操作';
      tagsWrap.appendChild(empty);
    }
    day.windows.forEach(w => {
      const isCharge = w.action === '充电';
      const tag = document.createElement('span');
      tag.className = 'tag ' + (isCharge ? 'charge' : 'discharge');
      tag.textContent = w.action + ' ' + w.start + '-' + w.end;
      tag.title = '电量 ' + fmt(w.energy) + ' MWh, 均价 ' + fmt(w.price) + ' 元/MWh';
      tagsWrap.appendChild(tag);
    });
    row.appendChild(tagsWrap);
    list.appendChild(row);
  });
  if(filtered.length === 0){
    list.innerHTML = '<div class="empty" style="padding:12px 0;">没有匹配的日期</div>';
  }
}
document.getElementById('monthFilter').addEventListener('input', e => render(e.target.value));
render('');
</script>
</body>
</html>
"""
    html = html.replace("__TITLE__", title_text).replace("__DATA_JSON__", data_json)
    if out_html:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
    return html


# ------------------------- 核心流程 (供CLI和GUI共用) -------------------------

def run_pipeline(xlsx_path: str, sheet_name=0, price_type: str | None = None,
                  capacity_mwh: float = 800.0, power_mw: float = 200.0,
                  annual_cycles: float = 330, soc_init: float = 0.5,
                  cycles_per_day: int | None = None,
                  date_start: str | None = None, date_end: str | None = None,
                  degradation_cost_per_mwh: float = DEGRADATION_COST_PER_MWH,
                  round_trip_eff: float = ROUND_TRIP_EFF,
                  annual_cycle_cap: float | None = ANNUAL_CYCLE_CAP) -> dict:
    """
    跑完整流程: 读数据 -> 逐日峰谷分析 -> 挑选全年可执行完整充放的天数(受循环预算约束)
    -> 按日期顺序求解(SOC跨天连续传递) -> 汇总。
    返回一个dict, 包含 daily / monthly / windows_df / all_schedule_df /
    skipped_dates / dt_hours / time_cols_ordered / 电站参数等, 不做任何文件读写,
    方便被命令行脚本和图形界面(GUI)共用。

    策略逻辑(对应"一天一充一放, 4小时系统充放都是4小时"的设计):
      1. 每个"启用日"必须充电总时长 = 放电总时长 = 额定时长(容量/功率), 硬约束,
         不会再出现"充3小时放4小时"这种不对称结果。
      2. 是否启用某一天完整充放, 由当天(假设完整充放)的净收益是否为正决定——
         盈利就启用, 不盈利就待机, 不设上限。
      3. annual_cycles(默认330次/年)是投资测算里的"最低保证利用率"目标(下限),
         按数据集实际跨度折算成 target_min_days 仅用于对比展示是否达标; 如果
         真正盈利的天数天生不足330天/年的比例, 会在结果里如实体现并提示缺口,
         不会为了凑数硬做亏本/保本的循环。
      4. 未启用的天保持待机, SOC不变; 启用日之间SOC按时间顺序连续传递
         (不再每天重置回soc_init), 避免"凭空生电"的问题。
    """
    # xlsx_path 可能是真实文件路径(命令行/*.bat用), 也可能是内存中的文件对象
    # (网页版上传文件后传的是 io.BytesIO, 不是路径字符串)。os.path.exists() 只认
    # 字符串/PathLike, 传BytesIO进去会直接抛TypeError而不是走下面的FileNotFoundError,
    # 导致网页版"开始测算"按钮每次点都崩溃——这里先判断类型, 只有"看起来像路径"时才做
    # exists() 检查, BytesIO这类文件对象直接放行交给 pandas 处理。
    _looks_like_path = isinstance(xlsx_path, (str, bytes, _os.PathLike))
    if _looks_like_path and not _os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"未找到电价文件: {xlsx_path}\n"
            f"请确认路径是否正确; 若是首次使用, 可先运行 scripts/generate_sample_data.py "
            f"生成示例数据, 或用网页版(price_trading_app.py)直接上传 Excel。"
        )
    df_long, dt_hours = load_price_data_xlsx_wide(xlsx_path, price_type=price_type, sheet_name=sheet_name)
    duration_h = capacity_mwh / power_mw
    if cycles_per_day is None:
        # 自动判定: 2小时及以下系统一天两充两放, 其余一天一充一放
        cycles_per_day = 2 if duration_h <= 2.5 else 1
    min_block_slots = max(1, round(MIN_BLOCK_HOURS / dt_hours))

    half_eff = round_trip_eff ** 0.5
    bat = BatteryConfig(
        capacity_mwh=capacity_mwh,
        max_power_mw=power_mw,
        soc_init=soc_init,
        degradation_cost_per_mwh=degradation_cost_per_mwh,
        charge_eff=half_eff,
        discharge_eff=half_eff,
    )
    risk = RiskConfig()

    dates = sorted(df_long["date"].unique())
    # 日期区间过滤: 只测算选中的时间段(数据量大时避免全量逐日求解)
    if date_start:
        dates = [d for d in dates if d >= date_start]
    if date_end:
        dates = [d for d in dates if d <= date_end]
    if not dates:
        raise ValueError(f"所选日期区间({date_start}~{date_end})内没有数据")

    # ---- 0. 过滤缺值日期 ----
    day_frames = {}
    skipped_dates = []
    for d in dates:
        day_df = df_long[df_long["date"] == d].sort_values("time").reset_index(drop=True)
        n_missing = day_df["price"].isna().sum()
        if n_missing > 0:
            skipped_dates.append((d, int(n_missing), len(day_df)))
            continue
        day_frames[d] = day_df
    valid_dates = sorted(day_frames.keys())
    if not valid_dates:
        raise ValueError("没有可用的数据行 (所有日期都被跳过或表格为空), 请检查Excel格式和数值是否正常。")

    # ---- 1. 逐日价差门槛快筛(不求解) ----
    # 门槛一(价差门槛): 最优N小时峰值窗口均价 - 谷值窗口均价 >= MIN_SPREAD_4H,
    #   不过门槛的天直接待机, 连MILP都不跑。
    # 门槛二(收益门槛): 过了价差门槛的天, 在下面主循环里正式求解(用跨天传递的
    #   真实SOC), 净收益>0才启用; <=0则当日转待机。单遍求解, 比"先预估再正式"快一倍。
    spread_4h_by_date = {}
    passes_gate = {}
    for d in valid_dates:
        w4 = best_4h_windows(day_frames[d], dt_hours, window_h=duration_h)
        spread_4h_by_date[d] = w4["spread_4h"]
        passes_gate[d] = bool(w4["spread_4h"] >= MIN_SPREAD_4H)

    # ---- 2. 330次/年是"最低保证利用率"目标(下限, 来自投资测算), 不是上限 ----
    n_dataset_days = len(valid_dates)
    # 年度目标是"循环次数"; 2小时系统一天2次, 折算成目标天数要除以每日次数
    target_min_days = round(annual_cycles / cycles_per_day * n_dataset_days / 365)

    # 盈亏平衡价差阈值(粗略参考值, 基于数据集平均买入价折算; 实际是否启用仍以
    # 逐日MILP的精确net_profit为准, 这个数只是给你一个直观的"至少要多大价差"概念):
    #   discharge_eff * P_sell = P_buy + degradation_cost_per_mwh
    #   => 盈亏平衡价差 ≈ P_buy*(1/discharge_eff - 1) + degradation_cost_per_mwh/discharge_eff
    avg_buy_price = float(df_long["price"].mean())
    breakeven_spread = avg_buy_price * (1 / bat.discharge_eff - 1) + bat.degradation_cost_per_mwh / bat.discharge_eff

    # ---- 2.5 年度循环硬上限(可选): 把"下限目标"折算成天数的同一套公式,
    #   用来算"上限最多能启用几天"。annual_cycle_cap=None时完全不影响现状行为。
    cycle_cap_days = None
    selected_dates = None
    if annual_cycle_cap is not None:
        cycle_cap_days = round(annual_cycle_cap / cycles_per_day * n_dataset_days / 365)
        # 给每个过了价差门槛的候选日估算一个"假设启用"的净收益(用统一的初始SOC估算,
        # 不依赖跨天链式SOC——链式SOC此时还没算出来, 只是排序用, 跟下面第3步的正式
        # 求解结果口径一致, 只是soc_init可能有细微差别不影响排序结论), 按收益从高到低
        # 排序, 只留最优cycle_cap_days天真正入选; 落选的天即使自己盈利也转待机。
        est_profit = {}
        for d in valid_dates:
            if not passes_gate[d]:
                continue
            try:
                _, est_sum = solve_day_with_cycle_rule(
                    day_frames[d], bat, risk, dt_hours=dt_hours,
                    duration_h=duration_h, min_block_slots=min_block_slots,
                    cycles_per_day=cycles_per_day, date_str=d)
                est_profit[d] = (est_sum.get("net_profit", -1.0)
                                  if est_sum.get("status") == "Optimal" else -1.0)
            except Exception:
                est_profit[d] = -1.0
        ranked = sorted((d for d in est_profit if est_profit[d] > 0),
                         key=lambda d: est_profit[d], reverse=True)
        selected_dates = set(ranked[:cycle_cap_days])

    # ---- 3. 按日期顺序正式求解, SOC跨天连续传递 ----
    rows = []
    all_windows = []
    all_schedules = []
    current_soc_frac = soc_init

    for d in valid_dates:
        day_df = day_frames[d]
        pv_stats = analyze_peak_valley(day_df, dt_hours, window_h=duration_h)
        bat_day = dc_replace(bat, soc_init=current_soc_frac)

        # 过了价差门槛, 但年度循环硬上限已开启且这天没排进前cycle_cap_days名
        # (被收益更高的其他交易日挤掉) -> 这天让位为待机, 不再正式求解
        capped_out = (selected_dates is not None) and passes_gate[d] and (d not in selected_dates)

        used_flag = False
        day_profit = float("nan")
        if passes_gate[d] and not capped_out:
            try:
                df_sol, sum_sol = solve_day_with_cycle_rule(
                    day_df, bat_day, risk, dt_hours=dt_hours,
                    duration_h=duration_h, min_block_slots=min_block_slots,
                    cycles_per_day=cycles_per_day, date_str=d)
            except Exception:
                df_sol, sum_sol = None, {"status": "Error", "net_profit": -1.0}
            day_profit = sum_sol.get("net_profit", -1.0)
            if sum_sol.get("status") == "Optimal" and day_profit > 0:
                df_out, summary = df_sol, sum_sol
                used_flag = True
            else:
                df_out, summary = make_idle_schedule(day_df, bat_day)
        else:
            df_out, summary = make_idle_schedule(day_df, bat_day)

        current_soc_frac = summary["end_soc_pct"] / 100

        warnings = run_risk_checks(summary, risk) if used_flag else []
        df_out["date"] = d
        all_schedules.append(df_out)
        all_windows.extend(extract_action_windows(df_out, d, dt_hours))

        # --- 策略实际执行口径的套利价差 (电量加权): AI实际选中的充/放电时点均价之差 ---
        ch_e = df_out["charge_mwh"].sum()
        dis_e = df_out["discharge_mwh"].sum()
        ai_charge_avg = (float((df_out["charge_mwh"] * df_out["buy_price"]).sum() / ch_e)
                          if ch_e > 1e-6 else float("nan"))
        ai_discharge_avg = (float((df_out["discharge_mwh"] * df_out["sell_price"]).sum() / dis_e)
                             if dis_e > 1e-6 else float("nan"))
        ai_spread = (ai_discharge_avg - ai_charge_avg
                      if ch_e > 1e-6 and dis_e > 1e-6 else float("nan"))

        rows.append({
            "date": d,
            "month": d[:7],
            **pv_stats,
            "used_full_cycle": used_flag,
            "ai_charge_avg_price": ai_charge_avg,
            "ai_discharge_avg_price": ai_discharge_avg,
            "ai_spread": ai_spread,
            "idle_reason": ("" if used_flag else
                             (f"峰谷价差{spread_4h_by_date[d]:.1f}<门槛{MIN_SPREAD_4H:.0f}元/MWh"
                              if not passes_gate[d]
                              else "超出年度循环预算上限, 让位给收益更高的交易日" if capped_out
                              else "完整充放净收益为负" if day_profit == day_profit and day_profit <= 0
                              else "求解不可行, 兜底待机")),
            "revenue": summary["total_revenue"],
            "cost": summary["total_cost"],
            "degradation": summary["total_degradation"],
            "net_profit": summary["net_profit"],
            "cycles_used": summary["cycles_used"],
            "n_cycles_final": summary.get("n_cycles_final", 0),
            "cycle2_spread": summary.get("cycle2_spread", float("nan")),
            "cycle_rule": summary.get("cycle_rule", ""),
            "warnings": "; ".join(warnings) if warnings else "",
        })

    daily = pd.DataFrame(rows)
    windows_df = pd.DataFrame(all_windows)
    all_schedule_df = pd.concat(all_schedules, ignore_index=True)

    monthly = daily.groupby("month").agg(
        days=("date", "count"),
        active_days=("used_full_cycle", "sum"),
        total_net_profit=("net_profit", "sum"),
        avg_net_profit=("net_profit", "mean"),
        avg_ai_charge_price=("ai_charge_avg_price", "mean"),
        avg_ai_discharge_price=("ai_discharge_avg_price", "mean"),
        avg_ai_spread=("ai_spread", "mean"),
        avg_spread_window=("spread_window_avg", "mean"),
        avg_peak_duration_h=("peak_duration_h", "mean"),
        avg_valley_duration_h=("valley_duration_h", "mean"),
        total_cycles=("cycles_used", "sum"),
    ).reset_index()

    time_cols_ordered = sorted(df_long["time"].unique(), key=lambda t: pd.to_datetime(t, format="%H:%M"))

    return {
        "daily": daily,
        "monthly": monthly,
        "windows_df": windows_df,
        "all_schedule_df": all_schedule_df,
        "skipped_dates": skipped_dates,
        "dt_hours": dt_hours,
        "time_cols_ordered": time_cols_ordered,
        "capacity_mwh": capacity_mwh,
        "power_mw": power_mw,
        "duration_h": duration_h,
        "annual_cycles": annual_cycles,
        "cycles_per_day": cycles_per_day,
        "target_min_days": target_min_days,
        "breakeven_spread": breakeven_spread,
        "avg_buy_price": avg_buy_price,
        "df_long": df_long,
        "degradation_cost_per_mwh": degradation_cost_per_mwh,
        "round_trip_eff": round_trip_eff,
        "annual_cycle_cap": annual_cycle_cap,
        "cycle_cap_days": cycle_cap_days,
    }


# ------------------------- 绘图 (返回 fig 对象, 由调用方决定保存还是显示) -------------------------

def fig_daily_peak_valley(day_df: pd.DataFrame, pv_stats: dict, date: str, dt_hours: float):
    """单日全部时点(如96个15分钟点)的价格曲线, 把当天识别出的峰段/谷段窗口
    用色块标注, 并把峰谷均价、价差、持续时长直接标注在图上。
    输入: day_df(单日价格, 含time/price列), pv_stats = analyze_peak_valley()的返回值。"""
    day_df = day_df.sort_values("time").reset_index(drop=True)
    x = range(len(day_df))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, day_df["price"], color="#1f77b4", linewidth=1.6, marker="o", markersize=2.5, zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    time_to_idx = {t: i for i, t in enumerate(day_df["time"])}
    if pv_stats["peak_window_start"] is not None:
        p0 = time_to_idx[pv_stats["peak_window_start"]]
        p1 = time_to_idx[pv_stats["peak_window_end"]]
        ax.axvspan(p0 - 0.5, p1 + 0.5, color="#F09595", alpha=0.5, zorder=2)
    if pv_stats["valley_window_start"] is not None:
        v0 = time_to_idx[pv_stats["valley_window_start"]]
        v1 = time_to_idx[pv_stats["valley_window_end"]]
        ax.axvspan(v0 - 0.5, v1 + 0.5, color="#97C459", alpha=0.5, zorder=2)

    ax.set_xticks(range(0, len(day_df), max(1, len(day_df) // 24)))
    ax.set_xticklabels(day_df["time"].iloc[::max(1, len(day_df) // 24)], rotation=90, fontsize=8,
                        fontproperties=_CJK_PROP)
    ax.set_ylabel("电价 (元/MWh)", fontproperties=_CJK_PROP)
    ax.set_xlabel(f"时段 (共{len(day_df)}个, {dt_hours*60:.0f}分钟颗粒度)", fontproperties=_CJK_PROP)

    legend_elems = [
        Patch(facecolor="#F09595", alpha=0.5, label="峰段(高于85分位)"),
        Patch(facecolor="#97C459", alpha=0.5, label="谷段(低于15分位)"),
    ]
    ax.legend(handles=legend_elems, prop=_CJK_PROP, loc="upper left")

    info = (
        f"峰段均价 {pv_stats['peak_window_avg_price']:.0f} 元/MWh, 持续 {pv_stats['peak_duration_h']:.2f} 小时\n"
        f"谷段均价 {pv_stats['valley_window_avg_price']:.0f} 元/MWh, 持续 {pv_stats['valley_duration_h']:.2f} 小时\n"
        f"峰谷价差(窗口均价) {pv_stats['spread_window_avg']:.0f} 元/MWh\n"
        f"峰谷价差(全天最高-最低) {pv_stats['spread_max_min']:.0f} 元/MWh"
    )
    ax.text(0.99, 0.98, info, transform=ax.transAxes, va="top", ha="right", fontsize=11,
            fontproperties=_CJK_PROP,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#cccccc"))

    ax.set_title(f"{date} 全天{len(day_df)}时点电价曲线 — 峰谷分析", fontproperties=_CJK_PROP, fontsize=13)
    fig.tight_layout()
    return fig


def fig_daily_revenue(daily: pd.DataFrame, tick_step: int, power_mw: float, capacity_mwh: float, annual_cycles: float):
    months = sorted(daily["month"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {m: cmap(i % 20) for i, m in enumerate(months)}
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(daily["date"], daily["net_profit"], color=[color_map[m] for m in daily["month"]])
    ax.set_ylabel("单日净收益 (元)", fontproperties=_CJK_PROP)
    ax.set_xticks(range(0, len(daily), tick_step))
    ax.set_xticklabels(daily["date"].iloc[::tick_step], rotation=90, fontsize=7, fontproperties=_CJK_PROP)
    legend_elems = [Patch(facecolor=color_map[m], label=m) for m in months]
    ax.legend(handles=legend_elems, prop=_CJK_PROP)
    ax.set_title(f"{power_mw:.0f}MW/{capacity_mwh:.0f}MWh 储能电站逐日净收益 (年最低利用率目标{annual_cycles:.0f}次)",
                 fontproperties=_CJK_PROP, fontsize=13)
    fig.tight_layout()
    return fig


def fig_peak_valley_duration(daily: pd.DataFrame, tick_step: int):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar([x - 0.2 for x in range(len(daily))], daily["peak_duration_h"], width=0.4, color="#d62728", label="峰段时长(小时)")
    ax.bar([x + 0.2 for x in range(len(daily))], daily["valley_duration_h"], width=0.4, color="#2ca02c", label="谷段时长(小时)")
    ax.set_ylabel("持续时长 (小时)", fontproperties=_CJK_PROP)
    ax.set_xticks(range(0, len(daily), tick_step))
    ax.set_xticklabels(daily["date"].iloc[::tick_step], rotation=90, fontsize=7, fontproperties=_CJK_PROP)
    ax.legend(prop=_CJK_PROP)
    ax.set_title("逐日峰段/谷段连续持续时长", fontproperties=_CJK_PROP, fontsize=13)
    fig.tight_layout()
    return fig


def fig_daily_price_curve(day_df: pd.DataFrame, windows_for_day: pd.DataFrame, date: str, dt_hours: float):
    """单日全部时点(如15分钟颗粒度=96个点)的价格曲线, 并把当天的充电/放电
    时间窗口用色块标注在图上, 方便直接对照价格曲线看调度决策是否合理。"""
    day_df = day_df.sort_values("time").reset_index(drop=True)
    x = range(len(day_df))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, day_df["price"], color="#1f77b4", linewidth=1.6, marker="o", markersize=2.5, zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    time_to_idx = {t: i for i, t in enumerate(day_df["time"])}
    for _, w in windows_for_day.iterrows():
        if w["action"] not in ("充电", "放电"):
            continue
        start_i = time_to_idx.get(w["start_time"])
        if start_i is None:
            continue
        # 右边界直接用独占索引 end_idx, 避免 "23:45+dt->00:00" 被解析成当天首格而跨图错位
        end_i = int(w["end_idx"])
        color = "#97C459" if w["action"] == "充电" else "#F09595"
        ax.axvspan(start_i - 0.5, end_i - 0.5, color=color, alpha=0.45, zorder=2)

    ax.set_xticks(range(0, len(day_df), max(1, len(day_df) // 24)))
    ax.set_xticklabels(day_df["time"].iloc[::max(1, len(day_df) // 24)], rotation=90, fontsize=8,
                        fontproperties=_CJK_PROP)
    ax.set_ylabel("电价 (元/MWh)", fontproperties=_CJK_PROP)
    ax.set_xlabel(f"时段 (共{len(day_df)}个, {dt_hours*60:.0f}分钟颗粒度)", fontproperties=_CJK_PROP)
    legend_elems = [
        Patch(facecolor="#97C459", alpha=0.45, label="充电时段"),
        Patch(facecolor="#F09595", alpha=0.45, label="放电时段"),
    ]
    ax.legend(handles=legend_elems, prop=_CJK_PROP, loc="upper left")
    ax.set_title(f"{date} 全天{len(day_df)}时点电价曲线", fontproperties=_CJK_PROP, fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------- 主流程 (命令行入口) -------------------------

def main():
    result = run_pipeline(
        XLSX_PATH, sheet_name=SHEET_NAME, price_type=PRICE_TYPE,
        capacity_mwh=CAPACITY_MWH, power_mw=POWER_MW, annual_cycles=ANNUAL_CYCLES,
        degradation_cost_per_mwh=DEGRADATION_COST_PER_MWH, round_trip_eff=ROUND_TRIP_EFF,
        annual_cycle_cap=ANNUAL_CYCLE_CAP,
    )
    daily = result["daily"]
    monthly = result["monthly"]
    windows_df = result["windows_df"]
    all_schedule_df = result["all_schedule_df"]
    skipped_dates = result["skipped_dates"]
    time_cols_ordered = result["time_cols_ordered"]
    dt_hours = result["dt_hours"]
    duration_h = result["duration_h"]
    target_min_days = result["target_min_days"]
    breakeven_spread = result["breakeven_spread"]
    df_long = result["df_long"]

    tick_step = max(1, len(daily) // 25)  # 自动控制横轴标签数量, 避免天数多时挤成一团
    n_active = int(daily["used_full_cycle"].sum())
    gap = target_min_days - n_active

    print("=" * 70)
    cpd = result["cycles_per_day"]
    print(f"电站参数: {POWER_MW:.0f}MW / {CAPACITY_MWH:.0f}MWh "
          f"({duration_h:.0f}小时系统, 每日{cpd}充{cpd}放, 每日充/放电总时长各{duration_h*cpd:.0f}小时)")
    print(f"电池参数: 损耗成本 {DEGRADATION_COST_PER_MWH:.0f} 元/MWh放电, 往返效率 {ROUND_TRIP_EFF*100:.0f}%"
          f" (经验默认值, 换电芯/换项目请改 DEGRADATION_COST_PER_MWH / ROUND_TRIP_EFF 常量或用CLI参数覆盖)")
    print(f"盈亏平衡价差参考值: 约 {breakeven_spread:.0f} 元/MWh (基于数据集平均买入价"
          f"{result['avg_buy_price']:.0f}元/MWh折算, 往返效率{ROUND_TRIP_EFF*100:.0f}%+损耗成本; 实际启用判断仍以逐日精确MILP净收益为准)")
    print(f"最低利用率目标: {ANNUAL_CYCLES}次/年(投资测算假设, 每日{cpd}次) -> 按本次数据跨度({len(daily)}天)折算约 {target_min_days} 天")
    if ANNUAL_CYCLE_CAP:
        print(f"年度循环硬上限: 已启用, {ANNUAL_CYCLE_CAP:.0f}次/年 -> 折算约 {result['cycle_cap_days']} 天"
              f" (超出上限的天即使自己盈利也会让位给收益更高的交易日, 详见idle_reason)")
    else:
        print("年度循环硬上限: 未启用(只要过门槛+当日盈利就启用, 不设上限; 如需硬上限改 ANNUAL_CYCLE_CAP 常量)")
    print(f"实际启用天数: {n_active} 天 (启用需同时满足: ①最优4h峰谷价差≥{MIN_SPREAD_4H:.0f}元/MWh ②完整充放净收益>0"
          + ("③未超年度循环硬上限" if ANNUAL_CYCLE_CAP else "") + ")")
    idle_days = daily[~daily["used_full_cycle"]]
    if len(idle_days):
        print(f"待机 {len(idle_days)} 天及原因:")
        for _, r in idle_days.iterrows():
            print(f"  - {r['date']}: {r['idle_reason']} (当日4h价差{r['spread_4h']:.1f}元/MWh)")
    if gap > 0:
        print(f"[提示] 实际盈利天数比目标少 {gap} 天——按当前价差水平, 天生达不到"
              f"{ANNUAL_CYCLES}次/年的最低利用率假设, 建议结合更长时间的历史数据复核, "
              f"或跟财务模型的人对一下这个假设是否需要下调")
    elif n_active > target_min_days:
        print(f"[提示] 实际盈利天数比目标多 {n_active - target_min_days} 天, 已超过最低利用率要求")
    print("=" * 70)
    print("\n【按月汇总】")
    print(monthly.to_string(index=False))
    print(f"\n全期({len(daily)}天)累计净收益: {daily['net_profit'].sum():,.0f} 元")
    print(f"全期日均净收益(按{len(daily)}天摊薄): {daily['net_profit'].mean():,.0f} 元")
    active_rows = daily[daily["used_full_cycle"]]
    if len(active_rows):
        print(f"策略实际执行口径(启用日, 电量加权): 平均充电价 {active_rows['ai_charge_avg_price'].mean():.1f} 元/MWh, "
              f"平均放电价 {active_rows['ai_discharge_avg_price'].mean():.1f} 元/MWh, "
              f"平均套利价差 {active_rows['ai_spread'].mean():.1f} 元/MWh")
    print(f"全期累计循环次数: {daily['cycles_used'].sum():.1f} 次")
    print("(注: 每个启用日的充电总时长与放电总时长都严格等于额定时长(硬约束), 不会再出现"
          "\"充3小时放4小时\"这种不对称结果; 未启用日SOC保持不变, 启用日之间SOC按时间顺序连续传递,"
          "不再每天重置。)")
    if skipped_dates:
        print(f"\n[提示] 以下 {len(skipped_dates)} 天因源数据缺值被跳过 (未计入统计):")
        for d, n_missing, n_total in skipped_dates:
            print(f"  - {d}: 缺 {n_missing}/{n_total} 个时段")

    out_daily = _out("station_200mw800mwh_daily.csv")
    out_monthly = _out("station_200mw800mwh_monthly.csv")
    out_windows = _out("station_200mw800mwh_windows.csv")
    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
    monthly.to_csv(out_monthly, index=False, encoding="utf-8-sig")
    windows_df.to_csv(out_windows, index=False, encoding="utf-8-sig")
    print(f"\n逐日结果已保存: {out_daily}")
    print(f"月度汇总已保存: {out_monthly}")
    print(f"充放电时间窗口已保存: {out_windows}")

    out_html = _out("station_schedule_tags.html")
    generate_html_tags(windows_df, out_html,
                        title_text=f"{POWER_MW:.0f}MW/{CAPACITY_MWH:.0f}MWh 储能电站充放电时段")
    print(f"充放电时段标签页面已保存: {out_html} (双击用浏览器打开, 不需要网络)")

    print("\n【充放电时间窗口示例(前5天)】")
    sample_dates = list(daily["date"])[:5]
    print(windows_df[windows_df["date"].isin(sample_dates)].to_string(index=False))

    # ---- 挑选要看哪一天的96时点数据 (用于峰谷分析图 + 充放电时段价格曲线图, 同一天) ----
    if TARGET_DATE and TARGET_DATE in set(daily["date"]):
        target_date = TARGET_DATE
    else:
        used_days = daily[daily["used_full_cycle"]]
        pool = used_days if len(used_days) else daily
        target_date = pool.loc[pool["spread_window_avg"].idxmax(), "date"]

    day_df_target = df_long[df_long["date"] == target_date].sort_values("time").reset_index(drop=True)
    pv_stats_target = analyze_peak_valley(day_df_target, dt_hours)
    windows_target = windows_df[windows_df["date"] == target_date]

    fig1 = fig_daily_peak_valley(day_df_target, pv_stats_target, target_date, dt_hours)
    fig1.savefig(_out("station_peak_valley_price.png"), dpi=150)
    plt.close(fig1)
    print(f"\n单日峰谷分析图已保存 (自动选择: {target_date}): station_peak_valley_price.png")
    print("  (在 TARGET_DATE 常量里填具体日期字符串, 可以指定看哪一天; 留 None 则自动选价差最大的一天)")

    fig2 = fig_daily_revenue(daily, tick_step, POWER_MW, CAPACITY_MWH, ANNUAL_CYCLES)
    fig2.savefig(_out("station_daily_revenue.png"), dpi=150)
    plt.close(fig2)

    fig3 = fig_peak_valley_duration(daily, tick_step)
    fig3.savefig(_out("station_peak_valley_duration.png"), dpi=150)
    plt.close(fig3)

    fig4 = fig_schedule_heatmap(all_schedule_df, list(daily["date"]), time_cols_ordered)
    fig4.savefig(_out("station_schedule_heatmap.png"), dpi=150)
    plt.close(fig4)

    fig5 = fig_daily_price_curve(day_df_target, windows_target, target_date, dt_hours)
    fig5.savefig(_out("station_daily_price_curve.png"), dpi=150)
    plt.close(fig5)

    print(f"\n图表已保存 (输出目录: {_OUT_DIR}):")
    print(" - station_peak_valley_price.png  (单日96时点峰谷分析)")
    print(" - station_daily_revenue.png")
    print(" - station_peak_valley_duration.png")
    print(" - station_schedule_heatmap.png")
    print(" - station_daily_price_curve.png  (单日96时点充放电时段)")


if __name__ == "__main__":
    main()
