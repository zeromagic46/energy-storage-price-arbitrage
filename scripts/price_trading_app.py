# -*- coding: utf-8 -*-
"""
电力交易节点电价分析 & 储能充放电策略收益测算 (交互式) - 科技感 UI 升级版
========================================================
运行方式:  streamlit run price_trading_app.py
依赖:      pip install streamlit plotly pandas pulp openpyxl matplotlib

功能:
  1. 上传节点电价Excel(宽表, 96时点/15分钟颗粒度, 支持日前/实时价格共存)
  2. 交互式输入电站规模(功率/容量), 充放电时长自动=容量/功率, 策略动态调整
  3. 选择日期区间 -> 逐日峰谷价差曲线(基于96时点数据计算)
  4. 选择单日 -> 96时点价格曲线 + 环比(前一日)/同比(上月同日)对比及各自峰谷价差
  5. 按策略(每启用日严格Nh充+Nh放, 4h价差门槛+盈利门槛, SOC跨天连续)测算收益
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
import io

try:
    import plotly.graph_objects as go
except ImportError:
    st.error("缺少 plotly, 请先运行: pip install plotly")
    st.stop()

import station_analysis as sa
from battery_arbitrage import load_price_data_xlsx_wide, load_price_wide_sheet, BatteryConfig, RiskConfig

st.set_page_config(page_title="节点电价分析与储能策略测算", layout="wide", page_icon=":material/bolt:")

# ---------------- 全局样式 (精简版: 仅保留 config.toml 无法配置的部分) ----------------
st.markdown("""
<style>
/* 顶部留白收紧 */
.block-container { padding-top: 1.5rem; max-width: 1400px; }

/* 背景: 极淡的网格科技底纹 */
body {
    background-color: #f8fafc;
    background-image: 
        linear-gradient(rgba(14, 111, 184, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(14, 111, 184, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* 主按钮 - 充能动画 */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0e6fb8 0%, #1fae5a 100%);
    border: none; font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 14px 0 rgba(14, 111, 184, 0.25);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px 0 rgba(14, 111, 184, 0.35);
}

</style>
""", unsafe_allow_html=True)

# 图表统一科技风格
_COLORWAY = ["#0e6fb8", "#1fae5a", "#d62728", "#f08a2c", "#6a51a3", "#0d9488"]

def style_fig(fig):
    fig.update_layout(
        font=dict(family="Microsoft YaHei, Noto Sans CJK SC, PingFang SC, sans-serif", size=13, color="#25405a"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        colorway=_COLORWAY, 
        hoverlabel=dict(font_size=13, bgcolor="#15314f", font_color="white", bordercolor="#0e6fb8"),
        margin=dict(t=fig.layout.margin.t or 30, r=16, l=8),
        transition=dict(duration=500, easing="cubic-in-out")
    )
    fig.update_xaxes(gridcolor="#f0f4f8", zerolinecolor="#dfe7ee", hoverformat=".1f")
    fig.update_yaxes(gridcolor="#e8eef4", zerolinecolor="#d5dee6", hoverformat=".1f")
    return fig

def fmt_money(v):
    """大金额用万元显示, 避免指标卡数字被截断"""
    return f"{v/1e4:,.1f} 万元" if abs(v) >= 1e6 else f"{v:,.0f} 元"

# ---------------------------------------------------------------- utilities

@st.cache_data(show_spinner=False)
def _detect_types(file_bytes: bytes, sheet_name):
    """读取表中包含哪些价格类型(日前/实时...), 以及日期范围。

    复用 load_price_wide_sheet 的同一套 sheet 自动探测逻辑, 保证"类型/日期"
    列表与正式加载数据读的是同一张表 (否则当电价宽表不在第一个 sheet 时,
    这里的类型/日期会读错)。
    """
    _bio = io.BytesIO(file_bytes)
    raw = load_price_wide_sheet(_bio, sheet_name)
    type_col = next((c for c in raw.columns if "类型" in str(c)), None)
    types = list(raw[type_col].dropna().unique()) if type_col is not None else []
    date_col = next((c for c in raw.columns if "日期" in str(c)), None)
    dates = sorted(raw[date_col].astype(str).str[:10].unique()) if date_col else []
    return types, dates

@st.cache_data(show_spinner=False)
def _load_long(file_bytes: bytes, sheet_name, price_type):
    _bio = io.BytesIO(file_bytes)
    df_long, dt_hours = load_price_data_xlsx_wide(_bio, price_type=price_type, sheet_name=sheet_name)
    return df_long, dt_hours

@st.cache_data(show_spinner="正在逐日求解MILP, 天数多时需要几分钟…")
def _run_pipeline(file_bytes: bytes, sheet_name, price_type,
                  capacity_mwh, power_mw, annual_cycles, min_spread, soc_init,
                  cycles_per_day, second_min_spread=200.0, second_fallback="skip",
                  date_start=None, date_end=None,
                  degradation_cost=60.0, round_trip_eff=0.87, cycle_cap=None):
    _bio = io.BytesIO(file_bytes)
    result = sa.run_pipeline(_bio, sheet_name=sheet_name, price_type=price_type,
                              capacity_mwh=capacity_mwh, power_mw=power_mw,
                              annual_cycles=annual_cycles, soc_init=soc_init,
                              cycles_per_day=cycles_per_day,
                              degradation_cost_per_mwh=degradation_cost,
                              round_trip_eff=round_trip_eff,
                              annual_cycle_cap=cycle_cap,
                              min_spread_4h=float(min_spread),
                              second_cycle_min_spread=float(second_min_spread),
                              second_cycle_fallback=second_fallback,
                              date_start=date_start, date_end=date_end)
    return {k: result[k] for k in
            ["daily", "monthly", "windows_df", "all_schedule_df", "skipped_dates",
             "dt_hours", "duration_h", "target_min_days", "breakeven_spread",
             "avg_buy_price", "df_long", "cycle_cap_days"]}

@st.cache_data(show_spinner="正在为该日求解最优充放方案…")
def _solve_day(file_bytes: bytes, sheet_name, price_type, day: str,
               capacity_mwh, power_mw, cycles_per_day, soc_init,
               second_min_spread=200.0, second_fallback="skip",
               degradation_cost=60.0, round_trip_eff=0.87):
    df_long, dt_hours = _load_long(file_bytes, sheet_name, price_type)
    day_df = df_long[df_long["date"] == day].sort_values("time").reset_index(drop=True)
    if len(day_df) == 0 or day_df["price"].isna().any():
        return None, None, dt_hours
    half_eff = round_trip_eff ** 0.5
    bat = BatteryConfig(capacity_mwh=capacity_mwh, max_power_mw=power_mw, soc_init=soc_init,
                         degradation_cost_per_mwh=degradation_cost,
                         charge_eff=half_eff, discharge_eff=half_eff)
    risk = RiskConfig()
    duration_h = capacity_mwh / power_mw
    min_block_slots = max(1, round(1.0 / dt_hours))
    try:
        df_out, summary = sa.solve_day_with_cycle_rule(
            day_df, bat, risk, dt_hours=dt_hours, duration_h=duration_h,
            min_block_slots=min_block_slots, cycles_per_day=cycles_per_day, date_str=day,
            second_cycle_min_spread=float(second_min_spread),
            second_cycle_fallback=second_fallback)
    except Exception:
        return None, None, dt_hours
    if summary["status"] != "Optimal":
        return None, None, dt_hours
    windows = sa.extract_action_windows(df_out, day, dt_hours)
    return windows, summary, dt_hours

def day_slice(df_long: pd.DataFrame, d: str) -> pd.DataFrame:
    return df_long[df_long["date"] == d].sort_values("time").reset_index(drop=True)

def spread_metrics_for_day(df_long, d, dt_hours, duration_h):
    ddf = day_slice(df_long, d)
    if len(ddf) == 0 or ddf["price"].isna().any():
        return None
    stats = sa.analyze_peak_valley(ddf, dt_hours, window_h=duration_h)
    stats.update(sa.best_4h_windows(ddf, dt_hours, window_h=duration_h))
    return stats

def prev_day(dates_sorted, d):
    target = (pd.to_datetime(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if target in dates_sorted:
        return target
    earlier = [x for x in dates_sorted if x < d]
    return earlier[-1] if earlier else None

def month_ago_day(dates_sorted, d):
    t = pd.to_datetime(d)
    try:
        target = (t - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    except Exception:
        return None
    return target if target in dates_sorted else None

def price_curve_trace(ddf, name, color=None, dash=None):
    return go.Scatter(x=ddf["time"], y=ddf["price"], mode="lines+markers",
                      name=name, marker=dict(size=4),
                      line=dict(color=color, dash=dash, width=2))

def build_range_df(df_long, date_list, dt_hours, duration_h):
    rows = []
    for d in date_list:
        m = None
        if d is not None and d in set(df_long["date"]):
            m = spread_metrics_for_day(df_long, d, dt_hours, duration_h)
        if m is None:
            rows.append({"date": d, "peak_avg": None, "valley_avg": None, "spread": None,
                          "peak_avg_pct": None, "valley_avg_pct": None, "spread_pct": None})
        else:
            rows.append({"date": d,
                          "peak_avg": m["peak4h_avg"], "valley_avg": m["valley4h_avg"],
                          "spread": m["spread_4h"],
                          "peak_avg_pct": m["peak_window_avg_price"],
                          "valley_avg_pct": m["valley_window_avg_price"],
                          "spread_pct": m["spread_window_avg"]})
    return pd.DataFrame(rows)

def shift_dates(date_list, months=0, days=0, years=0):
    out = []
    for d in date_list:
        t = pd.to_datetime(d)
        if years: t = t - pd.DateOffset(years=years)
        if months: t = t - pd.DateOffset(months=months)
        if days: t = t - pd.Timedelta(days=days)
        out.append(t.strftime("%Y-%m-%d"))
    return out

# ---------------------------------------------------------------- sidebar

st.sidebar.header("① 数据与电站参数")
uploaded = st.sidebar.file_uploader("上传节点电价Excel (宽表, 96时点)", type=["xlsx", "xls"])
sheet_name = st.sidebar.text_input("Sheet名称 (默认第一个)", value="") or 0
if isinstance(sheet_name, str) and sheet_name.isdigit():
    sheet_name = int(sheet_name)

if uploaded is None:
    st.title(":material/bolt: 节点电价分析与储能充放电策略测算")
    st.info("请在左侧上传电价Excel文件。格式: 每行一天, 列为 [类型/日期/00:00/00:15/.../23:45], 支持日前价格与实时价格共存(用'类型'列区分)。")
    st.stop()

file_bytes = uploaded.getvalue()
types, all_dates = _detect_types(file_bytes, sheet_name)

if types:
    price_type = st.sidebar.selectbox("价格类型", types, index=0)
else:
    price_type = None
    st.sidebar.caption("表中未发现'类型'列, 将直接使用全部行")

st.sidebar.markdown("---")
power_mw = st.sidebar.number_input("额定功率 (MW)", min_value=1.0, value=200.0, step=10.0)
capacity_mwh = st.sidebar.number_input("额定容量 (MWh)", min_value=1.0, value=800.0, step=50.0)
duration_h = capacity_mwh / power_mw
st.sidebar.metric("充/放电时长 (自动=容量/功率)", f"{duration_h:.2f} 小时")

_cpd_auto = 2 if duration_h <= 2.5 else 1
cycles_per_day = st.sidebar.segmented_control("每日充放次数", [1, 2], default=[1, 2][_cpd_auto - 1],
                                              help="2小时系统(容量/功率≤2h)通常一天两充两放; 4小时系统一天一充一放")
st.sidebar.caption(f"当前: {duration_h:.0f}h系统 × 每日{cycles_per_day}次 = 每日充/放电总时长各 {duration_h*cycles_per_day:.0f} 小时")
annual_cycles = st.sidebar.number_input("年最低利用率目标 (次/年)", min_value=1, value=330 * cycles_per_day,
                                         step=10, key=f"ac_{cycles_per_day}")
if cycles_per_day >= 2:
    second_min_spread = st.sidebar.number_input("第二循环最低价差", min_value=0.0, value=200.0, step=10.0)
    second_fallback = st.sidebar.selectbox("第二循环价差不足时", ["skip", "charge_only"],
                                            format_func=lambda x: "跳过第二循环(仅一充一放)" if x == "skip" else "只做低价充电(留到次日)")
else:
    second_min_spread, second_fallback = 200.0, "skip"
min_spread = st.sidebar.number_input("启用门槛: 最优时长窗口峰谷价差 ≥", min_value=0.0, value=150.0, step=10.0)
soc_init = st.sidebar.slider("初始SOC (%)", 10, 90, 50) / 100

st.sidebar.markdown("---")
st.sidebar.caption("电池经济性参数 (直接影响\"当日是否启用\"的净收益判断)")
degradation_cost = st.sidebar.number_input(
    "电池损耗成本 (元/MWh放电)", min_value=0.0, value=60.0, step=5.0,
    help="经验默认值, 不代表适用所有项目——建议按 电池采购成本/(循环寿命×放电深度) 重新估算后再填")
round_trip_eff = st.sidebar.slider(
    "往返效率 (%)", 70, 98, 87,
    help="充电效率×放电效率; 同时影响收入和\"是否过盈利门槛\"的判断") / 100

use_cycle_cap = st.sidebar.toggle(
    "启用年度循环次数硬上限", value=False,
    help="默认不限制: 只要过价差门槛且当日盈利就启用。勾选后, 超出预算的天即使自己"
         "盈利也会按估算净收益排序, 让位给收益更高的其他交易日(常见于电池质保/年度循环预算约束)")
cycle_cap = None
if use_cycle_cap:
    cycle_cap = st.sidebar.number_input("年度循环上限 (次/年)", min_value=1,
                                          value=int(annual_cycles), step=10)

df_long, dt_hours = _load_long(file_bytes, sheet_name, price_type)
dates_sorted = sorted(df_long["date"].unique())
n_slots = df_long.groupby("date").size().max()

st.title(":material/bolt: 节点电价分析与储能充放电策略测算")

c1, c2, c3, c4 = st.columns(4)
c1.metric("数据天数", f"{len(dates_sorted)} 天", border=True)
c2.metric("时点颗粒度", f"{dt_hours*60:.0f}分 · {n_slots}点/天", border=True)
c3.metric("日期范围", f"{dates_sorted[0]} ~ {dates_sorted[-1]}", border=True)
c4.metric("价格类型", price_type or "全部", border=True)

tab1, tab2, tab3 = st.tabs(["单日分析", "环比/同比", "充放电策略·收益测算"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("单日分析: 选择日期, 查看当日96时点电价与策略充放电时段")
    import datetime as _dt
    _dmin = _dt.date.fromisoformat(dates_sorted[0])
    _dmax = _dt.date.fromisoformat(dates_sorted[-1])
    cal_col, chart_col = st.columns([1, 3])
    with cal_col:
        picked = st.date_input("日历选择", value=_dmax, min_value=_dmin, max_value=_dmax)
        sel_day = picked.isoformat()
        if sel_day not in dates_sorted:
            st.warning(f"{sel_day} 无数据, 请换一天")
            sel_day = None
    day_windows, day_summary = None, None
    if sel_day:
        m1 = spread_metrics_for_day(df_long, sel_day, dt_hours, duration_h)
        day_windows, day_summary, _ = _solve_day(file_bytes, sheet_name, price_type, sel_day,
                                                  capacity_mwh, power_mw, cycles_per_day, soc_init,
                                                  second_min_spread, second_fallback,
                                                  degradation_cost, round_trip_eff)
    with cal_col:
        if sel_day and m1 is not None:
            gate_pass = m1['spread_4h'] >= min_spread
            st.metric(f"最优{duration_h:.0f}h价差(门槛口径)", f"{m1['spread_4h']:.1f} 元/MWh",
                       delta=f"{m1['spread_4h']-min_spread:+.1f} vs 门槛", delta_color="normal", border=True)
            if gate_pass:
                st.badge("已过价差门槛", icon=":material/check:", color="green")
            else:
                st.badge("未过价差门槛", color="red")
                st.error(f"价差 {m1['spread_4h']:.1f} < 门槛 {min_spread:.0f}, 当日按策略**待机**。下方方案仅供参考。")
            if day_summary:
                cw = [w for w in day_windows if w["action"] == "充电"]
                dw = [w for w in day_windows if w["action"] == "放电"]
                ce = sum(w["energy_mwh"] for w in cw)
                de = sum(w["energy_mwh"] for w in dw)
                c_avg = sum(w["energy_mwh"] * w["avg_price"] for w in cw) / ce if ce else 0
                d_avg = sum(w["energy_mwh"] * w["avg_price"] for w in dw) / de if de else 0
                st.metric("套利价差(策略执行口径)", f"{d_avg - c_avg:.1f} 元/MWh",
                           delta=f"净收益 {fmt_money(day_summary['net_profit'])}", delta_color="off", border=True)
                rule = day_summary.get("cycle_rule", "")
                st.caption(f"实际{day_summary.get('n_cycles_final', cycles_per_day)}次循环 · 循环 {day_summary['cycles_used']:.2f} 次"
                            + (f"  \n:material/warning: {rule}" if rule else ""))
    with chart_col:
        if sel_day and m1 is not None:
            ddf1 = day_slice(df_long, sel_day)
            figd = go.Figure()
            figd.add_trace(price_curve_trace(ddf1, f"{sel_day} 电价", color="#1f77b4"))
            times = ddf1["time"].tolist()
            if day_windows:
                if m1['spread_4h'] < min_spread:
                    st.warning("该日未过启用门槛, 按策略为待机日, 以下时段仅供参考", icon=":material/warning:")
                ci = di = 0
                for w in day_windows:
                    if w["action"] == "充电":
                        ci += 1
                        _x1 = times[min(int(w["end_idx"]) - 1, len(times) - 1)]
                        figd.add_vrect(x0=w["start_time"], x1=_x1,
                                        fillcolor="rgba(46, 160, 67, 0.32)", line_width=0,
                                        layer="below",
                                        annotation_text=f"充{ci}", annotation_position="top left")
                    elif w["action"] == "放电":
                        di += 1
                        _x1 = times[min(int(w["end_idx"]) - 1, len(times) - 1)]
                        figd.add_vrect(x0=w["start_time"], x1=_x1,
                                        fillcolor="rgba(228, 86, 73, 0.32)", line_width=0,
                                        layer="below",
                                        annotation_text=f"放{di}", annotation_position="top left")
            figd.update_layout(height=430, yaxis_title="电价 (元/MWh)", xaxis_title="时段",
                                xaxis=dict(type="category"), margin=dict(t=30),
                                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(style_fig(figd))
            if day_windows:
                wdf = pd.DataFrame(day_windows)
                wdf = wdf[["action", "start_time", "end_time", "duration_h", "energy_mwh", "avg_price"]].copy()
                wdf["action"] = wdf["action"].map(lambda a: "🟩 充电" if a == "充电" else ("🟥 放电" if a == "放电" else a))
                wdf["duration_h"] = wdf["duration_h"].round(2)
                wdf["energy_mwh"] = wdf["energy_mwh"].round(1)
                wdf["avg_price"] = wdf["avg_price"].round(1)
                wdf.columns = ["动作", "开始", "结束", "时长", "电量", "均价"]
                st.dataframe(wdf, hide_index=True)

    st.subheader("区间分析: 每日最高/最低均价曲线")
    dcol1, dcol2 = st.columns(2)
    d_start = dcol1.selectbox("起始日期", dates_sorted, index=0)
    d_end = dcol2.selectbox("结束日期", dates_sorted, index=len(dates_sorted) - 1)
    if d_end <= d_start:
        st.warning("起止时间必须大于一天")
    else:
        range_dates = [d for d in dates_sorted if d_start <= d <= d_end]
        rows = []
        for d in range_dates:
            m = spread_metrics_for_day(df_long, d, dt_hours, duration_h)
            if m is None: continue
            rows.append({
                "date": d, "最优窗口峰均价": m["peak4h_avg"], "最优窗口谷均价": m["valley4h_avg"],
                "最优时长窗口价差": m["spread_4h"], "峰段均价": m["peak_window_avg_price"],
                "谷段均价": m["valley_window_avg_price"], "峰谷窗口均价差": m["spread_window_avg"],
                "全天最高-最低": m["spread_max_min"], "峰段持续": m["peak_duration_h"], "谷段持续": m["valley_duration_h"],
            })
        spread_df = pd.DataFrame(rows)
        if len(spread_df):
            caliber = st.segmented_control("均价口径", [f"最优{duration_h:.0f}h窗口均价", "85/15分位峰谷段均价"],
                                           default=f"最优{duration_h:.0f}h窗口均价")
            if caliber.startswith("最优"):
                hi_col, lo_col, sp_col = "最优窗口峰均价", "最优窗口谷均价", "最优时长窗口价差"
                hi_name, lo_name = f"每日最高均价 (最优{duration_h:.0f}h峰窗)", f"每日最低均价 (最优{duration_h:.0f}h谷窗)"
            else:
                hi_col, lo_col, sp_col = "峰段均价", "谷段均价", "峰谷窗口均价差"
                hi_name, lo_name = "每日峰段均价 (>85分位窗口)", "每日谷段均价 (<15分位窗口)"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spread_df["date"], y=spread_df[hi_col], mode="lines+markers", name=hi_name, line=dict(color="#d62728", width=2)))
            fig.add_trace(go.Scatter(x=spread_df["date"], y=spread_df[lo_col], mode="lines+markers", name=lo_name, line=dict(color="#2ca02c", width=2)))
            _xcfg = dict(type="category")
            if len(spread_df) > 31: _xcfg["rangeslider"] = dict(visible=True, thickness=0.09)
            fig.update_layout(height=460 if len(spread_df) > 31 else 400, yaxis_title="电价 (元/MWh)", xaxis_title="日期",
                              xaxis=_xcfg, legend=dict(orientation="h", y=1.12), margin=dict(t=30))
            st.plotly_chart(style_fig(fig))
            if len(spread_df) > 31: st.caption(":material/lightbulb: 图下方滑块可拖拽/缩放查看长时间跨度")

            hi_idx = spread_df[hi_col].idxmax()
            lo_idx = spread_df[lo_col].idxmin()
            n_days_rng = spread_df[hi_col].notna().sum()
            t1c, t2c, t3c = st.columns(3)
            t1c.metric(f"区间最高均价 (Σ/{n_days_rng}天)", f"{spread_df[hi_col].mean():.1f}", delta=f"最高单日 {spread_df[hi_col].max():.0f}", delta_color="off", border=True)
            t2c.metric(f"区间最低均价 (Σ/{n_days_rng}天)", f"{spread_df[lo_col].mean():.1f}", delta=f"最低单日 {spread_df[lo_col].min():.0f}", delta_color="off", border=True)
            t3c.metric("平均价差", f"{spread_df[sp_col].mean():.1f}", delta=f"{spread_df[sp_col].mean()-min_spread:+.0f} vs 门槛", border=True)

            with st.expander("每日价差柱状图 / 峰谷持续时长 / 明细表下载"):
                fig_sp = go.Figure()
                fig_sp.add_trace(go.Bar(x=spread_df["date"], y=spread_df[sp_col], name="价差", marker_color="#6a51a3"))
                fig_sp.add_hline(y=min_spread, line_dash="dash", line_color="red", annotation_text=f"门槛 {min_spread:.0f}")
                fig_sp.update_layout(height=300, yaxis_title="价差 (元/MWh)", xaxis=dict(type="category"), margin=dict(t=30))
                st.plotly_chart(style_fig(fig_sp))

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=spread_df["date"], y=spread_df["峰段持续"], name="峰段持续", marker_color="#d62728"))
                fig2.add_trace(go.Bar(x=spread_df["date"], y=spread_df["谷段持续"], name="谷段持续", marker_color="#2ca02c"))
                fig2.update_layout(height=280, barmode="group", yaxis_title="小时", xaxis=dict(type="category"), legend=dict(orientation="h", y=1.15), margin=dict(t=30))
                st.plotly_chart(style_fig(fig2))
                st.dataframe(spread_df.round(1), hide_index=True)
                st.download_button("下载CSV", spread_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"峰谷均价_{d_start}_{d_end}.csv")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("单日对比: 当日96时点曲线 vs 环比日/同比日")
    import datetime as _dt2
    _dmin2 = _dt2.date.fromisoformat(dates_sorted[0])
    _dmax2 = _dt2.date.fromisoformat(dates_sorted[-1])
    cal2, chart2 = st.columns([1, 3])
    with cal2:
        picked2 = st.date_input("日历选择", value=_dmax2, min_value=_dmin2, max_value=_dmax2, key="cmp_cal")
        sel_d = picked2.isoformat()
        if sel_d not in dates_sorted:
            st.warning(f"{sel_d} 无数据")
            sel_d = None
        else:
            auto_prev = prev_day(dates_sorted, sel_d)
            auto_mom = month_ago_day(dates_sorted, sel_d)
            cmp_prev = st.selectbox("环比日", ["(不对比)"] + dates_sorted, index=(dates_sorted.index(auto_prev) + 1) if auto_prev else 0)
            cmp_mom = st.selectbox("同比日", ["(不对比)"] + dates_sorted, index=(dates_sorted.index(auto_mom) + 1) if auto_mom else 0)
            if auto_mom is None: st.caption("数据中无上月同日, 可手动选")
    with chart2:
        if sel_d:
            fig = go.Figure()
            fig.add_trace(price_curve_trace(day_slice(df_long, sel_d), f"{sel_d} (当日)", color="#1f77b4"))
            compare_days = [("当日", sel_d)]
            if cmp_prev != "(不对比)":
                fig.add_trace(price_curve_trace(day_slice(df_long, cmp_prev), f"{cmp_prev} (环比)", color="#ff7f0e", dash="dash"))
                compare_days.append(("环比", cmp_prev))
            if cmp_mom != "(不对比)":
                fig.add_trace(price_curve_trace(day_slice(df_long, cmp_mom), f"{cmp_mom} (同比)", color="#2ca02c", dash="dot"))
                compare_days.append(("同比", cmp_mom))
            fig.update_layout(height=420, yaxis_title="电价 (元/MWh)", xaxis_title="时段", xaxis=dict(type="category"), legend=dict(orientation="h", y=1.1), margin=dict(t=30))
            st.plotly_chart(style_fig(fig))

    if sel_d:
        mcols = st.columns(len(compare_days))
        base_spread = None
        for i, (label, d) in enumerate(compare_days):
            m = spread_metrics_for_day(df_long, d, dt_hours, duration_h)
            if m is None:
                mcols[i].warning(f"{d} 数据缺失")
                continue
            sp = m["spread_4h"]
            delta = None
            if label == "当日": base_spread = sp
            elif base_spread is not None: delta = f"{base_spread - sp:+.1f} 当日较此"
            mcols[i].metric(f"{label} {d} 最优{duration_h:.0f}h价差", f"{sp:.1f} 元/MWh", delta, border=True)
            mcols[i].caption(f"最高均价 {m['peak4h_avg']:.0f} · 最低均价 {m['valley4h_avg']:.0f}")

    st.subheader("区间对比: 充电/放电均价曲线 vs 环比期/同比期")
    rc1, rc2 = st.columns(2)
    r_start = rc1.selectbox("起始日期", dates_sorted, index=0, key="cmp_rs")
    r_end = rc2.selectbox("结束日期", dates_sorted, index=len(dates_sorted) - 1, key="cmp_re")
    if r_end <= r_start:
        st.warning("起止时间必须大于一天")
    else:
        cur_dates = [d for d in dates_sorted if r_start <= d <= r_end]
        hb_dates = shift_dates(cur_dates, months=1)
        tb_dates = shift_dates(cur_dates, years=1)
        _dset = set(dates_sorted)
        hb_avail = any(d in _dset for d in hb_dates)
        tb_avail = any(d in _dset for d in tb_dates)

        opt_c1, opt_c2 = st.columns(2)
        if hb_avail: show_hb = opt_c1.toggle(f"叠加环比期 ({hb_dates[0]} ~ {hb_dates[-1]})", value=True)
        else: show_hb = False; opt_c1.caption(f"环比期无数据")
        if tb_avail: show_tb = opt_c2.toggle(f"叠加同比期 ({tb_dates[0]} ~ {tb_dates[-1]})", value=True)
        else: show_tb = False; opt_c2.caption(f"同比期无数据")

        cur_df = build_range_df(df_long, cur_dates, dt_hours, duration_h)
        hb_df = build_range_df(df_long, hb_dates, dt_hours, duration_h) if show_hb else None
        tb_df = build_range_df(df_long, tb_dates, dt_hours, duration_h) if show_tb else None

        x_cur = cur_df["date"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_cur, y=cur_df["peak_avg"], mode="lines+markers", name="本期放电均价", line=dict(color="#d62728", width=2.6)))
        fig.add_trace(go.Scatter(x=x_cur, y=cur_df["valley_avg"], mode="lines+markers", name="本期充电均价", line=dict(color="#2ca02c", width=2.6)))
        if show_hb and hb_df["peak_avg"].notna().any():
            fig.add_trace(go.Scatter(x=x_cur, y=hb_df["peak_avg"], mode="lines", name="环比放电均价 (上月)", line=dict(color="#d62728", dash="dash", width=1.6),
                                      opacity=0.6, text=hb_df["date"], hovertemplate="环比 %{text}: %{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=x_cur, y=hb_df["valley_avg"], mode="lines", name="环比充电均价 (上月)", line=dict(color="#2ca02c", dash="dash", width=1.6),
                                      opacity=0.6, text=hb_df["date"], hovertemplate="环比 %{text}: %{y:.1f}<extra></extra>"))
        if show_tb and tb_df["peak_avg"].notna().any():
            fig.add_trace(go.Scatter(x=x_cur, y=tb_df["peak_avg"], mode="lines", name="同比放电均价 (去年)", line=dict(color="#d62728", dash="dot", width=1.6),
                                      opacity=0.6, text=tb_df["date"], hovertemplate="同比 %{text}: %{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=x_cur, y=tb_df["valley_avg"], mode="lines", name="同比充电均价 (去年)", line=dict(color="#2ca02c", dash="dot", width=1.6),
                                      opacity=0.6, text=tb_df["date"], hovertemplate="同比 %{text}: %{y:.1f}<extra></extra>"))
        _xcfg2 = dict(type="category")
        if len(cur_df) > 31: _xcfg2["rangeslider"] = dict(visible=True, thickness=0.09)
        fig.update_layout(height=490 if len(cur_df) > 31 else 440, yaxis_title="电价 (元/MWh)", xaxis_title="日期", xaxis=_xcfg2, legend=dict(orientation="h", y=1.18), margin=dict(t=30))
        st.plotly_chart(style_fig(fig))

        def _agg(df):
            if df is None or df["spread"].notna().sum() == 0: return None
            return {"平均放电均价": df["peak_avg"].mean(), "平均充电均价": df["valley_avg"].mean(), "平均价差": df["spread"].mean()}
        cur_a, hb_a, tb_a = _agg(cur_df), _agg(hb_df), _agg(tb_df)
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("本期平均放电均价", f"{cur_a['平均放电均价']:.1f} 元/MWh", border=True)
        m2c.metric("本期平均充电均价", f"{cur_a['平均充电均价']:.1f} 元/MWh", border=True)
        m3c.metric("本期平均价差", f"{cur_a['平均价差']:.1f} 元/MWh", border=True)
        tbl_rows = []
        for key in ["平均放电均价", "平均充电均价", "平均价差"]:
            row = {"指标": key, "本期": f"{cur_a[key]:.1f}"}
            for nm, agg in [("环比期(上月)", hb_a), ("同比期(去年)", tb_a)]:
                if agg is None: row[nm] = "-"; row[nm + "变化"] = "-"
                else:
                    row[nm] = f"{agg[key]:.1f}"
                    diff = cur_a[key] - agg[key]
                    pct = diff / abs(agg[key]) * 100 if agg[key] else 0
                    row[nm + "变化"] = f"{diff:+.1f} ({pct:+.1f}%)"
            tbl_rows.append(row)
        st.dataframe(pd.DataFrame(tbl_rows), hide_index=True)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("按充放电策略测算收益")
    bt1, bt2 = st.columns(2)
    bt_start = bt1.selectbox("测算起始日期", dates_sorted, index=0, key="bt_rs")
    bt_end = bt2.selectbox("测算结束日期", dates_sorted, index=len(dates_sorted) - 1, key="bt_re")
    if bt_end < bt_start:
        st.warning("结束日期需不早于起始日期")
    
    if st.button("开始测算", type="primary", icon=":material/rocket_launch:", disabled=(bt_end < bt_start)):
        res = _run_pipeline(file_bytes, sheet_name, price_type,
                             capacity_mwh, power_mw, annual_cycles, min_spread, soc_init,
                             cycles_per_day, second_min_spread, second_fallback,
                             bt_start, bt_end,
                             degradation_cost, round_trip_eff, cycle_cap)
        st.session_state["pipeline_result"] = res
        st.rerun()

    res = st.session_state.get("pipeline_result")
    if res:
        daily = res["daily"]
        monthly = res["monthly"]
        windows_df = res["windows_df"]
        n_active = int(daily["used_full_cycle"].sum())
        total_profit = daily["net_profit"].sum()

        with st.container(horizontal=True):
            st.metric("累计净收益", fmt_money(total_profit), border=True,
                      chart_data=daily["net_profit"].tolist(), chart_type="line")
            st.metric("日均净收益", fmt_money(daily["net_profit"].mean()), border=True)
            st.metric("启用天数", f"{n_active}/{len(daily)} 天", border=True)
            st.metric("最低利用率目标(折算)", f"{res['target_min_days']} 天",
                      delta=f"{n_active - res['target_min_days']:+d} 天", delta_color="normal", border=True)
            st.metric("盈亏平衡价差参考", f"{res['breakeven_spread']:.0f} 元/MWh", border=True)
        gap_days = n_active - res["target_min_days"]
        if gap_days >= 0:
            st.badge(f"利用率达标 (+{gap_days} 天)", icon=":material/check:", color="green")
        else:
            st.badge(f"利用率缺口 {abs(gap_days)} 天", color="orange")
        if res.get("cycle_cap_days") is not None:
            n_capped = int((daily["idle_reason"] == "超出年度循环预算上限, 让位给收益更高的交易日").sum())
            st.caption(f":material/warning: 已启用年度循环硬上限: 最多 {res['cycle_cap_days']} 天。"
                        + (f"其中 {n_capped} 天虽自身盈利, 但收益排名靠后被让位为待机。" if n_capped else "本次测算范围内未触发上限。"))

        active_rows = daily[daily["used_full_cycle"]]
        if len(active_rows):
            a1, a2, a3 = st.columns(3)
            a1.metric("AI平均充电价 (电量加权)", f"{active_rows['ai_charge_avg_price'].mean():.1f} 元/MWh", border=True)
            a2.metric("AI平均放电价 (电量加权)", f"{active_rows['ai_discharge_avg_price'].mean():.1f} 元/MWh", border=True)
            a3.metric("平均套利价差", f"{active_rows['ai_spread'].mean():.1f} 元/MWh", border=True)

            fig_sp = go.Figure()
            fig_sp.add_trace(go.Bar(x=active_rows["date"], y=active_rows["ai_spread"], name="当日套利价差", marker_color="#6a51a3"))
            fig_sp.add_hline(y=active_rows["ai_spread"].mean(), line_dash="dash", line_color="#e6550d", annotation_text=f"期均 {active_rows['ai_spread'].mean():.0f}")
            fig_sp.update_layout(height=300, yaxis_title="套利价差 (元/MWh)", xaxis_title="日期 (仅启用日)", xaxis=dict(type="category"), margin=dict(t=20))
            st.plotly_chart(style_fig(fig_sp))

        fig = go.Figure()
        colors = ["#1f77b4" if u else "#c9c9c9" for u in daily["used_full_cycle"]]
        fig.add_trace(go.Bar(x=daily["date"], y=daily["net_profit"], marker_color=colors, name="单日净收益"))
        fig.update_layout(height=380, yaxis_title="净收益 (元)", xaxis_title="日期(灰色=待机日)", xaxis=dict(type="category"), margin=dict(t=20))
        st.plotly_chart(style_fig(fig))

        cum = daily[["date", "net_profit"]].copy()
        cum["累计净收益"] = cum["net_profit"].cumsum()
        fig_c = go.Figure(go.Scatter(x=cum["date"], y=cum["累计净收益"], mode="lines", fill="tozeroy", name="累计净收益", line=dict(color="#0e6fb8", width=3)))
        fig_c.update_layout(height=300, yaxis_title="累计净收益 (元)", xaxis=dict(type="category"), margin=dict(t=20))
        st.plotly_chart(style_fig(fig_c))

        st.markdown("**按月汇总**")
        _m = monthly.copy()
        _mcols = {"month": "月份", "days": "天数", "active_days": "启用天数", "total_net_profit": "净收益合计(元)", "avg_net_profit": "日均净收益(元)"}
        _m = _m.rename(columns=_mcols)
        for c in _m.columns:
            if c not in ("月份", "天数", "启用天数"): _m[c] = _m[c].round(1)
        st.dataframe(
            _m,
            hide_index=True,
            column_config={
                "净收益合计(元)": st.column_config.NumberColumn("净收益合计(元)", format="accounting"),
                "日均净收益(元)": st.column_config.NumberColumn("日均净收益(元)", format="accounting"),
            },
        )

        st.markdown("**查看某一天的充放电时段与当日曲线**")
        active_dates = list(daily[daily["used_full_cycle"]]["date"])
        if active_dates:
            vd = st.selectbox("选择启用日", active_dates, index=0)
            vddf = day_slice(res["df_long"], vd)
            vwin = windows_df[windows_df["date"] == vd]
            figd = go.Figure()
            figd.add_trace(price_curve_trace(vddf, f"{vd} 电价", color="#1f77b4"))
            times = vddf["time"].tolist()
            for _, w in vwin.iterrows():
                color = "rgba(46, 160, 67, 0.32)" if w["action"] == "充电" else "rgba(228, 86, 73, 0.32)"
                _x1 = times[min(int(w["end_idx"]) - 1, len(times) - 1)]
                figd.add_vrect(x0=w["start_time"], x1=_x1, fillcolor=color, line_width=0, layer="below",
                                annotation_text=w["action"], annotation_position="top left")
            figd.update_layout(height=420, yaxis_title="电价", margin=dict(t=30))
            st.plotly_chart(style_fig(figd))
            _v = vwin[["action", "start_time", "end_time", "duration_h", "energy_mwh", "avg_price"]].copy()
            _v["duration_h"] = _v["duration_h"].round(2)
            _v["energy_mwh"] = _v["energy_mwh"].round(1)
            _v["avg_price"] = _v["avg_price"].round(1)
            st.dataframe(_v, hide_index=True)

        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("逐日结果CSV", daily.to_csv(index=False).encode("utf-8-sig"), file_name="daily_result.csv")
        dl2.download_button("月度汇总CSV", monthly.to_csv(index=False).encode("utf-8-sig"), file_name="monthly_result.csv")
        dl3.download_button("充放电窗口CSV", windows_df.to_csv(index=False).encode("utf-8-sig"), file_name="windows_result.csv")
    else:
        st.info("设置好左侧参数后, 点击上方按钮开始测算。参数变化后需重新点击。")
