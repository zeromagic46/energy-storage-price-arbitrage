# -*- coding: utf-8 -*-
"""
储能电站自动购售电优化程序 (MILP)
==================================
核心逻辑: 给定未来价格序列 (日前/实时电价), 在满足电池物理约束和风控条件下,
求解使"卖电收入 - 购电成本 - 电池损耗成本"最大化的充放电计划。

架构对应 (参考文档8层模型):
  数据层     -> load_price_data()
  预测层     -> (由外部提供 price 序列, 可接入你的预测模型输出)
  状态建模   -> BatteryConfig (SOC / 功率 / 寿命约束)
  收益函数   -> Objective 中的 revenue - cost - degradation
  优化模型   -> MILP (PuLP + CBC)
  AI策略     -> 求解结果即为等待/充电/放电决策序列
  实时学习   -> 每次用最新价格数据重新调用 solve() 即为"重新训练"
  风险控制   -> RiskConfig 中的硬约束 + run_risk_checks()

用法:
  python battery_arbitrage.py --csv your_price.csv --capacity 100 --power 50
  (不给 --csv 时使用内置示例数据跑通流程)
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# 中文字体: 跨平台自动探测常见路径 (Windows/macOS/Linux), 找不到则跳过并给出提示
# (图表仍会生成, 只是中文可能显示为方块 —— 到时候把你机器上的中文字体路径填进
#  _CJK_FONT_CANDIDATES 列表最前面即可, Windows常见路径已包含微软雅黑/黑体)
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (本沙盒环境)
    "C:/Windows/Fonts/msyh.ttc",       # Windows 微软雅黑
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",     # Windows 黑体
    "C:/Windows/Fonts/simsun.ttc",     # Windows 宋体
    "/System/Library/Fonts/PingFang.ttc",  # macOS
]
_CJK_PROP = None
for _path in _CJK_FONT_CANDIDATES:
    if os.path.exists(_path):
        try:
            fm.fontManager.addfont(_path)
            _CJK_PROP = fm.FontProperties(fname=_path)
            plt.rcParams["font.family"] = _CJK_PROP.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
if _CJK_PROP is None:
    print("[警告] 未找到中文字体, 图表中的中文可能显示为方块。"
          "请把你机器上的中文字体路径加入 _CJK_FONT_CANDIDATES 列表。")


# ------------------------- 配置 -------------------------

@dataclass
class BatteryConfig:
    capacity_mwh: float = 100.0       # 电池容量 (MWh)
    max_power_mw: float = 50.0        # 最大充/放功率 (MW)
    soc_init: float = 0.5             # 初始SOC (0~1)
    soc_min: float = 0.10             # SOC下限
    soc_max: float = 0.95             # SOC上限
    charge_eff: float = 0.9327        # 充电效率(单向)
    discharge_eff: float = 0.9327     # 放电效率(单向)。charge_eff × discharge_eff ≈ 0.87
                                       # (往返效率87%, 按你给的实测值设定, sqrt(0.87)≈0.9327)
    degradation_cost_per_mwh: float = 60.0  # 每MWh放电的电池损耗成本 (元/MWh)
    max_cycles_per_day: float = 2.0   # 每日最大循环次数


@dataclass
class RiskConfig:
    max_daily_loss_yuan: float = 500_000.0  # 单日最大允许亏损 (元), 超过则告警
    hard_soc_min: float = 0.05              # 绝对不可触碰的SOC下限 (安全冗余)
    hard_soc_max: float = 0.98


# ------------------------- 数据层 -------------------------

def load_price_data(csv_path: str | None) -> pd.DataFrame:
    """
    读取价格数据。CSV需包含列: time, price  (或 time, buy_price, sell_price)
    - 只有 price 列时, 买卖同价 (国内现货市场常见: 储能按节点电价结算)
    - dt_hours 按相邻两行时间差自动推算; 若只有一行/无法解析, 默认0.25h(15分钟)
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if "price" in df.columns and "buy_price" not in df.columns:
            df["buy_price"] = df["price"]
            df["sell_price"] = df["price"]
        assert {"buy_price", "sell_price"}.issubset(df.columns), \
            "CSV需包含 price 列, 或 buy_price + sell_price 列"
    else:
        # 内置示例数据: 与文档中江苏典型日前价格曲线量级一致 (元/kWh 已转换为 元/MWh)
        demo = {
            "time":  ["00:00","01:00","02:00","03:00","04:00","05:00","06:00",
                      "07:00","08:00","09:00","10:00","11:00","12:00","13:00",
                      "14:00","15:00","16:00","17:00","18:00","19:00","20:00",
                      "21:00","22:00","23:00"],
            "price": [250,230,220,180,180,180,250,350,480,550,480,420,250,220,
                      250,380,500,900,1080,1150,850,650,450,300],
        }
        df = pd.DataFrame(demo)
        df["buy_price"] = df["price"]
        df["sell_price"] = df["price"]

    df = df.reset_index(drop=True)
    return df


# ------------------------- 优化模型 (MILP) -------------------------

def solve_schedule(df: pd.DataFrame, bat: BatteryConfig, risk: RiskConfig,
                    dt_hours: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """[历史/宽松版求解器] 仅约束"总放电量<=次数×容量", 无精确充放时长、无严格交替。
    仅供对照参考, 正式测算请以 station_analysis.solve_schedule_fixed_duration 为准
    (网页版与命令行回测均走严格版)。"""
    n = len(df)
    prob = pulp.LpProblem("battery_arbitrage", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0)      # 从电网购电量 (MWh)
    discharge = pulp.LpVariable.dicts("discharge", range(n), lowBound=0)  # 电池放出电量 (MWh)
    is_charging = pulp.LpVariable.dicts("is_charging", range(n), cat="Binary")
    soc = pulp.LpVariable.dicts("soc", range(n), lowBound=0)

    max_e = bat.max_power_mw * dt_hours
    cap = bat.capacity_mwh

    # 目标函数: 卖电收入 - 购电成本 - 电池损耗成本
    revenue = pulp.lpSum(discharge[t] * bat.discharge_eff * df.loc[t, "sell_price"] for t in range(n))
    cost = pulp.lpSum(charge[t] * df.loc[t, "buy_price"] for t in range(n))
    degradation = pulp.lpSum(discharge[t] * bat.degradation_cost_per_mwh for t in range(n))
    prob += revenue - cost - degradation

    for t in range(n):
        # 充放电互斥 + 功率上限
        prob += charge[t] <= max_e * is_charging[t]
        prob += discharge[t] <= max_e * (1 - is_charging[t])

        # SOC动态方程
        prev_soc = bat.soc_init * cap if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + charge[t] * bat.charge_eff - discharge[t]

        # SOC硬约束 (风控)
        prob += soc[t] >= max(bat.soc_min, risk.hard_soc_min) * cap
        prob += soc[t] <= min(bat.soc_max, risk.hard_soc_max) * cap

    # 每日循环次数约束 (总放电量 / 容量 近似循环数)
    prob += pulp.lpSum(discharge[t] for t in range(n)) <= bat.max_cycles_per_day * cap

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
    }
    summary["net_profit"] = summary["total_revenue"] - summary["total_cost"] - summary["total_degradation"]

    return df_out, summary


# ------------------------- 风险控制层 -------------------------

def run_risk_checks(summary: dict, risk: RiskConfig) -> list[str]:
    warnings = []
    if summary["status"] != "Optimal":
        warnings.append(f"[严重] 求解器状态异常: {summary['status']}, 结果不可信")
    if summary["net_profit"] < -risk.max_daily_loss_yuan:
        warnings.append(
            f"[风控] 预期亏损 {abs(summary['net_profit']):,.0f} 元, "
            f"超过单日最大允许亏损 {risk.max_daily_loss_yuan:,.0f} 元 -> 建议停止自动交易, 转人工复核"
        )
    if summary["cycles_used"] > 2.05:
        warnings.append(f"[风控] 循环次数 {summary['cycles_used']:.2f} 超出预期, 请检查约束")
    return warnings


def plot_schedule(df_out: pd.DataFrame, out_png: str, title: str = "储能站MILP优化充放电计划"):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    colors = df_out["action"].apply(
        lambda a: "#2ca02c" if "充电" in a else ("#d62728" if "放电" in a else "#d9d9d9")
    )
    ax1.bar(df_out["time"], df_out["buy_price"], color=colors, zorder=2)
    ax1.set_ylabel("电价 (元/MWh)", fontproperties=_CJK_PROP)
    ax1.set_xlabel("时段", fontproperties=_CJK_PROP)
    ax1.set_xticks(range(len(df_out)))
    ax1.set_xticklabels(df_out["time"], rotation=45, fontproperties=_CJK_PROP)

    ax2 = ax1.twinx()
    ax2.plot(df_out["time"], df_out["soc_pct"], color="#1f77b4", marker="o", linewidth=2)
    ax2.set_ylabel("SOC (%)", fontproperties=_CJK_PROP)
    ax2.set_ylim(0, 100)

    legend_elems = [
        Patch(facecolor="#2ca02c", label="充电时段"),
        Patch(facecolor="#d62728", label="放电时段"),
        Patch(facecolor="#d9d9d9", label="待机时段"),
    ]
    ax1.legend(handles=legend_elems, loc="upper left", prop=_CJK_PROP)
    ax2.legend(["SOC (%)"], loc="upper right", prop=_CJK_PROP)
    plt.title(title, fontproperties=_CJK_PROP, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def load_price_data_xlsx_wide(xlsx_path: str, price_type: str | None = None,
                               sheet_name=0) -> tuple[pd.DataFrame, float]:
    """
    读取"宽表"格式的电价Excel: 每行一天, 列为 [类型/]日期/时间列...
    兼容两种时间列写法:
      - 字符串 "HH:MM" (如 "00:00")
      - Excel时间类型 datetime.time (如 time(0,0)), 常见于手工整理的表格
    多余的分析列(最高价/最低价/充放电价差等)会被自动忽略, 不影响解析。

    price_type: 若表中有多种类型(如"日前价格"和"实时价格"共存), 用此参数筛选,
                例如 price_type="日前价格"。不传则默认使用表中第一种类型。
    sheet_name: 数据所在sheet, 默认自动探测(遍历各sheet, 选第一个含时间列的)。
                也可显式指定, 例如 sheet_name="贵港运通变充放电分析"。
    """
    import datetime as _dt

    def _is_time_col(c):
        if isinstance(c, _dt.time):
            return True
        if isinstance(c, str):
            # 兼容 "00:00" 与 "00:00:00" 两种写法
            return bool(re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", c.strip()))
        return False

    # 自动定位含时间列的工作表: 若指定 sheet 没有时间列, 则遍历所有 sheet 找第一个匹配的
    # (实测很多文件电价宽表不在第一个 sheet, 比如前面有一页"价格汇总")
    xl = pd.ExcelFile(xlsx_path)
    if sheet_name is None:
        sheet_candidates = xl.sheet_names
    else:
        sheet_candidates = [sheet_name] if sheet_name in xl.sheet_names else xl.sheet_names

    raw = None
    for s in sheet_candidates:
        _df = xl.parse(s, header=0)
        if any(_is_time_col(c) for c in _df.columns):
            raw = _df
            break
    if raw is None:
        raise ValueError(
            "未在任一工作表中识别到时间列(HH:MM 或 HH:MM:SS 格式, 如 00:00 / 00:15 / 00:00:00)。\n"
            "请确认 Excel 是宽表: 每行一天, 列为 [类型] [日期] 00:00 00:15 ... 23:45。"
        )

    time_cols_raw = [c for c in raw.columns if _is_time_col(c)]

    # 统一转成 "HH:MM" 字符串列名 (自动丢弃后面的派生分析列)
    def _to_hhmm(c):
        if isinstance(c, _dt.time):
            return c.strftime("%H:%M")
        s = str(c).strip()
        # 归一化 "00:00:00" -> "00:00", 兼容各种带秒写法
        try:
            fmt = "%H:%M:%S" if s.count(":") == 2 else "%H:%M"
            return pd.to_datetime(s, format=fmt).strftime("%H:%M")
        except Exception:
            return s[:5]
    rename_map = {c: _to_hhmm(c) for c in time_cols_raw}
    time_cols = [rename_map[c] for c in time_cols_raw]

    label_cols = [c for c in raw.columns if c not in time_cols_raw]
    # 日期列: 优先匹配"日期"/"date"; 都没有则取第一个非时间列(避免误用最后一列)
    date_col = next((c for c in label_cols
                     if ("日期" in str(c) or "date" in str(c).lower())), None)
    if date_col is None and label_cols:
        date_col = label_cols[0]
    type_col = next((c for c in label_cols if "类型" in str(c)), None)

    raw = raw.rename(columns=rename_map)

    if type_col is not None and price_type:
        # 部分匹配, 避免"日前价格" vs "日前"这类拼写差异导致整表被静默清空
        mask = raw[type_col].astype(str).str.contains(str(price_type), na=False)
        if mask.any():
            raw = raw[mask]
        # 若没有任何匹配, 保留整表(不静默清空), 后续会在缺值检查中报错提示
    elif type_col is not None:
        first_type = raw[type_col].iloc[0]
        raw = raw[raw[type_col] == first_type]

    if len(raw) == 0:
        raise ValueError(
            f"筛选价格类型'{price_type}'后没有任何行。请检查 price_type 是否与表中"
            f"'类型'列的取值一致(可选值见 Excel 的'类型'列)。"
        )

    if date_col is None:
        # 极端情况: 没有任何标签列, 用行号代替日期, 保证流程不崩
        raw = raw.reset_index()
        date_col = "index"

    long_rows = []
    for _, row in raw.iterrows():
        date_val = row[date_col]
        for t in time_cols:
            long_rows.append({"date": str(date_val)[:10], "time": t, "price": row[t]})
    df_long = pd.DataFrame(long_rows)
    # 价格强制转数值(字符串型价格 isna() 检测不到, 会在后续运算埋雷)
    df_long["price"] = pd.to_numeric(df_long["price"], errors="coerce")
    df_long["buy_price"] = df_long["price"]
    df_long["sell_price"] = df_long["price"]

    # 推算 dt_hours (相邻时间戳间隔); 单列/无法解析时回退到 15 分钟
    if len(time_cols) >= 2:
        t0 = pd.to_datetime(time_cols[0])
        t1 = pd.to_datetime(time_cols[1])
        dt_hours = (t1 - t0).total_seconds() / 3600
        if dt_hours <= 0:
            dt_hours = 0.25
    else:
        dt_hours = 0.25

    return df_long, dt_hours


# ------------------------- 多日回测 -------------------------
# (原 run_backtest()/plot_daily_profit() 已删除: 经核查未被任何入口调用的孤儿代码,
#  且内部使用上面标注为"历史/宽松版"的 solve_schedule, 留着有被误用/误import的风险。
#  多日回测请用 station_analysis.run_pipeline —— main() 的 --xlsx 分支已是这样做的。)


# ------------------------- 主流程 -------------------------

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _out_dir = os.path.join(_script_dir, "..", "output")
    if not os.path.isdir(_out_dir):
        os.makedirs(_out_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="储能电站自动购售电 MILP 优化")
    parser.add_argument("--csv", type=str, default=None, help="单日价格数据CSV路径 (time, price 或 time, buy_price, sell_price)")
    parser.add_argument("--xlsx", type=str, default=None, help="多日宽表电价Excel路径 (逐日回测模式, 走与网页版一致的严格求解器)")
    parser.add_argument("--price_type", type=str, default=None, help="xlsx模式下筛选的价格类型, 如'日前价格'")
    parser.add_argument("--capacity", type=float, default=100.0, help="电池容量 MWh")
    parser.add_argument("--power", type=float, default=50.0, help="最大充放电功率 MW")
    parser.add_argument("--soc_init", type=float, default=0.5, help="初始SOC (0~1)")
    parser.add_argument("--dt_hours", type=float, default=1.0, help="每个时段的小时数 (1.0=按小时, 0.25=按15分钟)")
    parser.add_argument("--out", type=str, default=os.path.join(_out_dir, "schedule_result.csv"), help="输出结果CSV路径")
    parser.add_argument("--chart", type=str, default=os.path.join(_out_dir, "schedule_chart.png"), help="输出图表PNG路径")
    parser.add_argument("--degradation_cost", type=float, default=60.0,
                         help="电池损耗成本 (元/MWh放电); 经验默认值, 建议按实际电池采购价/循环寿命重新估算")
    parser.add_argument("--round_trip_eff", type=float, default=0.87, help="往返效率 (0~1), 如0.87=87%%")
    parser.add_argument("--cycle_cap", type=float, default=None,
                         help="[仅--xlsx模式] 年度循环次数硬上限(次/年); 不填则不限制(现状行为)")
    args = parser.parse_args()

    half_eff = args.round_trip_eff ** 0.5
    bat = BatteryConfig(capacity_mwh=args.capacity, max_power_mw=args.power, soc_init=args.soc_init,
                         degradation_cost_per_mwh=args.degradation_cost,
                         charge_eff=half_eff, discharge_eff=half_eff)
    risk = RiskConfig()

    if args.xlsx:
        # 多日回测: 复用 station_analysis 的完整流程(严格求解器 + SOC跨天传递),
        # 与网页版结果完全一致, 避免"双求解器口径打架"
        from station_analysis import run_pipeline
        if not os.path.exists(args.xlsx):
            print(f"[错误] 未找到电价文件: {args.xlsx}")
            return
        result = run_pipeline(args.xlsx, sheet_name=0, price_type=args.price_type,
                              capacity_mwh=args.capacity, power_mw=args.power,
                              soc_init=args.soc_init,
                              degradation_cost_per_mwh=args.degradation_cost,
                              round_trip_eff=args.round_trip_eff,
                              annual_cycle_cap=args.cycle_cap)
        daily = result["daily"]
        n_days = len(daily)
        total_profit = daily["net_profit"].sum()
        avg_profit = daily["net_profit"].mean()
        n_loss_days = (daily["net_profit"] < 0).sum()
        n_active = int(daily["used_full_cycle"].sum())

        print("=" * 60)
        print(f"回测天数  : {n_days} 天  (时段颗粒度 {result['dt_hours']*60:.0f} 分钟)")
        print(f"启用天数  : {n_active} 天")
        print(f"累计净收益: {total_profit:,.0f} 元")
        print(f"日均净收益: {avg_profit:,.0f} 元")
        print(f"亏损天数  : {n_loss_days} / {n_days}")
        print("=" * 60)
        out_daily = args.out.replace(".csv", "_daily_summary.csv") if args.out.endswith(".csv") else args.out + "_daily_summary.csv"
        daily.to_csv(out_daily, index=False, encoding="utf-8-sig")
        print(f"\n每日汇总已保存: {out_daily}")
        return

    # 单日演示: 使用与网页版一致的严格求解器 (充电/放电各精确 N 小时)
    from station_analysis import solve_schedule_fixed_duration
    df = load_price_data(args.csv)
    dt_hours = args.dt_hours
    duration_h = bat.capacity_mwh / bat.max_power_mw
    cycles_per_day = 2 if duration_h <= 2.5 else 1
    df_out, summary = solve_schedule_fixed_duration(
        df, bat, risk, dt_hours=dt_hours, duration_h=duration_h, cycles_per_day=cycles_per_day)
    warnings = run_risk_checks(summary, risk)

    print("=" * 60)
    print(f"求解状态: {summary['status']}")
    print(f"卖电收入: {summary['total_revenue']:,.0f} 元")
    print(f"购电成本: {summary['total_cost']:,.0f} 元")
    print(f"电池损耗: {summary['total_degradation']:,.0f} 元")
    print(f"净收益  : {summary['net_profit']:,.0f} 元")
    print(f"循环次数: {summary['cycles_used']:.2f}")
    print("=" * 60)
    if warnings:
        print("风控告警:")
        for w in warnings:
            print(" -", w)
    print()
    print(df_out[["time", "buy_price", "action", "soc_pct"]].to_string(index=False))

    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    try:
        plot_schedule(df_out, args.chart)
        print(f"\n图表已保存    : {args.chart}")
    except Exception as e:
        print(f"[提示] 图表生成跳过: {e}")
    print(f"完整结果已保存: {args.out}")


if __name__ == "__main__":
    main()
