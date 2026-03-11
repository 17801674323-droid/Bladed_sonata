#!/usr/bin/env python3
"""IEA-15-240-RWT OpenFAST vs Bladed 4.18 comparison utility.

This script compares key time-series channels exported from OpenFAST and Bladed.
Expected input format is CSV/TXT with a header row containing channel names.

Example:
  python compare_iea15mw_openfast_bladed.py \
    --openfast data/openfast_timeseries.csv \
    --bladed data/bladed_timeseries.csv \
    --output-dir results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib is optional
    plt = None


# 推荐的IEA 15MW常见对比通道映射（可按你的输出字段名调整）
DEFAULT_CHANNEL_MAP: Dict[str, str] = {
    "Wind1VelX": "Wind1VelX",          # hub-height wind speed [m/s]
    "RotSpeed": "RotSpeed",            # rotor speed [rpm]
    "GenPwr": "GenPwr",                # generator power [kW]
    "BldPitch1": "BldPitch1",          # collective pitch [deg]
    "TwrBsMyt": "TwrBsMyt",            # tower base fore-aft moment [kN-m]
    "RootMyb1": "RootMyb1",            # blade 1 root flapwise moment [kN-m]
    "NacYaw": "NacYaw",                # nacelle yaw angle [deg]
}


@dataclass
class ChannelMetrics:
    channel_openfast: str
    channel_bladed: str
    samples: int
    mean_openfast: float
    mean_bladed: float
    std_openfast: float
    std_bladed: float
    bias_bladed_minus_openfast: float
    mae: float
    rmse: float
    corrcoef: float
    max_abs_error: float


def parse_channel_map(channel_map_arg: str | None) -> Dict[str, str]:
    if not channel_map_arg:
        return DEFAULT_CHANNEL_MAP.copy()

    channel_map_path = Path(channel_map_arg)
    if channel_map_path.exists():
        with channel_map_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("channel map JSON必须是对象：{openfast_channel: bladed_channel}")
        return {str(k): str(v) for k, v in payload.items()}

    # inline "A:B,C:D" format
    mapping: Dict[str, str] = {}
    for pair in channel_map_arg.split(","):
        if ":" not in pair:
            raise ValueError(f"映射格式错误: {pair}，应为 openfast:bladed")
        left, right = pair.split(":", 1)
        mapping[left.strip()] = right.strip()
    return mapping


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f, "") for _ in range(3)])
    if "\t" in head:
        return "\t"
    if ";" in head and "," not in head:
        return ";"
    return ","


def load_table(path: Path, delimiter: str | None = None) -> pd.DataFrame:
    sep = delimiter or detect_delimiter(path)
    df = pd.read_csv(path, sep=sep, engine="python", comment="#")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare_time_column(df: pd.DataFrame, preferred: str | None = None) -> Tuple[pd.DataFrame, str]:
    if preferred and preferred in df.columns:
        time_col = preferred
    else:
        candidates = ["Time", "time", "t", "Seconds", "Sec"]
        found = [c for c in candidates if c in df.columns]
        if not found:
            raise KeyError(
                f"找不到时间列。可选参数 --openfast-time-col/--bladed-time-col 指定。当前列: {list(df.columns)}"
            )
        time_col = found[0]

    out = df.copy()
    out = out.sort_values(time_col)
    out = out.drop_duplicates(subset=[time_col])
    out = out.reset_index(drop=True)
    return out, time_col


def interpolate_to_common_time(
    of_df: pd.DataFrame,
    bl_df: pd.DataFrame,
    of_time_col: str,
    bl_time_col: str,
    of_channel: str,
    bl_channel: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if of_channel not in of_df.columns:
        raise KeyError(f"OpenFAST缺少通道: {of_channel}")
    if bl_channel not in bl_df.columns:
        raise KeyError(f"Bladed缺少通道: {bl_channel}")

    t0 = max(float(of_df[of_time_col].min()), float(bl_df[bl_time_col].min()))
    t1 = min(float(of_df[of_time_col].max()), float(bl_df[bl_time_col].max()))
    if t1 <= t0:
        raise ValueError(f"通道 {of_channel}->{bl_channel} 没有时间重叠区间")

    of_mask = (of_df[of_time_col] >= t0) & (of_df[of_time_col] <= t1)
    bl_mask = (bl_df[bl_time_col] >= t0) & (bl_df[bl_time_col] <= t1)

    t_ref = of_df.loc[of_mask, of_time_col].to_numpy(dtype=float)
    of_val = of_df.loc[of_mask, of_channel].to_numpy(dtype=float)
    bl_t = bl_df.loc[bl_mask, bl_time_col].to_numpy(dtype=float)
    bl_val = bl_df.loc[bl_mask, bl_channel].to_numpy(dtype=float)

    bl_interp = np.interp(t_ref, bl_t, bl_val)
    return t_ref, of_val, bl_interp


def calc_metrics(of_series: np.ndarray, bl_series: np.ndarray, of_ch: str, bl_ch: str) -> ChannelMetrics:
    diff = bl_series - of_series
    corr = float(np.corrcoef(of_series, bl_series)[0, 1]) if len(of_series) > 1 else np.nan
    return ChannelMetrics(
        channel_openfast=of_ch,
        channel_bladed=bl_ch,
        samples=int(len(of_series)),
        mean_openfast=float(np.mean(of_series)),
        mean_bladed=float(np.mean(bl_series)),
        std_openfast=float(np.std(of_series)),
        std_bladed=float(np.std(bl_series)),
        bias_bladed_minus_openfast=float(np.mean(diff)),
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        corrcoef=corr,
        max_abs_error=float(np.max(np.abs(diff))),
    )


def save_metrics(metrics: Iterable[ChannelMetrics], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [m.__dict__ for m in metrics]
    df = pd.DataFrame(rows)
    out_path = output_dir / "iea15mw_openfast_vs_bladed_metrics.csv"
    df.to_csv(out_path, index=False)
    return out_path


def maybe_plot(
    output_dir: Path,
    channel_metrics: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> List[Path]:
    if plt is None:
        return []

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for of_ch, bl_ch, t, of_v, bl_v in channel_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, of_v, label=f"OpenFAST:{of_ch}", lw=1.2)
        ax.plot(t, bl_v, label=f"Bladed:{bl_ch}", lw=1.0, alpha=0.9)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(of_ch)
        ax.set_title(f"IEA 15MW comparison - {of_ch} vs {bl_ch}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        out = figures_dir / f"{of_ch}__vs__{bl_ch}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved_paths.append(out)

    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IEA 15MW OpenFAST vs Bladed 4.18 对比脚本")
    p.add_argument("--openfast", required=True, help="OpenFAST导出时序文件(CSV/TXT)")
    p.add_argument("--bladed", required=True, help="Bladed 4.18导出时序文件(CSV/TXT)")
    p.add_argument("--output-dir", default="comparison_output", help="输出目录")
    p.add_argument("--openfast-time-col", default=None, help="OpenFAST时间列名")
    p.add_argument("--bladed-time-col", default=None, help="Bladed时间列名")
    p.add_argument("--openfast-delimiter", default=None, help="OpenFAST分隔符，可选: ',', ';', '\\t'")
    p.add_argument("--bladed-delimiter", default=None, help="Bladed分隔符，可选: ',', ';', '\\t'")
    p.add_argument(
        "--channel-map",
        default=None,
        help="通道映射: JSON文件路径或内联 'OpenFAST_A:Bladed_A,OpenFAST_B:Bladed_B'",
    )
    p.add_argument("--no-plot", action="store_true", help="仅输出指标CSV，不画图")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)

    channel_map = parse_channel_map(args.channel_map)

    of_df = load_table(Path(args.openfast), args.openfast_delimiter)
    bl_df = load_table(Path(args.bladed), args.bladed_delimiter)

    of_df, of_t = prepare_time_column(of_df, args.openfast_time_col)
    bl_df, bl_t = prepare_time_column(bl_df, args.bladed_time_col)

    metric_rows: List[ChannelMetrics] = []
    plot_payload: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for of_ch, bl_ch in channel_map.items():
        t_ref, of_v, bl_v = interpolate_to_common_time(of_df, bl_df, of_t, bl_t, of_ch, bl_ch)
        m = calc_metrics(of_v, bl_v, of_ch, bl_ch)
        metric_rows.append(m)
        plot_payload.append((of_ch, bl_ch, t_ref, of_v, bl_v))

    metrics_path = save_metrics(metric_rows, output_dir)
    print(f"[OK] 指标已保存: {metrics_path}")

    if not args.no_plot:
        plot_paths = maybe_plot(output_dir, plot_payload)
        if plot_paths:
            print("[OK] 图像输出:")
            for p in plot_paths:
                print(f"  - {p}")
        else:
            print("[WARN] matplotlib不可用，跳过绘图")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
