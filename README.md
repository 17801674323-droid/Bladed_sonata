# Bladed_sonata

IEA 15MW 风机在 **OpenFAST** 与 **Bladed 4.18** 下的时域结果对比脚本。

## 文件说明
- `compare_iea15mw_openfast_bladed.py`：读取两套仿真导出的时序数据，做统一时间轴插值，对关键通道输出统计误差并可选绘图。

## 依赖
```bash
pip install numpy pandas matplotlib
```

## 快速使用
```bash
python compare_iea15mw_openfast_bladed.py \
  --openfast data/openfast_timeseries.csv \
  --bladed data/bladed_timeseries.csv \
  --output-dir comparison_output
```

输出：
- `comparison_output/iea15mw_openfast_vs_bladed_metrics.csv`
- `comparison_output/figures/*.png`（未加 `--no-plot` 时）

## 通道映射
默认对比通道（可按你的导出字段改名）：
- `Wind1VelX`
- `RotSpeed`
- `GenPwr`
- `BldPitch1`
- `TwrBsMyt`
- `RootMyb1`
- `NacYaw`

自定义映射方式：
1. 内联：
```bash
--channel-map "RotSpeed:RotorSpeed,GenPwr:ElectricalPower"
```
2. JSON 文件（推荐）：
```json
{
  "RotSpeed": "RotorSpeed",
  "GenPwr": "ElectricalPower"
}
```

## 常见参数
- `--openfast-time-col` / `--bladed-time-col`：指定时间列。
- `--openfast-delimiter` / `--bladed-delimiter`：手动指定分隔符（`,` `;` `\t`）。
- `--no-plot`：只导出指标，不画图。
