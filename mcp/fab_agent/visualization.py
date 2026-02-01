"""
FabAgent Visualization Tools - Enhanced Version

支持的可视化类型:
1. visualize_predictions - 单材料预测结果条形图
2. visualize_series_trend - 系列材料趋势图（折线图）
3. visualize_comparison - 多材料对比条形图（分组）

Author: PSC_Agents Team
Date: 2026-01-30
"""

import os
from pathlib import Path
from typing import Any, List, Dict, Optional
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# 配置
# =============================================================================

# 性能指标配置 - 同时支持新旧两种命名格式
METRIC_CONFIG = {
    # 新格式 (小写)
    "pce": {"label": "PCE", "unit": "%", "color": "#2ecc71", "max_ref": 30},
    "voc": {"label": "Voc", "unit": "V", "color": "#3498db", "max_ref": 1.3},
    "jsc": {"label": "Jsc", "unit": "mA/cm2", "color": "#e74c3c", "max_ref": 30},
    "ff": {"label": "FF", "unit": "%", "color": "#9b59b6", "max_ref": 90},
    "dft_band_gap": {"label": "Band Gap", "unit": "eV", "color": "#f39c12", "max_ref": 3.0},
    "energy_above_hull": {"label": "E_hull", "unit": "eV/atom", "color": "#1abc9c", "max_ref": 0.5},
    # 旧格式 (兼容)
    "PCE_percent": {"label": "PCE", "unit": "%", "color": "#2ecc71", "max_ref": 30},
    "Voc_V": {"label": "Voc", "unit": "V", "color": "#3498db", "max_ref": 1.3},
    "Jsc_mA_cm2": {"label": "Jsc", "unit": "mA/cm2", "color": "#e74c3c", "max_ref": 30},
    "FF_percent": {"label": "FF", "unit": "%", "color": "#9b59b6", "max_ref": 90},
    "BandGap_eV": {"label": "Band Gap", "unit": "eV", "color": "#f39c12", "max_ref": 3.0},
    "T80_hours": {"label": "T80", "unit": "hours", "color": "#95a5a6", "max_ref": 2000},
}

# 旧格式到新格式的映射
LEGACY_METRIC_MAP = {
    "PCE_percent": "pce",
    "Voc_V": "voc",
    "Jsc_mA_cm2": "jsc",
    "FF_percent": "ff",
    "BandGap_eV": "dft_band_gap",
    "E_hull_eV": "energy_above_hull",
}


class PredictionVisualizer:
    """
    增强版可视化工具
    
    支持:
    - 单材料预测结果条形图 (visualize_predictions)
    - 系列材料趋势图/折线图 (visualize_series_trend)  
    - 多材料对比条形图 (visualize_comparison)
    """
    
    # 兼容旧代码
    METRIC_CONFIG = METRIC_CONFIG
    
    def __init__(self, output_dir: str = "prediction_plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. 单材料预测结果条形图 (保持向后兼容)
    # =========================================================================
    def visualize_predictions(
        self,
        predicted_metrics: dict[str, float],
        target_metrics: dict[str, float] | None = None,
        recipe_id: str = "Design",
        save_path: str | None = None,
        use_plotly: bool = True,
    ) -> dict[str, Any]:
        """
        创建单材料预测结果条形图
        
        Args:
            predicted_metrics: Dict of metric_name -> predicted_value
            target_metrics: Optional dict of metric_name -> target_value for comparison
            recipe_id: Identifier for this prediction batch
            save_path: Custom save path (optional)
            use_plotly: Use Plotly (interactive HTML) or Matplotlib (static PNG)
            
        Returns:
            Dict with status, file_path, and metrics summary
        """
        if use_plotly and HAS_PLOTLY:
            return self._visualize_plotly(predicted_metrics, target_metrics, recipe_id, save_path)
        elif HAS_MATPLOTLIB:
            return self._visualize_matplotlib(predicted_metrics, target_metrics, recipe_id, save_path)
        else:
            return {
                "status": "error",
                "message": "No visualization library available. Install matplotlib or plotly.",
                "metrics": predicted_metrics
            }
    
    # =========================================================================
    # 2. 系列材料趋势图（折线图）- 新增核心功能
    # =========================================================================
    def visualize_series_trend(
        self,
        series_data: List[Dict[str, Any]],
        x_label: str = "Composition Parameter",
        y_metric: str = "pce",
        title: str = "Property Trend",
        save_path: str | None = None,
    ) -> Dict[str, Any]:
        """
        创建系列材料趋势图（折线图）
        
        用于展示某个性能指标随组分变化的趋势，例如：
        - PCE 随 Cs 含量的变化
        - Band Gap 随 Br 比例的变化
        - Voc 随混合阳离子比例的变化
        
        Args:
            series_data: 系列数据列表，每项包含:
                - x_value: X轴数值（如 Cs 含量 0, 0.1, 0.2...）
                - x_label: X轴标签（如 "FAPbI3", "FA0.9Cs0.1PbI3"...）
                - predictions: 预测结果字典
            x_label: X轴标题
            y_metric: 要绘制的Y轴指标 (pce, voc, jsc, ff, dft_band_gap, energy_above_hull)
            title: 图表标题
            save_path: 自定义保存路径
            
        Example:
            series_data = [
                {"x_value": 0.0, "x_label": "FAPbI3", "predictions": {"pce": {"value": 20.1}}},
                {"x_value": 0.1, "x_label": "FA0.9Cs0.1PbI3", "predictions": {"pce": {"value": 21.2}}},
                {"x_value": 0.2, "x_label": "FA0.8Cs0.2PbI3", "predictions": {"pce": {"value": 21.8}}},
            ]
            
        Returns:
            Dict with status, file_path, trend analysis
        """
        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            return {"status": "error", "message": "需要安装 matplotlib 或 plotly"}
        
        if not series_data:
            return {"status": "error", "message": "没有系列数据"}
        
        # 提取数据
        x_values = []
        y_values = []
        x_labels = []
        
        for item in series_data:
            x_val = item.get("x_value", len(x_values))
            x_lab = item.get("x_label", str(x_val))
            predictions = item.get("predictions", {})
            
            # 检查 predictions 是否存在
            if not predictions:
                continue
            
            # 获取Y值 - 支持多种格式
            y_val = self._extract_metric_value(predictions, y_metric)
            
            if y_val is not None:
                x_values.append(x_val)
                y_values.append(y_val)
                x_labels.append(x_lab)
        
        if not y_values:
            # 给出更详细的错误信息
            missing_predictions = sum(1 for item in series_data if not item.get("predictions"))
            if missing_predictions == len(series_data):
                return {
                    "status": "error", 
                    "message": f"所有 series_data 项都缺少 'predictions' 字段。请确保每项都包含预测结果，格式如: {{'x_value': 0, 'x_label': 'MAPbI3', 'predictions': {{'pce': {{'value': 19.1}}, 'voc': {{'value': 1.05}}, ...}}}}"
                }
            return {"status": "error", "message": f"没有找到 {y_metric} 的有效数据。请检查 predictions 字典中是否包含该指标。"}
        
        # 获取指标配置
        cfg = METRIC_CONFIG.get(y_metric, {"label": y_metric, "unit": "", "color": "#3498db"})
        y_axis_label = f"{cfg['label']} ({cfg['unit']})"
        color = cfg['color']
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path:
            file_path = Path(save_path)
        else:
            safe_title = title.replace(" ", "_").replace("/", "_")[:30]
            file_path = self.output_dir / f"trend_{safe_title}_{timestamp}.png"
        
        # 绘图
        if HAS_PLOTLY:
            return self._line_chart_plotly(x_values, y_values, x_labels, x_label, y_axis_label, title, color, file_path)
        else:
            return self._line_chart_matplotlib(x_values, y_values, x_labels, x_label, y_axis_label, title, color, file_path)
    
    # =========================================================================
    # 3. 多材料对比条形图（分组）- 增强版
    # =========================================================================
    def visualize_comparison(
        self,
        materials_data: List[Dict[str, Any]] = None,
        results_list: list[dict[str, Any]] = None,  # 兼容旧接口
        metrics: List[str] = None,
        metric_key: str = "pce",  # 兼容旧接口
        title: str = "Materials Comparison",
        save_path: str | None = None,
    ) -> Dict[str, Any]:
        """
        创建多材料对比条形图（分组对比）
        
        用于对比多种材料的性能，例如：
        - MAPbI3 vs FAPbI3 vs CsPbI3 的 PCE 对比
        - 不同 Cs 含量配方的多指标对比
        
        Args:
            materials_data: 材料数据列表，每项包含:
                - name: 材料名称
                - predictions: 预测结果字典
            results_list: 兼容旧接口，同 materials_data
            metrics: 要对比的指标列表，默认 ["pce", "voc", "jsc", "ff"]
            metric_key: 兼容旧接口，单指标对比时使用
            title: 图表标题
            save_path: 自定义保存路径
            
        Example:
            materials_data = [
                {"name": "MAPbI3", "predictions": {"pce": {"value": 19.1}, "voc": {"value": 1.05}}},
                {"name": "FAPbI3", "predictions": {"pce": {"value": 20.5}, "voc": {"value": 1.08}}},
                {"name": "CsPbI3", "predictions": {"pce": {"value": 18.2}, "voc": {"value": 1.02}}},
            ]
            
        Returns:
            Dict with status, file_path, comparison data
        """
        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            return {"status": "error", "message": "需要安装 matplotlib 或 plotly"}
        
        # 兼容旧接口
        if materials_data is None and results_list is not None:
            # 转换旧格式
            materials_data = []
            for r in results_list:
                materials_data.append({
                    "name": r.get("recipe_id", "Unknown"),
                    "predictions": r.get("predicted_metrics", {})
                })
        
        if not materials_data:
            return {"status": "error", "message": "没有材料数据"}
        
        # 如果没有指定 metrics，使用默认或单指标
        if metrics is None:
            if metric_key:
                metrics = [metric_key]
            else:
                metrics = ["pce", "voc", "jsc", "ff"]
        
        # 提取数据
        material_names = []
        metric_values = {m: [] for m in metrics}
        
        for mat in materials_data:
            name = mat.get("name", mat.get("recipe_id", "Unknown"))
            predictions = mat.get("predictions", mat.get("predicted_metrics", {}))
            material_names.append(name)
            
            for m in metrics:
                val = self._extract_metric_value(predictions, m)
                metric_values[m].append(val if val is not None else 0)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path:
            file_path = Path(save_path)
        else:
            safe_title = title.replace(" ", "_").replace("/", "_")[:30]
            file_path = self.output_dir / f"comparison_{safe_title}_{timestamp}.png"
        
        # 绘图
        if HAS_PLOTLY:
            return self._grouped_bar_plotly(material_names, metric_values, title, file_path)
        else:
            return self._grouped_bar_matplotlib(material_names, metric_values, title, file_path)
    
    # =========================================================================
    # 内部方法 - Plotly 单材料条形图
    # =========================================================================
    def _visualize_plotly(
        self,
        predicted_metrics: dict[str, float],
        target_metrics: dict[str, float] | None,
        recipe_id: str,
        save_path: str | None,
    ) -> dict[str, Any]:
        """Create interactive Plotly bar chart."""
        # Prepare data
        metrics = []
        predicted_values = []
        target_values = []
        colors = []
        
        for metric_key, value in predicted_metrics.items():
            if metric_key in METRIC_CONFIG and value is not None:
                config = METRIC_CONFIG[metric_key]
                metrics.append(f"{config['label']} ({config['unit']})")
                predicted_values.append(value)
                colors.append(config['color'])
                
                if target_metrics and metric_key in target_metrics:
                    target_values.append(target_metrics[metric_key])
                else:
                    target_values.append(None)
        
        # Create figure
        fig = go.Figure()
        
        # Add predicted values bars
        fig.add_trace(go.Bar(
            name='Predicted',
            x=metrics,
            y=predicted_values,
            marker_color=colors,
            text=[f'{v:.2f}' if v else 'N/A' for v in predicted_values],
            textposition='outside',
        ))
        
        # Add target values if provided
        if target_metrics and any(t is not None for t in target_values):
            fig.add_trace(go.Bar(
                name='Target',
                x=metrics,
                y=[t if t else 0 for t in target_values],
                marker_color='rgba(100, 100, 100, 0.5)',
                marker_line_color='rgba(50, 50, 50, 0.8)',
                marker_line_width=2,
                text=[f'{t:.2f}' if t else '' for t in target_values],
                textposition='outside',
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'🔬 Prediction Results: {recipe_id}',
                font=dict(size=20),
                x=0.5
            ),
            xaxis_title='Performance Metrics',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white',
            showlegend=bool(target_metrics),
            height=500,
            width=800,
            font=dict(size=14),
        )
        
        # Add annotations for context
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.add_annotation(
            text=f"Generated: {timestamp}",
            xref="paper", yref="paper",
            x=1, y=-0.15,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        # Save
        if save_path:
            file_path = Path(save_path)
        else:
            file_path = self.output_dir / f"{recipe_id}_{timestamp.replace(':', '-').replace(' ', '_')}.html"
        
        fig.write_html(str(file_path))
        
        # Also save as PNG for easy embedding
        png_path = file_path.with_suffix('.png')
        try:
            fig.write_image(str(png_path), scale=2)
            image_saved = True
        except Exception:
            image_saved = False
            png_path = None
        
        return {
            "status": "success",
            "html_path": str(file_path),
            "png_path": str(png_path) if image_saved else None,
            "recipe_id": recipe_id,
            "metrics_visualized": list(predicted_metrics.keys()),
            "predicted_values": predicted_metrics,
            "target_values": target_metrics,
        }
    
    # =========================================================================
    # 内部方法 - Matplotlib 单材料条形图
    # =========================================================================
    def _visualize_matplotlib(
        self,
        predicted_metrics: dict[str, float],
        target_metrics: dict[str, float] | None,
        recipe_id: str,
        save_path: str | None,
    ) -> dict[str, Any]:
        """Create static Matplotlib bar chart."""
        # Prepare data
        metrics = []
        predicted_values = []
        target_values = []
        colors = []
        
        for metric_key, value in predicted_metrics.items():
            if metric_key in METRIC_CONFIG and value is not None:
                config = METRIC_CONFIG[metric_key]
                metrics.append(f"{config['label']}\n({config['unit']})")
                predicted_values.append(value)
                colors.append(config['color'])
                
                if target_metrics and metric_key in target_metrics:
                    target_values.append(target_metrics[metric_key])
                else:
                    target_values.append(None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(metrics))
        width = 0.35
        
        # Predicted bars
        bars1 = ax.bar(
            [i - width/2 if target_metrics else i for i in x],
            predicted_values,
            width if target_metrics else width * 1.5,
            label='Predicted',
            color=colors,
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels on predicted bars
        for bar, val in zip(bars1, predicted_values):
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
        
        # Target bars if provided
        if target_metrics and any(t is not None for t in target_values):
            bars2 = ax.bar(
                [i + width/2 for i in x],
                [t if t else 0 for t in target_values],
                width,
                label='Target',
                color='lightgray',
                edgecolor='black',
                linewidth=1,
                hatch='//'
            )
            for bar, val in zip(bars2, target_values):
                if val:
                    ax.annotate(
                        f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10
                    )
        
        # Styling
        ax.set_xlabel('Performance Metrics', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'🔬 Prediction Results: {recipe_id}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                 ha='right', va='bottom', fontsize=8, color='gray')
        
        plt.tight_layout()
        
        # Save
        if save_path:
            file_path = Path(save_path)
        else:
            file_path = self.output_dir / f"{recipe_id}_{timestamp.replace(':', '-').replace(' ', '_')}.png"
        
        plt.savefig(file_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            "status": "success",
            "png_path": str(file_path),
            "recipe_id": recipe_id,
            "metrics_visualized": list(predicted_metrics.keys()),
            "predicted_values": predicted_metrics,
            "target_values": target_metrics,
        }
    
    # =========================================================================
    # 内部方法 - 折线图（趋势图）
    # =========================================================================
    def _line_chart_matplotlib(self, x_values, y_values, x_labels, x_label, y_label, title, color, file_path):
        """Matplotlib 趋势折线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 折线图
        ax.plot(x_values, y_values, 'o-', color=color, linewidth=2.5, markersize=10, label=y_label)
        
        # 添加数据点标签
        for x, y, label in zip(x_values, y_values, x_labels):
            ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), textcoords="offset points",
                       ha='center', fontsize=10, fontweight='bold')
        
        # 如果x_labels有意义，使用它们作为刻度
        if x_labels and len(set(x_labels)) == len(x_labels):
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 添加趋势分析
        trend_info = None
        if len(y_values) >= 2:
            trend = "increasing" if y_values[-1] > y_values[0] else "decreasing"
            change = ((y_values[-1] - y_values[0]) / y_values[0] * 100) if y_values[0] != 0 else 0
            trend_info = f'{trend} ({change:+.1f}%)'
            ax.text(0.02, 0.98, f'Trend: {trend_info}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            "status": "success",
            "chart_type": "line_chart",
            "file_path": str(file_path),
            "title": title,
            "data_points": len(y_values),
            "x_values": x_values,
            "y_values": y_values,
            "trend": trend_info,
        }
    
    def _line_chart_plotly(self, x_values, y_values, x_labels, x_label, y_label, title, color, file_path):
        """Plotly 趋势折线图"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode='lines+markers+text',
            line=dict(color=color, width=3), marker=dict(size=12),
            text=[f'{v:.2f}' for v in y_values], textposition='top center',
            hovertext=x_labels, name=y_label
        ))
        
        # 趋势信息
        trend_info = None
        if len(y_values) >= 2:
            trend = "increasing" if y_values[-1] > y_values[0] else "decreasing"
            change = ((y_values[-1] - y_values[0]) / y_values[0] * 100) if y_values[0] != 0 else 0
            trend_info = f'{trend} ({change:+.1f}%)'
            fig.add_annotation(
                text=f'Trend: {trend_info}',
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray"
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18), x=0.5),
            xaxis_title=x_label, yaxis_title=y_label,
            template='plotly_white', height=500, width=900
        )
        
        # 设置X轴刻度
        if x_labels:
            fig.update_xaxes(tickvals=x_values, ticktext=x_labels, tickangle=45)
        
        # 保存
        html_path = file_path.with_suffix('.html')
        fig.write_html(str(html_path))
        
        png_path = file_path.with_suffix('.png')
        try:
            fig.write_image(str(png_path), scale=2)
        except:
            png_path = None
        
        return {
            "status": "success",
            "chart_type": "line_chart",
            "html_path": str(html_path),
            "png_path": str(png_path) if png_path else None,
            "title": title,
            "data_points": len(y_values),
            "trend": trend_info,
        }
    
    # =========================================================================
    # 内部方法 - 分组条形图（多材料对比）
    # =========================================================================
    def _grouped_bar_matplotlib(self, material_names, metric_values, title, file_path):
        """Matplotlib 分组条形图"""
        fig, ax = plt.subplots(figsize=(max(10, len(material_names) * 2), 6))
        
        metrics = list(metric_values.keys())
        n_materials = len(material_names)
        n_metrics = len(metrics)
        
        x = np.arange(n_materials)
        width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            cfg = METRIC_CONFIG.get(metric, {"label": metric, "color": "#888888"})
            offset = (i - n_metrics/2 + 0.5) * width
            values = metric_values[metric]
            bars = ax.bar(x + offset, values, width, label=cfg['label'], color=cfg['color'], edgecolor='black')
            
            # 数值标签
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        ax.set_xlabel('Materials', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(material_names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            "status": "success",
            "chart_type": "grouped_bar_chart",
            "file_path": str(file_path),
            "title": title,
            "materials": material_names,
            "metrics": metrics,
        }
    
    def _grouped_bar_plotly(self, material_names, metric_values, title, file_path):
        """Plotly 分组条形图"""
        fig = go.Figure()
        
        for metric, values in metric_values.items():
            cfg = METRIC_CONFIG.get(metric, {"label": metric, "color": "#888888"})
            fig.add_trace(go.Bar(
                name=cfg['label'], x=material_names, y=values,
                marker_color=cfg['color'], text=[f'{v:.2f}' for v in values], textposition='outside'
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18), x=0.5),
            xaxis_title='Materials', yaxis_title='Value',
            barmode='group', template='plotly_white', height=500, width=900
        )
        
        # 保存
        html_path = file_path.with_suffix('.html')
        fig.write_html(str(html_path))
        
        png_path = file_path.with_suffix('.png')
        try:
            fig.write_image(str(png_path), scale=2)
        except:
            png_path = None
        
        return {
            "status": "success",
            "chart_type": "grouped_bar_chart",
            "html_path": str(html_path),
            "png_path": str(png_path) if png_path else None,
            "title": title,
            "materials": material_names,
            "metrics": list(metric_values.keys()),
        }
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    def _extract_metric_value(self, predictions: Dict, metric_key: str) -> Optional[float]:
        """
        从预测结果中提取指标值，支持多种格式
        
        支持的格式：
        - {"pce": 20.5}  直接值
        - {"pce": {"value": 20.5}}  嵌套值
        - {"PCE_percent": 20.5}  旧格式
        """
        if not predictions:
            return None
        
        # 直接查找
        if metric_key in predictions:
            val = predictions[metric_key]
            if isinstance(val, dict):
                return val.get("value")
            return val
        
        # 尝试旧格式映射
        for old_key, new_key in LEGACY_METRIC_MAP.items():
            if new_key == metric_key and old_key in predictions:
                val = predictions[old_key]
                if isinstance(val, dict):
                    return val.get("value")
                return val
        
        return None


# =============================================================================
# 便捷函数接口 - 供 FabAgent 调用
# =============================================================================

def visualize_prediction_results(
    predicted_metrics: dict[str, float],
    target_metrics: dict[str, float] | None = None,
    recipe_id: str = "Design_Batch",
    output_dir: str = "src/test/prediction_results",
) -> dict[str, Any]:
    """
    单材料预测结果条形图 - 保持向后兼容
    
    Args:
        predicted_metrics: Dict with keys like PCE_percent, Voc_V, Jsc_mA_cm2, FF_percent, T80_hours
        target_metrics: Optional target values for comparison
        recipe_id: Identifier for this prediction batch
        output_dir: Directory to save outputs
        
    Returns:
        Dict with visualization paths and status
    """
    visualizer = PredictionVisualizer(output_dir=output_dir)
    return visualizer.visualize_predictions(
        predicted_metrics=predicted_metrics,
        target_metrics=target_metrics,
        recipe_id=recipe_id,
    )


def visualize_series_trend(
    series_data: List[Dict[str, Any]],
    x_label: str = "Composition Parameter",
    y_metric: str = "pce",
    title: str = "Property Trend",
    output_dir: str = "src/test/prediction_results",
) -> Dict[str, Any]:
    """
    系列材料趋势图（折线图）- 新增功能
    
    用于可视化某个性能指标随组分变化的趋势
    
    Args:
        series_data: 系列数据，格式：
            [
                {"x_value": 0.0, "x_label": "FAPbI3", "predictions": {"pce": {"value": 20.1}}},
                {"x_value": 0.1, "x_label": "FA0.9Cs0.1PbI3", "predictions": {"pce": {"value": 21.2}}},
                ...
            ]
        x_label: X轴标题
        y_metric: Y轴指标 (pce, voc, jsc, ff, dft_band_gap, energy_above_hull)
        title: 图表标题
        output_dir: 输出目录
        
    Returns:
        包含文件路径和状态的字典
    """
    visualizer = PredictionVisualizer(output_dir=output_dir)
    return visualizer.visualize_series_trend(
        series_data=series_data,
        x_label=x_label,
        y_metric=y_metric,
        title=title,
    )


def visualize_comparison(
    materials_data: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: str = "Materials Comparison",
    output_dir: str = "src/test/prediction_results",
) -> Dict[str, Any]:
    """
    多材料对比条形图（分组）- 新增功能
    
    用于对比多种材料的性能
    
    Args:
        materials_data: 材料数据，格式：
            [
                {"name": "MAPbI3", "predictions": {"pce": {"value": 19.1}, "voc": {"value": 1.05}}},
                {"name": "FAPbI3", "predictions": {"pce": {"value": 20.5}, "voc": {"value": 1.08}}},
                ...
            ]
        metrics: 要对比的指标列表，默认 ["pce", "voc", "jsc", "ff"]
        title: 图表标题
        output_dir: 输出目录
        
    Returns:
        包含文件路径和状态的字典
    """
    visualizer = PredictionVisualizer(output_dir=output_dir)
    return visualizer.visualize_comparison(
        materials_data=materials_data,
        metrics=metrics,
        title=title,
    )


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced Visualization Tools")
    print("=" * 60)
    
    # 测试1: 单材料条形图
    print("\n1. Testing single prediction bar chart...")
    result1 = visualize_prediction_results(
        predicted_metrics={"PCE_percent": 22.5, "Voc_V": 1.15, "Jsc_mA_cm2": 24.3, "FF_percent": 80.5},
        target_metrics={"PCE_percent": 25.0, "Voc_V": 1.2},
        recipe_id="Test_MAPbI3",
        output_dir="test_output"
    )
    print(f"   Status: {result1.get('status')}")
    print(f"   File: {result1.get('png_path') or result1.get('html_path')}")
    
    # 测试2: 趋势图 - 这是Q030-Q035需要的功能
    print("\n2. Testing series trend line chart (NEW)...")
    series_data = [
        {"x_value": 0.0, "x_label": "FAPbI3", "predictions": {"pce": {"value": 20.1}, "dft_band_gap": {"value": 1.52}}},
        {"x_value": 0.1, "x_label": "FA0.9Cs0.1PbI3", "predictions": {"pce": {"value": 21.2}, "dft_band_gap": {"value": 1.55}}},
        {"x_value": 0.2, "x_label": "FA0.8Cs0.2PbI3", "predictions": {"pce": {"value": 21.8}, "dft_band_gap": {"value": 1.58}}},
        {"x_value": 0.3, "x_label": "FA0.7Cs0.3PbI3", "predictions": {"pce": {"value": 21.5}, "dft_band_gap": {"value": 1.61}}},
    ]
    result2 = visualize_series_trend(
        series_data=series_data,
        x_label="Cs Content (x)",
        y_metric="pce",
        title="PCE vs Cs Content in FA(1-x)Cs(x)PbI3",
        output_dir="test_output"
    )
    print(f"   Status: {result2.get('status')}")
    print(f"   File: {result2.get('file_path') or result2.get('png_path')}")
    print(f"   Trend: {result2.get('trend')}")
    
    # 测试3: 多材料对比
    print("\n3. Testing materials comparison chart (NEW)...")
    materials_data = [
        {"name": "MAPbI3", "predictions": {"pce": {"value": 19.1}, "voc": {"value": 1.05}, "jsc": {"value": 22.5}, "ff": {"value": 78.0}}},
        {"name": "FAPbI3", "predictions": {"pce": {"value": 20.5}, "voc": {"value": 1.08}, "jsc": {"value": 24.1}, "ff": {"value": 79.5}}},
        {"name": "CsPbI3", "predictions": {"pce": {"value": 18.2}, "voc": {"value": 1.02}, "jsc": {"value": 21.8}, "ff": {"value": 76.5}}},
    ]
    result3 = visualize_comparison(
        materials_data=materials_data,
        metrics=["pce", "voc", "jsc", "ff"],
        title="PCE Comparison: MAPbI3 vs FAPbI3 vs CsPbI3",
        output_dir="test_output"
    )
    print(f"   Status: {result3.get('status')}")
    print(f"   File: {result3.get('file_path') or result3.get('png_path')}")
    print(f"   Materials: {result3.get('materials')}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
