from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QRectF, QLineF, QSettings
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPainterPath, QPixmap, QPen, QPdfWriter, QPageSize, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
    QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QDoubleSpinBox, QToolTip,
    QScrollArea, QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from sem_metrology.calibration import Calibration, detect_footer, detect_scale_bar, read_sem_companion_metadata
from sem_metrology.io import to_gray
from sem_metrology.lines import analyze_line_space, estimate_line_direction
from sem_metrology.overlay import circle_overlay, cross_section_overlay, line_space_overlay, oriented_line_space_overlay, rectangular_overlay, shape_overlay
from sem_metrology.reporting import measurement_statistics
from sem_metrology.roughness import spatial_descriptors
from sem_metrology.shapes import analyze_circular_outlines, analyze_rectangular_features, analyze_shapes, infer_lattice_periods, population_summary
from sem_metrology.models import Measurement
from sem_metrology.measurements import measure_cross_section, measure_layer_stack
from sem_metrology.populations import infer_feature_classes
from sem_metrology.profile import measure_lateral_profile
from sem_metrology.studies import common_study_metrics, dose_response_summary
from sem_metrology.waveguide import analyze_waveguide, waveguide_overlay

REFERENCE_TAUGHT_RECIPES = frozenset((0, 1, 2, 4))
APP_VERSION = "1.2.0"

APP_STYLE = """
QMainWindow, QWidget { background: #10151c; color: #dce5ee; font-family: 'Segoe UI'; font-size: 10pt; }
QToolBar { background: #171e27; border-bottom: 1px solid #2d3947; spacing: 4px; padding: 3px; }
QToolButton, QPushButton { background: #243141; border: 1px solid #3a4b5e; border-radius: 4px; padding: 7px 12px; }
QToolButton:hover, QPushButton:hover { background: #304258; }
QPushButton#runButton { background: #007f86; border-color: #19a4aa; font-weight: 600; }
QGroupBox { border: 1px solid #2e3a47; border-radius: 5px; margin-top: 10px; padding-top: 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #8fd8dc; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #18212b; border: 1px solid #394958; border-radius: 3px; padding: 5px; }
QTableWidget { background: #111820; alternate-background-color: #17212b; gridline-color: #2d3945; border: 0; }
QHeaderView::section { background: #202c38; color: #c9d6e2; border: 0; border-right: 1px solid #354454; padding: 6px; font-weight: 600; }
QTabWidget::pane { border: 1px solid #2e3a47; }
QTabBar::tab { background: #18222c; padding: 8px 16px; border: 1px solid #2e3a47; }
QTabBar::tab:selected { background: #24404a; color: #9ee7e8; }
QLabel#metricValue { font-size: 17pt; font-weight: 650; color: #f1f6fa; }
QLabel#metricName { color: #8fa1b2; }
QFrame#metricCard { background: #17212b; border: 1px solid #2e3a47; border-radius: 5px; }
QToolButton#sectionHeader { background: #19232e; border: 1px solid #2e3a47; border-radius: 4px; padding: 7px; text-align: left; color: #8fd8dc; font-weight: 600; }
QToolButton#sectionHeader:checked { background: #22303d; }
QLabel#panelTitle { color: #dce5ee; font-size: 11pt; font-weight: 650; padding: 3px 2px; }
QLabel#qaGood { color: #65d38e; font-weight: 650; }
QLabel#qaWarn { color: #f0ba59; font-weight: 650; }
QLabel#qaBad { color: #ef7474; font-weight: 650; }
"""


def array_to_pixmap(rgb: np.ndarray) -> QPixmap:
    data = np.ascontiguousarray(rgb)
    image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def bundled_path(relative: str) -> Path:
    root=Path(getattr(sys,"_MEIPASS",Path(__file__).resolve().parent))
    return root/relative


class DocumentationDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent); self.setWindowTitle(f"SEM Metrology Workbench {APP_VERSION} — Tutorials and troubleshooting"); self.resize(980,760)
        layout=QVBoxLayout(self); browser=QTextBrowser(); browser.setOpenExternalLinks(True)
        path=bundled_path("docs/user_guide.html")
        try: browser.setHtml(path.read_text(encoding="utf-8"))
        except OSError as exc: browser.setHtml(f"<h2>Documentation unavailable</h2><p>{exc}</p>")
        layout.addWidget(browser); buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); buttons.rejected.connect(self.reject); layout.addWidget(buttons)


class ImageView(QGraphicsView):
    referenceSelected = Signal(tuple)
    analysisSelected = Signal(tuple)
    analysisMaskSelected = Signal(object)
    lineSelected = Signal(tuple)
    interfacePointSelected = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#090d12"))
        self.teach_mode = False; self.analysis_mode = False; self.mask_mode=False; self.line_mode=False; self.interface_mode=False; self._teach_start = None; self._teach_rect = None; self._analysis_start = None; self._analysis_rect = None; self._mask_points=[]; self._mask_item=None; self._line_start=None; self._line_item=None

    def set_rgb(self, rgb: np.ndarray):
        self.item.setPixmap(array_to_pixmap(rgb))
        self.scene().setSceneRect(self.item.boundingRect())
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def begin_reference_selection(self):
        self.teach_mode = True; self.setCursor(Qt.CursorShape.CrossCursor)

    def begin_analysis_selection(self):
        if self._mask_item is not None: self.scene().removeItem(self._mask_item); self._mask_item=None
        self.analysis_mode = True; self.setCursor(Qt.CursorShape.CrossCursor)

    def begin_mask_selection(self):
        if self._analysis_rect is not None: self.scene().removeItem(self._analysis_rect); self._analysis_rect=None
        self.mask_mode=True; self.setCursor(Qt.CursorShape.CrossCursor)

    def begin_line_selection(self):
        self.line_mode=True; self.setCursor(Qt.CursorShape.CrossCursor)

    def begin_interface_edit(self):
        self.interface_mode=True; self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if self.interface_mode and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(event.position().toPoint())
            if self.item.boundingRect().contains(point):
                self.interfacePointSelected.emit((round(point.x()),round(point.y())))
            event.accept(); return
        if self.line_mode and event.button()==Qt.MouseButton.LeftButton:
            self._line_start=self.mapToScene(event.position().toPoint())
            if self._line_item is not None: self.scene().removeItem(self._line_item)
            self._line_item=QGraphicsLineItem(QLineF(self._line_start,self._line_start)); self._line_item.setPen(QPen(QColor("#49f5ff"),3)); self._line_item.setZValue(22); self.scene().addItem(self._line_item); event.accept(); return
        if self.mask_mode and event.button()==Qt.MouseButton.LeftButton:
            point=self.mapToScene(event.position().toPoint()); self._mask_points=[point]
            if self._mask_item is not None: self.scene().removeItem(self._mask_item)
            self._mask_item=QGraphicsPathItem(); self._mask_item.setPen(QPen(QColor("#ffb347"),2,Qt.PenStyle.DashLine)); self._mask_item.setZValue(19); self.scene().addItem(self._mask_item); event.accept(); return
        if self.analysis_mode and event.button() == Qt.MouseButton.LeftButton:
            self._analysis_start = self.mapToScene(event.position().toPoint())
            if self._analysis_rect is not None: self.scene().removeItem(self._analysis_rect)
            self._analysis_rect = QGraphicsRectItem(QRectF(self._analysis_start,self._analysis_start)); self._analysis_rect.setPen(QPen(QColor("#ffd34d"),2,Qt.PenStyle.DashLine)); self.scene().addItem(self._analysis_rect)
            event.accept(); return
        if self.teach_mode and event.button() == Qt.MouseButton.LeftButton:
            self._teach_start = self.mapToScene(event.position().toPoint())
            if self._teach_rect is not None: self.scene().removeItem(self._teach_rect)
            self._teach_rect = QGraphicsRectItem(QRectF(self._teach_start, self._teach_start))
            self._teach_rect.setPen(QPen(QColor("#00f0ff"), 2, Qt.PenStyle.DashLine)); self.scene().addItem(self._teach_rect)
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.line_mode and self._line_start is not None:
            self._line_item.setLine(QLineF(self._line_start,self.mapToScene(event.position().toPoint()))); event.accept(); return
        if self.mask_mode and self._mask_points:
            point=self.mapToScene(event.position().toPoint()); self._mask_points.append(point); path=QPainterPath(self._mask_points[0])
            for p in self._mask_points[1:]: path.lineTo(p)
            self._mask_item.setPath(path); event.accept(); return
        if self.analysis_mode and self._analysis_start is not None:
            end=self.mapToScene(event.position().toPoint()); self._analysis_rect.setRect(QRectF(self._analysis_start,end).normalized()); event.accept(); return
        if self.teach_mode and self._teach_start is not None:
            end = self.mapToScene(event.position().toPoint()); self._teach_rect.setRect(QRectF(self._teach_start, end).normalized())
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.line_mode and self._line_start is not None:
            end=self.mapToScene(event.position().toPoint()); start=self._line_start; self._line_start=None; self.line_mode=False; self.unsetCursor()
            bounds=self.item.boundingRect(); start=bounds.topLeft() if not bounds.contains(start) else start; end=bounds.bottomRight() if not bounds.contains(end) else end
            self._line_item.setLine(QLineF(start,end))
            if QLineF(start,end).length()>=3: self.lineSelected.emit((start.x(),start.y(),end.x(),end.y()))
            event.accept(); return
        if self.mask_mode and self._mask_points:
            if len(self._mask_points)>=3:
                path=QPainterPath(self._mask_points[0])
                for p in self._mask_points[1:]: path.lineTo(p)
                path.closeSubpath(); self._mask_item.setPath(path)
                self.analysisMaskSelected.emit([(round(p.x()),round(p.y())) for p in self._mask_points])
            self._mask_points=[]; self.mask_mode=False; self.unsetCursor(); event.accept(); return
        if self.analysis_mode and self._analysis_start is not None:
            rect=self._analysis_rect.rect().intersected(self.item.boundingRect()); self._analysis_start=None; self.analysis_mode=False; self.unsetCursor()
            if rect.width()>=12 and rect.height()>=12:
                self._analysis_rect.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable,True); self._analysis_rect.setZValue(19)
                self.analysisSelected.emit((round(rect.x()),round(rect.y()),round(rect.width()),round(rect.height())))
            event.accept(); return
        if self.teach_mode and self._teach_start is not None:
            rect = self._teach_rect.rect().intersected(self.item.boundingRect())
            self._teach_start = None; self.teach_mode = False; self.unsetCursor()
            if rect.width() >= 12 and rect.height() >= 12:
                self._teach_rect.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
                self._teach_rect.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
                self._teach_rect.setZValue(20)
                self.referenceSelected.emit((round(rect.x()), round(rect.y()), round(rect.width()), round(rect.height())))
            event.accept(); return
        super().mouseReleaseEvent(event)

    def reference_geometry(self):
        if self._teach_rect is None: return None
        rect = self._teach_rect.sceneBoundingRect().intersected(self.item.boundingRect())
        return (round(rect.x()), round(rect.y()), round(rect.width()), round(rect.height()))

    def clear_reference(self):
        if self._teach_rect is not None:
            self.scene().removeItem(self._teach_rect); self._teach_rect = None
        self._teach_start = None; self.teach_mode = False; self.unsetCursor()

    def clear_analysis_region(self):
        if self._analysis_rect is not None: self.scene().removeItem(self._analysis_rect); self._analysis_rect=None
        if self._mask_item is not None: self.scene().removeItem(self._mask_item); self._mask_item=None
        if self._line_item is not None: self.scene().removeItem(self._line_item); self._line_item=None
        self._analysis_start=None; self.analysis_mode=False; self.mask_mode=False; self.line_mode=False; self.interface_mode=False; self._line_start=None; self._mask_points=[]; self.unsetCursor()


class MetricCard(QFrame):
    def __init__(self, name: str):
        super().__init__(); self.setObjectName("metricCard")
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 8, 12, 8)
        self.name = QLabel(name); self.name.setObjectName("metricName")
        self.value = QLabel("—"); self.value.setObjectName("metricValue")
        layout.addWidget(self.name); layout.addWidget(self.value)


class CollapsibleSection(QWidget):
    """Compact settings section suited to short laptop displays."""
    def __init__(self, title: str, content: QWidget, expanded: bool = True):
        super().__init__()
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(3)
        self.header=QToolButton(); self.header.setObjectName("sectionHeader"); self.header.setCheckable(True); self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly); self.content=content
        if isinstance(content,QGroupBox): content.setTitle("")
        self.header.toggled.connect(self._toggle); layout.addWidget(self.header); layout.addWidget(content)
        self.title=title; self._toggle(expanded)

    def _toggle(self, expanded: bool):
        self.content.setVisible(expanded); self.header.setText(("▼  " if expanded else "▶  ")+self.title)


class DistributionPlot(QWidget):
    """Dependency-free metrology histogram and individuals/control plot."""
    def __init__(self):
        super().__init__(); self.values = np.array([]); self.metric_name = "No results"; self.point_info=[]; self._point_positions=[]; self._hover_index=None
        self.setMinimumHeight(245)
        self.setMouseTracking(True)

    def set_values(self, values, metric, point_info=None):
        raw=np.asarray(values,dtype=float); finite=np.isfinite(raw)
        self.values=raw[finite]; supplied=list(point_info or [{} for _ in raw])
        if len(supplied)<len(raw): supplied += [{} for _ in range(len(raw)-len(supplied))]
        self.point_info=[supplied[index] for index in np.flatnonzero(finite)]
        self.metric_name=metric; self._point_positions=[]; self._hover_index=None; self.update()

    def mouseMoveEvent(self,event):
        if not self._point_positions:
            QToolTip.hideText(); return
        point=event.position(); distances=[(point.x()-x)**2+(point.y()-y)**2 for x,y in self._point_positions]
        index=int(np.argmin(distances))
        if distances[index]>100:
            if self._hover_index is not None: self._hover_index=None; self.update()
            QToolTip.hideText(); return
        self._hover_index=index; self.update()
        info=self.point_info[index] if index<len(self.point_info) else {}
        lines=[str(info.get("label",f"Point {index+1}")),f"{self.metric_name}: {self.values[index]:.6g}"]
        if info.get("class"): lines.append(str(info["class"]))
        if info.get("details"): lines.append(str(info["details"]))
        QToolTip.showText(event.globalPosition().toPoint(),"\n".join(lines),self)

    def leaveEvent(self,event):
        self._hover_index=None; self.update(); QToolTip.hideText(); super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#0c1218")); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#dce5ee")); p.drawText(14, 22, f"{self.metric_name} — distribution and individuals")
        if not len(self.values): p.setPen(QColor("#8fa1b2")); p.drawText(14, 48, "Run an analysis to populate statistical plots."); return
        w, h = self.width(), self.height(); mid = w // 2; top, bottom = 38, h - 28
        values = self.values; lo, hi = float(values.min()), float(values.max())
        if hi <= lo: lo -= 0.5; hi += 0.5
        bins = min(16, max(5, int(np.sqrt(len(values)) + 1))); counts, edges = np.histogram(values, bins=bins)
        p.setPen(QColor("#40505f")); p.drawRect(12, top, mid - 24, bottom - top)
        barw = (mid - 28) / bins; maxc = max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            bh = (bottom - top - 12) * count / maxc; p.fillRect(int(15 + i * barw), int(bottom - bh), max(1, int(barw - 2)), int(bh), QColor("#18a6ad"))
        p.setPen(QColor("#9fb0bf")); p.drawText(14, h - 8, f"{lo:.2f}"); p.drawText(mid - 92, h - 8, f"{hi:.2f}")
        left, right = mid + 18, w - 14; p.setPen(QColor("#40505f")); p.drawRect(left, top, right-left, bottom-top)
        mean = float(values.mean()); sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        plo, phi = min(lo, mean - 3.5*sd), max(hi, mean + 3.5*sd)
        if phi <= plo: phi = plo + 1
        def yy(v): return int(bottom - (v-plo)/(phi-plo)*(bottom-top))
        for v, color in ((mean, "#f5d76e"), (mean+3*sd, "#ef7474"), (mean-3*sd, "#ef7474")):
            p.setPen(QPen(QColor(color), 1, Qt.PenStyle.DashLine)); p.drawLine(left, yy(v), right, yy(v))
        p.setPen(QColor("#67d8dc")); n=max(len(values)-1,1)
        pts=[]
        for i,v in enumerate(values):
            x=int(left + i*(right-left)/n); y=yy(float(v)); p.drawEllipse(x-2,y-2,4,4); pts.append((x,y))
        self._point_positions=pts
        for a,b in zip(pts,pts[1:]): p.drawLine(a[0],a[1],b[0],b[1])
        if self._hover_index is not None and self._hover_index<len(pts):
            x,y=pts[self._hover_index]; p.setPen(QPen(QColor("#ffffff"),2)); p.setBrush(QColor("#18a6ad")); p.drawEllipse(x-6,y-6,12,12)
        p.setPen(QColor("#9fb0bf")); p.drawText(left, h-8, "Object sequence"); p.drawText(right-190, 22, f"mean {mean:.2f}   3σ {3*sd:.2f}   N {len(values)}")


class StudyEntryDialog(QDialog):
    def __init__(self,parent=None,factor_name="E-beam dose",factor_unit="µC/cm²",lock_factor=False):
        super().__init__(parent); self.setWindowTitle("Add reviewed result to process study")
        form=QFormLayout(self)
        self.study_name=QLineEdit("Dose study"); self.factor_name=QLineEdit(factor_name); self.factor_unit=QLineEdit(factor_unit)
        self.factor_value=QDoubleSpinBox(); self.factor_value.setRange(-1e12,1e12); self.factor_value.setDecimals(6)
        self.sample=QLineEdit(); self.condition=QLineEdit(); self.replicate=QSpinBox(); self.replicate.setRange(1,999)
        if lock_factor: self.factor_name.setEnabled(False); self.factor_unit.setEnabled(False)
        form.addRow("Study name",self.study_name); form.addRow("Comparison factor",self.factor_name); form.addRow("Factor unit",self.factor_unit)
        form.addRow("Factor value",self.factor_value); form.addRow("Sample / field label",self.sample); form.addRow("Condition / split",self.condition); form.addRow("Replicate",self.replicate)
        note=QLabel("Dose cannot be inferred from SEM pixels. Enter the exposure value from the process record or trusted metadata."); note.setWordWrap(True); form.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def values(self):
        return {"study_name":self.study_name.text().strip() or "Process study",
                "factor_name":self.factor_name.text().strip() or "Process factor",
                "factor_unit":self.factor_unit.text().strip(),
                "factor_value":self.factor_value.value(),
                "sample":self.sample.text().strip(),"condition":self.condition.text().strip(),
                "replicate":self.replicate.value()}


class ProcessTrendPlot(QWidget):
    def __init__(self):
        super().__init__(); self.records=[]; self.metric_name=""; self.points=[]; self.hover_index=None
        self.setMinimumHeight(330); self.setMouseTracking(True)

    def set_data(self,records,metric):
        self.records=list(records); self.metric_name=metric; self.points=[]; self.hover_index=None; self.update()

    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor("#0c1218")); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#dce5ee")); p.drawText(14,22,self.metric_name or "Select a common metric")
        data=[]
        for record in self.records:
            metric=record.get("metrics",{}).get(self.metric_name,{})
            if np.isfinite(record.get("factor_value",np.nan)) and np.isfinite(metric.get("mean",np.nan)):
                data.append((record,metric))
        if not data: p.setPen(QColor("#8fa1b2")); p.drawText(14,48,"At least two reviewed image results with a common metric are required."); return
        left,right,top,bottom=72,self.width()-24,42,self.height()-54
        xs=np.asarray([item[0]["factor_value"] for item in data],float); ys=np.asarray([item[1]["mean"] for item in data],float)
        finite_std=np.asarray([item[1].get("std",np.nan) for item in data],float)
        xlo,xhi=float(xs.min()),float(xs.max()); ylo=float(np.nanmin(ys-np.nan_to_num(finite_std))); yhi=float(np.nanmax(ys+np.nan_to_num(finite_std)))
        if xhi<=xlo: xlo-=.5; xhi+=.5
        if yhi<=ylo: ylo-=.5; yhi+=.5
        xpad=.06*(xhi-xlo); ypad=.12*(yhi-ylo); xlo-=xpad; xhi+=xpad; ylo-=ypad; yhi+=ypad
        xx=lambda value:int(left+(value-xlo)/(xhi-xlo)*(right-left))
        yy=lambda value:int(bottom-(value-ylo)/(yhi-ylo)*(bottom-top))
        p.setPen(QColor("#40505f")); p.drawRect(left,top,right-left,bottom-top)
        summary=dose_response_summary(self.records,self.metric_name)
        if summary.get("n",0)>=2 and np.isfinite(summary.get("slope",np.nan)):
            xa,xb=xlo,xhi; ya=summary["slope"]*xa+summary["intercept"]; yb=summary["slope"]*xb+summary["intercept"]
            p.setPen(QPen(QColor("#f5d76e"),1,Qt.PenStyle.DashLine)); p.drawLine(xx(xa),yy(ya),xx(xb),yy(yb))
        palette=(QColor("#67d8dc"),QColor("#ffbe4a"),QColor("#7fe18e"),QColor("#d29cff"))
        conditions=[]; self.points=[]
        for index,(record,metric) in enumerate(data):
            condition=record.get("condition","") or "default"
            if condition not in conditions: conditions.append(condition)
            color=palette[conditions.index(condition)%len(palette)]; x,y=xx(record["factor_value"]),yy(metric["mean"])
            std=metric.get("std",np.nan)
            if np.isfinite(std) and std>0:
                p.setPen(QPen(color,1)); p.drawLine(x,yy(metric["mean"]-std),x,yy(metric["mean"]+std)); p.drawLine(x-4,yy(metric["mean"]-std),x+4,yy(metric["mean"]-std)); p.drawLine(x-4,yy(metric["mean"]+std),x+4,yy(metric["mean"]+std))
            p.setPen(QPen(color,2)); p.setBrush(color); p.drawEllipse(x-4,y-4,8,8); self.points.append((x,y,record,metric))
        if self.hover_index is not None and self.hover_index<len(self.points):
            x,y,_,_=self.points[self.hover_index]; p.setPen(QPen(QColor("#ffffff"),2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawEllipse(x-8,y-8,16,16)
        factor_name=data[0][0].get("factor_name","Factor"); factor_unit=data[0][0].get("factor_unit",""); unit=data[0][1].get("unit","")
        p.setPen(QColor("#9fb0bf")); p.drawText(left,bottom+28,f"{factor_name} ({factor_unit})" if factor_unit else factor_name); p.drawText(8,top-8,f"Y: {unit}")
        if summary.get("n",0)>=2 and np.isfinite(summary.get("slope",np.nan)):
            p.drawText(max(left,right-390),22,f"slope {summary['slope']:.5g} {unit}/{factor_unit or 'factor'}   R² {summary['r_squared']:.3f}   Δ {summary['change']:.4g} {unit}")

    def mouseMoveEvent(self,event):
        if not self.points: QToolTip.hideText(); return
        distances=[(event.position().x()-item[0])**2+(event.position().y()-item[1])**2 for item in self.points]; index=int(np.argmin(distances))
        if distances[index]>144:
            self.hover_index=None; self.update(); QToolTip.hideText(); return
        self.hover_index=index; self.update(); _,_,record,metric=self.points[index]
        std=metric.get("std",np.nan); spread=f" ± {std:.5g}" if np.isfinite(std) else ""
        text=(f"{record.get('sample') or Path(record.get('image','image')).name}\n"
              f"{record.get('factor_name')}: {record.get('factor_value'):.6g} {record.get('factor_unit','')}\n"
              f"{self.metric_name}: {metric.get('mean'):.6g}{spread} {metric.get('unit','')}  N={metric.get('n',0)}\n"
              f"Condition: {record.get('condition') or '—'} · replicate {record.get('replicate',1)}\n"
              f"{record.get('recipe','')} · QA {record.get('qa_status','')}")
        QToolTip.showText(event.globalPosition().toPoint(),text,self)

    def leaveEvent(self,event):
        self.hover_index=None; self.update(); QToolTip.hideText(); super().leaveEvent(event)


class ProcessStudyDialog(QDialog):
    def __init__(self,owner):
        super().__init__(owner); self.owner=owner; self.setWindowTitle("Multi-image process study"); self.resize(1250,780)
        layout=QVBoxLayout(self); controls=QHBoxLayout()
        self.metric_combo=QComboBox(); self.metric_combo.currentTextChanged.connect(self.refresh)
        for label,slot in (("Save study…",self.save_study),("Load study…",self.load_study),("Export long CSV…",self.export_csv),("Remove selected",self.remove_selected),("Clear study",self.clear_study)):
            button=QPushButton(label); button.clicked.connect(slot); controls.addWidget(button)
        controls.addWidget(QLabel("Compared metric")); controls.addWidget(self.metric_combo,1); layout.addLayout(controls)
        self.summary=QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.table=QTableWidget(); self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); layout.addWidget(self.table,2)
        self.plot=ProcessTrendPlot(); layout.addWidget(self.plot,3)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.rebuild_metrics()

    def rebuild_metrics(self):
        previous=self.metric_combo.currentText(); self.metric_combo.blockSignals(True); self.metric_combo.clear(); self.metric_combo.addItems(common_study_metrics(self.owner.study_records,2))
        index=self.metric_combo.findText(previous); self.metric_combo.setCurrentIndex(max(0,index)); self.metric_combo.blockSignals(False); self.refresh()

    def refresh(self):
        metric=self.metric_combo.currentText(); rows=[]
        for record in sorted(self.owner.study_records,key=lambda item:(item.get("factor_value",0),item.get("replicate",1))):
            data=record.get("metrics",{}).get(metric,{})
            rows.append({"Factor":record.get("factor_value"),"Unit":record.get("factor_unit",""),"Sample":record.get("sample",""),"Condition":record.get("condition",""),"Replicate":record.get("replicate",1),"Recipe":record.get("recipe",""),"QA":record.get("qa_status",""),"Mean":data.get("mean",np.nan),"Std":data.get("std",np.nan),"N":data.get("n",0),"Metric unit":data.get("unit",""),"Image":Path(record.get("image","")).name})
        self.owner._set_table(self.table,rows); self.plot.set_data(self.owner.study_records,metric)
        summary=dose_response_summary(self.owner.study_records,metric)
        if summary.get("n",0)>=2 and np.isfinite(summary.get("slope",np.nan)):
            self.summary.setText(f"{self.owner.study_name} · {summary['n']} images · slope {summary['slope']:.6g} · R² {summary['r_squared']:.3f} · endpoint change {summary['change']:.6g} ({summary.get('percent_change',np.nan):.2f}%). Error bars are within-image feature Std; repeated images remain separate study points.")
        else: self.summary.setText(f"{self.owner.study_name} · {len(self.owner.study_records)} reviewed image result(s). Select a metric present in at least two images.")

    def save_study(self):
        path,_=QFileDialog.getSaveFileName(self,"Save process study","SEM_process_study.json","JSON (*.json)")
        if path:
            with open(path,"w",encoding="utf-8") as handle: json.dump({"schema":1,"study_name":self.owner.study_name,"factor_name":self.owner.study_factor_name,"factor_unit":self.owner.study_factor_unit,"records":self.owner.study_records},handle,indent=2)

    def load_study(self):
        path,_=QFileDialog.getOpenFileName(self,"Load process study","","JSON (*.json)")
        if not path:return
        try:
            with open(path,encoding="utf-8") as handle:data=json.load(handle)
            self.owner.study_name=data.get("study_name","Process study"); self.owner.study_factor_name=data.get("factor_name","E-beam dose"); self.owner.study_factor_unit=data.get("factor_unit","µC/cm²"); self.owner.study_records=list(data.get("records",[])); self.rebuild_metrics()
        except Exception as exc: QMessageBox.critical(self,"Cannot load study",str(exc))

    def export_csv(self):
        if not self.owner.study_records:return
        path,_=QFileDialog.getSaveFileName(self,"Export long-form process study","SEM_process_study.csv","CSV (*.csv)")
        if not path:return
        rows=[]
        for record in self.owner.study_records:
            for metric,data in record.get("metrics",{}).items():
                rows.append({"study":self.owner.study_name,"factor":record.get("factor_name"),"factor_unit":record.get("factor_unit"),"factor_value":record.get("factor_value"),"sample":record.get("sample"),"condition":record.get("condition"),"replicate":record.get("replicate"),"image":record.get("image"),"recipe":record.get("recipe"),"qa":record.get("qa_status"),"metric":metric,"metric_unit":data.get("unit"),"N":data.get("n"),"mean":data.get("mean"),"std":data.get("std"),"median":data.get("median"),"cv_percent":data.get("cv")})
        with open(path,"w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def remove_selected(self):
        selected=sorted({index.row() for index in self.table.selectionModel().selectedRows()},reverse=True)
        ordered=sorted(enumerate(self.owner.study_records),key=lambda item:(item[1].get("factor_value",0),item[1].get("replicate",1)))
        for row in selected:
            if row<len(ordered): self.owner.study_records.pop(ordered[row][0])
        self.rebuild_metrics()

    def clear_study(self):
        if QMessageBox.question(self,"Clear study","Remove all stored study results?")==QMessageBox.StandardButton.Yes:
            self.owner.study_records=[]; self.rebuild_metrics()


class MetrologyWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SEM Metrology Workbench {APP_VERSION}")
        self.resize(1360, 820)
        self.setMinimumSize(960, 600)
        self.rgb = self.analysis_rgb = self.overlay_rgb = None
        self.image_path = None; self.calibration = None; self.footer_start = None; self.sem_metadata={}; self.metadata_sampling_um=None; self.source_warning=None
        self.feature_rows = []; self.measurements = []; self.reference_roi = None; self.analysis_roi = None; self.analysis_polygon=None; self.exclusion_polygon=None; self.plot_series = {}; self.plot_metadata={}; self.pixel_mode=False; self.cross_section_records=[]; self.cross_section_overlays=[]
        self.layer_candidate_interfaces=[]; self.layer_active_interfaces=[]; self.layer_roi=None; self.class_model=None; self._drawing_scale_bar=False
        self.study_records=[]; self.study_name="Dose study"; self.study_factor_name="E-beam dose"; self.study_factor_unit="µC/cm²"
        self._build_ui()

    def _build_ui(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open SEM image…", self); open_action.setShortcut("Ctrl+O"); open_action.triggered.connect(self.open_image)
        export_action = QAction("Export &report CSV…", self); export_action.setShortcut("Ctrl+E"); export_action.triggered.connect(self.export_report)
        export_plot_action = QAction("Export current &plot PNG…", self); export_plot_action.triggered.connect(self.export_plot)
        export_pdf_action=QAction("Export professional PDF report…",self); export_pdf_action.triggered.connect(self.export_professional_report)
        add_study_action=QAction("Add reviewed result to process study…",self); add_study_action.triggered.connect(self.add_current_to_study)
        open_study_action=QAction("Open process study…",self); open_study_action.triggered.connect(self.open_process_study)
        save_recipe_action=QAction("Save measurement recipe…",self); save_recipe_action.triggered.connect(self.save_recipe)
        load_recipe_action=QAction("Load measurement recipe…",self); load_recipe_action.triggered.connect(self.load_recipe)
        exit_action = QAction("E&xit", self); exit_action.setShortcut("Alt+F4"); exit_action.triggered.connect(self.close)
        file_menu.addActions([open_action, export_action, export_plot_action,export_pdf_action,save_recipe_action,load_recipe_action]); file_menu.addSeparator(); file_menu.addActions([add_study_action,open_study_action]); file_menu.addSeparator(); file_menu.addAction(exit_action)
        edit_menu = self.menuBar().addMenu("&Edit")
        reset_action = QAction("&Reset identification", self); reset_action.setShortcut("Ctrl+R"); reset_action.triggered.connect(self.reset_analysis)
        redraw_action = QAction("&Redraw reference feature", self); redraw_action.triggered.connect(self._begin_teach)
        apply_reference_action = QAction("&Apply moved reference", self); apply_reference_action.triggered.connect(self._apply_adjusted_reference)
        remove_cross_action=QAction("Remove last cross-section",self); remove_cross_action.triggered.connect(self._remove_last_cross_section)
        clear_cross_action=QAction("Clear cross-section set",self); clear_cross_action.triggered.connect(self._clear_cross_section_set)
        edit_menu.addActions([reset_action, redraw_action, apply_reference_action,remove_cross_action,clear_cross_action])
        view_menu = self.menuBar().addMenu("&View")
        fit_action = QAction("&Fit image", self); fit_action.setShortcut("F"); fit_action.triggered.connect(lambda: self.image_view.fitInView(self.image_view.item, Qt.AspectRatioMode.KeepAspectRatio))
        plots_action = QAction("Show &distribution plots", self); plots_action.setShortcut("Ctrl+P"); plots_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        view_menu.addActions([fit_action, plots_action])
        help_menu = self.menuBar().addMenu("&Help")
        tutorials_action=QAction("Complete &tutorials and troubleshooting",self); tutorials_action.setShortcut("F1"); tutorials_action.triggered.connect(lambda: DocumentationDialog(self).exec())
        workflow_action = QAction("Measurement &workflow", self); workflow_action.triggered.connect(self.show_help)
        texture_action = QAction("Surface texture (&Ra/Sa)", self); texture_action.triggered.connect(self.run_surface_texture)
        line_gauge_action = QAction("Draw &line gauge", self); line_gauge_action.triggered.connect(self._begin_line_gauge)
        capabilities_action = QAction("Supported &structures",self); capabilities_action.triggered.connect(self.show_capabilities)
        definitions_action = QAction("Measurement &definitions",self); definitions_action.triggered.connect(self.show_definitions)
        cross_definitions_action=QAction("Cross-section definitions",self); cross_definitions_action.triggered.connect(self.show_cross_section_definitions)
        study_help_action=QAction("Process and dose studies",self); study_help_action.triggered.connect(self.show_study_help)
        about_action = QAction("&About / cite", self); about_action.triggered.connect(self.show_about)
        help_menu.addActions([tutorials_action,workflow_action,capabilities_action,definitions_action,cross_definitions_action,study_help_action,about_action])

        toolbar = QToolBar("Main"); toolbar.setMovable(False); toolbar.setFloatable(False); self.addToolBar(toolbar)
        toolbar.addAction(open_action); toolbar.addAction(reset_action); toolbar.addAction(fit_action); toolbar.addSeparator(); toolbar.addAction(workflow_action); toolbar.addAction(texture_action); toolbar.addAction(line_gauge_action)
        toolbar.addSeparator(); toolbar.addWidget(QLabel("Settings"))
        self.settings_width_slider=QSlider(Qt.Orientation.Horizontal); self.settings_width_slider.setRange(220,420); self.settings_width_slider.setValue(290); self.settings_width_slider.setFixedWidth(75); toolbar.addWidget(self.settings_width_slider)
        toolbar.addWidget(QLabel("Diagnostics")); self.diagnostics_width_slider=QSlider(Qt.Orientation.Horizontal); self.diagnostics_width_slider.setRange(220,420); self.diagnostics_width_slider.setValue(290); self.diagnostics_width_slider.setFixedWidth(75); toolbar.addWidget(self.diagnostics_width_slider)
        self.reset_widths_button=QPushButton("Reset widths"); toolbar.addWidget(self.reset_widths_button)

        root = QWidget(); root_layout = QVBoxLayout(root); root_layout.setContentsMargins(5, 5, 5, 5)
        self.image_view = ImageView(); self.image_view.referenceSelected.connect(self._reference_selected); self.image_view.analysisSelected.connect(self._analysis_selected); self.image_view.analysisMaskSelected.connect(self._analysis_mask_selected); self.image_view.lineSelected.connect(self._line_selected); self.image_view.interfacePointSelected.connect(self._layer_interface_clicked)
        controls = QWidget(); controls_layout = QVBoxLayout(controls)
        settings_title=QLabel("1 · Measurement settings"); settings_title.setObjectName("panelTitle"); controls_layout.addWidget(settings_title)

        calibration_group = QGroupBox("Calibration & traceability"); cal_form = QFormLayout(calibration_group)
        self.scale_length = QDoubleSpinBox(); self.scale_length.setRange(0.001, 1_000_000); self.scale_length.setValue(200); self.scale_length.setDecimals(3)
        self.scale_unit = QComboBox(); self.scale_unit.addItems(["nm", "µm"])
        self.scale_length.valueChanged.connect(self._refresh_base_display); self.scale_unit.currentIndexChanged.connect(self._refresh_base_display)
        scale_row = QWidget(); scale_layout = QHBoxLayout(scale_row); scale_layout.setContentsMargins(0,0,0,0); scale_layout.addWidget(self.scale_length); scale_layout.addWidget(self.scale_unit)
        self.detected_bar = QLabel("No image loaded"); self.sampling_label = QLabel("—")
        self.confirm_calibration = QCheckBox("Calibration source reviewed and confirmed")
        self.allow_pixel_mode=QCheckBox("Allow original-image pixel units when scale is unavailable")
        self.redraw_scale_button=QPushButton("Redraw scale-bar endpoints"); self.redraw_scale_button.clicked.connect(self._begin_scale_bar_redraw)
        cal_form.addRow("Scale-bar value", scale_row); cal_form.addRow("Detected bar", self.detected_bar); cal_form.addRow(self.redraw_scale_button); cal_form.addRow("Sampling", self.sampling_label); cal_form.addRow(self.confirm_calibration); cal_form.addRow(self.allow_pixel_mode)
        controls_layout.addWidget(CollapsibleSection("Calibration & traceability", calibration_group, True))

        recipe_group = QGroupBox("Physical measurement model"); recipe_form = QFormLayout(recipe_group)
        self.recipe = QComboBox(); self.recipe.addItems(["Circular outlines / holes", "Rectangular / elongated periodic array", "Irregular particles / grains", "Line-space / grating", "Single feature / contour ROI", "Curved / tapered waveguide ROI", "Cross-section feature ROI", "Layer stack / film cross-section ROI"])
        self.recipe.currentIndexChanged.connect(self._recipe_changed)
        self.min_size = QDoubleSpinBox(); self.min_size.setRange(1, 1_000_000); self.min_size.setValue(100); self.min_size.setSuffix(" nm")
        self.max_size = QDoubleSpinBox(); self.max_size.setRange(2, 1_000_000); self.max_size.setValue(1000); self.max_size.setSuffix(" nm")
        self.sensitivity = QDoubleSpinBox(); self.sensitivity.setRange(0.50, 0.98); self.sensitivity.setValue(0.82); self.sensitivity.setSingleStep(0.01)
        self.class_mode=QComboBox(); self.class_mode.addItems(["Auto population model","Treat as one population","Expect 2 classes","Expect 3 classes"])
        self.confirm_classes=QCheckBox("Class assignments reviewed and confirmed")
        self.line_orientation = QComboBox(); self.line_orientation.addItems(["Auto", "Vertical lines", "Horizontal lines"])
        self.line_polarity = QComboBox(); self.line_polarity.addItems(["Auto", "Bright lines", "Dark lines"])
        self.period_x_hint = QDoubleSpinBox(); self.period_y_hint = QDoubleSpinBox()
        for control in (self.period_x_hint, self.period_y_hint): control.setRange(0,1_000_000); control.setDecimals(2); control.setSpecialValueText("Auto"); control.setSuffix(" nm")
        self.layer_period_count=QSpinBox(); self.layer_period_count.setRange(0,20); self.layer_period_count.setSpecialValueText("Auto"); self.layer_period_count.setSuffix(" periods")
        self.reject_border = QCheckBox("Reject partial border features"); self.reject_border.setChecked(True)
        self.recommendation_label = QLabel("Model recommendation appears after loading"); self.recommendation_label.setWordWrap(True)
        recipe_form.addRow("Recipe", self.recipe); recipe_form.addRow("Recommendation", self.recommendation_label); recipe_form.addRow("Line orientation", self.line_orientation); recipe_form.addRow("Feature / line polarity", self.line_polarity); recipe_form.addRow("Expected period X", self.period_x_hint); recipe_form.addRow("Expected period Y", self.period_y_hint); recipe_form.addRow("Periods spanned by layer ROI",self.layer_period_count); recipe_form.addRow("Population classes",self.class_mode); recipe_form.addRow(self.confirm_classes); recipe_form.addRow("Minimum credible size", self.min_size); recipe_form.addRow("Maximum credible size", self.max_size); recipe_form.addRow("Detection sensitivity", self.sensitivity); recipe_form.addRow(self.reject_border)
        self.teach_button = QPushButton("Draw one reference feature"); self.teach_button.clicked.connect(self._begin_teach)
        self.apply_reference_button = QPushButton("Apply moved reference"); self.apply_reference_button.clicked.connect(self._apply_adjusted_reference)
        self.reference_label = QLabel("No reference taught — automatic proposal mode"); self.reference_label.setWordWrap(True)
        teach_row = QWidget(); teach_layout = QHBoxLayout(teach_row); teach_layout.setContentsMargins(0,0,0,0); teach_layout.addWidget(self.teach_button); teach_layout.addWidget(self.apply_reference_button)
        recipe_form.addRow("Operator guidance", teach_row); recipe_form.addRow("Reference", self.reference_label)
        self.interface_selector=QComboBox(); self.interface_selector.setToolTip("Select a numbered interface as a reliable alternative to clicking its line.")
        self.edit_interfaces_button=QPushButton("Enable click-to-remove"); self.edit_interfaces_button.clicked.connect(self._begin_layer_interface_edit)
        self.remove_interface_button=QPushButton("Remove selected interface"); self.remove_interface_button.clicked.connect(self._remove_selected_layer_interface)
        self.restore_interfaces_button=QPushButton("Restore all proposed interfaces"); self.restore_interfaces_button.clicked.connect(self._restore_layer_interfaces)
        self.interface_edit_status=QLabel("Run layer analysis to review interfaces"); self.interface_edit_status.setWordWrap(True)
        interface_row=QWidget(); interface_layout=QVBoxLayout(interface_row); interface_layout.setContentsMargins(0,0,0,0)
        interface_layout.addWidget(self.interface_selector); interface_layout.addWidget(self.edit_interfaces_button); interface_layout.addWidget(self.remove_interface_button); interface_layout.addWidget(self.restore_interfaces_button); interface_layout.addWidget(self.interface_edit_status)
        recipe_form.addRow("Layer review",interface_row)
        self.analysis_region_button=QPushButton("Draw rectangular analysis region"); self.analysis_region_button.clicked.connect(self.image_view.begin_analysis_selection)
        self.analysis_mask_button=QPushButton("Draw freehand mask"); self.analysis_mask_button.clicked.connect(self.image_view.begin_mask_selection)
        self.mask_behavior=QComboBox(); self.mask_behavior.addItems(["Include inside mask","Exclude inside mask"])
        self.full_region_button=QPushButton("Use full specimen area"); self.full_region_button.clicked.connect(self._clear_analysis_region)
        region_row=QWidget(); region_layout=QHBoxLayout(region_row); region_layout.setContentsMargins(0,0,0,0); region_layout.addWidget(self.analysis_region_button); region_layout.addWidget(self.analysis_mask_button); region_layout.addWidget(self.mask_behavior); region_layout.addWidget(self.full_region_button)
        self.analysis_region_label=QLabel("Full specimen area"); recipe_form.addRow("Analysis region",region_row); recipe_form.addRow("Region status",self.analysis_region_label)
        controls_layout.addWidget(CollapsibleSection("Measurement settings", recipe_group, True))

        qa_group = QGroupBox("Measurement release gate"); qa_layout = QVBoxLayout(qa_group)
        self.qa_label = QLabel("WAITING — load and calibrate an image"); self.qa_label.setObjectName("qaWarn"); self.qa_label.setWordWrap(True)
        self.qa_detail = QLabel("Results are released only when calibration, sampling, population, and model-fit checks pass."); self.qa_detail.setWordWrap(True)
        self.model_warning_label = QLabel("Model compatibility not yet evaluated"); self.model_warning_label.setWordWrap(True)
        qa_layout.addWidget(self.qa_label); qa_layout.addWidget(self.qa_detail)
        model_group=QGroupBox("Model guidance"); model_layout=QVBoxLayout(model_group); model_layout.addWidget(self.model_warning_label)
        self.run_button = QPushButton("Run model-based analysis"); self.run_button.setObjectName("runButton"); self.run_button.clicked.connect(self.run_analysis)
        controls_layout.addStretch()
        control_scroll = QScrollArea(); control_scroll.setWidgetResizable(True); control_scroll.setFrameShape(QFrame.Shape.NoFrame); control_scroll.setMinimumWidth(235); control_scroll.setWidget(controls)

        self.tabs = QStackedWidget(); self.results_table = QTableWidget(); self.stats_table = QTableWidget(); self.qa_table = QTableWidget()
        plot_page = QWidget(); plot_layout = QVBoxLayout(plot_page); plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Plotted metric")); self.plot_metric = QComboBox(); self.plot_metric.currentIndexChanged.connect(self._update_plot); plot_controls.addWidget(self.plot_metric); plot_controls.addStretch()
        self.distribution_plot = DistributionPlot(); plot_layout.addLayout(plot_controls); plot_layout.addWidget(self.distribution_plot)
        for page in (self.results_table,self.stats_table,plot_page,self.qa_table): self.tabs.addWidget(page)
        self.result_view=QComboBox(); self.result_view.addItems(["Measurements","Statistical summary","Distribution & control plots","QA & traceability"])
        self.result_view.currentIndexChanged.connect(self.tabs.setCurrentIndex); self.tabs.currentChanged.connect(self.result_view.setCurrentIndex)
        result_header=QHBoxLayout(); result_header.addWidget(QLabel("Centre view")); result_header.addWidget(self.result_view,1)

        centre=QWidget(); centre_layout=QVBoxLayout(centre); centre_layout.setContentsMargins(4,0,4,0); centre_layout.setSpacing(5)
        centre_title=QLabel("2 · Image and results"); centre_title.setObjectName("panelTitle"); centre_layout.addWidget(centre_title)
        cards_layout=QGridLayout(); cards_layout.setSpacing(5); self.cards=[MetricCard(x) for x in ("Objects / bands","Primary mean","Variation (CV)","Release status")]
        for index,card in enumerate(self.cards): cards_layout.addWidget(card,index//2,index%2)
        centre_layout.addLayout(cards_layout)
        image_panel=QWidget(); image_layout=QVBoxLayout(image_panel); image_layout.setContentsMargins(0,0,0,0); image_layout.addWidget(self.image_view)
        results_panel=QWidget(); results_layout=QVBoxLayout(results_panel); results_layout.setContentsMargins(0,0,0,0); results_layout.addLayout(result_header); results_layout.addWidget(self.tabs)
        self.vertical_splitter=QSplitter(Qt.Orientation.Vertical); self.vertical_splitter.addWidget(image_panel); self.vertical_splitter.addWidget(results_panel); self.vertical_splitter.setSizes([430,310]); self.vertical_splitter.setChildrenCollapsible(False)
        centre_layout.addWidget(self.vertical_splitter,1)

        diagnostics=QWidget(); diagnostics_layout=QVBoxLayout(diagnostics); diagnostics_layout.setContentsMargins(4,0,4,0)
        diagnostics_title=QLabel("3 · Diagnostics and release"); diagnostics_title.setObjectName("panelTitle"); diagnostics_layout.addWidget(diagnostics_title)
        diagnostics_layout.addWidget(self.run_button)
        diagnostics_layout.addWidget(CollapsibleSection("Release diagnostics",qa_group,True))
        diagnostics_layout.addWidget(CollapsibleSection("Model compatibility",model_group,True))
        quick_help=QGroupBox("Workspace guidance"); quick_help_layout=QVBoxLayout(quick_help)
        quick_help_text=QLabel("1. Select a recipe and calibration on the left.\n2. Review the overlay and selectable result view in the centre.\n3. Resolve warnings here before accepting or exporting results.")
        quick_help_text.setWordWrap(True); quick_help_layout.addWidget(quick_help_text); diagnostics_layout.addWidget(CollapsibleSection("Guidance",quick_help,False)); diagnostics_layout.addStretch()
        diagnostic_scroll=QScrollArea(); diagnostic_scroll.setWidgetResizable(True); diagnostic_scroll.setFrameShape(QFrame.Shape.NoFrame); diagnostic_scroll.setMinimumWidth(230); diagnostic_scroll.setWidget(diagnostics)

        self.workspace_splitter=QSplitter(Qt.Orientation.Horizontal); self.workspace_splitter.addWidget(control_scroll); self.workspace_splitter.addWidget(centre); self.workspace_splitter.addWidget(diagnostic_scroll)
        self.workspace_splitter.setChildrenCollapsible(False); self.workspace_splitter.setSizes([290,760,290]); self.workspace_splitter.setStretchFactor(0,0); self.workspace_splitter.setStretchFactor(1,1); self.workspace_splitter.setStretchFactor(2,0)
        self.settings_width_slider.valueChanged.connect(self._set_settings_width); self.diagnostics_width_slider.valueChanged.connect(self._set_diagnostics_width); self.reset_widths_button.clicked.connect(self._reset_workspace_widths); self.workspace_splitter.splitterMoved.connect(self._sync_width_sliders)
        root_layout.addWidget(self.workspace_splitter,1); self.setCentralWidget(root); self.setStatusBar(QStatusBar())
        settings=QSettings("KofiOsae","SEM Metrology Workbench")
        workspace_state=settings.value("workspaceSplitter"); result_state=settings.value("resultSplitter")
        if workspace_state is not None: self.workspace_splitter.restoreState(workspace_state)
        if result_state is not None: self.vertical_splitter.restoreState(result_state)
        self._sync_width_sliders()

    def _set_settings_width(self,width):
        sizes=self.workspace_splitter.sizes(); total=sum(sizes); right=sizes[2]; centre=max(300,total-int(width)-right); self.workspace_splitter.setSizes([int(width),centre,right])

    def _set_diagnostics_width(self,width):
        sizes=self.workspace_splitter.sizes(); total=sum(sizes); left=sizes[0]; centre=max(300,total-left-int(width)); self.workspace_splitter.setSizes([left,centre,int(width)])

    def _sync_width_sliders(self,*_):
        sizes=self.workspace_splitter.sizes()
        if len(sizes)!=3: return
        for slider,value in ((self.settings_width_slider,sizes[0]),(self.diagnostics_width_slider,sizes[2])):
            slider.blockSignals(True); slider.setValue(max(slider.minimum(),min(slider.maximum(),value))); slider.blockSignals(False)

    def _reset_workspace_widths(self):
        total=max(960,sum(self.workspace_splitter.sizes())); side=min(290,max(230,(total-420)//2)); self.workspace_splitter.setSizes([side,max(420,total-2*side),side]); self.vertical_splitter.setSizes([430,310]); self._sync_width_sliders()

    def closeEvent(self,event):
        settings=QSettings("KofiOsae","SEM Metrology Workbench")
        settings.setValue("workspaceSplitter",self.workspace_splitter.saveState()); settings.setValue("resultSplitter",self.vertical_splitter.saveState())
        super().closeEvent(event)

    def _recipe_changed(self):
        taught_model = self.recipe.currentIndex() in REFERENCE_TAUGHT_RECIPES
        self.sensitivity.setEnabled(taught_model)
        self.teach_button.setEnabled(taught_model); self.apply_reference_button.setEnabled(taught_model)
        self.line_orientation.setEnabled(self.recipe.currentIndex() == 3)
        self.line_polarity.setEnabled(self.recipe.currentIndex() in (2,3))
        self.period_x_hint.setEnabled(self.recipe.currentIndex() in (0,1)); self.period_y_hint.setEnabled(self.recipe.currentIndex() in (0,1))
        self.layer_period_count.setEnabled(self.recipe.currentIndex()==7)
        class_applicable=self.recipe.currentIndex() in (0,1,2,3,6)
        self.class_mode.setEnabled(class_applicable); self.confirm_classes.setEnabled(class_applicable)
        self.edit_interfaces_button.setEnabled(self.recipe.currentIndex()==7 and bool(self.layer_active_interfaces))
        self.remove_interface_button.setEnabled(self.recipe.currentIndex()==7 and len(self.layer_active_interfaces)>2)
        self.restore_interfaces_button.setEnabled(self.recipe.currentIndex()==7 and bool(self.layer_candidate_interfaces))
        hints = {
            0: "Diameter, projected area, fitted perimeter, pitch and circle-fit residual",
            1: "Length, mean width, width 3σ, X/Y period, LER 3σ and template confidence",
            2: "Area, perimeter, Feret axes, aspect ratio, circularity and solidity",
            3: "Line CD, pitch, duty cycle and line-width roughness",
            4: "One isolated regular/irregular feature: pixel contour, area, perimeter, Feret and axes",
            5: "One operator-bounded filled waveguide: medial path length, centerline-normal width and taper",
            6: "One cross-section feature: height, top/middle/bottom CD and nominal sidewall angle",
            7: "Laterally continuous film stack: reviewed vertical interfaces and layer thicknesses",
        }
        self.recipe.setToolTip(hints.get(self.recipe.currentIndex(), ""))

    def _begin_layer_interface_edit(self):
        if self.recipe.currentIndex()!=7 or not self.layer_active_interfaces:
            QMessageBox.information(self,"No layer interfaces","Run the Layer stack / film cross-section ROI model first."); return
        self.image_view.begin_interface_edit()
        self.edit_interfaces_button.setText("Click-to-remove is ACTIVE")
        self.edit_interfaces_button.setStyleSheet("background:#8a3f3f;border-color:#ef7474;font-weight:600")
        self.interface_edit_status.setText("ACTIVE — click within the yellow ROI near a cyan line. The nearest interface will be removed.")
        self.statusBar().showMessage("Interface edit active: click near a numbered cyan interface, or select its ID and press Remove selected interface.")

    def _layer_interface_clicked(self, point):
        if self.layer_roi is None or not self.layer_active_interfaces: return
        x,y,w,h=self.layer_roi
        if not (x<=point[0]<=x+w and y<=point[1]<=y+h): return
        local_y=point[1]-y
        nearest=min(self.layer_active_interfaces,key=lambda row:abs(row-local_y))
        tolerance=max(12,round(h*.045))
        if abs(nearest-local_y)>tolerance:
            self.statusBar().showMessage("No detected interface near that click; click a numbered cyan line.",5000); return
        if len(self.layer_active_interfaces)<=2:
            QMessageBox.warning(self,"Cannot remove interface","At least two interfaces are required to define one layer."); return
        self.layer_active_interfaces.remove(nearest)
        self._render_layer_stack()
        self.statusBar().showMessage(f"Removed interface near image Y={y+nearest}px. Layer IDs and measurements were recalculated.",6000)

    def _remove_selected_layer_interface(self):
        if self.layer_roi is None or len(self.layer_active_interfaces)<=2:
            QMessageBox.warning(self,"Cannot remove interface","At least two interfaces are required to define one layer."); return
        index=self.interface_selector.currentIndex()
        if index<0 or index>=len(self.layer_active_interfaces):
            QMessageBox.information(self,"Select interface","Choose I1, I2… in the Layer review list first."); return
        removed=self.layer_active_interfaces.pop(index)
        self._render_layer_stack()
        self.statusBar().showMessage(f"Removed selected interface at image Y={self.layer_roi[1]+removed}px.",6000)

    def _restore_layer_interfaces(self):
        if self.layer_roi is None or not self.layer_candidate_interfaces: return
        self.layer_active_interfaces=list(self.layer_candidate_interfaces)
        self._render_layer_stack()
        self.image_view.interface_mode=False; self.image_view.unsetCursor()
        self.edit_interfaces_button.setText("Enable click-to-remove"); self.edit_interfaces_button.setStyleSheet("")
        self.interface_edit_status.setText("All proposed interfaces restored. Review I1…I" + str(len(self.layer_active_interfaces)) + ".")
        self.statusBar().showMessage("All automatically proposed interfaces restored.",5000)

    def _render_layer_stack(self):
        if self.layer_roi is None or self.analysis_rgb is None or self.calibration is None: return
        x,y,w,h=self.layer_roi; scale=self.calibration.microns_per_pixel; gray=to_gray(self.analysis_rgb)
        crop_gray=gray[y:y+h,x:x+w]
        measurements,diagnostics=measure_layer_stack(crop_gray,scale,self.layer_active_interfaces)
        self._apply_layer_period_guidance(measurements,diagnostics,w,scale)
        diagnostics["unit_scale"]=1.0 if self.pixel_mode else scale*1000
        diagnostics["unit"]="px" if self.pixel_mode else "nm"
        self.measurements=measurements
        crop_rgb=self.analysis_rgb[y:y+h,x:x+w]; annotated=np.asarray(cross_section_overlay(crop_rgb,diagnostics))
        overlay=self.analysis_rgb.copy(); overlay[y:y+h,x:x+w]=annotated
        interfaces=diagnostics.get("interface_rows",[])
        self.feature_rows=[{"Layer ID":f"L{i+1}","Upper interface":f"I{i+1}","Lower interface":f"I{i+2}",
                            "Upper interface Y (px)":y+upper,"Lower interface Y (px)":y+lower,
                            "Thickness (µm)":(lower-upper)*scale}
                           for i,(upper,lower) in enumerate(zip(interfaces,interfaces[1:]))]
        self.model_warning=None if len(interfaces)>=2 else "Fewer than two operator-accepted interfaces remain; no layer thickness is releasable."
        self.model_warning_label.setText("MODEL REVIEW: "+self.model_warning if self.model_warning else "Interfaces are operator-reviewed; numbered image labels correspond to table interface IDs.")
        self.overlay_rgb=self._full_frame_display(overlay); self.image_view.set_rgb(self.overlay_rgb)
        display_rows=self._practical_rows(self.feature_rows); self._set_table(self.results_table,display_rows)
        stats=self._practical_stats(measurement_statistics(self.measurements)); self._set_table(self.stats_table,stats)
        self._populate_plot_metrics(display_rows); self._update_qa(stats)
        self.interface_selector.blockSignals(True); self.interface_selector.clear()
        for index,row in enumerate(self.layer_active_interfaces,start=1):
            self.interface_selector.addItem(f"I{index} — image Y={y+row}px")
        self.interface_selector.blockSignals(False)
        self.edit_interfaces_button.setEnabled(bool(self.layer_active_interfaces)); self.remove_interface_button.setEnabled(len(self.layer_active_interfaces)>2); self.restore_interfaces_button.setEnabled(bool(self.layer_candidate_interfaces))
        if not self.image_view.interface_mode: self.interface_edit_status.setText(f"{len(self.layer_active_interfaces)} accepted interfaces. Select an ID or enable click-to-remove.")

    def _apply_layer_period_guidance(self,measurements,diagnostics,roi_width_px,scale):
        count=self.layer_period_count.value()
        if count<=0: return
        measurements[:]=[m for m in measurements if m.name!="Lateral repeat period"]
        period_px=roi_width_px/count
        diagnostics["period_px"]=period_px; diagnostics["period_confidence"]=1.0
        measurements.append(Measurement("Operator-guided lateral period",period_px*scale,"µm",
                                        f"analysis ROI width divided by operator-declared {count} periods",1.0,
                                        {"period_px":period_px,"period_count":count,"roi_width_px":roi_width_px}))

    def _apply_class_aware_analysis(self,recipe,overlay):
        """Propose size populations for repeated, physically comparable features."""
        signatures={
            0:[("Diameter","Diameter (µm)","µm")],
            1:[("Feature length","Length (µm)","µm"),("Mean width","Mean width (µm)","µm")],
            2:[("Feret max","Feret max (µm)","µm"),("Feret min","Feret min (µm)","µm")],
            3:[("Line CD","Mean CD (µm)","µm")],
            6:[("Feature height","Feature height (µm)","µm"),("Middle width","Middle width (µm)","µm")],
        }
        fields=signatures.get(recipe)
        self.class_model=None
        if self.class_mode.currentIndex()==1: return overlay
        if not fields or len(self.feature_rows)<8: return overlay
        valid_fields=[field for field in fields if all(field[1] in row and np.isfinite(row[field[1]]) for row in self.feature_rows)]
        if not valid_fields: return overlay
        values=np.asarray([[row[field[1]] for field in valid_fields] for row in self.feature_rows],float)
        requested={2:2,3:3}.get(self.class_mode.currentIndex())
        model=infer_feature_classes(values,requested_classes=requested)
        if model.get("labels") is None:
            if requested:
                note=f"Operator expected {requested} classes, but the measured population does not support a sufficiently separated {requested}-class model."
                self.model_warning=(self.model_warning+"; " if self.model_warning else "")+note
            return overlay
        self.class_model=model; labels=model["labels"]; confidence=model["confidence"]; letters="ABC"
        for row,label in zip(self.feature_rows,labels):
            if label<0: continue
            row["Size class"]=f"Class {letters[label]}"; row["Class confidence (%)"]=confidence*100
        for class_index in range(model["class_count"]):
            class_name=f"Class {letters[class_index]}"
            for _,key,_ in valid_fields:
                class_key=key.replace(" (µm)",f" — {class_name} (µm)").replace(" (µm²)",f" — {class_name} (µm²)")
                for row,label in zip(self.feature_rows,labels):
                    row[class_key]=float(row[key]) if label==class_index else np.nan
        class_periods={}
        def base_period(axis):
            key=f"Fundamental period {axis} (µm)"
            candidates=[row.get(key,np.nan) for row in self.feature_rows]
            candidates=np.asarray(candidates,float); candidates=candidates[np.isfinite(candidates)&(candidates>0)]
            return float(np.median(candidates)) if len(candidates) else np.nan
        base_x,base_y=base_period("X"),base_period("Y")

        def registered_sublattice(points,axis_index,base_axis,base_other):
            if len(points)<4 or not np.isfinite(base_axis) or not np.isfinite(base_other): return None
            steps=[]; residuals=[]; pairs=[]
            for first in range(len(points)):
                for second in range(first+1,len(points)):
                    along=abs(points[first,axis_index]-points[second,axis_index])
                    across=abs(points[first,1-axis_index]-points[second,1-axis_index])
                    if across>.35*base_other or along<.65*base_axis: continue
                    ratio=along/base_axis; step=max(1,int(round(ratio))); residual=abs(ratio-step)
                    if residual<=.20: steps.append(step); residuals.append(residual); pairs.append((first,second))
            if len(steps)<3: return None
            steps=np.asarray(steps,int); residuals=np.asarray(residuals,float)
            candidates=[]
            for candidate in range(1,int(np.percentile(steps,60))+1):
                exact=int(np.sum(steps==candidate)); divisible=float(np.mean(steps%candidate==0))
                if exact>=2 and divisible>=.72: candidates.append((candidate,exact,divisible))
            if not candidates: return None
            candidate,exact,divisible=max(candidates,key=lambda item:(item[0],item[1]))
            support=float(np.clip((len(steps)/max(3,len(points)-1))*.35+divisible*.45+(1-np.mean(residuals)/.20)*.20,0,1))
            if support<.55: return None
            matching=[i for i,step in enumerate(steps) if step==candidate]
            representative=min(matching,key=lambda i:residuals[i])
            first,second=pairs[representative]
            return candidate*base_axis,candidate,support,(points[first],points[second])

        for class_index in range(model["class_count"]):
            member_indices=np.flatnonzero(labels==class_index); class_name=f"Class {letters[class_index]}"
            for metric_name,key,unit in valid_fields:
                for index in member_indices:
                    self.measurements.append(Measurement(f"{metric_name} — {class_name}",float(self.feature_rows[index][key]),unit,
                                                         f"class-aware population; {model['reason']}",confidence,
                                                         {"class":class_name,"class_count":model["class_count"]}))
            self.measurements.append(Measurement(f"Population fraction — {class_name}",len(member_indices)/len(labels)*100,"%",
                                                 "accepted feature count fraction",confidence))
            # Class sublattices must register to the accepted global lattice.
            # This prevents sparse missing-site gaps from becoming false periods.
            if len(member_indices)>=4 and all("Centroid X (px)" in self.feature_rows[i] and "Centroid Y (px)" in self.feature_rows[i] for i in member_indices):
                points=np.asarray([[self.feature_rows[i]["Centroid X (px)"],self.feature_rows[i]["Centroid Y (px)"]] for i in member_indices],float)
                scale=self.calibration.microns_per_pixel
                registered={}
                for axis_index,axis,base_axis,base_other in ((0,"X",base_x/scale,base_y/scale),(1,"Y",base_y/scale,base_x/scale)):
                    result=registered_sublattice(points,axis_index,base_axis,base_other)
                    if result:
                        period_px,multiple,period_confidence,gauge=result; period_um=period_px*scale
                        registered[axis]=(period_um,multiple,period_confidence,gauge)
                        self.measurements.append(Measurement(f"{class_name} sublattice period {axis}",period_um,"µm",
                                                             f"class-centroid repeat registered as {multiple}× global period {axis}",
                                                             min(confidence,period_confidence)))
                class_periods[class_index]=registered
        canvas=np.asarray(overlay).copy(); colors=((0,255,255),(255,190,40),(80,255,110))
        for row,label in zip(self.feature_rows,labels):
            if label<0: continue
            if "Centroid X (px)" in row and "Centroid Y (px)" in row:
                center=(round(row["Centroid X (px)"]),round(row["Centroid Y (px)"]))
                cv2.circle(canvas,center,6,colors[label%len(colors)],2,cv2.LINE_AA)
        for index in range(model["class_count"]):
            primary_mean=float(np.mean(values[labels==index,0])); unit_text="nm" if valid_fields[0][2]=="µm" else valid_fields[0][2]
            primary_display=primary_mean*1000 if valid_fields[0][2]=="µm" else primary_mean
            period_text=""
            for axis in ("X","Y"):
                if axis in class_periods.get(index,{}): period_text+=f" P{axis}={class_periods[index][axis][0]*1000:.1f}nm"
            label=f"Class {letters[index]}: N={int(model['counts'][index])}, {valid_fields[0][0]}={primary_display:.1f}{unit_text}{period_text}"
            cv2.putText(canvas,label,(8,64+index*18),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,colors[index%len(colors)],1,cv2.LINE_AA)
            for axis,entry in class_periods.get(index,{}).items():
                first,second=entry[3]; a=tuple(np.rint(first).astype(int)); b=tuple(np.rint(second).astype(int))
                cv2.line(canvas,a,b,colors[index%len(colors)],2,cv2.LINE_AA)
                midpoint=((a[0]+b[0])//2+4,(a[1]+b[1])//2-5)
                cv2.putText(canvas,f"{letters[index]} P{axis} {entry[0]*1000:.1f}nm",midpoint,
                            cv2.FONT_HERSHEY_SIMPLEX,.38,colors[index%len(colors)],1,cv2.LINE_AA)
        note=f"{model['class_count']} size populations proposed ({confidence:.0%} confidence)"
        if self.confirm_classes.isChecked(): note+="; assignments operator-confirmed"
        else:
            note+="; operator confirmation required"
            self.model_warning=(self.model_warning+"; " if self.model_warning else "")+note
        self.recommendation_label.setText(note+". Class colors and class-specific statistics are shown.")
        return canvas

    def _begin_teach(self):
        if self.analysis_rgb is None:
            QMessageBox.information(self, "Load an image", "Open an SEM image before teaching a reference feature."); return
        if self.recipe.currentIndex() not in REFERENCE_TAUGHT_RECIPES:
            QMessageBox.information(self, "Select a taught model", "Reference teaching is available for circles, rectangular arrays, irregular particles and a single contour feature."); return
        instruction = ("Drag a tight rectangle around one complete, isolated particle with a small background margin."
                       if self.recipe.currentIndex() == 2 else
                       "Drag a tight rectangle around one complete representative feature.")
        self.statusBar().showMessage(instruction)
        self.image_view.begin_reference_selection()

    def _recommend_model(self):
        if self.analysis_rgb is None: return None
        gray = to_gray(self.analysis_rgb)
        if max(gray.shape)>1200:
            factor=1200/max(gray.shape); gray=cv2.resize(gray,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        work = cv2.GaussianBlur(gray,(3,3),0)
        ex = float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,1,0,ksize=3))))
        ey = float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,0,1,ksize=3))))
        total = ex+ey+1e-9; vertical_conf, horizontal_conf = ex/total, ey/total
        orientation = "vertical" if vertical_conf >= horizontal_conf else "horizontal"
        confidence = max(vertical_conf,horizontal_conf)
        if confidence >= .72:
            bright,_,_ = analyze_line_space(gray,1.0,"bright",2.0,max(gray.shape)/2,orientation)
            dark,_,_ = analyze_line_space(gray,1.0,"dark",2.0,max(gray.shape)/2,orientation)
            if max(len(bright),len(dark)) >= 4: return (3, orientation, confidence)
        return None

    def _reference_selected(self, roi):
        self.reference_roi = tuple(int(v) for v in roi)
        x, y, w, h = self.reference_roi
        if self.calibration:
            if self.recipe.currentIndex() == 0:
                nominal = 0.5 * (w + h) * self.calibration.microns_per_pixel * 1000
                self.min_size.setValue(max(1, nominal * 0.72)); self.max_size.setValue(nominal * 1.28)
                detail = f"ROI {w}×{h} px · nominal diameter {nominal:.1f} nm"
            elif self.recipe.currentIndex() == 2:
                nominal = np.sqrt(w*h)*self.calibration.microns_per_pixel*1000
                detail = f"ROI {w}×{h} px · nominal particle scale {nominal:.1f} nm"
            else:
                sx = w * self.calibration.microns_per_pixel * 1000; sy = h * self.calibration.microns_per_pixel * 1000
                detail = f"ROI {w}×{h} px · physical box {sx:.1f}×{sy:.1f} nm"
        else: detail = f"ROI {w}×{h} px · calibration applied at run"
        purpose="contact-splitting marker scale" if self.recipe.currentIndex()==2 else "centre proposals and local edge refitting"
        self.reference_label.setText(detail+f" · used for {purpose}")
        self.statusBar().showMessage("Reference feature accepted. Review size bounds, then run analysis.", 6000)

    def _analysis_selected(self,roi):
        x,y,w,h=(int(v) for v in roi); h=min(h,max(0,self.footer_start-y)) if self.footer_start is not None else h
        self.analysis_polygon=None; self.analysis_roi=(x,y,w,h); self.analysis_region_label.setText(f"ROI x={x}, y={y}, {w}×{h} px · full image remains displayed")

    def _analysis_mask_selected(self,points):
        clipped=[(max(0,min(self.analysis_rgb.shape[1]-1,int(x))),max(0,min(self.analysis_rgb.shape[0]-1,int(y)))) for x,y in points]
        polygon=np.asarray(clipped,dtype=np.int32)
        if self.mask_behavior.currentIndex()==1:
            self.exclusion_polygon=polygon; self.analysis_region_label.setText(f"Exclusion mask · {len(clipped)} points · features touching it rejected")
        elif recipe == 4:
            self.analysis_roi=None; self.analysis_polygon=polygon; self.analysis_region_label.setText(f"Inclusion mask · {len(clipped)} boundary points · complete features only")

    def _clear_analysis_region(self):
        self.analysis_roi=None; self.analysis_polygon=None; self.exclusion_polygon=None; self.image_view.clear_analysis_region(); self.analysis_region_label.setText("Full specimen area")

    def _remove_last_cross_section(self):
        if not self.cross_section_records: QMessageBox.information(self,"Cross-section set","No stored cross-section measurement."); return
        self.cross_section_records.pop(); self.cross_section_overlays.pop(); self._show_cross_section_set()

    def _clear_cross_section_set(self):
        self.cross_section_records=[]; self.cross_section_overlays=[]
        if self.recipe.currentIndex()==6: self.feature_rows=[]; self.measurements=[]; self._show_cross_section_set()

    def _show_cross_section_set(self):
        self.feature_rows=[record["row"] for record in self.cross_section_records]; self.measurements=[m for record in self.cross_section_records for m in record["measurements"]]
        overlay=self.analysis_rgb.copy() if self.analysis_rgb is not None else None
        if overlay is not None:
            for x,y,w,h,annotated in self.cross_section_overlays: overlay[y:y+h,x:x+w]=annotated
            self.overlay_rgb=self._full_frame_display(overlay); self.image_view.set_rgb(self.overlay_rgb)
        display=self._practical_rows(self.feature_rows); self._set_table(self.results_table,display); stats=self._practical_stats(measurement_statistics(self.measurements)); self._set_table(self.stats_table,stats); self._populate_plot_metrics(display)
        if stats: self._update_qa(stats)

    def _apply_adjusted_reference(self):
        roi = self.image_view.reference_geometry()
        if roi is None:
            QMessageBox.information(self, "No reference", "Draw a reference feature first."); return
        self._reference_selected(roi)

    def _full_frame_display(self, analysis_layer=None):
        """Preserve the complete SEM frame while keeping its footer out of analysis."""
        if self.rgb is None: return analysis_layer
        canvas = self.rgb.copy()
        if analysis_layer is not None:
            height = min(self.footer_start, analysis_layer.shape[0])
            canvas[:height] = analysis_layer[:height]
        detection = getattr(self, "bar_detection", None)
        if detection:
            start = tuple(int(v) for v in detection["start_xy"]); end = tuple(int(v) for v in detection["end_xy"])
            cv2.line(canvas,start,end,(255,0,255),3)
            cv2.circle(canvas,start,4,(0,255,255),-1); cv2.circle(canvas,end,4,(0,255,255),-1)
            label = f"detected scale: {self.scale_length.value():g} {self.scale_unit.currentText()} / {detection['pixel_length']} px"
            ty = max(18,start[1]-10); cv2.putText(canvas,label,(max(5,start[0]),ty),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,0,255),1,cv2.LINE_AA)
        elif self.metadata_sampling_um:
            cv2.putText(canvas,f"metadata calibration: {self.metadata_sampling_um*1000:.6g} nm/px",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,0,255),1,cv2.LINE_AA)
        return canvas

    def _refresh_base_display(self):
        if self.analysis_rgb is not None and not self.feature_rows:
            self.image_view.set_rgb(self._full_frame_display(self.analysis_rgb))

    def reset_analysis(self):
        self.feature_rows = []; self.measurements = []; self.overlay_rgb = None; self.reference_roi = None; self.analysis_roi=None; self.analysis_polygon=None; self.exclusion_polygon=None; self.plot_series = {}; self.cross_section_records=[]; self.cross_section_overlays=[]; self.layer_candidate_interfaces=[]; self.layer_active_interfaces=[]; self.layer_roi=None; self._drawing_scale_bar=False
        self.image_view.clear_reference(); self.image_view.clear_analysis_region(); self.reference_label.setText("No reference taught — automatic proposal mode"); self.analysis_region_label.setText("Full specimen area")
        self.interface_selector.clear(); self.interface_edit_status.setText("Run layer analysis to review interfaces"); self.edit_interfaces_button.setText("Enable click-to-remove"); self.edit_interfaces_button.setStyleSheet("")
        for table in (self.results_table, self.stats_table, self.qa_table): self._set_table(table, [])
        self.plot_metric.clear(); self.distribution_plot.set_values([], "No results")
        for card in self.cards: card.value.setText("—")
        self.qa_label.setText("WAITING — run a reviewed measurement model"); self.qa_detail.setText("Identification and operator guidance were reset; calibration was retained.")
        if self.analysis_rgb is not None: self.image_view.set_rgb(self._full_frame_display(self.analysis_rgb))
        self.statusBar().showMessage("Identification reset", 4000)

    def show_help(self):
        QMessageBox.information(self, "Measurement workflow", "1. Open an image. The magenta scale overlay is an automatic proposal, not an accepted calibration. Verify that both endpoint dots coincide with the physical bar. If not, press Redraw scale-bar endpoints and draw from the first physical endpoint to the last; enter the printed value and confirm. Companion metadata remains preferred.\n2. Select the physical recipe and review its overlay.\n3. Cross-section population: draw ROI 1 and run; draw ROI 2 and run; continue for every structure. Stored ROIs remain overlaid and feed combined statistics and plots.\n4. Layer stacks: draw the yellow ROI through the stack. If its width spans a known number of repeats, enter that count. For reliable editing, select an interface ID and press Remove selected interface, or enable click-to-remove.\n5. Line profile: draw through particles to measure calibrated lateral FWHM-like chords, edge gaps, centre spacing and coverage. Chord equals diameter only for a central transect; SEM intensity is not height.\n6. Inclusion/exclusion masks reject labels and artefacts. The SEM footer is excluded from dimensional analysis.\n7. In Distribution & control plots, move the mouse over an individual point to see its object/structure ID, exact value, unit, class and original-image centroid where available.\n8. Inspect measurements, statistics, plots and QA before export.\n\nAll pixel coordinates, scale endpoints and ROI widths are original-image pixels, never resized display pixels. Mouse wheel: zoom. Drag: pan. F: fit image.")

    def show_about(self):
        QMessageBox.about(self,"About / cite SEM Metrology Workbench",
            f"SEM Metrology Workbench {APP_VERSION}\n"
            "Operator-guided dimensional metrology for SEM images\n\n"
            "Copyright © 2026 SEM Metrology Workbench Contributors\n"
            "Source code: BSD-3-Clause · Documentation: CC BY 4.0\n\n"
            f"Citation: cite SEM Metrology Workbench, version {APP_VERSION}, using the repository CITATION.cff/Zenodo DOI. "
            "Report the recipe, calibration source, sampling, population N and exclusions.\n\n"
            "Privacy: the desktop app sends no telemetry and never uploads images or results. Aggregate GitHub traffic and release-download counts may be used to measure adoption.\n\n"
            "Research software—not a certified metrology instrument. Operator review and a laboratory uncertainty budget remain required.")

    def show_capabilities(self):
        QMessageBox.information(self,"Supported structures","RELEASED MODELS\n• circular holes/rings/discs\n• bright or dark irregular apertures and particles, with Feret length/width\n• rectangular periodic islands\n• horizontal, vertical or diagonal line-space gratings\n• one isolated contour or one non-branching curved/tapered waveguide\n• isolated cross-section pillars with a local substrate datum\n• laterally continuous layer/film cross-sections\n• line gauges and rectangular/freehand signal-texture regions\n\nCALIBRATION AND LABELS\nHitachi-style companion TXT PixelSize metadata is used when present and still requires operator confirmation. Raster dimension labels are not measurements generated by this app; prefer raw acquisitions or exclude label regions with a freehand exclusion mask.\n\nMODEL DISTINCTIONS\nUse Cross-section feature ROI for isolated pillars/trenches. Use Layer stack / film cross-section ROI for continuous horizontal films. Use Irregular particles/grains with Dark polarity for non-circular dark metalens apertures. Diagonal gratings are internally deskewed and report normal CD/pitch plus line direction.\n\nLIMITATIONS\nDesktop/application screenshots receive a source warning. Touching irregular particles are marker-split but remain REVIEW because hidden overlap boundaries cannot be reconstructed from one projection; severe overlaps should be excluded or re-imaged. Branching/overlapping waveguides, severe junctions and occluded cross-sections are not release models.")

    def show_definitions(self):
        QMessageBox.information(self,"Measurement definitions","REGULAR FEATURES\nDiameter: twice the fitted-circle radius. Length / width: long / short local dimensions for the selected regular-feature model.\n\nCLASS-AWARE POPULATIONS\nFor repeated circles, rectangles, particles, line CDs and multi-ROI cross-sections, Auto compares one-, two- and three-population size models. A split is proposed only with N≥8, at least three members per class, adequate model improvement and separation. Class A is the smaller primary-size population. Pooled results remain visible, while class-specific N, mean, Std, CV and population fraction are reported separately. When centroids exist, class-specific X/Y sublattice periods are also proposed. Treat as one population disables splitting; Expect 2/3 supplies operator knowledge but still requires measurable separation. Confirmation is required before release.\n\nIRREGULAR FEATURES\nFeret max / min: maximum / minimum caliper distance; these are useful size descriptors but are not automatically the design length and width. Equivalent diameter: diameter of a circle having the same segmented area. Major / minor axes: fitted-ellipse axes. Circularity = 4πA/P². Solidity = area / convex-hull area. Draw a tight reference around one complete representative particle to supply nominal marker spacing. Orange contact contours and rows marked REVIEW describe only the visible segment and never reconstruct a hidden overlap boundary.\n\nEDGE ROUGHNESS\nLER: RMS or 3σ of edge-position residual after removing a best-fit line. LWR: RMS or 3σ of local linewidth minus its mean. Raw results remain SEM-noise inclusive. Filtered results use the disclosed [1,2,1]/4 low-pass.\n\nPERIODS\nFundamental period is the smallest credible lattice step explaining adjacent gaps, including integer missing-site multiples. Secondary periods are non-integer gap/superlattice families and are reported separately with confidence.\n\nSEM SIGNAL TEXTURE\nRa/Sa are SEM intensity residual statistics, not calibrated height roughness. Lateral calibration supplies sampled length and area only.\n\nPIXELS\nAny pixel result refers to original image pixels, never resized display pixels.")

    def show_cross_section_definitions(self):
        QMessageBox.information(self,"Cross-section definitions","ISOLATED FEATURE HEIGHT\nFeature height is measured vertically from the detected feature top to the first sustained local substrate-interface row. The green HEIGHT gauge shows both endpoints.\n\nLAYER STACKS\nInterfaces are labeled I1, I2… from top to bottom. Layer L1 is bounded by I1–I2, L2 by I2–I3, and so on; these same IDs appear in the table. Total stack height spans the outermost accepted interfaces. Automatic candidates require both transition strength and lateral persistence, reducing internal-texture false edges. They remain candidate material boundaries and require review.\n\nLATERAL PERIOD\nFor a deliberately drawn ROI spanning N repeats, set Periods spanned by layer ROI to N. The reported period is the ROI's original-image width divided by N. With Auto, period is released only when autocorrelation contains at least three repeats with adequate confidence. The operator-guided method is explicitly named in the report.\n\nCALIBRATION AND FOOTER\nCompanion PixelSize metadata is preferred. Without metadata, green bars and white tick-ended bars in the lower footer can be detected, but the physical bar value remains operator-confirmed. Magnification, kV, detector, date and other footer text are not analyzed.\n\nSTATISTICS\nStd is structure-to-structure dispersion. Std error = Std/√N. Mean 95% CI is ±1.96×standard error and is not a complete calibration/instrument uncertainty budget.")

    def show_study_help(self):
        QMessageBox.information(self,"Process and dose studies","PURPOSE\nA process study compares reviewed measurements from separate SEM images against one numeric process factor. E-beam dose is the default, but focus, temperature, etch time, development time, wafer position or another controlled split can be used.\n\nWORKFLOW\n1. Analyze and review one image normally, including calibration, recipe, overlays and class assignments.\n2. Click Add reviewed result to process study. Enter the trusted dose/factor value, sample label, process condition and replicate number.\n3. Repeat for every image. The recipe should normally remain fixed across a study.\n4. Open process study and select any metric common to at least two images.\n5. Hover points for image, dose/factor, mean, within-image Std, feature N, condition, replicate, recipe and QA.\n\nSTATISTICS\nEach plotted point is one reviewed image result. Error bars are the within-image feature Std, not uncertainty of the image mean and not between-image reproducibility. Replicate images remain separate points. The displayed slope and R² are descriptive unweighted linear fits; they do not prove causality or define a process window. Exported long-form CSV preserves every image/metric combination for advanced modeling.\n\nCLASS-AWARE SERIES\nClass A/B/C metrics and registered class sublattice periods are compared separately when common across images. Classes are ordered by primary size to reduce label swapping. Review warnings where classes merge, disappear or are unsupported.\n\nTRACEABILITY\nDose cannot be inferred from SEM pixels. Enter it from the exposure job, process traveler or trusted metadata. REVIEW images may be stored for investigation, but their QA status remains visible.")

    def add_current_to_study(self):
        if not self.measurements or self.image_path is None:
            QMessageBox.information(self,"No reviewed result","Run and review an image analysis before adding it to a process study."); return
        dialog=StudyEntryDialog(self,self.study_factor_name,self.study_factor_unit,bool(self.study_records)); dialog.study_name.setText(self.study_name)
        if dialog.exec()!=QDialog.DialogCode.Accepted:return
        entry=dialog.values()
        if self.study_records and (entry["factor_name"]!=self.study_factor_name or entry["factor_unit"]!=self.study_factor_unit):
            QMessageBox.warning(self,"Study factor mismatch","All records in one study must use the same comparison factor and unit."); return
        self.study_name=entry["study_name"]; self.study_factor_name=entry["factor_name"]; self.study_factor_unit=entry["factor_unit"]
        stats=self._practical_stats(measurement_statistics(self.measurements)); metrics={}
        for row in stats:
            mean=row.get("Mean",np.nan)
            if not isinstance(mean,(int,float,np.number)) or not np.isfinite(mean):continue
            metrics[row["Metric"]]={"unit":row["Unit"],"n":int(row.get("N",0)),"mean":float(mean),
                                    "std":float(row.get("Std",np.nan)) if isinstance(row.get("Std"),(int,float,np.number)) else np.nan,
                                    "median":float(row.get("Median",np.nan)) if isinstance(row.get("Median"),(int,float,np.number)) else np.nan,
                                    "cv":float(row.get("CV (%)",np.nan)) if isinstance(row.get("CV (%)"),(int,float,np.number)) else np.nan}
        record={**entry,"image":str(self.image_path),"recipe":self.recipe.currentText(),"qa_status":self.qa_label.text(),
                "calibration":self.sampling_label.text(),"class_confirmed":self.confirm_classes.isChecked(),"metrics":metrics}
        self.study_records.append(record); self.statusBar().showMessage(f"Added {self.image_path.name} at {entry['factor_value']} {entry['factor_unit']} to {self.study_name}",7000)
        ProcessStudyDialog(self).exec()

    def open_process_study(self):
        ProcessStudyDialog(self).exec()

    def _begin_line_gauge(self):
        if self.analysis_rgb is None: QMessageBox.information(self,"No image","Open an SEM image first."); return
        self._drawing_scale_bar=False
        self.statusBar().showMessage("Drag through the particles of interest. The gauge reports calibrated lateral chords, gaps and spacing plus the SEM intensity profile; it does not measure height.")
        self.image_view.begin_line_selection()

    def _begin_scale_bar_redraw(self):
        if self.rgb is None: QMessageBox.information(self,"No image","Open a SEM image first."); return
        if self.metadata_sampling_um:
            answer=QMessageBox.question(self,"Metadata available","Companion pixel-size metadata is available and normally takes precedence. Draw a raster scale bar instead?")
            if answer!=QMessageBox.StandardButton.Yes: return
            self.metadata_sampling_um=None
        self._drawing_scale_bar=True; self.confirm_calibration.setChecked(False); self.calibration=None
        self.image_view.begin_line_selection()
        self.statusBar().showMessage("Draw exactly from one physical scale-bar endpoint to the other; exclude its text and unrelated footer marks.")

    def _line_selected(self,line):
        if self._drawing_scale_bar:
            self._drawing_scale_bar=False
            x1,y1,x2,y2=[float(value) for value in line]; pixels=float(np.hypot(x2-x1,y2-y1))
            if pixels<8:
                QMessageBox.warning(self,"Scale bar too short","Draw a longer line between the two physical scale-bar endpoints."); return
            self.bar_detection={"start_xy":(x1,y1),"end_xy":(x2,y2),"pixel_length":round(pixels,2),"confidence":1.0,
                                "footer_start":self.footer_start,"style":"operator_drawn"}
            self.detected_bar.setText(f"{pixels:.2f} px · operator-drawn endpoints")
            self.sampling_label.setText("Awaiting value confirmation")
            self.image_view.set_rgb(self._full_frame_display(self.analysis_rgb))
            self.statusBar().showMessage("Endpoints stored. Verify the printed value and unit, then confirm the calibration source.")
            return
        x1,y1,x2,y2=(float(v) for v in line); height,width=self.analysis_rgb.shape[:2]
        x1,x2=np.clip([x1,x2],0,width-1); y1,y2=np.clip([y1,y2],0,height-1)
        length_px=float(np.hypot(x2-x1,y2-y1)); count=max(2,int(np.ceil(length_px))+1)
        xs=np.linspace(x1,x2,count).astype(np.float32); ys=np.linspace(y1,y2,count).astype(np.float32)
        gray=to_gray(self.analysis_rgb).astype(np.float32); signal=cv2.remap(gray,xs.reshape(1,-1),ys.reshape(1,-1),cv2.INTER_LINEAR).ravel()
        distance_px=np.linspace(0,length_px,count); trend=np.polyval(np.polyfit(distance_px,signal,1),distance_px); residual=(signal-trend)/255*100
        ra=float(np.mean(np.abs(residual))); rq=float(np.sqrt(np.mean(residual**2)))
        if self.calibration is None and self.bar_detection and self.confirm_calibration.isChecked():
            physical=self.scale_length.value()/1000 if self.scale_unit.currentText()=="nm" else self.scale_length.value(); source="operator_drawn_scale_bar" if self.bar_detection.get("style")=="operator_drawn" else "confirmed_automatic_scale_bar"; self.calibration=Calibration.from_line(physical,self.bar_detection["start_xy"],self.bar_detection["end_xy"],source)
        scale=self.calibration.microns_per_pixel if self.calibration else 1.0; unit="µm" if self.calibration else "original px"
        distance=distance_px*scale; length=length_px*scale
        correlation,wavelength=spatial_descriptors(residual,scale)
        expected_min_px=max(4.0,self._model_size(self.min_size)/max(scale,1e-12))
        requested_polarity=self.line_polarity.currentText().lower().split()[0]
        chord_rows,profile_diagnostics=measure_lateral_profile(signal,scale,expected_min_px,requested_polarity)
        self.feature_rows=[]
        for row in chord_rows:
            self.feature_rows.append({"Chord":row["Chord"],f"Center position ({unit})":row["Center position"],f"Lateral chord ({unit})":row["Lateral chord"],f"Center spacing ({unit})":row["Center spacing"],f"Edge-to-edge gap ({unit})":row["Edge-to-edge gap"],"Peak prominence (intensity)":row["Peak prominence (intensity)"],"Detection confidence (%)":row["Detection confidence (%)"],"Polarity":row["Polarity"]})
        self.measurements=[Measurement("Drawn line length",length,unit,"operator endpoints in original image",1.0),Measurement("Profile Ra signal",ra,"% full-scale intensity","bilinear samples; linear detrend",.8),Measurement("Profile Rq signal",rq,"% full-scale intensity","bilinear samples; linear detrend",.8),Measurement("Profile correlation length",correlation,unit,"1/e autocorrelation crossing",.7),Measurement("Profile dominant wavelength",wavelength,unit,"largest non-DC FFT component",.65)]
        for row in chord_rows:
            confidence=row["Detection confidence (%)"]/100
            self.measurements.append(Measurement("Line-profile lateral particle chord",row["Lateral chord"],unit,"local half-prominence edge crossings; not particle height",confidence))
            if np.isfinite(row["Center spacing"]): self.measurements.append(Measurement("Line-profile center spacing",row["Center spacing"],unit,"successive chord centres",confidence))
            if np.isfinite(row["Edge-to-edge gap"]): self.measurements.append(Measurement("Line-profile edge-to-edge gap",row["Edge-to-edge gap"],unit,"successive half-prominence edges",confidence))
        canvas=self.analysis_rgb.copy(); a=(int(round(x1)),int(round(y1))); b=(int(round(x2)),int(round(y2))); cv2.line(canvas,a,b,(0,255,255),4,cv2.LINE_AA); cv2.circle(canvas,a,5,(255,255,0),-1); cv2.circle(canvas,b,5,(255,255,0),-1)
        for index,row in enumerate(chord_rows):
            left=row["Left edge (px)"]/max(length_px,1e-9); right=row["Right edge (px)"]/max(length_px,1e-9)
            start=(int(round(x1+left*(x2-x1))),int(round(y1+left*(y2-y1)))); end=(int(round(x1+right*(x2-x1))),int(round(y1+right*(y2-y1))))
            colour=(0,220,80) if index%2==0 else (255,170,0); cv2.line(canvas,start,end,colour,7,cv2.LINE_AA); cv2.circle(canvas,start,3,(255,255,255),-1); cv2.circle(canvas,end,3,(255,255,255),-1)
        mean_chord=float(np.mean([row["Lateral chord"] for row in chord_rows])) if chord_rows else np.nan
        label=f"{len(chord_rows)} lateral chords"+(f" · mean {mean_chord:.3f} {unit}" if np.isfinite(mean_chord) else "")+f" · Ra(signal) {ra:.3f}%"; cv2.putText(canvas,label,(min(a[0],b[0])+6,max(18,min(a[1],b[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
        self.overlay_rgb=self._full_frame_display(canvas); self.image_view.set_rgb(self.overlay_rgb); rows=self._practical_rows(self.feature_rows); self._set_table(self.results_table,rows)
        stats=self._practical_stats(measurement_statistics(self.measurements)); self._set_table(self.stats_table,stats); self._populate_plot_metrics(rows)
        intensity_key="SEM intensity (0–255)"; self.plot_series[intensity_key]=signal.tolist(); self.plot_metadata[intensity_key]=[{"label":f"Sample {i+1}","details":f"Distance {d:.5g} {unit}"} for i,d in enumerate(distance)]
        residual_key="Detrended intensity residual (%)"; self.plot_series[residual_key]=residual.tolist(); self.plot_metadata[residual_key]=self.plot_metadata[intensity_key]
        self.plot_metric.addItems([key for key in (intensity_key,residual_key) if self.plot_metric.findText(key)<0]); self.plot_metric.setCurrentIndex(0); self._update_plot(); self.tabs.setCurrentIndex(2)
        chord_values=np.asarray([row["Lateral chord"] for row in chord_rows],float); chord_cv=float(np.std(chord_values,ddof=1)/np.mean(chord_values)*100) if len(chord_values)>1 and np.mean(chord_values) else np.nan
        self.cards[0].value.setText(f"{len(chord_rows)} chords / {count} samples"); self.cards[1].value.setText(f"{mean_chord:.3f} {unit}" if np.isfinite(mean_chord) else f"{length:.3f} {unit} line"); self.cards[2].value.setText(f"Chord CV {chord_cv:.2f}%" if np.isfinite(chord_cv) else f"Ra {ra:.3f}%"); self.cards[3].value.setText("LATERAL LINE PROFILE")
        self.qa_label.setText("OPERATOR LATERAL LINE PROFILE"); self.qa_detail.setText(f"Detected {len(chord_rows)} {profile_diagnostics.get('polarity','')} signal lobes using local half-prominence crossings; coverage {profile_diagnostics.get('coverage_percent',0):.1f}%. Chords, gaps and spacing are calibrated lateral dimensions. A chord equals diameter only for a central transect of an approximately circular particle. Ra/Rq describe SEM signal intensity and are never height.")

    def run_surface_texture(self):
        """Quantify texture of a structure-free rectangular SEM signal region.

        SEM intensity is not calibrated height, so these are signal-texture indices rather
        than ISO topographic Ra/Sa in length units. A fitted intensity plane is removed first.
        """
        if self.analysis_rgb is None:
            QMessageBox.information(self,"No image","Open an SEM image, then draw a rectangular analysis region."); return
        if self.analysis_roi is None and self.analysis_polygon is None:
            QMessageBox.information(self,"Analysis region required","Draw a rectangular or freehand polygon region over structure-free background, then select Surface texture (Ra/Sa) again."); return
        if self.analysis_polygon is not None:
            x,y,w,h=cv2.boundingRect(self.analysis_polygon.reshape((-1,1,2)))
            region_mask=np.zeros((h,w),np.uint8); local=self.analysis_polygon-np.array([x,y]); cv2.fillPoly(region_mask,[local.astype(np.int32)],1)
            region_name="polygon"
        elif recipe == 6:
            x,y,w,h=self.analysis_roi; region_mask=np.ones((h,w),np.uint8); region_name="rectangle"
        crop=to_gray(self.analysis_rgb[y:y+h,x:x+w]).astype(float)
        if crop.size < 100 or min(crop.shape) < 5:
            QMessageBox.warning(self,"Region too small","Use a background rectangle at least 5 × 5 original-image pixels."); return
        yy,xx=np.indices(crop.shape); selected=region_mask.astype(bool)
        design=np.column_stack((xx[selected],yy[selected],np.ones(int(selected.sum()))))
        coefficients=np.linalg.lstsq(design,crop[selected],rcond=None)[0]
        plane=coefficients[0]*xx+coefficients[1]*yy+coefficients[2]
        residual=(crop-plane)/255.0*100.0; values=residual[selected]
        sa=float(np.mean(np.abs(values))); sq=float(np.sqrt(np.mean(values**2)))
        ssk=float(np.mean(values**3)/(sq**3)) if sq else np.nan; sku=float(np.mean(values**4)/(sq**4)) if sq else np.nan
        if self.calibration is None and self.bar_detection and self.confirm_calibration.isChecked():
            length_um=self.scale_length.value()/1000 if self.scale_unit.currentText()=="nm" else self.scale_length.value(); source="operator_drawn_scale_bar" if self.bar_detection.get("style")=="operator_drawn" else "confirmed_automatic_scale_bar"; self.calibration=Calibration.from_line(length_um,self.bar_detection["start_xy"],self.bar_detection["end_xy"],source)
        lateral_spacing=self.calibration.microns_per_pixel if self.calibration else 1.0; lateral_unit="µm" if self.calibration else "original px"
        row_ra=[]; row_correlation=[]; row_wavelength=[]
        axis=np.arange(w,dtype=float)
        for trace,row_mask in zip(crop,selected):
            positions=axis[row_mask]
            if len(positions)<5: continue
            samples=trace[row_mask]; detrended=samples-np.polyval(np.polyfit(positions,samples,1),positions)
            row_ra.append(float(np.mean(np.abs(detrended))/255.0*100.0))
            corr,wave=spatial_descriptors(detrended,lateral_spacing); row_correlation.append(corr); row_wavelength.append(wave)
        self.feature_rows=[{"Profile row":i+1,"Ra signal (%)":v,f"Correlation length ({lateral_unit})":c,f"Dominant wavelength ({lateral_unit})":wv} for i,(v,c,wv) in enumerate(zip(row_ra,row_correlation,row_wavelength))]
        self.measurements=[Measurement("Profile Ra signal",v,"% full-scale intensity","row linear detrend",.7) for v in row_ra]
        self.measurements += [Measurement("Areal Sa signal",sa,"% full-scale intensity","2-D fitted-plane residual",.7),Measurement("Areal Sq signal",sq,"% full-scale intensity","2-D fitted-plane residual",.7),Measurement("Areal Ssk signal",ssk,"ratio","2-D fitted-plane residual",.65),Measurement("Areal Sku signal",sku,"ratio","2-D fitted-plane residual",.65)]
        self.measurements += [Measurement("Profile correlation length",v,lateral_unit,"1/e autocorrelation crossing",.65) for v in row_correlation]+[Measurement("Profile dominant wavelength",v,lateral_unit,"largest non-DC FFT component",.6) for v in row_wavelength]
        if self.calibration:
            sampling=self.calibration.microns_per_pixel; area=float(selected.sum()*sampling**2); mean_length=float(np.mean(selected.sum(axis=1)[selected.sum(axis=1)>0])*sampling)
            self.measurements += [Measurement("Sampled lateral area",area,"µm²","selected original-image pixels",1.0),Measurement("Mean profile sampling length",mean_length,"µm","selected row spans",1.0)]
        canvas=self.analysis_rgb.copy()
        if self.analysis_polygon is not None: cv2.polylines(canvas,[self.analysis_polygon.astype(np.int32)],True,(255,180,0),4,cv2.LINE_AA)
        else: cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),(255,180,0),4)
        cv2.putText(canvas,f"SEM signal texture: Sa {sa:.3f}%  Sq {sq:.3f}%",(x+5,max(18,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,180,0),1,cv2.LINE_AA)
        self.overlay_rgb=self._full_frame_display(canvas); self.image_view.set_rgb(self.overlay_rgb)
        rows=self._practical_rows(self.feature_rows); self._set_table(self.results_table,rows)
        stats=self._practical_stats(measurement_statistics(self.measurements)); self._set_table(self.stats_table,stats); self._populate_plot_metrics(rows)
        self.cards[0].value.setText(f"{int(selected.sum())} px · {region_name}"); self.cards[1].value.setText(f"Sa {sa:.3f}%"); self.cards[2].value.setText(f"Sq {sq:.3f}%"); self.cards[3].value.setText("SIGNAL TEXTURE")
        self.qa_label.setText("SEM SIGNAL TEXTURE — not calibrated surface height"); self.qa_label.setObjectName("qaWarn")
        self.qa_detail.setText("Ra/Sa are reported as percent full-scale SEM intensity after linear/planar form removal. Do not report them as nm roughness without calibrated height/topography data.")
        self.model_warning_label.setText("Review that the selected area contains only the intended structure-free background and no labels, scale bar, charging gradient or detector artefact. Lateral calibration supplies area/length only—not height amplitude.")
        self.tabs.setCurrentIndex(2); self.statusBar().showMessage("Surface-signal texture analysis complete",6000)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEM image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.webp)")
        if not path: return
        try:
            source_image=Image.open(path)
            source_aspect=max(source_image.size)/max(1,min(source_image.size))
            if source_aspect>3.2:
                QMessageBox.warning(self,"Unsupported source image","This unusually wide file appears to be a desktop/application screenshot rather than a SEM acquisition. Export the actual SEM image before loading."); return
            self.rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open image", str(exc)); return
        early_aspect=max(self.rgb.shape[:2])/max(1,min(self.rgb.shape[:2])); early_colourfulness=float(np.mean(np.std(self.rgb[::max(1,self.rgb.shape[0]//500),::max(1,self.rgb.shape[1]//500)].astype(float),axis=2)))
        if early_aspect>2.8 and early_colourfulness>12:
            self.rgb=None
            QMessageBox.warning(self,"Unsupported source image","This file appears to be a colour desktop/application screenshot rather than a grayscale SEM micrograph. Export the actual SEM acquisition or crop it externally before loading."); return
        self.image_path = Path(path); self.footer_start = detect_footer(self.rgb); self.sem_metadata=read_sem_companion_metadata(path); self.metadata_sampling_um=(self.sem_metadata.get("pixel_size_nm",0)/1000 or None); self._drawing_scale_bar=False
        self.image_view.clear_reference(); self.image_view.clear_analysis_region(); self.calibration = None; self.pixel_mode=False; self.reference_roi = None; self.analysis_roi=None; self.analysis_polygon=None; self.exclusion_polygon=None; self.feature_rows = []; self.measurements = []; self.overlay_rgb = None; self.cross_section_records=[]; self.cross_section_overlays=[]; self.layer_candidate_interfaces=[]; self.layer_active_interfaces=[]; self.layer_roi=None
        self.reference_label.setText("No reference taught — automatic proposal mode")
        self.analysis_region_label.setText("Full specimen area")
        self.interface_selector.clear(); self.interface_edit_status.setText("Run layer analysis to review interfaces"); self.edit_interfaces_button.setText("Enable click-to-remove"); self.edit_interfaces_button.setStyleSheet("")
        self.analysis_rgb = self.rgb[:self.footer_start].copy() if self.footer_start < self.rgb.shape[0] else self.rgb.copy()
        aspect=max(self.rgb.shape[:2])/max(1,min(self.rgb.shape[:2])); colourfulness=float(np.mean(np.std(self.rgb.astype(float),axis=2)))
        warnings=[]
        if aspect>2.8 and colourfulness>12: warnings.append("The file appears to be a desktop/application screenshot rather than a grayscale SEM micrograph")
        if self.sem_metadata and "_raw" not in self.image_path.stem.lower(): warnings.append("Raster measurement labels may be present; prefer the matching raw acquisition or draw an exclusion mask")
        self.source_warning="; ".join(warnings) if warnings else None
        self.model_recommendation = None if (self.source_warning and "desktop/application screenshot" in self.source_warning) else self._recommend_model()
        if self.model_recommendation:
            _, orientation, confidence = self.model_recommendation
            self.recipe.setCurrentIndex(3); self.line_orientation.setCurrentIndex(1 if orientation == "vertical" else 2)
            self.recommendation_label.setText(f"Line-space/grating · {orientation} lines · confidence {confidence:.0%}. Finite feature length is not reported.")
        else:
            self.recommendation_label.setText("No high-confidence infinite-line signature; select the physical feature family and review its overlay.")
        detection = detect_scale_bar(self.rgb, self.footer_start)
        self.bar_detection = detection
        if self.metadata_sampling_um:
            self.detected_bar.setText(f"Companion metadata · {self.metadata_sampling_um*1000:.6g} nm/px · {self.sem_metadata.get('InstructName','SEM')}")
        elif detection:
            self.detected_bar.setText(f"{detection['pixel_length']} px · automatic proposal · confidence {detection['confidence']:.0%}")
        else:
            self.detected_bar.setText("Not detected — calibration cannot be released")
        self.confirm_calibration.setChecked(False); self.image_view.set_rgb(self._full_frame_display(self.analysis_rgb))
        for table in (self.results_table, self.stats_table, self.qa_table): self._set_table(table, [])
        self.plot_metric.clear(); self.distribution_plot.set_values([], "No results")
        for card in self.cards: card.value.setText("—")
        self.statusBar().showMessage(str(self.image_path))

    def _set_table(self, table: QTableWidget, rows: list[dict]):
        table.clear()
        if not rows: table.setRowCount(0); table.setColumnCount(0); return
        columns = list(rows[0].keys()); table.setColumnCount(len(columns)); table.setRowCount(len(rows)); table.setHorizontalHeaderLabels(columns)
        for r, row in enumerate(rows):
            for c, column in enumerate(columns):
                value = row.get(column, "")
                text = "—" if isinstance(value, float) and not np.isfinite(value) else (f"{value:.3f}" if isinstance(value, float) else str(value))
                table.setItem(r, c, QTableWidgetItem(text))
        table.setAlternatingRowColors(True); table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents); table.horizontalHeader().setStretchLastSection(True)

    def _calibrate(self) -> bool:
        if self.rgb is None: return False
        if self.metadata_sampling_um:
            if not self.confirm_calibration.isChecked():
                QMessageBox.warning(self,"Review required","Confirm the companion-metadata pixel size before analysis."); return False
            self.pixel_mode=False; self.calibration=Calibration(self.metadata_sampling_um,self.metadata_sampling_um,1.0,(0,0),(1,0),"companion_sem_metadata",True); self.sampling_label.setText(f"{self.metadata_sampling_um*1000:.6g} nm/px · companion metadata"); return True
        if not self.bar_detection:
            if not self.allow_pixel_mode.isChecked():
                QMessageBox.warning(self, "Calibration required", "A scale bar was not detected. Enable original-image pixel units to continue without physical calibration."); return False
            self.pixel_mode=True; self.calibration=Calibration(1.0,1.0,1.0,(0,0),(1,0),"original_image_pixels",True); self.sampling_label.setText("1 original-image px/px"); return True
        if not self.confirm_calibration.isChecked():
            QMessageBox.warning(self, "Review required", "Confirm the detected scale bar and its physical value before analysis."); return False
        self.pixel_mode=False
        length_um = self.scale_length.value() / 1000 if self.scale_unit.currentText() == "nm" else self.scale_length.value()
        source="operator_drawn_scale_bar" if self.bar_detection.get("style")=="operator_drawn" else "confirmed_automatic_scale_bar"
        self.calibration = Calibration.from_line(length_um, self.bar_detection["start_xy"], self.bar_detection["end_xy"],source)
        source_label="operator-drawn bar" if source.startswith("operator") else "confirmed automatic bar"
        self.sampling_label.setText(f"{self.calibration.microns_per_pixel * 1000:.3f} nm/px · {source_label}")
        return True

    def _model_size(self,control):
        return control.value() if self.pixel_mode else control.value()/1000

    def _accepted_region(self, cx, cy, margin_x=0, margin_y=0):
        probes=((cx,cy),(cx-margin_x,cy-margin_y),(cx+margin_x,cy-margin_y),(cx+margin_x,cy+margin_y),(cx-margin_x,cy+margin_y))
        if self.exclusion_polygon is not None:
            excluded=self.exclusion_polygon.reshape((-1,1,2)).astype(np.float32)
            if any(cv2.pointPolygonTest(excluded,(float(px),float(py)),False)>=0 for px,py in probes): return False
        if self.analysis_polygon is not None:
            contour=self.analysis_polygon.reshape((-1,1,2)).astype(np.float32)
            return all(cv2.pointPolygonTest(contour,(float(px),float(py)),False)>=0 for px,py in probes)
        if self.analysis_roi:
            x,y,w,h=self.analysis_roi; left,top,right,bottom=x,y,x+w,y+h
        else:
            left,top,right,bottom=0,0,self.analysis_rgb.shape[1],self.analysis_rgb.shape[0]
        guard=2 if self.reject_border.isChecked() else 0
        return cx-margin_x>left+guard and cy-margin_y>top+guard and cx+margin_x<right-guard and cy+margin_y<bottom-guard

    def run_analysis(self):
        if not self._calibrate(): return
        scale = self.calibration.microns_per_pixel; gray = to_gray(self.analysis_rgb); recipe = self.recipe.currentIndex()
        self.measurements = []; self.feature_rows = []; overlay = None; self.model_warning = self.source_warning
        if getattr(self,"model_recommendation",None) and recipe != self.model_recommendation[0]:
            orientation = self.model_recommendation[1]
            self.model_warning = f"Image has a high-confidence {orientation} infinite-line signature, but {self.recipe.currentText()} was selected. Use Line-space / grating unless a different physical model is intentionally justified."
            QMessageBox.warning(self,"Possible wrong recipe",self.model_warning)
        if recipe == 0:
            rows, _, circles = analyze_circular_outlines(gray, scale, self._model_size(self.min_size), self._model_size(self.max_size), self.sensitivity.value(), self.reject_border.isChecked(), self.reference_roi)
            kept=[(row,circle) for row,circle in zip(rows,circles) if self._accepted_region(circle[0],circle[1],circle[2],circle[2])]
            rows=[p[0] for p in kept]; circles=np.asarray([p[1] for p in kept],float)
            if self.analysis_roi and len(circles)>1:
                extent=2*float(np.median(circles[:,2])); points=circles[:,:2]
                for axis_index,axis in ((0,"X"),(1,"Y")):
                    model=infer_lattice_periods(points,axis_index,.55*extent,extent)
                    for row in rows:
                        row[f"Fundamental period {axis} (µm)"]=model["fundamental_px"]*scale; row[f"Period {axis} confidence (%)"]=model["confidence"]*100
                        for family in (1,2): row[f"Secondary period {axis} #{family} (µm)"]=(model["secondary_px"][family-1]*scale if family<=len(model["secondary_px"]) else np.nan)
            overlay = np.asarray(circle_overlay(self.analysis_rgb, circles, rows)); self.feature_rows = rows
            for axis, hint in (("X",self.period_x_hint.value()),("Y",self.period_y_hint.value())):
                if hint > 0:
                    for row in rows: row[f"Fundamental period {axis} (µm)"] = hint if self.pixel_mode else hint/1000; row[f"Period {axis} confidence (%)"] = 100.0
            for row in rows:
                self.measurements += [Measurement("Diameter", row["Diameter (µm)"], "µm", "fitted circle", 0.8), Measurement("Pitch", row["Nearest-neighbor pitch (µm)"], "µm", "nearest center", 0.75),Measurement("Circle edge roughness RMS",row["Circle edge roughness RMS (µm)"],"µm","radial residual from best-fit circle",.7),Measurement("Circle edge roughness 3σ",row["Circle edge roughness 3σ (µm)"],"µm","raw radial residual; noise inclusive",.7),Measurement("Filtered circle edge roughness 3σ",row["Filtered circle edge roughness 3σ (µm)"],"µm","3-point disclosed low-pass",.65)]
            if rows:
                for axis in ("X", "Y"):
                    confidence = rows[0].get(f"Period {axis} confidence (%)", 0)/100
                    fundamental = rows[0].get(f"Fundamental period {axis} (µm)", np.nan)
                    if np.isfinite(fundamental): self.measurements.append(Measurement(f"Fundamental period {axis}", fundamental, "µm", "missing-site lattice inference", confidence))
                    for family in (1,2):
                        secondary = rows[0].get(f"Secondary period {axis} #{family} (µm)", np.nan)
                        if np.isfinite(secondary): self.measurements.append(Measurement(f"Secondary period {axis} #{family}", secondary, "µm", "gap/superlattice family", confidence))
        elif recipe == 1:
            if self.reference_roi is None:
                QMessageBox.information(self, "Reference required", "Draw a tight box around one complete, representative rectangular feature. This prevents internal texture from being segmented as separate objects."); return
            if self.reference_roi[3] >= .70*gray.shape[0] or self.reference_roi[2] >= .70*gray.shape[1]:
                self.recipe.setCurrentIndex(3)
                QMessageBox.warning(self, "Infinite line detected", "The reference spans most of the field in one direction. It is not a finite rectangle, so length in that direction is undefined. The model was changed to Line-space / grating; run again for CD, pitch, duty cycle and LER/LWR.")
                return
            rows, boxes, period_model = analyze_rectangular_features(gray, scale, self.reference_roi, self.sensitivity.value(), self.reject_border.isChecked())
            kept=[]
            for row,geometry in zip(rows,boxes):
                box=geometry["box"] if isinstance(geometry,dict) else geometry
                if self._accepted_region((box[0]+box[2])/2,(box[1]+box[3])/2,(box[2]-box[0])/2,(box[3]-box[1])/2): kept.append((row,geometry))
            rows=[p[0] for p in kept]; boxes=[p[1] for p in kept]
            if self.analysis_roi and len(rows)>1:
                points=np.asarray([[row["Centroid X (px)"],row["Centroid Y (px)"]] for row in rows],float)
                period_model={"period_x":infer_lattice_periods(points,0,.55*self.reference_roi[3],self.reference_roi[2]),"period_y":infer_lattice_periods(points,1,.55*self.reference_roi[2],self.reference_roi[3])}
                for axis in ("x","y"):
                    model=period_model[f"period_{axis}"]; fundamental=model["fundamental_px"]*scale
                    secondary=model["secondary_px"][0]*scale if model["secondary_px"] else np.nan
                    for row in rows: row[f"Period {axis.upper()} (µm)"]=fundamental; row[f"Secondary period {axis.upper()} #1 (µm)"]=secondary
            for axis, hint in (("x",self.period_x_hint.value()),("y",self.period_y_hint.value())):
                if hint > 0 and period_model.get(f"period_{axis}"):
                    physical_hint=hint if self.pixel_mode else hint/1000; period_model[f"period_{axis}"]["fundamental_px"] = physical_hint/scale; period_model[f"period_{axis}"]["confidence"] = 1.0
                    for row in rows: row[f"Period {axis.upper()} (µm)"] = physical_hint
            overlay = np.asarray(rectangular_overlay(self.analysis_rgb, boxes, rows)); self.feature_rows = rows
            for row in rows:
                self.measurements += [
                    Measurement("Feature length", row["Length (µm)"], "µm", "four-edge local gauge", 0.85),
                    Measurement("Mean width", row["Mean width (µm)"], "µm", "multi-row edge gauge", 0.85),
                    Measurement("LER 3σ", row["LER 3σ (µm)"], "µm", "edge residual", 0.75),
                ]
            for axis in ("x", "y"):
                model = period_model.get(f"period_{axis}", {})
                fundamental = model.get("fundamental_px", np.nan)
                if np.isfinite(fundamental): self.measurements.append(Measurement(f"Fundamental period {axis.upper()}", fundamental*scale, "µm", "missing-site lattice inference", model.get("confidence",0)))
                for index, secondary in enumerate(model.get("secondary_px", []), start=1):
                    self.measurements.append(Measurement(f"Secondary period {axis.upper()} #{index}", secondary*scale, "µm", "gap/superlattice family", model.get("confidence",0)))
                for observed in model.get("observed_px", []):
                    self.measurements.append(Measurement(f"Observed spacing {axis.upper()}", observed*scale, "µm", "adjacent detected sites", 0.75))
        elif recipe == 2:
            min_area = max(3, int(np.pi * (self._model_size(self.min_size)/scale/2)**2)); max_area = int(np.pi * (self._model_size(self.max_size)/scale/2)**2)
            selected_polarity=self.line_polarity.currentText().lower().split()[0]
            nominal_diameter=None; reference_template=None
            if self.reference_roi:
                rx,ry,rw,rh=self.reference_roi; reference_crop=gray[ry:ry+rh,rx:rx+rw]
                global_contrast=float(np.percentile(gray,95)-np.percentile(gray,5))
                reference_contrast=float(np.percentile(reference_crop,95)-np.percentile(reference_crop,5)) if reference_crop.size else 0.0
                if min(rw,rh)>=8 and reference_contrast>=.60*max(global_contrast,1):
                    nominal_diameter=float(np.sqrt(rw*rh)); reference_template=reference_crop.copy()
                else: self.model_warning="The taught irregular-particle reference contains insufficient feature contrast and was ignored; redraw tightly around one complete particle."
            if selected_polarity=="auto":
                proposals=[analyze_shapes(gray,scale,p,min_area,max_area,self.reject_border.isChecked(),nominal_diameter,self.sensitivity.value(),reference_template) for p in ("bright","dark")]
                rows,_,contours=max(proposals,key=lambda item: len(item[0]))
            else: rows, _, contours = analyze_shapes(gray, scale, selected_polarity, min_area, max_area,self.reject_border.isChecked(),nominal_diameter,self.sensitivity.value(),reference_template)
            kept=[]
            for row,contour in zip(rows,contours):
                x,y,w,h=cv2.boundingRect(contour)
                if self._accepted_region(x+w/2,y+h/2,w/2,h/2): kept.append((row,contour))
            rows=[p[0] for p in kept]; contours=np.asarray([p[1] for p in kept],dtype=object)
            overlay = np.asarray(shape_overlay(self.analysis_rgb, contours, rows)); self.feature_rows = rows
            contact_count=sum(bool(row.get("Contact split")) for row in rows)
            for row in rows:
                contact=bool(row.get("Contact split")); prefix="Contact candidate — " if contact else "Isolated particle — "
                method_prefix="watershed visible segment; " if contact else ""
                confidence=.50 if contact else .80
                self.measurements += [Measurement(prefix+"Equivalent diameter",row["Equivalent diameter (µm)"],"µm",method_prefix+"segmented area",confidence),Measurement(prefix+"Length / Feret max",row["Feret max (µm)"],"µm",method_prefix+"maximum caliper",confidence),Measurement(prefix+"Width / Feret min",row["Feret min (µm)"],"µm",method_prefix+"minimum caliper",confidence),Measurement(prefix+"Projected area",row["Area (µm²)"],"µm²",method_prefix+"segmented contour",confidence),Measurement(prefix+"Perimeter",row["Perimeter (µm)"],"µm",method_prefix+"pixel contour",confidence),Measurement(prefix+"Circularity",row["Circularity"],"ratio",method_prefix+"contour geometry",confidence)]
                if contact_count:
                    combined="Combined visible segments — "; combined_method="mixed isolated/contact visible contours; "
                    self.measurements += [Measurement(combined+"Equivalent diameter",row["Equivalent diameter (µm)"],"µm",combined_method+"segmented area",.50),Measurement(combined+"Length / Feret max",row["Feret max (µm)"],"µm",combined_method+"maximum caliper",.50),Measurement(combined+"Width / Feret min",row["Feret min (µm)"],"µm",combined_method+"minimum caliper",.50),Measurement(combined+"Projected area",row["Area (µm²)"],"µm²",combined_method+"segmented contour",.50),Measurement(combined+"Perimeter",row["Perimeter (µm)"],"µm",combined_method+"pixel contour",.50),Measurement(combined+"Circularity",row["Circularity"],"ratio",combined_method+"contour geometry",.50)]
            if contact_count:
                contact_warning=f"{contact_count} of {len(rows)} contours touch neighbours and were watershed-split; visible-segment dimensions require operator review and do not reconstruct hidden boundaries."
                self.model_warning=(self.model_warning+" "+contact_warning) if self.model_warning else contact_warning
        elif recipe == 3:
            orientation = self.line_orientation.currentText().lower().split()[0]
            work_gray=gray; deskew=None; direction_deg=np.nan
            if orientation == "auto":
                direction_deg,direction_conf=estimate_line_direction(gray)
                horizontal_distance=min(direction_deg,180-direction_deg) if np.isfinite(direction_deg) else np.inf; vertical_distance=abs(direction_deg-90) if np.isfinite(direction_deg) else np.inf
                if min(horizontal_distance,vertical_distance)<=8: orientation="horizontal" if horizontal_distance<vertical_distance else "vertical"
                elif np.isfinite(direction_deg) and direction_conf>=.10:
                    rotation=90-direction_deg; center=(gray.shape[1]/2,gray.shape[0]/2); matrix=cv2.getRotationMatrix2D(center,rotation,1.0); deskew=(matrix,cv2.invertAffineTransform(matrix)); work_gray=cv2.warpAffine(gray,matrix,(gray.shape[1],gray.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT); orientation="vertical"
                else: orientation = self.model_recommendation[1] if getattr(self,"model_recommendation",None) else "vertical"
            polarity = self.line_polarity.currentText().lower().split()[0]
            if polarity == "auto":
                candidates = [analyze_line_space(work_gray,scale,p,self._model_size(self.min_size),self._model_size(self.max_size),orientation) for p in ("bright","dark")]
                rows, _, bands = max(candidates, key=lambda item: len(item[0]))
            else:
                rows, _, bands = analyze_line_space(work_gray, scale, polarity, self._model_size(self.min_size), self._model_size(self.max_size), orientation)
            if self.analysis_roi and deskew is None:
                x,y,w,h=self.analysis_roi
                kept=[]
                for row,band in zip(rows,bands):
                    center=(band["left"]+band["right"])/2 if isinstance(band,dict) else (band[0]+band[1])/2
                    inside_region = (x<center<x+w) if orientation=="vertical" else (y<center<y+h)
                    if inside_region: kept.append((row,band))
                rows=[p[0] for p in kept]; bands=[p[1] for p in kept]
            if deskew is not None:
                for row in rows: row["Line direction (°)"]=direction_deg; row["Orientation"]="diagonal"
                overlay=np.asarray(oriented_line_space_overlay(self.analysis_rgb,bands,deskew[1],direction_deg))
            else: overlay = np.asarray(line_space_overlay(self.analysis_rgb, bands, orientation))
            self.feature_rows = rows
            for row in rows:
                self.measurements += [Measurement("Line CD", row["Mean CD (µm)"], "µm", "multi-row gauge", 0.8), Measurement("LWR 3σ", row["LWR 3σ (µm)"], "µm", "raw row-width residual", 0.75),Measurement("Filtered LWR 3σ",row["Filtered LWR 3σ (µm)"],"µm","3-point disclosed low-pass",.7),Measurement("Left LER 3σ",row["Left LER 3σ (µm)"],"µm","raw fitted-edge residual",.75),Measurement("Right LER 3σ",row["Right LER 3σ (µm)"],"µm","raw fitted-edge residual",.75)]
                if np.isfinite(row["Pitch to next (µm)"]): self.measurements.append(Measurement("Pitch", row["Pitch to next (µm)"], "µm", "band centers", 0.8))
        elif recipe == 4:
            if self.reference_roi is None:
                QMessageBox.information(self,"Single-feature ROI required","Draw a tight reference rectangle around the complete isolated feature, leaving a small background margin."); return
            x,y,w,h=self.reference_roi; crop=gray[y:y+h,x:x+w]
            candidates=[]
            for polarity in ("bright","dark"):
                candidate_rows,_,candidate_contours=analyze_shapes(crop,scale,polarity,max(5,int(crop.size*.002)),int(crop.size*.85),False)
                for row,contour in zip(candidate_rows,candidate_contours):
                    if row.get("Touches border"): continue
                    candidates.append((row,contour))
            if candidates:
                row,contour=max(candidates,key=lambda pair: pair[0].get("Area (µm²)",0))
                contour=np.asarray(contour,dtype=np.int32).copy(); contour[:,:,0]+=x; contour[:,:,1]+=y
                row["Centroid X (px)"]+=x; row["Centroid Y (px)"]+=y; row["Object"]=1; row["Touches border"]=False
                rows=[row]; contours=np.asarray([contour],dtype=object); self.feature_rows=rows
                overlay=np.asarray(shape_overlay(self.analysis_rgb,contours,rows))
                metric_map=(("Area","Area (µm²)","µm²"),("Perimeter","Perimeter (µm)","µm"),("Feret max","Feret max (µm)","µm"),("Feret min","Feret min (µm)","µm"),("Major axis","Major axis (µm)","µm"),("Minor axis","Minor axis (µm)","µm"),("Aspect ratio","Aspect ratio","ratio"),("Circularity","Circularity","ratio"),("Solidity","Solidity","ratio"))
                for name,key,unit in metric_map:
                    if np.isfinite(row.get(key,np.nan)): self.measurements.append(Measurement(name,row[key],unit,"reviewed pixel contour",.8))
            else:
                rows=[]; contours=np.asarray([],dtype=object); self.feature_rows=[]; overlay=self.analysis_rgb.copy(); self.model_warning="No closed, non-border contour was found inside the single-feature ROI. Review contrast or redraw with background margin."
        elif recipe == 5:
            if self.analysis_roi is None and self.analysis_polygon is None:
                QMessageBox.information(self,"Waveguide region required","Draw a rectangle or freehand inclusion mask enclosing one complete, non-branching waveguide and some background."); return
            region=np.zeros_like(gray,np.uint8)
            if self.analysis_polygon is not None: cv2.fillPoly(region,[self.analysis_polygon.astype(np.int32)],1)
            else:
                x,y,w,h=self.analysis_roi; region[y:y+h,x:x+w]=1
            rows,diagnostics=analyze_waveguide(gray,scale,region); self.feature_rows=rows; overlay=waveguide_overlay(self.analysis_rgb,diagnostics) if rows else self.analysis_rgb.copy()
            if rows:
                self.measurements=[Measurement("Waveguide normal width",row["Normal width (µm)"],"µm","distance-transform normal gauge",.72) for row in rows]
                self.measurements += [Measurement("Waveguide path length",diagnostics["path_length"],"µm","longest medial path",.75),Measurement("Waveguide minimum width",diagnostics["min_width"],"µm","centerline-normal gauge",.72),Measurement("Waveguide maximum width",diagnostics["max_width"],"µm","centerline-normal gauge",.72),Measurement("Waveguide taper",diagnostics["taper"],"ratio","linear width change per path length",.65)]
                if diagnostics.get("branch_pixels",0)>3: self.model_warning="The extracted medial axis appears branched. This released model is for one non-branching waveguide; redraw the region or treat the result as REVIEW."
            else: self.model_warning="No coherent filled waveguide centerline was found. Review inclusion region, contrast and whether the structure is an outline rather than a filled guide."
        elif recipe == 6:
            if self.analysis_roi is None:
                QMessageBox.information(self,"Cross-section ROI required","Draw a rectangular analysis region tightly around one cross-section feature, including a small background margin."); return
            x,y,w,h=self.analysis_roi; crop_gray=gray[y:y+h,x:x+w]; measurements,diagnostics=measure_cross_section(crop_gray,scale)
            crop_rgb=self.analysis_rgb[y:y+h,x:x+w]; annotated=np.asarray(cross_section_overlay(crop_rgb,diagnostics)); overlay=self.analysis_rgb.copy()
            if measurements:
                identifier=len(self.cross_section_records)+1; row={"Structure":identifier,"ROI X (px)":x,"ROI Y (px)":y,"Height datum":diagnostics.get("height_datum","")}
                for m in measurements: row[f"{m.name} ({m.unit})"]=m.value; m.details={**m.details,"structure":identifier,"roi":(x,y,w,h),"height_datum":diagnostics.get("height_datum","")}
                self.cross_section_records.append({"row":row,"measurements":measurements}); self.cross_section_overlays.append((x,y,w,h,annotated))
                self.feature_rows=[record["row"] for record in self.cross_section_records]; self.measurements=[m for record in self.cross_section_records for m in record["measurements"]]
                for ox,oy,ow,oh,layer in self.cross_section_overlays: overlay[oy:oy+oh,ox:ox+ow]=layer
            else: self.feature_rows=[record["row"] for record in self.cross_section_records]; self.measurements=[m for record in self.cross_section_records for m in record["measurements"]]; self.model_warning="No coherent cross-section feature was found inside the ROI. Review polarity and include only one feature."
        else:
            if self.analysis_roi is None:
                QMessageBox.information(self,"Layer-stack ROI required","Draw a clean rectangular ROI through a laterally continuous film stack. Avoid raster labels and dimension arrows."); return
            x,y,w,h=self.analysis_roi; crop_gray=gray[y:y+h,x:x+w]; measurements,diagnostics=measure_layer_stack(crop_gray,scale); self.measurements=measurements
            self._apply_layer_period_guidance(measurements,diagnostics,w,scale)
            self.layer_roi=(x,y,w,h); self.layer_candidate_interfaces=list(diagnostics.get("interface_rows",[])); self.layer_active_interfaces=list(self.layer_candidate_interfaces)
            diagnostics["unit_scale"]=1.0 if self.pixel_mode else scale*1000; diagnostics["unit"]="px" if self.pixel_mode else "nm"
            crop_rgb=self.analysis_rgb[y:y+h,x:x+w]; annotated=np.asarray(cross_section_overlay(crop_rgb,diagnostics)); overlay=self.analysis_rgb.copy(); overlay[y:y+h,x:x+w]=annotated
            interfaces=diagnostics.get("interface_rows",[]); self.feature_rows=[{"Layer ID":f"L{i+1}","Upper interface":f"I{i+1}","Lower interface":f"I{i+2}","Upper interface Y (px)":y+upper,"Lower interface Y (px)":y+lower,"Thickness (µm)":(lower-upper)*scale} for i,(upper,lower) in enumerate(zip(interfaces,interfaces[1:]))]
            self.interface_selector.clear()
            for index,row in enumerate(self.layer_active_interfaces,start=1): self.interface_selector.addItem(f"I{index} — image Y={y+row}px")
            self.edit_interfaces_button.setEnabled(bool(interfaces)); self.remove_interface_button.setEnabled(len(interfaces)>2); self.restore_interfaces_button.setEnabled(bool(interfaces))
            self.interface_edit_status.setText(f"{len(interfaces)} proposed interfaces. Select an ID or enable click-to-remove.")
            if len(interfaces)<2: self.model_warning="Fewer than two credible horizontal interfaces were detected. Redraw a clean ROI or use manual line gauges; no layer thickness is releasable."
            else: self.model_warning="Candidate interfaces are numbered I1…I" + str(len(interfaces)) + ". Review them and use Edit detected interfaces to remove false lines before release."
        if overlay is None: return
        if not self.feature_rows:
            self.model_warning = self.model_warning or "Selected recipe produced no valid features. Review polarity, size limits, reference and feature family."
        elif recipe == 0 and self.reference_roi and max(self.reference_roi[2:])/max(1,min(self.reference_roi[2:])) > 1.55:
            self.model_warning = "Circular recipe selected, but the taught reference is strongly elongated. Review the recipe."
        elif recipe == 1 and len(self.feature_rows) < 3:
            self.model_warning = "Rectangular periodic-array recipe found fewer than three features; period statistics are not releasable."
        elif recipe == 3 and len(self.feature_rows) < 3:
            self.model_warning = "Line-space recipe found fewer than three bands; this may be a cross-section, isolated line, or wrong polarity rather than a grating."
        overlay=self._apply_class_aware_analysis(recipe,overlay)
        self.model_warning_label.setText("MODEL REVIEW: "+self.model_warning if self.model_warning else "Model compatibility checks passed")
        self.overlay_rgb = self._full_frame_display(overlay); self.image_view.set_rgb(self.overlay_rgb)
        display_rows = self._practical_rows(self.feature_rows); self._set_table(self.results_table, display_rows)
        stats = self._practical_stats(measurement_statistics(self.measurements)); self._set_table(self.stats_table, stats)
        self._populate_plot_metrics(display_rows)
        self._update_qa(stats)

    def _populate_plot_metrics(self, rows):
        previous = self.plot_metric.currentText(); self.plot_metric.blockSignals(True); self.plot_metric.clear(); self.plot_series = {}; self.plot_metadata={}
        if rows:
            excluded = {"Object", "Chord", "Centroid X (px)", "Centroid Y (px)", "Touches border", "Contact split"}
            for key, value in rows[0].items():
                if key not in excluded and isinstance(value, (int, float, np.number)):
                    groups=[("",rows)]
                    if self.recipe.currentIndex()==2 and any("Contact split" in row for row in rows):
                        groups=[("Combined visible segments — ",rows),
                                ("Isolated particle — ",[row for row in rows if not row.get("Contact split")]),
                                ("Contact candidate — ",[row for row in rows if row.get("Contact split")])]
                    for prefix,group_rows in groups:
                        if not group_rows: continue
                        plot_key=prefix+key; self.plot_series[plot_key]=[row.get(key,np.nan) for row in group_rows]
                        metadata=[]
                        for index,row in enumerate(group_rows):
                            identifier=next((f"{name} {row[name]}" for name in ("Object","Chord","Band","Structure","Layer ID","Profile row","Sample") if name in row),f"Point {index+1}")
                            coordinates=""
                            if "Centroid X (px)" in row and "Centroid Y (px)" in row: coordinates=f"Centroid: ({row['Centroid X (px)']}, {row['Centroid Y (px)']}) original px"
                            metadata.append({"label":identifier,"class":row.get("Size class",""),"details":coordinates})
                        self.plot_metadata[plot_key]=metadata
        for measurement in self.measurements:
            if not np.isfinite(measurement.value): continue
            unit = ("px" if self.pixel_mode else "nm") if measurement.unit == "µm" else (("px²" if self.pixel_mode else "nm²") if measurement.unit=="µm²" else measurement.unit)
            key = f"{measurement.name} ({unit})"
            if key in self.plot_series: continue
            value = measurement.value if self.pixel_mode else (measurement.value*1000 if measurement.unit == "µm" else (measurement.value*1_000_000 if measurement.unit=="µm²" else measurement.value))
            self.plot_series.setdefault(key,[]).append(value)
            self.plot_metadata.setdefault(key,[]).append({"label":measurement.details.get("class","Measurement"),"class":measurement.details.get("class",""),"details":measurement.method})
        self.plot_metric.addItems(list(self.plot_series))
        index = self.plot_metric.findText(previous); self.plot_metric.setCurrentIndex(max(0, index)); self.plot_metric.blockSignals(False); self._update_plot()

    def _update_plot(self):
        key = self.plot_metric.currentText()
        values = self.plot_series.get(key,[])
        self.distribution_plot.set_values(values,key or "No results",self.plot_metadata.get(key,[]))

    def _practical_rows(self, rows):
        converted = []
        for row in rows:
            out = {}
            for key, value in row.items():
                if key.endswith("(µm)"): out[key.replace("(µm)", "(px)" if self.pixel_mode else "(nm)")] = value if self.pixel_mode else value * 1000
                elif key.endswith("(µm²)"): out[key.replace("(µm²)", "(px²)" if self.pixel_mode else "(nm²)")] = value if self.pixel_mode else value * 1_000_000
                else: out[key] = value
            converted.append(out)
        return converted

    def _practical_stats(self, rows):
        for row in rows:
            if row["Unit"] == "µm":
                if not self.pixel_mode:
                    for key in ("Mean", "Std", "3σ", "Std error", "Mean 95% CI ±", "Median", "Min", "Max", "Range"): row[key] *= 1000
                row["Unit"] = "px" if self.pixel_mode else "nm"
            elif row["Unit"] == "µm²":
                if not self.pixel_mode:
                    for key in ("Mean", "Std", "3σ", "Std error", "Mean 95% CI ±", "Median", "Min", "Max", "Range"): row[key] *= 1_000_000
                row["Unit"] = "px²" if self.pixel_mode else "nm²"
        return rows

    def _update_qa(self, stats):
        count = len(self.feature_rows); sampling = 1.0 if self.pixel_mode else self.calibration.microns_per_pixel * 1000
        warnings = []
        if count < 3 and self.recipe.currentIndex() not in (4,6,7): warnings.append("fewer than 3 features; population dispersion is not reliable")
        length_unit="px" if self.pixel_mode else "nm"
        if self.recipe.currentIndex()==2:
            primary=next((r for r in stats if r["Unit"]==length_unit and r["Metric"].startswith("Isolated particle — ")),None)
            if primary is None: primary=next((r for r in stats if r["Unit"]==length_unit),None)
        else: primary=next((r for r in stats if r["Unit"]==length_unit),None)
        if primary and primary["Mean"] / sampling < 10: warnings.append("primary dimension is sampled by fewer than 10 pixels")
        if primary and np.isfinite(primary["CV (%)"]) and self.recipe.currentIndex() == 0 and primary["CV (%)"] > 12 and not self.class_model:
            warnings.append("diameter population CV exceeds 12%; review model limits or mixed populations")
        if self.recipe.currentIndex() == 0 and self.feature_rows:
            fit_support = float(np.mean([row.get("Edge fit support (%)", 0) for row in self.feature_rows]))
            if fit_support < 60: warnings.append(f"mean circular edge support is only {fit_support:.1f}%")
            relative_fit = np.mean([row.get("Circle fit RMSE (µm)", np.inf) / max(row.get("Diameter (µm)", 0), 1e-12) for row in self.feature_rows]) * 100
            if relative_fit > 3: warnings.append(f"mean circle-fit residual is {relative_fit:.1f}% of diameter")
        if self.recipe.currentIndex() == 1 and self.feature_rows:
            match = float(np.mean([row.get("Template match (%)", 0) for row in self.feature_rows]))
            if match < 55: warnings.append(f"mean taught-feature match is only {match:.1f}%")
            if not self.reference_roi: warnings.append("rectangular-array model requires an operator reference")
        if not stats: warnings.append("no valid measurements")
        if getattr(self,"model_warning",None): warnings.append(self.model_warning)
        status = "REVIEW" if warnings else "RELEASE CANDIDATE"
        self.qa_label.setText(status); self.qa_label.setObjectName("qaWarn" if warnings else "qaGood"); self.qa_label.style().unpolish(self.qa_label); self.qa_label.style().polish(self.qa_label)
        self.qa_detail.setText("; ".join(warnings) if warnings else "Calibration, sampling, population, and model checks passed. Operator acceptance is still required.")
        if self.recipe.currentIndex()==2 and self.feature_rows:
            contact_count=sum(bool(row.get("Contact split")) for row in self.feature_rows)
            self.cards[0].value.setText(f"{count-contact_count} isolated / {contact_count} contact")
        else: self.cards[0].value.setText(str(count))
        self.cards[1].value.setText(f"{primary['Mean']:.2f} {'px' if self.pixel_mode else 'nm'}" if primary else "—")
        self.cards[2].value.setText(f"{primary['CV (%)']:.2f}%" if primary and np.isfinite(primary["CV (%)"]) else "—")
        if self.class_model:
            class_stats=[row for row in stats if " — Class " in row["Metric"] and row["Unit"]==("px" if self.pixel_mode else "nm")]
            if class_stats:
                self.cards[1].value.setText(" / ".join(f"{row['Metric'].split('Class ')[-1]} {row['Mean']:.1f}" for row in class_stats)+f" {'px' if self.pixel_mode else 'nm'}")
                self.cards[2].value.setText(" / ".join(f"{row['Metric'].split('Class ')[-1]} {row['CV (%)']:.1f}%" for row in class_stats if np.isfinite(row["CV (%)"])))
        self.cards[3].value.setText(status)
        population_ok=count>=3 or (self.recipe.currentIndex() in (4,6,7) and count>=1)
        qa_rows = [{"Check": "Calibration", "Result": "PASS", "Evidence": self.sampling_label.text()}, {"Check": "Population", "Result": "PASS" if population_ok else "REVIEW", "Evidence": f"N={count}"}, {"Check": "Sampling", "Result": "PASS" if primary and primary["Mean"]/sampling >= 10 else "REVIEW", "Evidence": f"{primary['Mean']/sampling:.1f} px" if primary else "—"}, {"Check": "Reference model", "Result": "GUIDED" if self.reference_roi else "AUTO", "Evidence": self.reference_label.text()}, {"Check": "Release", "Result": status, "Evidence": self.qa_detail.text()}]
        if self.class_model:
            qa_rows.insert(-1,{"Check":"Population classes","Result":"CONFIRMED" if self.confirm_classes.isChecked() else "REVIEW",
                               "Evidence":f"{self.class_model['class_count']} classes; confidence {self.class_model['confidence']:.0%}; counts {self.class_model['counts'].tolist()}"})
        self._set_table(self.qa_table, qa_rows); self.tabs.setCurrentWidget(self.stats_table)

    def export_report(self):
        if not self.measurements: QMessageBox.information(self, "Nothing to export", "Run an analysis first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export statistical report", "metrology_statistics.csv", "CSV (*.csv)")
        if not path: return
        rows = self._practical_stats(measurement_statistics(self.measurements));
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if self.cross_section_records:
            detail_path=str(Path(path).with_name(Path(path).stem+"_structures.csv")); detail_rows=self._practical_rows([record["row"] for record in self.cross_section_records]); columns=[]
            for row in detail_rows:
                for key in row:
                    if key not in columns: columns.append(key)
            with open(detail_path,"w",newline="",encoding="utf-8-sig") as handle:
                writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore"); writer.writeheader(); writer.writerows(detail_rows)
            self.statusBar().showMessage(f"Exported statistics and {detail_path}",5000)
        else: self.statusBar().showMessage(f"Exported {path}", 5000)

    def save_recipe(self):
        path,_=QFileDialog.getSaveFileName(self,"Save measurement recipe","sem_metrology_recipe.json","JSON (*.json)")
        if not path: return
        data={"schema":1,"recipe":self.recipe.currentText(),"minimum_size":self.min_size.value(),"maximum_size":self.max_size.value(),"sensitivity":self.sensitivity.value(),"line_orientation":self.line_orientation.currentText(),"line_polarity":self.line_polarity.currentText(),"period_x_hint":self.period_x_hint.value(),"period_y_hint":self.period_y_hint.value(),"layer_period_count":self.layer_period_count.value(),"class_mode":self.class_mode.currentText(),"confirm_classes":self.confirm_classes.isChecked(),"reject_border":self.reject_border.isChecked(),"allow_pixel_mode":self.allow_pixel_mode.isChecked()}
        with open(path,"w",encoding="utf-8") as handle: json.dump(data,handle,indent=2)
        self.statusBar().showMessage(f"Saved recipe {path}",5000)

    def load_recipe(self):
        path,_=QFileDialog.getOpenFileName(self,"Load measurement recipe","","JSON (*.json)")
        if not path: return
        try:
            with open(path,"r",encoding="utf-8") as handle: data=json.load(handle)
            index=self.recipe.findText(data["recipe"])
            if index<0: raise ValueError("Recipe type is not supported by this application version")
            self.recipe.setCurrentIndex(index); self.min_size.setValue(float(data["minimum_size"])); self.max_size.setValue(float(data["maximum_size"])); self.sensitivity.setValue(float(data["sensitivity"])); self.line_orientation.setCurrentText(data.get("line_orientation","Auto")); self.line_polarity.setCurrentText(data.get("line_polarity","Auto")); self.period_x_hint.setValue(float(data.get("period_x_hint",0))); self.period_y_hint.setValue(float(data.get("period_y_hint",0))); self.layer_period_count.setValue(int(data.get("layer_period_count",0))); self.class_mode.setCurrentText(data.get("class_mode","Auto population model")); self.confirm_classes.setChecked(bool(data.get("confirm_classes",False))); self.reject_border.setChecked(bool(data.get("reject_border",True))); self.allow_pixel_mode.setChecked(bool(data.get("allow_pixel_mode",False)))
        except Exception as exc: QMessageBox.critical(self,"Cannot load recipe",str(exc)); return
        self.statusBar().showMessage(f"Loaded recipe {path}. Calibration and ROIs remain image-specific.",6000)

    def export_plot(self):
        if not len(self.distribution_plot.values):
            QMessageBox.information(self, "No plot", "Run an analysis and choose a plotted metric first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export current plot", "metrology_distribution.png", "PNG image (*.png)")
        if path and not self.distribution_plot.grab().save(path, "PNG"):
            QMessageBox.warning(self, "Export failed", "The plot image could not be written.")

    def export_professional_report(self):
        if not self.measurements: QMessageBox.information(self,"Nothing to report","Run an analysis first."); return
        dialog=QDialog(self); dialog.setWindowTitle("PDF report contents"); layout=QVBoxLayout(dialog); layout.addWidget(QLabel("Select report sections:"))
        options={}
        for key,label,default in (("overlay","Calibrated image and measurement overlay",True),("measurements","Per-structure / per-feature table",True),("statistics","Statistical summary table",True),("plot","Current distribution/control plot",True),("qa","QA, calibration and method notes",True)):
            box=QCheckBox(label); box.setChecked(default); options[key]=box; layout.addWidget(box)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if dialog.exec()!=QDialog.DialogCode.Accepted: return
        path,_=QFileDialog.getSaveFileName(self,"Export professional report","SEM_metrology_report.pdf","PDF (*.pdf)")
        if not path: return
        writer=QPdfWriter(path); writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4)); writer.setResolution(144); writer.setTitle("SEM Metrology Report"); painter=QPainter(writer); painter.setWindow(0,0,1200,1700)
        normal=QFont("Segoe UI",10); heading=QFont("Segoe UI",20,QFont.Weight.Bold); subheading=QFont("Segoe UI",14,QFont.Weight.Bold)
        def header(title):
            painter.setPen(QColor("#123447")); painter.setFont(heading); painter.drawText(70,90,title); painter.setPen(QColor("#2c7c83")); painter.drawLine(70,112,1130,112); painter.setFont(normal); painter.setPen(Qt.GlobalColor.black)
        def new_page(title): writer.newPage(); painter.setWindow(0,0,1200,1700); header(title)
        def table_page(title,rows):
            if not rows: return
            columns=list(rows[0]); chunks=[rows[i:i+24] for i in range(0,len(rows),24)]
            for page_index,chunk in enumerate(chunks):
                new_page(title+(f" ({page_index+1}/{len(chunks)})" if len(chunks)>1 else "")); usable=1060; colw=max(90,int(usable/max(1,len(columns)))); painter.setFont(QFont("Segoe UI",7,QFont.Weight.Bold)); y=155
                for c,name in enumerate(columns): painter.drawText(70+c*colw,y,colw-4,42,Qt.TextFlag.TextWordWrap,str(name)[:30])
                painter.setFont(QFont("Segoe UI",7)); y+=48
                for row in chunk:
                    painter.setPen(QColor("#d0d7dc")); painter.drawLine(70,y-15,1130,y-15); painter.setPen(Qt.GlobalColor.black)
                    for c,name in enumerate(columns):
                        value=row.get(name,""); text_value=(f"{value:.4g}" if isinstance(value,float) else str(value)); painter.drawText(70+c*colw,y,colw-4,36,Qt.TextFlag.TextWordWrap,text_value[:42])
                    y+=55
        header("SEM Metrology Report"); painter.setFont(subheading); painter.drawText(70,170,self.recipe.currentText()); painter.setFont(normal)
        painter.drawText(70,220,f"Image: {self.image_path.name if self.image_path else 'unsaved image'}"); painter.drawText(70,255,f"Software: SEM Metrology Workbench {APP_VERSION}"); painter.drawText(70,290,f"Calibration: {self.sampling_label.text()}"); painter.drawText(70,325,f"Population: N={len(self.feature_rows)}"); painter.drawText(70,360,f"Release status: {self.qa_label.text()}"); painter.drawText(70,410,"This report preserves operator-selected calibration, model, overlays, statistics and QA state.")
        if options["overlay"].isChecked() and self.overlay_rgb is not None:
            new_page("Measurement overlay"); pix=array_to_pixmap(self.overlay_rgb); target=QRectF(70,150,1060,1150); scaled=pix.scaled(int(target.width()),int(target.height()),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation); painter.drawPixmap(70,150,scaled); painter.drawText(70,1360,"Overlay colours and gauge definitions are documented in Help → Measurement definitions.")
        if options["measurements"].isChecked(): table_page("Individual measurements",self._practical_rows(self.feature_rows))
        if options["statistics"].isChecked(): table_page("Statistical summary",self._practical_stats(measurement_statistics(self.measurements)))
        if options["plot"].isChecked() and len(self.distribution_plot.values):
            new_page("Distribution / control plot"); plot=self.distribution_plot.grab(); scaled=plot.scaled(1060,1000,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation); painter.drawPixmap(70,170,scaled); painter.drawText(70,1230,f"Metric: {self.plot_metric.currentText()}")
        if options["qa"].isChecked():
            new_page("QA and traceability"); painter.setFont(subheading); painter.drawText(70,170,self.qa_label.text()); painter.setFont(normal); painter.drawText(QRectF(70,210,1060,240),Qt.TextFlag.TextWordWrap,self.qa_detail.text()); painter.drawText(QRectF(70,460,1060,180),Qt.TextFlag.TextWordWrap,self.model_warning_label.text()); painter.drawText(70,680,f"Scale-bar detection: {self.detected_bar.text()}"); painter.drawText(70,720,f"Sampling: {self.sampling_label.text()}"); painter.drawText(70,760,f"Analysis region: {self.analysis_region_label.text()}"); painter.drawText(QRectF(70,820,1060,500),Qt.TextFlag.TextWordWrap,"Release note: automated SEM measurements remain sensitive to contrast, charging, focus, edge model, ROI placement and calibration. Candidate layer interfaces and REVIEW results require operator disposition. Confidence intervals quantify population-mean dispersion and are not a complete instrument uncertainty budget.")
        painter.end(); self.statusBar().showMessage(f"Exported professional report {path}",6000)


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(APP_STYLE)
    window = MetrologyWorkbench(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
