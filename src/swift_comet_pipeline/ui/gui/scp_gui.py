# from __future__ import annotations
# from typing import Mapping, Callable, Optional, Dict, Any
# from dataclasses import dataclass
#
#
# from PySide6.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QWidget,
#     QScrollArea,
#     QVBoxLayout,
#     QGridLayout,
#     QPushButton,
#     QComboBox,
#     QLineEdit,
#     QHBoxLayout,
#     QLabel,
# )
# from PySide6.QtCore import Qt
#
#
# # ---- Status palette & helpers ----------------------------------------------
#
#
# def _canon_status_name(name: str) -> str:
#     n = name.upper()
#     if "ERROR" in n or "FAIL" in n or "BLOCK" in n:
#         return "ERROR"
#     if "REGEN" in n or "REBUILD" in n or "FORCE" in n:
#         return "NEED_REGEN"
#     if "MISSING" in n or "ABSENT" in n or "NOT_FOUND" in n:
#         return "MISSING"
#     if "STALE" in n or "OUTDATED" in n:
#         return "STALE"
#     if "READY" in n or "BUILDABLE" in n or "PENDING" in n:
#         return "READY"
#     if "COMPLETE" in n or "DONE" in n or "FRESH" in n:
#         return "COMPLETE"
#     return "UNKNOWN"
#
#
# PALETTE: Dict[str, str] = {
#     "ERROR": "#b00020",
#     "NEED_REGEN": "#8e24aa",
#     "MISSING": "#d32f2f",
#     "STALE": "#f9a825",
#     "READY": "#0288d1",
#     "COMPLETE": "#2e7d32",
#     "UNKNOWN": "#9e9e9e",
# }
# LABELS: Dict[str, str] = {
#     "ERROR": "Error",
#     "NEED_REGEN": "Needs Regen",
#     "MISSING": "Missing",
#     "STALE": "Stale",
#     "READY": "Ready",
#     "COMPLETE": "Complete",
#     "UNKNOWN": "Unknown",
# }
# ORDER = {
#     "ERROR": 0,
#     "NEED_REGEN": 1,
#     "MISSING": 2,
#     "STALE": 3,
#     "READY": 4,
#     "COMPLETE": 5,
#     "UNKNOWN": 6,
# }
#
#
# def _canon_from_enum(status_obj: Any) -> str:
#     name = getattr(status_obj, "name", str(status_obj))
#     return _canon_status_name(name)
#
#
# def _contrast_text(hex_color: str) -> str:
#     """Return '#000' or '#fff' based on perceived luminance."""
#     c = hex_color.lstrip("#")
#     r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
#     # WCAG relative luminance-ish
#     luminance = (
#         0.2126 * (r / 255) ** 2.2
#         + 0.7152 * (g / 255) ** 2.2
#         + 0.0722 * (b / 255) ** 2.2
#     )
#     return "#000000" if luminance > 0.5 else "#ffffff"
#
#
# # ---- UI ---------------------------------------------------------------------
#
#
# @dataclass
# class ProductButtonInfo:
#     ref: Any
#     status_bucket: str
#     label: str
#
#
# class ProductStatusWindow(QMainWindow):
#     def __init__(
#         self,
#         status_map: Mapping[Any, Any],
#         *,
#         name_fn: Callable[[Any], str] = str,
#         on_product_clicked: Optional[Callable[[Any], None]] = None,
#         columns: int = 3,
#         title: str = "Product Status",
#     ) -> None:
#         super().__init__()
#         self.setWindowTitle(title)
#         self._name_fn = name_fn
#         self._on_click = on_product_clicked or (lambda ref: print(f"clicked: {ref}"))
#         self._columns = max(1, columns)
#
#         # Normalize to display info objects
#         self._items: list[ProductButtonInfo] = []
#         for ref, st in status_map.items():
#             bucket = _canon_from_enum(st)
#             self._items.append(
#                 ProductButtonInfo(ref=ref, status_bucket=bucket, label=name_fn(ref))
#             )
#         self._items.sort(
#             key=lambda x: (ORDER.get(x.status_bucket, 99), x.label.lower())
#         )
#
#         # Top controls
#         top = QWidget()
#         top_layout = QHBoxLayout(top)
#         top_layout.setContentsMargins(4, 4, 4, 4)
#         top_layout.setSpacing(8)
#
#         self.filter_combo = QComboBox()
#         self.filter_combo.addItem("All statuses")
#         for bucket in sorted(
#             {i.status_bucket for i in self._items}, key=lambda b: ORDER.get(b, 99)
#         ):
#             self.filter_combo.addItem(LABELS[bucket], userData=bucket)
#         self.filter_combo.currentIndexChanged.connect(self._rebuild_grid)
#
#         self.search = QLineEdit()
#         self.search.setPlaceholderText("Search products…")
#         self.search.textChanged.connect(self._rebuild_grid)
#
#         legend = self._build_legend()
#
#         top_layout.addWidget(QLabel("Filter:"))
#         top_layout.addWidget(self.filter_combo, 0)
#         top_layout.addSpacing(12)
#         top_layout.addWidget(self.search, 1)
#         top_layout.addStretch(1)
#         top_layout.addWidget(legend, 0, Qt.AlignRight)
#
#         # Scroll area with grid of buttons
#         self.grid_container = QWidget()
#         self.grid_layout = QGridLayout(self.grid_container)
#         self.grid_layout.setContentsMargins(8, 8, 8, 8)
#         self.grid_layout.setHorizontalSpacing(8)
#         self.grid_layout.setVerticalSpacing(8)
#
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)
#         scroll.setWidget(self.grid_container)
#
#         # Main layout
#         central = QWidget()
#         v = QVBoxLayout(central)
#         v.setContentsMargins(6, 6, 6, 6)
#         v.setSpacing(8)
#         v.addWidget(top)
#         v.addWidget(scroll, 1)
#         self.setCentralWidget(central)
#
#         self._rebuild_grid()
#
#     def _build_legend(self) -> QWidget:
#         w = QWidget()
#         h = QHBoxLayout(w)
#         h.setContentsMargins(0, 0, 0, 0)
#         h.setSpacing(6)
#         for key in sorted(PALETTE.keys(), key=lambda k: ORDER.get(k, 99)):
#             swatch = QLabel(LABELS[key])
#             bg = PALETTE[key]
#             fg = _contrast_text(bg)
#             swatch.setStyleSheet(
#                 f"QLabel {{ background-color: {bg}; color: {fg}; "
#                 f"padding: 4px 8px; border-radius: 6px; font-weight: 600; }}"
#             )
#             h.addWidget(swatch)
#         return w
#
#     def _rebuild_grid(self) -> None:
#         # Clear existing widgets
#         while self.grid_layout.count():
#             item = self.grid_layout.takeAt(0)
#             w = item.widget()
#             if w is not None:
#                 w.setParent(None)
#
#         # Apply filters
#         text = self.search.text().strip().lower()
#         idx = self.filter_combo.currentIndex()
#         bucket_filter = self.filter_combo.itemData(idx) if idx > 0 else None
#
#         filtered = [
#             i
#             for i in self._items
#             if (not bucket_filter or i.status_bucket == bucket_filter)
#             and (not text or text in i.label.lower())
#         ]
#
#         # Populate buttons
#         for n, info in enumerate(filtered):
#             btn = QPushButton(info.label)
#             btn.setCursor(Qt.PointingHandCursor)
#             btn.setToolTip(f"{LABELS[info.status_bucket]}")
#             self._apply_button_style(btn, info.status_bucket)
#             # capture ref in lambda default
#             btn.clicked.connect(lambda _=False, ref=info.ref: self._on_click(ref))
#             r, c = divmod(n, self._columns)
#             self.grid_layout.addWidget(btn, r, c)
#
#         # Add stretch to keep left/top aligned
#         rows = (len(filtered) + self._columns - 1) // self._columns
#         self.grid_layout.setRowStretch(rows, 1)
#         self.grid_layout.setColumnStretch(self._columns, 1)
#
#     def _apply_button_style(self, btn: QPushButton, bucket: str) -> None:
#         bg = PALETTE[bucket]
#         fg = _contrast_text(bg)
#         btn.setStyleSheet(
#             "QPushButton {"
#             f"background-color: {bg}; color: {fg};"
#             "padding: 8px 12px; border: none; border-radius: 10px;"
#             "font-weight: 600;"
#             "}"
#             "QPushButton:hover {"
#             "filter: brightness(1.1);"
#             "}"
#             "QPushButton:pressed {"
#             "transform: scale(0.99);"
#             "}"
#         )
#
#     # Optional: live update for a product’s status
#     def update_product_status(self, ref: Any, new_status_obj: Any) -> None:
#         bucket = _canon_from_enum(new_status_obj)
#         for info in self._items:
#             if info.ref == ref:
#                 info.status_bucket = bucket
#                 break
#         self._rebuild_grid()
#
#
# def main():
#     # # Mock types
#     # class ProductStatus:
#     #     def __init__(self, name):
#     #         self.name = name
#
#     # Pretend ProductReference -> ProductStatus mapping
#     demo_statuses = {
#         ("stack", "epoch_001"): ProductStatus("READY"),
#         ("stack", "epoch_002"): ProductStatus("MISSING"),
#         ("phot", "epoch_001"): ProductStatus("STALE"),
#         ("phot", "epoch_002"): ProductStatus("COMPLETE"),
#         ("oh", "epoch_001"): ProductStatus("NEED_REGEN"),
#         ("oh", "epoch_002"): ProductStatus("ERROR"),
#         ("profile", "epoch_001"): ProductStatus("COMPLETE"),
#         ("profile", "epoch_002"): ProductStatus("READY"),
#     }
#
#     def name_fn(ref):  # nice labels
#         kind, key = ref
#         return f"{kind}:{key}"
#
#     def on_click(ref):
#         print(f"[GUI] Clicked {ref}")
#
#     app = QApplication([])
#     win = ProductStatusWindow(
#         demo_statuses, name_fn=name_fn, on_product_clicked=on_click, columns=3
#     )
#     win.resize(900, 600)
#     win.show()
#     app.exec()
#
#
# if __name__ == "__main__":
#     main()

# animated_stacked_demo.py
import sys
from PySide6.QtCore import (
    Qt,
    QPoint,
    QPropertyAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QStackedWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
)


class AnimatedStackedWidget(QStackedWidget):
    """QStackedWidget with simple page transitions using overlayed pixmap labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._animating = False

    def _grab_pixmap(self):
        # Snapshot of just the stacked area
        return self.grab()

    def setCurrentIndexAnimated(
        self, index: int, mode: str = "fade", duration: int = 5000
    ):
        if (
            self._animating
            or index == self.currentIndex()
            or not (0 <= index < self.count())
        ):
            self.setCurrentIndex(index)
            return

        self._animating = True

        # Snapshots before/after
        old_pix = self._grab_pixmap()
        old_index = self.currentIndex()
        super().setCurrentIndex(index)  # switch instantly
        self.repaint()  # ensure render
        new_pix = self._grab_pixmap()

        # Overlay two labels on top of the stacked area (self)
        top = QLabel(self)
        bot = QLabel(self)
        top.setAttribute(Qt.WA_TransparentForMouseEvents)
        bot.setAttribute(Qt.WA_TransparentForMouseEvents)
        top.setPixmap(old_pix)
        bot.setPixmap(new_pix)
        top.setGeometry(0, 0, old_pix.width(), old_pix.height())
        bot.setGeometry(0, 0, new_pix.width(), new_pix.height())
        top.show()
        bot.show()

        # Start from correct z-order: old on top initially
        top.raise_()
        bot.lower()

        group = QParallelAnimationGroup(self)

        if mode.startswith("slide"):
            # Determine direction vector
            w, h = self.width(), self.height()
            dx, dy = 0, 0
            if mode == "slide-left":
                dx = -w
            elif mode == "slide-right":
                dx = w
            elif mode == "slide-up":
                dy = -h
            elif mode == "slide-down":
                dy = h
            else:
                dx = -w  # default to left

            # Position the new pixmap offscreen opposite to the old’s target
            bot.move(QPoint(dx, dy) * -1)  # e.g., if old moves left, new starts right
            bot.raise_()  # new above old as it slides in

            anim_old = QPropertyAnimation(top, b"pos", self)
            anim_old.setStartValue(QPoint(0, 0))
            anim_old.setEndValue(QPoint(dx, dy))
            anim_old.setDuration(duration)
            anim_old.setEasingCurve(QEasingCurve.OutBack)

            anim_new = QPropertyAnimation(bot, b"pos", self)
            anim_new.setStartValue(QPoint(-dx, -dy))
            anim_new.setEndValue(QPoint(0, 0))
            anim_new.setDuration(duration)
            anim_new.setEasingCurve(QEasingCurve.OutBack)

            group.addAnimation(anim_old)
            group.addAnimation(anim_new)

        else:  # "fade"
            from PySide6.QtWidgets import QGraphicsOpacityEffect

            eff_old = QGraphicsOpacityEffect(top)
            eff_new = QGraphicsOpacityEffect(bot)
            top.setGraphicsEffect(eff_old)
            bot.setGraphicsEffect(eff_new)

            # new underneath fades in; raise it so we see it
            bot.raise_()
            anim_old = QPropertyAnimation(eff_old, b"opacity", self)
            anim_old.setStartValue(1.0)
            anim_old.setEndValue(0.0)
            anim_old.setDuration(duration)
            anim_old.setEasingCurve(QEasingCurve.OutBack)

            anim_new = QPropertyAnimation(eff_new, b"opacity", self)
            anim_new.setStartValue(0.0)
            anim_new.setEndValue(1.0)
            anim_new.setDuration(duration)
            anim_new.setEasingCurve(QEasingCurve.OutBack)

            group.addAnimation(anim_old)
            group.addAnimation(anim_new)

        def cleanup():
            # Remove overlays; final page already set
            top.deleteLater()
            bot.deleteLater()
            self._animating = False

        group.finished.connect(cleanup)
        group.start()

    # Convenience wrappers
    def setCurrentWidgetAnimated(self, widget: QWidget, mode="fade", duration=300):
        self.setCurrentIndexAnimated(self.indexOf(widget), mode, duration)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animated QStackedWidget (PySide6)")

        self.stack = AnimatedStackedWidget()

        # Create three example pages
        page1 = self._make_page("Page 1", "#1e293b", "#93c5fd")
        page2 = self._make_page("Page 2", "#065f46", "#a7f3d0")
        page3 = self._make_page("Page 3", "#7c2d12", "#fed7aa")
        for p in (page1, page2, page3):
            self.stack.addWidget(p)

        # Controls
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["fade", "slide-left", "slide-right", "slide-up", "slide-down"]
        )

        btn1 = QPushButton("Go to Page 1")
        btn2 = QPushButton("Go to Page 2")
        btn3 = QPushButton("Go to Page 3")
        for i, b in enumerate((btn1, btn2, btn3)):
            b.clicked.connect(lambda _, idx=i: self.switch_to(idx))

        topbar = QHBoxLayout()
        topbar.addWidget(self.mode_combo)
        topbar.addStretch()
        topbar.addWidget(btn1)
        topbar.addWidget(btn2)
        topbar.addWidget(btn3)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(topbar)
        layout.addWidget(self.stack)
        self.setCentralWidget(root)
        self.resize(800, 500)

    def _make_page(self, title: str, bg: str, fg: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{bg};")
        lab = QLabel(title)
        lab.setStyleSheet(f"color:{fg}; font-size:28px; font-weight:600;")
        lab.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        lay.addWidget(lab)
        lay.addStretch(1)
        return w

    def switch_to(self, index: int):
        mode = self.mode_combo.currentText()
        self.stack.setCurrentIndexAnimated(index, mode=mode, duration=350)


def main():
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
