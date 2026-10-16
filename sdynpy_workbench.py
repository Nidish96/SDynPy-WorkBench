#!/usr/bin/env python3
"""
SDynPy Workbench
================
One main window that hosts the whole SDynPy test workflow:

    Load FRFs (UFF)  ->  build / edit geometry (nodes, coordinate systems,
    tracelines)  ->  inspect FRFs  ->  PolyPy / SMAC curve fitting
    ->  collect shapes  ->  plot shapes on the geometry, MAC, save.

Every tool window SDynPy opens (Geometry.edit(), PolyPy_GUI, SMAC_GUI,
GUIPlot, ShapePlotter, matplotlib figures, ...) is moved into a QMdiArea in
the main window, so nothing floats around the desktop.  Windows that SDynPy
creates on its own are caught by an application-wide event filter, so this
also works for windows I did not wrap explicitly.

Geometry creation uses SDynPy's own Geometry.edit() editor (node / coordinate
system / traceline / element tables + click-to-connect in the 3-D view).
"""
import os
import sys
import time
import functools
import traceback

if sys.platform.startswith('linux'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')   # must precede QApplication

import numpy as np
from qtpy.QtCore import Qt, QObject, QEvent, QTimer
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QDockWidget, QTreeWidget,
    QTreeWidgetItem, QPlainTextEdit, QAction, QActionGroup, QFileDialog, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QMenu, QWidget, QLabel, QVBoxLayout)

import matplotlib
try:
    matplotlib.use('QtAgg')
except Exception:                       # older matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor
import sdynpy as sdn

from serep_tools import SerepMixin
from project_io import ProjectMixin

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def guarded(fn):
    """Run a menu/toolbar handler; show + log any exception instead of crashing
    (PyQt aborts the process on exceptions that escape a slot)."""
    @functools.wraps(fn)
    def wrapper(self, *_args, **_kwargs):      # QAction.triggered passes `checked`
        try:
            return fn(self)
        except Exception as exc:               # noqa: BLE001
            self.log(traceback.format_exc())
            QMessageBox.critical(self, 'Error', '{}: {}\n\nDetails are in the log.'.format(
                type(exc).__name__, exc))
    return wrapper


def alive(widget):
    """True if the Qt object still exists and is visible."""
    if widget is None:
        return False
    try:
        return widget.isVisible()
    except RuntimeError:
        return False


class LogStream:
    """File-like object that mirrors stdout/stderr into the log dock."""

    def __init__(self, sink, original):
        self.sink, self.original = sink, original

    def write(self, text):
        if self.original is not None:
            self.original.write(text)
        if text.strip():
            self.sink(text.rstrip('\n'))
        return len(text)

    def flush(self):
        if self.original is not None:
            self.original.flush()


# --------------------------------------------------------------------------- #
# MDI host: moves SDynPy's top-level windows into sub-windows
# --------------------------------------------------------------------------- #
class MdiHost(QObject):
    def __init__(self, main_window, mdi):
        super().__init__(main_window)
        self.main, self.mdi = main_window, mdi
        self.embed_3d = True
        self.mdi.subWindowActivated.connect(
            lambda s: s is not None and s.widget() is not None and self._nudge(s.widget()))        
        QApplication.instance().installEventFilter(self)

    # -- automatic capture of any new top-level window ------------------- #
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QWidget) and self._eligible(obj):
            QTimer.singleShot(0, lambda o=obj: self._adopt_later(o))
        t = event.type()
        if t in (QEvent.Resize, QEvent.WindowStateChange, QEvent.Show) \
           and isinstance(obj, QMdiSubWindow):
            w = obj.widget()
            if w is not None and w.property('_in_mdi') and self.is_3d(w):
                QTimer.singleShot(50, lambda w=w: self._nudge(w))
        return False

    def _eligible(self, w):
        try:
            return (w.isWindow() and w is not self.main and w.windowType() == Qt.Window
                    and not isinstance(w, (QDialog, QMenu, QMdiSubWindow))
                    and not w.property('_in_mdi'))
        except RuntimeError:
            return False

    def _adopt_later(self, w):
        try:
            if self._eligible(w):
                self.adopt(w)
        except RuntimeError:                   # window died before we got to it
            pass

    # -- explicit adoption ------------------------------------------------ #
    def is_3d(self, w):
        return bool(w.findChildren(QtInteractor))

    def adopt(self, w, title=None):
        """Re-parent top-level window `w` into the MDI area (idempotent)."""
        if w is None or w.property('_in_mdi'):
            return None
        if self.is_3d(w) and not self.embed_3d:
            return None
        w.setProperty('_in_mdi', True)
        title = title or w.windowTitle() or type(w).__name__
        size = w.size()
        sub = self.mdi.addSubWindow(w)
        sub.setWindowTitle(title)
        if not self.is_3d(w):
            # keep the Python object usable after the user closes the window
            # (PolyPy checks `window.isVisible()` on its resynthesis plots)
            sub.setAttribute(Qt.WA_DeleteOnClose, False)
        vp = self.mdi.viewport().size()
        sub.resize(max(480, min(size.width(), vp.width() - 30)),
                   max(320, min(size.height(), vp.height() - 30)))
        w.show()
        sub.show()
        self.mdi.setActiveSubWindow(sub)
        QTimer.singleShot(200, lambda: self._nudge(w))
        return sub

    def adopt_plotter(self, plotter, title=None):
        return self.adopt(getattr(plotter, 'app_window', None), title)

    def adopt_current_figure(self, title=None):
        fig = plt.gcf()
        win = fig.canvas.manager.window
        sub = self.adopt(win, title or (fig.get_label() or 'Figure'))
        fig.canvas.draw_idle()
        return sub

    def show_again(self, w):
        sub = w.parentWidget()
        if isinstance(sub, QMdiSubWindow):
            sub.show()
            w.show()
            self.mdi.setActiveSubWindow(sub)
        else:
            w.show()
            w.raise_()
            w.activateWindow()

    def close_window(self, w):
        if w is None:
            return
        try:
            sub = w.parentWidget()
            if isinstance(sub, QMdiSubWindow):
                sub.close()
            else:
                w.close()
        except RuntimeError:                       # already deleted
            pass
    
    @staticmethod
    def _nudge(w):
        try:
            for ii in w.findChildren(QtInteractor):
                ii.render()
                ii.update()
        except Exception:                      # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Dialogs
# --------------------------------------------------------------------------- #
class LoadFrfDialog(QDialog):
    DIRECTIONS = [('Keep directions from file', None), ('X+  (1)', 1),
                  ('Y+  (2)', 2), ('Z+  (3)', 3)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load FRFs from UFF')
        self.setMinimumWidth(520)
        form = QFormLayout(self)

        row = QHBoxLayout()
        self.path = QLineEdit()
        browse = QPushButton('Browse...')
        browse.clicked.connect(self._browse)
        row.addWidget(self.path)
        row.addWidget(browse)
        form.addRow('UFF file', row)

        self.direction = QComboBox()
        for text, _ in self.DIRECTIONS:
            self.direction.addItem(text)
        self.direction.setCurrentIndex(1)
        self.direction.setToolTip('Overwrite reference_direction and response_direction of every '
                                  'dataset-58 record.  Use this when the file stores a generic '
                                  'direction but the sensors are really along X+ of the '
                                  'node displacement coordinate system.')
        form.addRow('Force direction to', self.direction)

        self.swap = QCheckBox('Swap response / reference columns')
        self.swap.setChecked(True)
        self.swap.setToolTip('SDynPy expects coordinate = (response, reference).  Tick this if '
                             'your export has them the other way around.')
        form.addRow('', self.swap)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open UFF', '',
                                              'Universal files (*.uff *.unv);;All files (*)')
        if path:
            self.path.setText(path)

    def values(self):
        return (self.path.text().strip(), self.DIRECTIONS[self.direction.currentIndex()][1],
                self.swap.isChecked())


class SymmetrizeFrfDialog(QDialog):
    """Ask whether to symmetrize the full square FRF matrix or a square block."""

    def __init__(self, n_response, n_reference, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Symmetrize FRFs')
        self.setMinimumWidth(480)
        self._n_response = n_response
        self._n_reference = n_reference

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            'Symmetrization assumes reciprocal FRFs measured in compatible '
            'coordinates and directions. It averages H(i,j) and H(j,i) using '
            'the ordinary transpose (not the conjugate transpose).'
        ))

        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItems([
            'Do not symmetrize',
            'Square submatrix (choose size and starting indices)'
        ])
        form.addRow('Action', self.mode)

        self.size = QSpinBox()
        self.size.setRange(1, max(1, min(n_response, n_reference)))
        self.size.setValue(min(n_response, n_reference))
        form.addRow('Submatrix size', self.size)

        self.row_start = QSpinBox()
        self.row_start.setRange(1, max(1, n_response))
        self.row_start.setValue(1)
        form.addRow('Response start index', self.row_start)

        self.col_start = QSpinBox()
        self.col_start.setRange(1, max(1, n_reference))
        self.col_start.setValue(1)
        form.addRow('Reference start index', self.col_start)
        layout.addLayout(form)

        self.mode.currentIndexChanged.connect(self._update_controls)
        self.size.valueChanged.connect(self._update_ranges)
        self.row_start.valueChanged.connect(self._update_ranges)
        self.col_start.valueChanged.connect(self._update_ranges)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_controls()

    def _update_controls(self, *_):
        enabled = self.mode.currentIndex() == 1
        self.size.setEnabled(enabled)
        self.row_start.setEnabled(enabled)
        self.col_start.setEnabled(enabled)
        self._update_ranges()

    def _update_ranges(self, *_):
        # A block must fit completely within both the response and reference axes.
        n = self.size.value()
        self.row_start.setMaximum(max(1, self._n_response - n + 1))
        self.col_start.setMaximum(max(1, self._n_reference - n + 1))

    def values(self):
        if self.mode.currentIndex() == 0:
            return ('none', None)
        return ('block', (self.size.value(),
                          self.row_start.value() - 1,
                          self.col_start.value() - 1))


class RingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Generate ring of sensors')
        form = QFormLayout(self)
        self.n = QSpinBox()
        self.n.setRange(2, 1000)
        self.n.setValue(8)
        self.dia = QDoubleSpinBox()
        self.dia.setDecimals(5)
        self.dia.setRange(1e-5, 1e6)
        self.dia.setValue(0.20)
        self.dia.setSuffix(' m')
        self.z = QDoubleSpinBox()
        self.z.setDecimals(5)
        self.z.setRange(-1e6, 1e6)
        self.z.setSuffix(' m')
        self.cs = QComboBox()
        self.cs.addItems(['Cylindrical (X+ = radial)', 'Cartesian (global)'])
        self.close_loop = QCheckBox('Connect nodes with a closed traceline')
        self.close_loop.setChecked(True)
        form.addRow('Number of nodes', self.n)
        form.addRow('Diameter', self.dia)
        form.addRow('Height (z)', self.z)
        form.addRow('Coordinate system', self.cs)
        form.addRow('', self.close_loop)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        return dict(n=self.n.value(), dia=self.dia.value(), z=self.z.value(),
                    cylindrical=self.cs.currentIndex() == 0, close=self.close_loop.isChecked())


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class Workbench(ProjectMixin, SerepMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SDynPy Workbench')
        self.resize(1500, 950)

        # ---- project state ---------------------------------------------- #
        self.geom = sdn.Geometry()
        self.frfs = None
        self.frf_source = ''
        self.shapes = None
        self.polypy = None
        self.smac = None
        self.editor_win = None
        self._keep = []                        # keep Python wrappers alive

        # ---- central MDI area ------------------------------------------- #
        self.mdi = QMdiArea()
        self.mdi.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi.setTabsClosable(True)
        self.mdi.setTabsMovable(True)
        self.setCentralWidget(self.mdi)
        self.host = MdiHost(self, self.mdi)

        self.init_serep_state()        
        self._build_docks()
        self._build_menus()
        self._build_project_menu()
        self._build_toolbar()
        self._build_serep_toolbar()

        # route prints / tracebacks into the log dock
        sys.stdout = LogStream(self.log, sys.__stdout__)
        sys.stderr = LogStream(self.log, sys.__stderr__)
        sys.excepthook = self._excepthook

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_project)
        self._timer.start(1000)                # editor edits geometry in place
        self.refresh_project()
        self.log('SDynPy {}  |  ready.'.format(getattr(sdn, '__version__', '')))

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_docks(self):
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Project', 'Details'])
        self.tree.setColumnWidth(0, 130)
        self.tree.itemDoubleClicked.connect(self._tree_activated)
        dock = QDockWidget('Project', self)
        dock.setWidget(self.tree)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        dock.setMinimumWidth(300)

        self.logview = QPlainTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setMaximumBlockCount(5000)
        mono = QFont('Monospace')
        mono.setStyleHint(QFont.TypeWriter)
        mono.setPointSize(9)
        self.logview.setFont(mono)
        ldock = QDockWidget('Log', self)
        ldock.setWidget(self.logview)
        ldock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, ldock)
        ldock.setMinimumHeight(110)
        self.resizeDocks([ldock], [150], Qt.Vertical)

    def _action(self, menu, text, slot, shortcut=None, tip=None, checkable=False):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.setCheckable(checkable)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_menus(self):
        mb = self.menuBar()

        m = mb.addMenu('&File')
        self._action(m, 'Load FRFs (UFF)...', self.load_frfs, 'Ctrl+O',
                     'Read a UFF file containing dataset-58 FRFs')
        self._action(m, 'Load shapes...', self.load_shapes)
        self._action(m, 'Save shapes...', self.save_shapes)
        m.addSeparator()
        self._action(m, 'Quit', self.close, 'Ctrl+Q')

        m = mb.addMenu('&Geometry')
        self._action(m, 'New (empty)', self.new_geometry)
        self._action(m, 'Open...', self.open_geometry)
        self._action(m, 'Import UFF/UNV geometry...', self.import_geometry)
        self._action(m, 'Save...', self.save_geometry)
        m.addSeparator()
        self._action(m, 'Generate ring of sensors...', self.ring_geometry)
        self._action(m, 'Import nodes from CSV...', self.import_nodes_csv,
                     tip='Columns: id, x, y, z [, def_cs, disp_cs]')
        m.addSeparator()
        self._action(m, 'Edit geometry (nodes / tracelines)...', self.edit_geometry, 'Ctrl+G',
                     'Open SDynPy\'s interactive geometry editor')
        self._action(m, 'Plot geometry', self.plot_geometry)

        m = mb.addMenu('&Data')
        self._action(m, 'Plot FRFs', self.plot_frfs)
        self._action(m, 'Plot CMIF', self.plot_cmif)
        self._action(m, 'Check channels against geometry', self.check_channels)

        m = mb.addMenu('&Modal')
        self._action(m, 'PolyPy curve fitter', self.open_polypy, 'Ctrl+P')
        self._action(m, 'SMAC curve fitter', self.open_smac)
        m.addSeparator()
        self._action(m, 'Collect shapes from PolyPy', self.collect_shapes, 'Ctrl+K',
                     'Copy the shapes currently fitted in PolyPy into the project')

        m = mb.addMenu('&Shapes')
        self._action(m, 'Plot shapes on geometry', self.plot_shapes, 'Ctrl+3')
        self._action(m, 'Auto-MAC', self.plot_mac)
        self._action(m, 'Print mode table', self.print_mode_table)

        self._build_serep_menu()
        self.window_menu = mb.addMenu('&Window')
        self.window_menu.aboutToShow.connect(self._fill_window_menu)

        m = mb.addMenu('&Help')
        self._action(m, 'Workflow', self.show_help)

    def _build_toolbar(self):
        tb = self.addToolBar('Workflow')
        tb.setMovable(False)
        for text, slot in [('Load FRFs', self.load_frfs), ('Geometry Editor', self.edit_geometry),
                           ('Plot FRFs', self.plot_frfs), ('PolyPy', self.open_polypy),
                           ('Collect Shapes', self.collect_shapes),
                           ('Plot Shapes', self.plot_shapes), ('MAC', self.plot_mac)]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)
            if text in ('Geometry Editor', 'Plot FRFs', 'Collect Shapes'):
                tb.addSeparator()

    def _fill_window_menu(self):
        m = self.window_menu
        m.clear()
        m.addAction('Cascade', self.mdi.cascadeSubWindows)
        m.addAction('Tile', self.mdi.tileSubWindows)
        tabbed = m.addAction('Tabbed view')
        tabbed.setCheckable(True)
        tabbed.setChecked(self.mdi.viewMode() == QMdiArea.TabbedView)
        tabbed.toggled.connect(lambda on: self.mdi.setViewMode(
            QMdiArea.TabbedView if on else QMdiArea.SubWindowView))
        embed = m.addAction('Embed 3-D windows')
        embed.setCheckable(True)
        embed.setChecked(self.host.embed_3d)
        embed.setStatusTip('Turn off if 3-D views misbehave on your GPU/driver; they then open '
                           'as normal top-level windows.')
        embed.toggled.connect(lambda on: setattr(self.host, 'embed_3d', on))
        m.addSeparator()
        m.addAction('Close all windows', self._close_all)
        m.addSeparator()
        for sub in self.mdi.subWindowList():
            if sub.isVisible():
                act = m.addAction(sub.windowTitle())
                act.triggered.connect(lambda _=False, s=sub: self.mdi.setActiveSubWindow(s))

    def _close_all(self):
        for sub in self.mdi.subWindowList():
            sub.close()

    # ------------------------------------------------------------------ #
    # Logging / status
    # ------------------------------------------------------------------ #
    def log(self, text):
        self.logview.appendPlainText('[{}] {}'.format(time.strftime('%H:%M:%S'), text))

    def _excepthook(self, etype, value, tb):
        self.log(''.join(traceback.format_exception(etype, value, tb)))

    def refresh_project(self):
        g = self.geom
        rows = [('Geometry', 'geometry',
                 '{} nodes, {} CS, {} tracelines, {} elements'.format(
                     g.node.size, g.coordinate_system.size, g.traceline.size, g.element.size))]
        if self.frfs is None:
            rows.append(('FRFs', 'frfs', 'none loaded'))
        else:
            f = self.frfs.abscissa
            rows.append(('FRFs', 'frfs', '{} resp x {} ref, {} lines, {:.1f}-{:.1f} Hz'.format(
                self.frfs.shape[0], self.frfs.shape[1], self.frfs.num_elements,
                f.min(), f.max())))
        if self.shapes is None or self.shapes.size == 0:
            rows.append(('Shapes', 'shapes', 'none'))
        else:
            fr = self.shapes.frequency
            rows.append(('Shapes', 'shapes', '{} modes, {:.1f}-{:.1f} Hz'.format(
                self.shapes.size, fr.min(), fr.max())))
        rows += self.serep_project_rows()
        if [self.tree.topLevelItem(i).text(1) for i in range(self.tree.topLevelItemCount())] \
                == [r[2] for r in rows]:
            return
        self.tree.clear()
        for name, key, detail in rows:
            item = QTreeWidgetItem([name, detail])
            item.setData(0, Qt.UserRole, key)
            self.tree.addTopLevelItem(item)
        if self.frf_source and self.frfs is not None:
            QTreeWidgetItem(self.tree.topLevelItem(1), ['file', os.path.basename(self.frf_source)])
            self.tree.expandAll()

    def _tree_activated(self, item, _col):
        key = item.data(0, Qt.UserRole)
        {'geometry': self.edit_geometry, 'frfs': self.plot_frfs,
         'shapes': self.plot_shapes}.get(key, lambda: None)()

    # ------------------------------------------------------------------ #
    # Requirements
    # ------------------------------------------------------------------ #
    def _need_frfs(self):
        if self.frfs is None:
            raise RuntimeError('Load FRFs first (File > Load FRFs).')

    def _need_geometry(self):
        if self.geom.node.size == 0:
            raise RuntimeError('The geometry has no nodes yet (Geometry menu).')

    def _need_shapes(self):
        if self.shapes is None or self.shapes.size == 0:
            if self.polypy is not None and self.polypy.shapes is not None \
                    and self.polypy.shapes.size > 0:
                self.collect_shapes()
            else:
                raise RuntimeError('No shapes yet: fit modes in PolyPy and use '
                                   'Modal > Collect shapes, or File > Load shapes.')

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    @guarded
    def load_frfs(self):
        dlg = LoadFrfDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        path, direction, swap = dlg.values()
        self.set_frfs_from_uff(path, direction, swap)

    def set_frfs_from_uff(self, path, direction=1, swap=True):
        dat = sdn.uff.readuff(path)
        if 58 not in dat:
            raise ValueError('No dataset 58 (frequency response) records in {}'.format(path))
        if direction is not None:
            for d in dat[58]:
                d.reference_direction = direction
                d.response_direction = direction
        frfs = sdn.TransferFunctionArray.from_uff(dat).reshape_to_matrix()
        if swap:
            frfs.coordinate = frfs.coordinate[..., ::-1]
        self.set_frfs(frfs, path)

    def _symmetrize_frf_block(self, frfs, row_start, col_start, size):
        """Average a square FRF block with its transpose at every frequency."""
        data = frfs.ordinate
        block = data[row_start:row_start + size, col_start:col_start + size, ...]
        block[...] = 0.5 * (block + np.swapaxes(block, 0, 1))
        self.log(
            'Symmetrized {} x {} FRF block (response indices {}-{}, '
            'reference indices {}-{}).'.format(
                size, size, row_start + 1, row_start + size,
                col_start + 1, col_start + size)
        )

    def set_frfs(self, frfs, source=''):
        # Optionally enforce reciprocity on a full square FRF matrix or a
        # square response/reference block selected by the user.
        n_response, n_reference = frfs.shape[:2]
        dlg = SymmetrizeFrfDialog(n_response, n_reference, self)
        if dlg.exec_() == QDialog.Accepted:
            mode, block = dlg.values()
            if mode == 'block':
                size, row_start, col_start = block
                if row_start + size > n_response or col_start + size > n_reference:
                    raise ValueError('Selected square FRF block is outside the matrix bounds.')
                self._symmetrize_frf_block(frfs, row_start, col_start, size)

        self.frfs, self.frf_source = frfs, source
        # a fit on old data is meaningless
        for attr in ('polypy', 'smac'):
            self.host.close_window(getattr(self, attr))
            setattr(self, attr, None)
        self.log('Loaded FRFs: {} responses x {} references, {} lines'.format(
            frfs.shape[0], frfs.shape[1], frfs.num_elements))
        self.refresh_project()
        if self.geom.node.size:
            self.check_channels()

    @guarded
    def check_channels(self):
        self._need_frfs()
        self._need_geometry()
        nodes = np.concatenate([self.frfs.coordinate[..., 0].node.ravel(),
                                self.frfs.coordinate[..., 1].node.ravel()])
        missing = np.setdiff1d(np.unique(nodes), self.geom.node.id)
        if missing.size:
            self.log('WARNING: FRF node IDs not in geometry: {}'.format(missing.tolist()))
        else:
            self.log('All FRF node IDs are present in the geometry.')

    @guarded
    def plot_frfs(self):
        self._need_frfs()
        win = sdn.GUIPlot(self.frfs)
        self._keep.append(win)
        self.host.adopt(win, 'FRFs')

    @guarded
    def plot_cmif(self):
        self._need_frfs()
        win = sdn.GUIPlot(CMIF=self.frfs.compute_cmif())
        win.ordinate_log = True
        win.actionOrdinate_Log.setChecked(True)
        self._keep.append(win)
        self.host.adopt(win, 'CMIF')

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def replace_geometry(self, new):
        """Swap contents in place so PolyPy / the editor keep the same object."""
        self.host.close_window(self.editor_win)
        self.editor_win = None
        for name in ('node', 'coordinate_system', 'traceline', 'element'):
            setattr(self.geom, name, getattr(new, name))
        self.refresh_project()

    def _confirm_replace(self):
        return (self.geom.node.size == 0 or QMessageBox.question(
            self, 'Replace geometry', 'This replaces the current geometry. Continue?')
            == QMessageBox.Yes)

    @guarded
    def new_geometry(self):
        if self._confirm_replace():
            self.replace_geometry(sdn.Geometry())

    @guarded
    def open_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open geometry', '', 'SDynPy geometry (*.npz)')
        if path and self._confirm_replace():
            self.replace_geometry(sdn.Geometry.load(path))
            self.log('Opened geometry ' + path)

    @guarded
    def import_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import geometry', '',
                                              'Universal files (*.uff *.unv)')
        if path and self._confirm_replace():
            self.replace_geometry(sdn.Geometry.from_uff(sdn.uff.readuff(path)))
            self.log('Imported geometry from ' + path)

    @guarded
    def save_geometry(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save geometry', '', 'SDynPy geometry (*.npz)')
        if path:
            self.geom.save(path)
            self.log('Saved geometry ' + path)

    @guarded
    def ring_geometry(self):
        dlg = RingDialog(self)
        if dlg.exec_() == QDialog.Accepted and self._confirm_replace():
            self.make_ring(**dlg.values())

    def make_ring(self, n=8, dia=0.2, z=0.0, cylindrical=True, close=True):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ids = 1 + np.arange(n)
        if cylindrical:     # (r, theta [deg], z); node X+ is radial
            coords = np.stack([np.full(n, dia / 2), np.degrees(th), np.full(n, z)], axis=-1)
            csa = sdn.coordinate_system_array(id=1, name='cylindrical', color=1, cs_type=1,
                                              matrix=np.eye(4, 3))
        else:
            coords = np.stack([dia / 2 * np.cos(th), dia / 2 * np.sin(th), np.full(n, z)], axis=-1)
            csa = sdn.coordinate_system_array(id=1, name='global', color=1, cs_type=0,
                                              matrix=np.eye(4, 3))
        nodes = sdn.node_array(id=ids, coordinate=coords, color=1, def_cs=1, disp_cs=1)
        g = sdn.Geometry(nodes, csa)
        if close:
            g.add_traceline(list(ids) + [ids[0]], color=1)
        self.replace_geometry(g)
        self.log('Generated ring geometry with {} nodes'.format(n))

    @guarded
    def import_nodes_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import nodes', '', 'Text (*.csv *.txt)')
        if not path:
            return
        with open(path) as fh:
            first = next((ln for ln in fh if ln.strip() and not ln.lstrip().startswith('#')), '')
        data = np.atleast_2d(np.genfromtxt(path, delimiter=',' if ',' in first else None,
                                           comments='#'))
        data = data[~np.isnan(data[:, 0])]               # drops a header row
        if data.shape[1] < 4:
            raise ValueError('Need columns: id, x, y, z [, def_cs, disp_cs]')
        if self.geom.coordinate_system.size == 0:
            self.geom.add_coordinate_system()
        existing, added = set(self.geom.node.id.tolist()), 0
        for row in data:
            nid = int(row[0])
            if nid in existing:
                self.log('Skipping duplicate node id {}'.format(nid))
                continue
            def_cs = int(row[4]) if data.shape[1] > 4 else 1
            disp_cs = int(row[5]) if data.shape[1] > 5 else def_cs
            self.geom.add_node(row[1:4], id=nid, def_cs=def_cs, disp_cs=disp_cs)
            added += 1
        unknown = (set(self.geom.node.def_cs.tolist()) | set(self.geom.node.disp_cs.tolist())) \
            - set(self.geom.coordinate_system.id.tolist())
        if unknown:
            self.log('WARNING: coordinate systems {} do not exist yet - add them in the '
                     'editor (Coord Systems tab).'.format(sorted(unknown)))
        self.log('Imported {} nodes from {}'.format(added, os.path.basename(path)))
        self.refresh_project()

    # @guarded
    # def edit_geometry(self):
    #     if alive(self.editor_win):
    #         self.host.show_again(self.editor_win)
    #         return
    #     if self.editor_win is not None:
    #         self.host.close_window(self.editor_win)
    #         try:
    #             self.editor_win.close()
    #         except RuntimeError:
    #             pass
    #     # called on the instance: edits self.geom in place and returns the QMainWindow
    #     self.editor_win = self.geom.edit()
    #     self.host.adopt(self.editor_win, 'Geometry Editor')
    @guarded
    def edit_geometry(self):
        if alive(self.editor_win):
            self.host.show_again(self.editor_win)
            return
        self.host.close_window(self.editor_win)    # safe now; ignores deleted windows
        self.editor_win = None
        before = set(QApplication.topLevelWidgets())
        self.editor_win = self.geom.edit()
        # everything the editor opened stays a normal floating window
        for w in (set(QApplication.topLevelWidgets()) - before) | {self.editor_win}:
            w.setProperty('_in_mdi', True)
        
    @guarded
    def plot_geometry(self):
        self._need_geometry()
        plotter = self.geom.plot(label_nodes=self.geom.node.size <= 200)
        self.host.adopt_plotter(plotter, 'Geometry')

    # ------------------------------------------------------------------ #
    # Modal fitting
    # ------------------------------------------------------------------ #
    def _open_fitter(self, attr, cls, title):
        self._need_frfs()
        existing = getattr(self, attr)
        if existing is not None and alive(existing.parentWidget()):
            self.host.show_again(existing)
            return
        if existing is not None:                 # was closed -> restart cleanly
            self.host.close_window(existing)
        win = cls(self.frfs)
        if self.geom.node.size and hasattr(win, 'set_geometry'):
            win.set_geometry(self.geom)
        setattr(self, attr, win)
        self.host.adopt(win, title)

    @guarded
    def open_polypy(self):
        self._open_fitter('polypy', sdn.PolyPy_GUI, 'PolyPy')

    @guarded
    def open_smac(self):
        self._open_fitter('smac', sdn.SMAC_GUI, 'SMAC')

    @guarded
    def collect_shapes(self):
        src = None
        for fitter in (self.polypy, self.smac):
            if fitter is not None and getattr(fitter, 'shapes', None) is not None \
                    and fitter.shapes.size > 0:
                src = fitter.shapes
                break
        if src is None:
            raise RuntimeError('No fitted shapes yet - select poles and compute shapes first.')
        self.shapes = np.sort(src).copy()
        self.log('Collected {} shapes from the curve fitter.'.format(self.shapes.size))
        self.refresh_project()

    @guarded
    def load_shapes(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load shapes', '', 'SDynPy shapes (*.npy)')
        if path:
            self.shapes = sdn.shape.load(path)
            self.log('Loaded {} shapes from {}'.format(self.shapes.size, path))
            self.refresh_project()

    @guarded
    def save_shapes(self):
        self._need_shapes()
        path, _ = QFileDialog.getSaveFileName(self, 'Save shapes', '', 'SDynPy shapes (*.npy)')
        if path:
            self.shapes.save(path)
            self.log('Saved shapes to ' + path)

    # ------------------------------------------------------------------ #
    # Shapes
    # ------------------------------------------------------------------ #
    @guarded
    def plot_shapes(self):
        self._need_geometry()
        self._need_shapes()
        plotter = self.geom.plot_shape(self.shapes)
        self.host.adopt_plotter(plotter, 'Mode shapes')

    @guarded
    def plot_mac(self):
        self._need_shapes()
        self.shapes.plot_mac()
        self.host.adopt_current_figure('Auto-MAC')

    @guarded
    def print_mode_table(self):
        self._need_shapes()
        self.log('Mode   Freq (Hz)   Damping (%)')
        for i, (f, z) in enumerate(zip(self.shapes.frequency, self.shapes.damping), 1):
            self.log('{:>4}  {:>10.3f}  {:>10.3f}'.format(i, f, 100 * z))

    # ------------------------------------------------------------------ #
    @guarded
    def show_help(self):
        QMessageBox.information(self, 'Workflow', (
            '1. File > Load FRFs: pick the UFF file, direction override and column swap.\n'
            '2. Geometry: generate a ring, import nodes from CSV, or open the editor\n'
            '   (nodes, coordinate systems, tracelines - click nodes, then Create).\n'
            '3. Data > Check channels: verifies FRF node IDs exist in the geometry.\n'
            '4. Modal > PolyPy: pick poles, compute shapes.\n'
            '5. Modal > Collect shapes, then Shapes > Plot / Auto-MAC.\n\n'
            'Window > Tabbed view switches between sub-windows and tabs.'))

    def closeEvent(self, event):
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        sys.excepthook = sys.__excepthook__
        self._timer.stop()
        super().closeEvent(event)


def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    except Exception:                          # noqa: BLE001
        pass
    try:    
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    except Exception:                          # noqa: BLE001
        pass        
    app = QApplication.instance() or QApplication(sys.argv)   # BEFORE any SDynPy GUI call
    app.setStyle('Fusion')
    win = Workbench()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
