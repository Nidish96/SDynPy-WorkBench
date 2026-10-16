"""
serep_tools.py - FE mesh / mode import, mesh-group sensor picking and SEREP
expansion for the SDynPy Workbench.

Used as a mixin:   class Workbench(SerepMixin, QMainWindow): ...
"""
import functools
import traceback

import numpy as np
import h5py
import meshio
import matplotlib.pyplot as plt
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange, QVBoxLayout, QWidget, QAction)
import sdynpy as sdn


# --------------------------------------------------------------------------- #
# Small helpers (same behaviour as in the main script)
# --------------------------------------------------------------------------- #
def guarded(fn):
    @functools.wraps(fn)
    def wrapper(self, *_a, **_k):
        try:
            return fn(self)
        except Exception as exc:               # noqa: BLE001
            self.log(traceback.format_exc())
            QMessageBox.critical(self, 'Error', '{}: {}\n\nDetails are in the log.'.format(
                type(exc).__name__, exc))
    return wrapper


def alive(widget):
    if widget is None:
        return False
    try:
        return widget.isVisible()
    except RuntimeError:
        return False


# --------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------- #
SHELL_TYPES = (('triangle', 3, 91), ('quad', 4, 94))   # SDynPy: 91 = tri, 94 = quad


def shell_cells(mesh):
    """1-based corner-node connectivity + SDynPy element types of all shells."""
    conn, types = [], []
    for block in mesh.cells:
        for prefix, ncorner, etype in SHELL_TYPES:
            if block.type.startswith(prefix):
                for c in block.data:
                    conn.append(np.asarray(c[:ncorner]) + 1)
                    types.append(etype)
    return conn, types


def _as_index(v, n):
    v = np.asarray(v)
    if v.dtype == bool:
        return np.flatnonzero(v)
    return np.unique(v.astype(np.int64).ravel())


def extract_groups(mesh):
    """Return {label: 0-based node index array} for every group meshio found.

    Handles (a) named point sets, (b) point tags (+ their names), and
    (c) cell sets / cell tags, converted to the nodes they touch.
    """
    n = len(mesh.points)
    groups = {}

    def add(label, idx):
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size == 0:
            return
        key, k = label, 2
        while key in groups:
            key = '{} #{}'.format(label, k)
            k += 1
        groups[key] = idx

    # (a) named point sets
    for name, v in (getattr(mesh, 'point_sets', None) or {}).items():
        add('[nodes] ' + str(name), _as_index(v, n))

    # (b) point tags
    names = getattr(mesh, 'point_tags', None) or {}
    pt = mesh.point_data.get('point_tags') if mesh.point_data else None
    if pt is not None:
        pt = np.asarray(pt).ravel()
        for tag in np.unique(pt):
            if tag == 0:
                continue
            nm = names.get(int(tag))
            label = ', '.join(nm) if nm else 'tag {}'.format(int(tag))
            if '[nodes] ' + label not in groups:
                add('[nodes] ' + label, np.flatnonzero(pt == tag))

    # (c) cell sets -> nodes
    for name, blocks in (getattr(mesh, 'cell_sets', None) or {}).items():
        nodes = []
        for block, sel in zip(mesh.cells, blocks):
            if sel is None:
                continue
            sel = _as_index(sel, len(block.data))
            if sel.size:
                nodes.append(np.asarray(block.data)[sel].ravel())
        if nodes:
            add('[cells] ' + str(name), np.concatenate(nodes))

    # (d) cell tags -> nodes
    cnames = getattr(mesh, 'cell_tags', None) or {}
    ctags = (mesh.cell_data or {}).get('cell_tags')
    if ctags is not None:
        for tag in np.unique(np.concatenate([np.asarray(t).ravel() for t in ctags])):
            if tag == 0:
                continue
            nodes = [np.asarray(b.data)[np.asarray(t).ravel() == tag].ravel()
                     for b, t in zip(mesh.cells, ctags)]
            nm = cnames.get(int(tag))
            label = ', '.join(nm) if nm else 'tag {}'.format(int(tag))
            if '[cells] ' + label not in groups:
                add('[cells] ' + label, np.concatenate(nodes))
    return groups


# --------------------------------------------------------------------------- #
# SEREP dialog
# --------------------------------------------------------------------------- #
class SerepDialog(QDialog):
    """Tick the FE modes that form the SEREP expansion basis."""

    def __init__(self, freqs, preselected=None, n_test_dofs=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('SEREP expansion - choose FE modes')
        self.setMinimumSize(460, 560)
        self.freqs = np.asarray(freqs, dtype=float)
        n = len(self.freqs)
        self.n_test_dofs = n_test_dofs
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Tick the FE modes to use as the expansion basis.'))

        self.listw = QListWidget()
        for i, f in enumerate(self.freqs):
            it = QListWidgetItem('Mode {:>3}     {:>10.2f} Hz'.format(i + 1, f))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.listw.addItem(it)
        lay.addWidget(self.listw, 1)

        row = QHBoxLayout()
        for text, fn in (('All', lambda: self._set_all(True)),
                         ('None', lambda: self._set_all(False)),
                         ('Invert', self._invert)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Type modes:'))
        self.spec = QLineEdit()
        self.spec.setPlaceholderText('e.g.  1, 2, 4   or   7-10, 14')
        self.spec.returnPressed.connect(self._apply_spec)
        row.addWidget(self.spec, 1)
        b = QPushButton('Set')
        b.clicked.connect(self._apply_spec)
        row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Select all modes above'))
        self.fmin = QDoubleSpinBox()
        self.fmin.setRange(0, 1e9)
        self.fmin.setDecimals(2)
        self.fmin.setSuffix(' Hz')
        self.fmin.setValue(1.0)
        row.addWidget(self.fmin)
        b = QPushButton('Select')
        b.setToolTip('Handy for skipping rigid-body modes')
        b.clicked.connect(self._above)
        row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.count = QLabel('')
        lay.addWidget(self.count)
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addWidget(self.bb)

        if preselected is not None:
            sel = [i for i in preselected if 0 <= i < n]
        else:
            sel = [i for i in range(6, 10) if i < n] or list(range(n))
        self._set_checked(sel)
        self.listw.itemChanged.connect(self._update_count)
        self._update_count()

    # -- helpers ----------------------------------------------------------- #
    def _set_checked(self, indices):
        wanted = set(indices)
        self.listw.blockSignals(True)
        for i in range(self.listw.count()):
            self.listw.item(i).setCheckState(Qt.Checked if i in wanted else Qt.Unchecked)
        self.listw.blockSignals(False)
        self._update_count()

    def _set_all(self, on):
        self._set_checked(range(self.listw.count()) if on else [])

    def _invert(self):
        have = set(self.values())
        self._set_checked([i for i in range(self.listw.count()) if i not in have])

    def _above(self):
        self._set_checked(np.flatnonzero(self.freqs >= self.fmin.value()).tolist())

    def _apply_spec(self):
        n = self.listw.count()
        try:
            idx = []
            for tok in self.spec.text().replace(',', ' ').split():
                if '-' in tok:
                    lo, hi = (int(t) for t in tok.split('-', 1))
                    idx += list(range(min(lo, hi), max(lo, hi) + 1))
                else:
                    idx.append(int(tok))
            if not idx:
                return
            if min(idx) < 1 or max(idx) > n:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, 'Modes', 'Use mode numbers 1..{}, e.g. "1, 2, 4" or '
                                '"7-10".'.format(n))
            return
        self._set_checked([i - 1 for i in idx])

    def _update_count(self, *_):
        sel = self.values()
        txt = 'Selected {} mode(s){}'.format(
            len(sel), ': ' + ', '.join(str(i + 1) for i in sel[:25]) +
            (' ...' if len(sel) > 25 else '') if sel else '')
        if self.n_test_dofs is not None and len(sel) > self.n_test_dofs:
            txt += '\nWARNING: more modes than test DOFs ({}) - SEREP will be ill-posed.'.format(
                self.n_test_dofs)
        self.count.setText(txt)
        self.bb.button(QDialogButtonBox.Ok).setEnabled(bool(sel))

    def values(self):
        """Sorted 0-based indices of the ticked FE modes."""
        return [i for i in range(self.listw.count())
                if self.listw.item(i).checkState() == Qt.Checked]


# --------------------------------------------------------------------------- #
# Group picker + sensor mapping widget (lives in the MDI area)
# --------------------------------------------------------------------------- #
class SensorGroupPicker(QWidget):
    def __init__(self, wb):
        super().__init__()
        self.wb = wb
        self.setWindowTitle('FE groups / sensor mapping')
        pts = np.asarray(wb.fe_mesh.points, dtype=float)
        if pts.shape[1] < 3:
            pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
        self.pts = pts[:, :3]
        self.groups = dict(wb.fe_groups)
        self._busy = False

        root = QHBoxLayout(self)
        left = QWidget()
        left.setMaximumWidth(400)
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel('Groups found in the mesh (click one to show it):'))
        self.listw = QListWidget()
        lv.addWidget(self.listw, 2)
        custom = QPushButton('Custom node IDs...')
        custom.clicked.connect(self._custom)
        lv.addWidget(custom)

        self.chk_angle = QCheckBox('Order by angle around Z (ring); start nearest 0 deg')
        self.chk_angle.setChecked(True)
        self.chk_rev = QCheckBox('Reverse direction')
        self.spin_off = QSpinBox()
        self.spin_off.setRange(-999, 999)
        row = QHBoxLayout()
        row.addWidget(QLabel('Start offset'))
        row.addWidget(self.spin_off)
        lv.addWidget(self.chk_angle)
        lv.addWidget(self.chk_rev)
        lv.addLayout(row)

        self.status = QLabel('')
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Test node', 'FE node'])
        self.table.horizontalHeader().setStretchLastSection(True)
        lv.addWidget(self.table, 3)
        apply_btn = QPushButton('Apply: renumber test FRFs / shapes / geometry')
        apply_btn.clicked.connect(self.apply)
        undo_btn = QPushButton('Undo last renumbering')
        undo_btn.clicked.connect(self.undo)
        lv.addWidget(apply_btn)
        lv.addWidget(undo_btn)
        root.addWidget(left, 0)

        self.plotter = QtInteractor(self)
        root.addWidget(self.plotter.interactor, 1)

        for ii, (name, idx) in enumerate(self.groups.items()):
            it = QListWidgetItem('{}  ({} nodes)'.format(name, idx.size))
            it.setData(Qt.UserRole, name)
            self.listw.addItem(it)
        self.listw.currentItemChanged.connect(lambda *_: self._refresh(reset_camera=True))
        self.chk_angle.toggled.connect(lambda *_: self._refresh())
        self.chk_rev.toggled.connect(lambda *_: self._refresh())
        self.spin_off.valueChanged.connect(lambda *_: self._refresh())
        self.table.itemChanged.connect(self._table_edited)
        self.resize(1200, 700)
        if not self.groups:
            self.status.setText('No groups found in the mesh - use "Custom node IDs...".')
        self._draw(reset_camera=True)

    # -- plumbing --------------------------------------------------------- #
    def log(self, text):
        self.wb.log(text)

    def dispose(self):
        try:
            self.plotter.close()
        except Exception:                      # noqa: BLE001
            pass

    def closeEvent(self, event):
        self.dispose()
        super().closeEvent(event)

    # -- selection -------------------------------------------------------- #
    def _group_idx(self):
        it = self.listw.currentItem()
        return None if it is None else self.groups[it.data(Qt.UserRole)]

    def _ordered(self, idx):
        if idx is None:
            return np.array([], dtype=np.int64)
        idx = np.asarray(idx)
        if self.chk_angle.isChecked() and idx.size:
            xy = self.pts[idx]
            th = np.arctan2(xy[:, 1], xy[:, 0])
            si = np.argsort(th)
            pos = np.flatnonzero(si == np.argmin(np.abs(th)))[0]
            si = np.roll(si, -pos)
            if self.chk_rev.isChecked():
                si = np.r_[si[:1], si[:0:-1]]
            si = np.roll(si, -self.spin_off.value())
            return idx[si]
        out = idx[::-1] if self.chk_rev.isChecked() else idx
        return np.roll(out, -self.spin_off.value())

    def _refresh(self, reset_camera=False):
        self._populate_table()
        self._draw(reset_camera)

    def _populate_table(self):
        order = self._ordered(self._group_idx()) + 1          # FE node ids
        test = self.wb.test_node_ids()
        self._busy = True
        self.table.setRowCount(len(test))
        for r, t in enumerate(test):
            a = QTableWidgetItem(str(int(t)))
            a.setFlags(a.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, a)
            self.table.setItem(r, 1, QTableWidgetItem(str(int(order[r])) if r < len(order) else ''))
        self._busy = False
        if self._group_idx() is not None:
            self.status.setText('Group: {} nodes, test data: {} nodes.{}'.format(
                len(order), len(test),
                '' if len(order) == len(test) else '  COUNTS DIFFER - check the mapping.'))

    def _pairs(self, strict=False):
        pairs = []
        for r in range(self.table.rowCount()):
            t = int(self.table.item(r, 0).text())
            it = self.table.item(r, 1)
            txt = it.text().strip() if it else ''
            try:
                fe = int(txt) if txt else None
            except ValueError:
                if strict:
                    raise ValueError('"{}" is not a valid FE node id'.format(txt))
                fe = None
            pairs.append((t, fe))
        return pairs

    def _table_edited(self, *_):
        if not self._busy:
            self._draw()

    # -- drawing ---------------------------------------------------------- #
    def _draw(self, reset_camera=False):
        p = self.plotter
        p.clear()
        faces = self.wb.fe_faces
        if faces is not None:
            p.add_mesh(pv.PolyData(self.pts, faces=faces), color='lightgray',
                       show_edges=True, opacity=0.5)
        else:
            p.add_mesh(pv.PolyData(self.pts), color='gray', point_size=3)
        idx = self._group_idx()
        if idx is not None and len(idx):
            p.add_mesh(pv.PolyData(self.pts[idx]), color='orange', point_size=9,
                       render_points_as_spheres=True)
            pairs = [(t, fe) for t, fe in self._pairs() if fe is not None and 1 <= fe <= len(self.pts)]
            if pairs:
                fe_idx = np.array([fe - 1 for _, fe in pairs])
                labels = ['T{}->{}'.format(t, fe) for t, fe in pairs]
            else:
                fe_idx = self._ordered(idx)
                labels = ['#{}: {}'.format(k + 1, i + 1) for k, i in enumerate(fe_idx)]
            p.add_point_labels(self.pts[fe_idx], labels, point_size=14, point_color='red',
                               font_size=12, shape_opacity=0.3, always_visible=True,
                               render_points_as_spheres=True)
        p.add_axes()
        if reset_camera:
            p.reset_camera()
        p.render()

    # -- actions ---------------------------------------------------------- #
    def _custom(self):
        text, ok = QInputDialog.getText(self, 'Custom nodes',
                                        'FE node IDs (comma / space separated):')
        if not ok or not text.strip():
            return
        ids = np.array([int(t) for t in text.replace(',', ' ').split()]) - 1
        if ids.min() < 0 or ids.max() >= len(self.pts):
            QMessageBox.warning(self, 'Custom nodes', 'Node ids must be 1..{}'.format(len(self.pts)))
            return
        name = 'custom ({} nodes)'.format(ids.size)
        self.groups[name] = np.unique(ids) if not self.chk_angle.isChecked() else ids
        self.wb.fe_groups[name] = self.groups[name]
        it = QListWidgetItem('{}'.format(name))
        it.setData(Qt.UserRole, name)
        self.listw.addItem(it)
        self.listw.setCurrentItem(it)

    @guarded
    def apply(self):
        pairs = self._pairs(strict=True)
        mapping = {t: fe for t, fe in pairs if fe is not None}
        if not mapping:
            raise RuntimeError('Select a group first (nothing to map).')
        n = len(self.pts)
        bad = [v for v in mapping.values() if v < 1 or v > n]
        if bad:
            raise ValueError('FE node ids out of range 1..{}: {}'.format(n, bad))
        final = [mapping.get(t, t) for t, _ in pairs]
        if len(set(final)) != len(final):
            raise ValueError('The mapping would give two test nodes the same id.')
        self.wb.apply_node_map(mapping)
        self._refresh()

    @guarded
    def undo(self):
        self.wb.undo_node_map()
        self._refresh()


# --------------------------------------------------------------------------- #
# Test-geometry helpers (pure numpy / SDynPy, no GUI)
# --------------------------------------------------------------------------- #
def local_frame(normal, axis=0):
    """Right-handed local axes (rows = x, y, z in global coordinates).

    The surface normal becomes local axis `axis` (0 = X, 1 = Y, 2 = Z).  One of
    the two remaining axes is the in-surface direction closest to global Z (or
    to global Y when the normal itself is the Z axis); the third follows from
    the right-hand rule.  SDynPy stores these rows as coordinate_system.matrix[:3].
    """
    n = np.asarray(normal, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError('zero-length normal')
    n = n / nn
    up = 2 if axis != 2 else 1                 # local axis that should point "up"
    h = np.zeros(3)
    h[up] = 1.0                                # same direction in the global frame
    if abs(n @ h) > 0.95:                      # normal (anti)parallel to it -> use another
        h = np.zeros(3)
        h[0 if up != 0 else 1] = 1.0
    t = h - (h @ n) * n
    t /= np.linalg.norm(t)
    A = [None, None, None]
    A[axis] = n
    A[up] = t
    w = ({0, 1, 2} - {axis, up}).pop()
    A[w] = np.cross(A[(w + 1) % 3], A[(w + 2) % 3])
    return np.array(A)


def compute_point_normals(mesh, pts, faces, orient='asis'):
    """Per-node unit surface normals of an FE mesh.

    orient: 'asis' (as stored in the mesh), 'centroid' (flip so they point away
    from the centroid of the surface nodes) or 'auto' (VTK auto-orientation,
    only reliable for closed surfaces).  Returns (normals (n, 3), valid (n,)).
    Nodes that are not on a surface get a zero vector and valid = False.
    """
    n = len(pts)
    normals = np.zeros((n, 3))
    if faces is not None:
        poly = pv.PolyData(pts, faces=faces)
        ids = np.arange(n)
    else:                                       # solids: use their outer skin
        try:
            surf = pv.from_meshio(mesh).extract_surface()
            ids = np.asarray(surf.point_data['vtkOriginalPointIds'])
            poly = surf
        except Exception:                       # noqa: BLE001
            return normals, np.zeros(n, dtype=bool)
    kw = dict(cell_normals=False, point_normals=True, split_vertices=False,
              consistent_normals=True, auto_orient_normals=(orient == 'auto'))
    try:
        poly = poly.compute_normals(**kw)
    except Exception:                           # noqa: BLE001
        kw['auto_orient_normals'] = False
        poly = poly.compute_normals(**kw)
    normals[ids] = np.asarray(poly.point_data['Normals'])
    valid = np.linalg.norm(normals, axis=1) > 1e-9
    normals[valid] /= np.linalg.norm(normals[valid], axis=1)[:, None]
    if orient == 'centroid' and valid.any():
        c = pts[valid].mean(axis=0)
        flip = np.einsum('ij,ij->i', normals, pts - c) < 0
        normals[flip & valid] *= -1
    return normals, valid


def make_test_geometry(positions, test_ids, frames, connect=False, close=False):
    """Geometry whose nodes sit at `positions` (global) with one Cartesian
    displacement coordinate system per node given by `frames` (n, 3, 3 rows=axes)."""
    n = len(test_ids)
    cs_ids = np.arange(n) + 2                   # cs 1 = global
    mats = np.zeros((n + 1, 4, 3))
    mats[0, :3, :] = np.eye(3)
    mats[1:, :3, :] = frames
    names = ['Global'] + ['sensor {}'.format(t) for t in test_ids]
    csa = sdn.coordinate_system_array(id=np.r_[1, cs_ids], name=names, color=1,
                                      cs_type=0, matrix=mats)
    nodes = sdn.node_array(id=np.asarray(test_ids), coordinate=np.asarray(positions),
                           color=1, def_cs=1, disp_cs=cs_ids)
    g = sdn.Geometry(nodes, csa)
    if connect and n > 1:
        ids = list(np.asarray(test_ids))
        g.add_traceline(ids + ([ids[0]] if close else []), color=1)
    return g


# --------------------------------------------------------------------------- #
# Test geometry builder (lives in the MDI area)
# --------------------------------------------------------------------------- #
class TestGeometryBuilder(QWidget):
    AXES = ['X+', 'Y+', 'Z+']
    COLORS = ['red', 'green', 'blue']

    def __init__(self, wb):
        super().__init__()
        self.wb = wb
        self.setWindowTitle('Test geometry from FE nodes')
        pts = np.asarray(wb.fe_mesh.points, dtype=float)
        if pts.shape[1] < 3:
            pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
        self.pts = pts[:, :3]
        self.centroid = self.pts.mean(axis=0)
        self.diag = float(np.linalg.norm(self.pts.max(0) - self.pts.min(0))) or 1.0
        self.rows = []               # dicts: fe, test, axis, flip, override
        self._busy = False
        self._picking = False
        self.raw, self.valid = compute_point_normals(wb.fe_mesh, self.pts, wb.fe_faces, 'asis')
        self.normals = self.raw.copy()
        if not self.valid.any():
            wb.log('WARNING: no surface normals available for this mesh - edit the normal '
                   'vectors in the table by hand.')

        root = QHBoxLayout(self)
        left = QWidget()
        left.setMaximumWidth(560)
        lv = QVBoxLayout(left)

        # --- adding nodes
        lv.addWidget(QLabel('1. Choose nodes'))
        row = QHBoxLayout()
        self.group_combo = QComboBox()
        self.group_combo.addItems(list(wb.fe_groups.keys()))
        add_group = QPushButton('Add group')
        add_group.clicked.connect(self._add_group)
        row.addWidget(self.group_combo, 1)
        row.addWidget(add_group)
        lv.addLayout(row)
        row = QHBoxLayout()
        ids_btn = QPushButton('Add node IDs...')
        ids_btn.clicked.connect(self._add_ids)
        self.chk_pick = QCheckBox('Click mesh to add / remove nodes')
        self.chk_pick.toggled.connect(self._toggle_pick)
        row.addWidget(ids_btn)
        row.addWidget(self.chk_pick, 1)
        lv.addLayout(row)

        # --- defaults
        lv.addWidget(QLabel('2. Direction defaults'))
        form = QFormLayout()
        self.orient = QComboBox()
        self.orient.addItems(['Outward = away from mesh centroid', 'As stored in the mesh',
                              'Auto-orient (closed surfaces only)'])
        self.orient.currentIndexChanged.connect(self._reorient)
        form.addRow('Normal orientation', self.orient)
        row = QHBoxLayout()
        self.default_axis = QComboBox()
        self.default_axis.addItems(['X+  (matches the "force X+" FRF default)', 'Y+', 'Z+'])
        row.addWidget(self.default_axis, 1)
        apply_axis = QPushButton('Apply to all')
        apply_axis.clicked.connect(self._axis_all)
        row.addWidget(apply_axis)
        form.addRow('Normal = local axis', row)
        self.num_mode = QComboBox()
        self.num_mode.addItems(['Sequential 1, 2, 3...', 'Same as FE node id'])
        form.addRow('New test node ids', self.num_mode)
        lv.addLayout(form)

        # --- table
        lv.addWidget(QLabel('3. Review / edit (double-click Test node or Normal to edit)'))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['FE node', 'Test node', 'Normal axis', 'Flip',
                                              'Normal (x, y, z)', 'Position'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.itemChanged.connect(self._cell_edited)
        lv.addWidget(self.table, 1)

        row = QHBoxLayout()
        for text, fn in (('Remove selected', self._remove_selected), ('Clear', self._clear),
                         ('Flip all', self._flip_all)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lv.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('Order:'))
        for text, fn, tip in (('Up', lambda: self._move(-1), 'Move the selected rows up'),
                              ('Down', lambda: self._move(+1), 'Move the selected rows down'),
                              ('Reverse', self._reverse, 'Reverse the whole order')):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['Sort by...', 'Test node id', 'FE node id',
                                  'Angle around Z (start nearest 0 deg)',
                                  'Angle around Z, reversed'])
        self.sort_combo.currentIndexChanged.connect(self._sort)
        row.addWidget(self.sort_combo, 1)
        lv.addLayout(row)
        row = QHBoxLayout()
        for text, fn in (('Renumber 1..N', self._renumber), ('Use FE ids', self._use_fe_ids),
                         ('From FRF nodes', self._from_frf)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lv.addLayout(row)

        # --- build
        self.chk_trace = QCheckBox('Connect nodes with a traceline (in table order)')
        self.chk_close = QCheckBox('Close the loop')
        lv.addWidget(self.chk_trace)
        lv.addWidget(self.chk_close)
        build = QPushButton('Build test geometry')
        build.clicked.connect(self.build)
        lv.addWidget(build)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        lv.addWidget(self.status)
        root.addWidget(left, 0)

        self.plotter = QtInteractor(self)
        root.addWidget(self.plotter.interactor, 1)
        self.resize(1300, 760)
        saved = getattr(wb, '_saved_builder_state', None)
        if saved:
            for combo, key in ((self.orient, 'orient'), (self.default_axis, 'axis'),
                               (self.num_mode, 'num')):
                combo.blockSignals(True)
                combo.setCurrentIndex(int(saved.get(key, 0)))
                combo.blockSignals(False)
            self.chk_trace.setChecked(bool(saved.get('trace', False)))
            self.chk_close.setChecked(bool(saved.get('close', False)))
        self._reorient()                       # (clears normal overrides, so restore rows after)
        if saved:
            for r in saved.get('rows', []):
                if 1 <= int(r['fe']) <= len(self.pts):
                    self.rows.append(dict(
                        fe=int(r['fe']), test=int(r['test']), axis=int(r['axis']),
                        flip=bool(r['flip']),
                        override=None if r['override'] is None
                        else np.array(r['override'], dtype=float)))
            self._fill_table()
        self._draw(reset_camera=True)

    def state(self):
        """JSON-able snapshot of the table and options (for save / reopen)."""
        out = dict(rows=[dict(fe=int(r['fe']), test=int(r['test']), axis=int(r['axis']),
                              flip=bool(r['flip']),
                              override=None if r['override'] is None
                              else [float(v) for v in r['override']]) for r in self.rows])
        try:
            out.update(orient=self.orient.currentIndex(), axis=self.default_axis.currentIndex(),
                       num=self.num_mode.currentIndex(), trace=self.chk_trace.isChecked(),
                       close=self.chk_close.isChecked())
        except RuntimeError:                   # widgets already deleted
            pass
        return out

    # -- plumbing ----------------------------------------------------------- #
    def log(self, text):
        self.wb.log(text)

    def dispose(self):
        try:
            self.plotter.disable_picking()
        except Exception:                      # noqa: BLE001
            pass
        try:
            self.plotter.close()
        except Exception:                      # noqa: BLE001
            pass

    def closeEvent(self, event):
        self.dispose()
        super().closeEvent(event)

    # -- normals ------------------------------------------------------------- #
    def _reorient(self, *_):
        mode = self.orient.currentIndex()
        if mode == 1:
            self.normals = self.raw.copy()
        elif mode == 0:
            self.normals, _ = compute_point_normals(self.wb.fe_mesh, self.pts,
                                                    self.wb.fe_faces, 'centroid')
        else:
            self.normals, _ = compute_point_normals(self.wb.fe_mesh, self.pts,
                                                    self.wb.fe_faces, 'auto')
        for r in self.rows:
            r['override'] = None
        self._fill_table()
        self._draw()

    def row_normal(self, r):
        """(unit normal, has_surface_normal) for a table row."""
        i = r['fe'] - 1
        ok = True
        if r['override'] is not None:
            n = np.array(r['override'], dtype=float)
        else:
            n = self.normals[i].copy()
            if np.linalg.norm(n) < 1e-9:                 # not on a surface -> radial guess
                n = self.pts[i] - self.centroid
                ok = False
        if np.linalg.norm(n) < 1e-12:
            n = np.array([1.0, 0.0, 0.0])
            ok = False
        if r['flip']:
            n = -n
        return n / np.linalg.norm(n), ok

    def frames(self):
        return np.array([local_frame(self.row_normal(r)[0], r['axis']) for r in self.rows])

    # -- adding / removing nodes ------------------------------------------- #
    def add_nodes(self, fe_ids):
        n = len(self.pts)
        have = {r['fe'] for r in self.rows}
        nxt = max([r['test'] for r in self.rows], default=0) + 1
        added = 0
        for fe in fe_ids:
            fe = int(fe)
            if fe in have or not 1 <= fe <= n:
                continue
            test = fe if self.num_mode.currentIndex() == 1 else nxt
            nxt += 1
            self.rows.append(dict(fe=fe, test=test, axis=self.default_axis.currentIndex(),
                                  flip=False, override=None))
            have.add(fe)
            added += 1
        self._fill_table()
        self._draw()
        return added

    @guarded
    def _add_group(self):
        name = self.group_combo.currentText()
        if name:
            self.add_nodes(self.wb.fe_groups[name] + 1)

    @guarded
    def _add_ids(self):
        text, ok = QInputDialog.getText(self, 'Add nodes', 'FE node IDs (comma / space separated, '
                                        'ranges like 10-20 allowed):')
        if not ok or not text.strip():
            return
        ids = []
        for tok in text.replace(',', ' ').split():
            if '-' in tok:
                lo, hi = (int(t) for t in tok.split('-', 1))
                ids += list(range(min(lo, hi), max(lo, hi) + 1))
            else:
                ids.append(int(tok))
        bad = [i for i in ids if not 1 <= i <= len(self.pts)]
        if bad:
            raise ValueError('FE node ids out of range 1..{}: {}'.format(len(self.pts), bad[:10]))
        self.add_nodes(ids)

    def _remove_selected(self):
        gone = {i.row() for i in self.table.selectionModel().selectedRows()}
        self.rows = [r for i, r in enumerate(self.rows) if i not in gone]
        self._fill_table()
        self._draw()

    def _clear(self):
        self.rows = []
        self._fill_table()
        self._draw()

    def _flip_all(self):
        for r in self.rows:
            r['flip'] = not r['flip']
        self._fill_table()
        self._draw()

    def _axis_all(self):
        for r in self.rows:
            r['axis'] = self.default_axis.currentIndex()
        self._fill_table()
        self._draw()

    # -- ordering ------------------------------------------------------------ #
    def _selected_rows(self):
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()})

    def _reselect(self, row_objs):
        self.table.clearSelection()
        for k, r in enumerate(self.rows):
            if any(r is o for o in row_objs):
                self.table.setRangeSelected(QTableWidgetSelectionRange(k, 0, k, 5), True)

    def _move(self, delta):
        sel = set(self._selected_rows())
        if not sel:
            return
        chosen = [self.rows[i] for i in sorted(sel)]
        n = len(self.rows)
        if delta < 0:
            for i in range(1, n):
                if i in sel and (i - 1) not in sel:
                    self.rows[i - 1], self.rows[i] = self.rows[i], self.rows[i - 1]
                    sel = (sel - {i}) | {i - 1}
        else:
            for i in range(n - 2, -1, -1):
                if i in sel and (i + 1) not in sel:
                    self.rows[i + 1], self.rows[i] = self.rows[i], self.rows[i + 1]
                    sel = (sel - {i}) | {i + 1}
        self._fill_table()
        self._reselect(chosen)
        self._draw()

    def _reverse(self):
        chosen = [self.rows[i] for i in self._selected_rows()]
        self.rows.reverse()
        self._fill_table()
        self._reselect(chosen)
        self._draw()

    def _sort(self, mode):
        if mode == 0 or not self.rows:
            return
        if mode == 1:
            self.rows.sort(key=lambda r: r['test'])
        elif mode == 2:
            self.rows.sort(key=lambda r: r['fe'])
        else:
            xy = np.array([self.pts[r['fe'] - 1][:2] for r in self.rows])
            th = np.arctan2(xy[:, 1], xy[:, 0])
            order = np.argsort(th)
            pos = int(np.flatnonzero(order == np.argmin(np.abs(th)))[0])
            order = np.roll(order, -pos)
            if mode == 4:
                order = np.r_[order[:1], order[:0:-1]]
            self.rows = [self.rows[i] for i in order]
        self.sort_combo.blockSignals(True)
        self.sort_combo.setCurrentIndex(0)
        self.sort_combo.blockSignals(False)
        self._fill_table()
        self._draw()

    # -- numbering ------------------------------------------------------------ #
    def _renumber(self):
        for k, r in enumerate(self.rows):
            r['test'] = k + 1
        self._fill_table()
        self._draw()

    def _use_fe_ids(self):
        for r in self.rows:
            r['test'] = r['fe']
        self._fill_table()
        self._draw()

    @guarded
    def _from_frf(self):
        self.wb._need_frfs()
        ids = np.unique(np.asarray(self.wb.frfs.coordinate[..., 0]['node']).astype(np.int64))
        for r, t in zip(self.rows, ids):
            r['test'] = int(t)
        msg = 'FRF response nodes: {}'.format(ids.tolist())
        if len(ids) != len(self.rows):
            msg += '  (table has {} rows - counts differ)'.format(len(self.rows))
        self.log(msg)
        self._fill_table()
        self._draw()

    # -- table ----------------------------------------------------------------- #
    def _fill_table(self):
        self._busy = True
        self.table.setRowCount(len(self.rows))
        for k, r in enumerate(self.rows):
            a = QTableWidgetItem(str(r['fe']))
            a.setFlags(a.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, 0, a)
            self.table.setItem(k, 1, QTableWidgetItem(str(r['test'])))
            combo = QComboBox()
            combo.addItems(self.AXES)
            combo.setCurrentIndex(r['axis'])
            combo.currentIndexChanged.connect(lambda idx, r=r: self._set(r, 'axis', idx))
            self.table.setCellWidget(k, 2, combo)
            chk = QCheckBox()
            chk.setChecked(r['flip'])
            chk.toggled.connect(lambda on, r=r: self._set(r, 'flip', bool(on)))
            self.table.setCellWidget(k, 3, chk)
            n, ok = self.row_normal(r)
            self.table.setItem(k, 4, QTableWidgetItem(
                '{:.4f}, {:.4f}, {:.4f}{}'.format(*n, '' if ok else '  *')))
            p = QTableWidgetItem('{:.4g}, {:.4g}, {:.4g}'.format(*self.pts[r['fe'] - 1]))
            p.setFlags(p.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, 5, p)
        self._busy = False
        self.table.resizeColumnsToContents()
        n_bad = sum(1 for r in self.rows if not self.row_normal(r)[1])
        self.status.setText('{} nodes selected.{}'.format(
            len(self.rows), '  * = no surface normal here (radial guess) - check or edit it.'
            if n_bad else ''))

    def _set(self, row, key, value):
        row[key] = value
        QTimer.singleShot(0, self._refresh_after_edit)

    def _refresh_after_edit(self):
        self._fill_table()
        self._draw()

    def _cell_edited(self, item):
        if self._busy:
            return
        k, col = item.row(), item.column()
        if k >= len(self.rows):
            return
        r = self.rows[k]
        try:
            if col == 1:
                v = int(item.text())
                if v < 1:
                    raise ValueError
                r['test'] = v
            elif col == 4:
                vec = np.array([float(t) for t in item.text().replace('*', '').replace(
                    ',', ' ').split()])
                if vec.size != 3 or np.linalg.norm(vec) < 1e-12:
                    raise ValueError
                r['override'] = vec / np.linalg.norm(vec)
                r['flip'] = False
        except ValueError:
            pass                                   # bad input -> table is rebuilt below
        QTimer.singleShot(0, self._refresh_after_edit)

    # -- picking ----------------------------------------------------------------- #
    def _toggle_pick(self, on):
        p = self.plotter
        try:
            if on:
                try:
                    p.enable_point_picking(callback=self._picked, show_message=False,
                                           show_point=False, left_clicking=True, picker='cell')
                except TypeError:
                    p.enable_point_picking(callback=self._picked, show_message=False,
                                           show_point=False)
            else:
                p.disable_picking()
        except Exception as exc:                   # noqa: BLE001
            self.log('Could not toggle picking: {}'.format(exc))
            self.chk_pick.blockSignals(True)
            self.chk_pick.setChecked(False)
            self.chk_pick.blockSignals(False)

    def _picked(self, *args):
        try:
            pt = np.asarray(args[0], dtype=float).ravel()[:3]
        except Exception:                          # noqa: BLE001
            return
        if pt.size < 3 or not np.all(np.isfinite(pt)):
            return
        fe = int(np.argmin(np.linalg.norm(self.pts - pt, axis=1))) + 1
        QTimer.singleShot(0, lambda: self._toggle_node(fe))

    def _toggle_node(self, fe):
        for k, r in enumerate(self.rows):
            if r['fe'] == fe:
                del self.rows[k]
                self._fill_table()
                self._draw()
                return
        self.add_nodes([fe])

    # -- drawing ---------------------------------------------------------------- #
    def _draw(self, reset_camera=False):
        p = self.plotter
        try:
            p.clear()
            faces = self.wb.fe_faces
            if faces is not None:
                p.add_mesh(pv.PolyData(self.pts, faces=faces), color='lightgray',
                           show_edges=True, opacity=0.5, pickable=True)
            else:
                p.add_mesh(pv.PolyData(self.pts), color='gray', point_size=3, pickable=True)
            if self.rows:
                P = self.pts[[r['fe'] - 1 for r in self.rows]]
                p.add_mesh(pv.PolyData(P), color='orange', point_size=11,
                           render_points_as_spheres=True, pickable=False)
                p.add_point_labels(P, [str(r['test']) for r in self.rows], font_size=12,
                                   shape_opacity=0.3, always_visible=True, pickable=False)
                F = self.frames()
                for i, col in enumerate(self.COLORS):
                    p.add_arrows(P, F[:, i, :], mag=0.05 * self.diag, color=col)
            p.add_axes()
            if reset_camera:
                p.reset_camera()
            p.render()
        except Exception as exc:                   # noqa: BLE001
            self.log('Draw failed: {}'.format(exc))

    # -- build ------------------------------------------------------------------ #
    @guarded
    def build(self):
        if not self.rows:
            raise RuntimeError('Select some nodes first.')
        test_ids = [r['test'] for r in self.rows]
        dup = sorted({t for t in test_ids if test_ids.count(t) > 1})
        if dup:
            raise ValueError('Test node ids must be unique; duplicated: {}'.format(dup))
        if not self.wb._confirm_replace():
            return
        positions = self.pts[[r['fe'] - 1 for r in self.rows]]
        g = make_test_geometry(positions, test_ids, self.frames(),
                               connect=self.chk_trace.isChecked(),
                               close=self.chk_close.isChecked())
        self.wb.replace_geometry(g)
        self.wb.test_fe_map = {r['test']: r['fe'] for r in self.rows}
        self.wb._node_map_history = []
        self.log('Built test geometry: {} nodes with local coordinate systems (normal along {}).'
                 .format(len(self.rows), ', '.join(sorted({self.AXES[r['axis']] for r in self.rows}))))
        self.log('Test -> FE node map: ' + ', '.join(
            '{}->{}'.format(t, f) for t, f in self.wb.test_fe_map.items()))
        if self.wb.frfs is not None:
            self.wb.check_channels()


# --------------------------------------------------------------------------- #
# Mixin for the Workbench main window
# --------------------------------------------------------------------------- #
class SerepMixin:
    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def init_serep_state(self):
        self.fe_mesh = None
        self.fe_geom = None
        self.fe_faces = None
        self.fe_groups = {}
        self.fe_shapes = None
        self.expanded = None
        self.picker = None
        self.builder = None
        self._saved_builder_state = None
        self.test_fe_map = {}
        self._node_map_history = []

    def _build_serep_menu(self):
        m = self.menuBar().addMenu('&FE / SEREP')
        self._action(m, 'Load FE mesh (.med)...', self.load_fe_mesh, 'Ctrl+M',
                     'Read a Code_Aster / Salome .med mesh and list its groups')
        self._action(m, 'Load FE modes (.rmed)...', self.load_fe_modes)
        m.addSeparator()
        self._action(m, 'Mesh groups / sensor mapping...', self.open_sensor_picker, 'Ctrl+Shift+G',
                     'Show mesh groups, pick one and renumber the test nodes to it')
        self._action(m, 'Undo test node renumbering', self.undo_node_map)
        self._action(m, 'Build test geometry from FE nodes...', self.open_geometry_builder,
                     'Ctrl+B', 'Pick FE nodes, get surface-normal local axes, edit '
                     'numbers/directions, and use them as the test geometry')
        m.addSeparator()
        self._action(m, 'Plot FE mesh', self.plot_fe_mesh)
        self._action(m, 'Plot FE modes', self.plot_fe_modes)
        m.addSeparator()
        self._action(m, 'Run SEREP expansion...', self.run_serep, 'Ctrl+E')
        self._action(m, 'Plot expanded shapes', self.plot_expanded)
        self._action(m, 'Plot test vs expanded overlay', self.plot_overlay)
        self._action(m, 'Plot MAC: expanded vs FE modes', self.plot_expanded_fe_mac,
                     tip='MAC between the SEREP-expanded test shapes and all FE modes')
        self._action(m, 'Save expanded shapes...', self.save_expanded)

    def _build_serep_toolbar(self):
        tb = self.addToolBar('SEREP')
        tb.setMovable(False)
        for text, slot in [('FE Mesh', self.load_fe_mesh), ('FE Modes', self.load_fe_modes),
                           ('Groups', self.open_sensor_picker), ('Test Geom', self.open_geometry_builder),
                           ('SEREP', self.run_serep),
                           ('Overlay', self.plot_overlay)]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)

    def serep_project_rows(self):
        rows = []
        if self.fe_mesh is None:
            rows.append(('FE mesh', 'fe_mesh', 'none'))
        else:
            rows.append(('FE mesh', 'fe_mesh', '{} nodes, {} groups'.format(
                len(self.fe_mesh.points), len(self.fe_groups))))
        rows.append(('FE modes', 'fe_modes', 'none' if self.fe_shapes is None else
                     '{} modes'.format(self.fe_shapes.size)))
        rows.append(('Expanded', 'expanded', 'none' if self.expanded is None else
                     '{} shapes'.format(self.expanded.size)))
        return rows

    # ------------------------------------------------------------------ #
    # requirements
    # ------------------------------------------------------------------ #
    def _need_fe_mesh(self):
        if self.fe_mesh is None:
            raise RuntimeError('Load an FE mesh first (FE / SEREP > Load FE mesh).')

    def _need_fe_modes(self):
        if self.fe_shapes is None:
            raise RuntimeError('Load FE modes first (FE / SEREP > Load FE modes).')

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @guarded
    def load_fe_mesh(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load FE mesh', '',
                                              'MED files (*.med *.rmed);;All files (*)')
        if not path:
            return
        mesh = meshio.read(path, file_format='med')
        pts = np.asarray(mesh.points, dtype=float)
        if pts.shape[1] < 3:
            pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
        node_ids = np.arange(len(pts)) + 1             # meshio is 0-based, SDynPy ids > 0

        cs = sdn.coordinate_system_array(id=1, cs_type=0, name='Global')
        nodes = sdn.node_array(id=node_ids, coordinate=pts, color=1, def_cs=1, disp_cs=1)
        conn, types = shell_cells(mesh)
        if conn:
            elements = sdn.element_array(id=np.arange(len(conn)) + 1, type=types,
                                         color=1, connectivity=conn)
            geom = sdn.Geometry(node=nodes, coordinate_system=cs, element=elements)
            self.fe_faces = np.concatenate([np.r_[len(c), c - 1] for c in conn]).astype(np.int64)
        else:
            geom = sdn.Geometry(node=nodes, coordinate_system=cs)
            self.fe_faces = None

        self._dispose_fe_tools()
        self.test_fe_map = {}
        self._saved_builder_state = None
        self.fe_mesh, self.fe_geom = mesh, geom
        self.fe_groups = extract_groups(mesh)
        self.fe_shapes = self.expanded = None
        self.log('Loaded FE mesh {}: {} nodes, {} shell elements'.format(
            path, len(pts), len(conn)))
        if self.fe_groups:
            self.log('Groups found:')
            for name, idx in self.fe_groups.items():
                self.log('   {}  ({} nodes)'.format(name, idx.size))
        else:
            self.log('No groups found in the mesh (you can still enter node IDs by hand).')
        self.refresh_project()
        self.open_sensor_picker()

    @guarded
    def load_fe_modes(self):
        self._need_fe_mesh()
        path, _ = QFileDialog.getOpenFileName(self, 'Load FE modes', '',
                                              'MED results (*.rmed *.med);;All files (*)')
        if not path:
            return
        n_nodes = len(self.fe_mesh.points)
        phi, freqs = [], []
        with h5py.File(path, 'r') as f:
            cha = f['CHA']
            names = list(cha.keys())
            if not names:
                raise RuntimeError('No result fields (CHA) in the file.')
            default = next((i for i, n in enumerate(names) if n.endswith('DEPL')), 0)
            name = names[0]
            if len(names) > 1:
                name, ok = QInputDialog.getItem(self, 'Field', 'Displacement field:',
                                                names, default, False)
                if not ok:
                    return
            fld = cha[name]
            ncomp = int(fld.attrs['NCO'])
            for s in sorted(fld.keys()):
                g = fld[s]
                co = g['NOE']['MED_NO_PROFILE_INTERNAL']['CO'][:]
                if co.size != ncomp * n_nodes:
                    raise ValueError('Step {}: {} values, expected {} comps x {} nodes. The results '
                                     'do not match the loaded mesh.'.format(
                                         s, co.size, ncomp, n_nodes))
                arr = co.reshape(ncomp, n_nodes).T     # component-major storage
                phi.append(arr[:, :3])                 # DX, DY, DZ; drop rotations
                freqs.append(float(g.attrs['PDT']))
        phi, freqs = np.array(phi), np.array(freqs)
        coords = sdn.coordinate_array(self.fe_geom.node.id, ['X+', 'Y+', 'Z+'],
                                      force_broadcast=True).flatten()
        self.fe_shapes = sdn.shape_array(coords, phi.reshape(len(freqs), -1),
                                         frequency=freqs, damping=0.0)
        self.expanded = None
        self.log('Loaded {} FE modes from field {}:'.format(len(freqs), name))
        self.log('   ' + ', '.join('{}: {:.2f}'.format(i + 1, f) for i, f in enumerate(freqs[:30])))
        self.refresh_project()

    @guarded
    def open_geometry_builder(self):
        self._need_fe_mesh()
        if self.builder is not None and alive(self.builder):
            self.host.show_again(self.builder)
            return
        if self.builder is not None:
            try:
                self._saved_builder_state = self.builder.state()
            except Exception:                      # noqa: BLE001
                pass
            self.host.close_window(self.builder)
            self.builder.dispose()
        self.builder = TestGeometryBuilder(self)
        self.host.adopt(self.builder, 'Test geometry from FE nodes')

    def _dispose_fe_tools(self):
        for attr in ('picker', 'builder'):
            w = getattr(self, attr)
            if w is not None:
                self.host.close_window(w)
                w.dispose()
                setattr(self, attr, None)

    def _serep_id_map(self):
        """id_map (FE ids -> test ids) when the test geometry was built from FE nodes."""
        m = getattr(self, 'test_fe_map', None)
        if not m or all(k == v for k, v in m.items()):
            return None
        test_ids = set(np.asarray(self.geom.node.id).astype(int).tolist())
        pairs = [(fe, t) for t, fe in m.items() if t in test_ids]
        if not pairs:
            return None
        return sdn.id_map(np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))

    # ------------------------------------------------------------------ #
    # group picker
    # ------------------------------------------------------------------ #
    @guarded
    def open_sensor_picker(self):
        self._need_fe_mesh()
        if self.picker is not None and alive(self.picker):
            self.host.show_again(self.picker)
            return
        if self.picker is not None:
            self.host.close_window(self.picker)
            self.picker.dispose()
        self.picker = SensorGroupPicker(self)
        self.host.adopt(self.picker, 'FE groups / sensor mapping')

    # ------------------------------------------------------------------ #
    # node renumbering
    # ------------------------------------------------------------------ #
    def test_node_ids(self):
        """All distinct test node ids in the FRFs, shapes and test geometry."""
        ids = []
        if self.frfs is not None:
            ids.append(np.asarray(self.frfs.coordinate['node']).astype(np.int64).ravel())
        if self.shapes is not None and self.shapes.size:
            ids.append(np.asarray(self.shapes['coordinate'][0]['node']).astype(np.int64).ravel())
        if self.geom.node.size:
            ids.append(np.asarray(self.geom.node['id']).astype(np.int64).ravel())
        return np.unique(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)

    @staticmethod
    def _remap(a, mapping):
        a = np.asarray(a)
        out = a.copy()
        for k, v in mapping.items():          # masks use the ORIGINAL ids -> no chaining
            out[a == k] = v
        return out

    def _remap_shapes(self, shp, mapping):
        c0 = shp['coordinate'][0].copy()
        c0['node'] = self._remap(c0['node'], mapping)
        return sdn.shape_array(c0, shp['shape_matrix'], frequency=shp.frequency,
                               damping=shp.damping, modal_mass=shp.modal_mass,
                               comment1=shp.comment1, comment2=shp.comment2,
                               comment3=shp.comment3, comment4=shp.comment4,
                               comment5=shp.comment5)

    def apply_node_map(self, mapping, record=True):
        mapping = {int(k): int(v) for k, v in mapping.items()}
        if self.frfs is not None:
            c = self.frfs.coordinate.copy()
            c['node'] = self._remap(c['node'], mapping)
            self.frfs.coordinate = c
            for attr in ('polypy', 'smac'):        # fits on the old numbering are stale
                self.host.close_window(getattr(self, attr))
                setattr(self, attr, None)
        if self.shapes is not None and self.shapes.size:
            self.shapes = self._remap_shapes(self.shapes, mapping)
        g = self.geom
        if g.node.size:
            g.node['id'] = self._remap(g.node['id'], mapping)
            try:                                   # tracelines / elements reference node ids
                for arr in (g.traceline, g.element):
                    for i in np.ndindex(arr.shape):
                        arr['connectivity'][i] = self._remap(arr['connectivity'][i], mapping)
            except Exception as exc:               # noqa: BLE001
                self.log('WARNING: could not renumber traceline/element connectivity: '
                         '{}'.format(exc))
        self.expanded = None
        if self.test_fe_map:
            self.test_fe_map = {mapping.get(k, k): v for k, v in self.test_fe_map.items()}
        if record:
            self._node_map_history.append(mapping)
        self.log('Renumbered test nodes: ' + ', '.join(
            '{}->{}'.format(k, v) for k, v in mapping.items()))
        self.refresh_project()

    def undo_node_map(self):
        if not self._node_map_history:
            raise RuntimeError('Nothing to undo.')
        last = self._node_map_history.pop()
        self.apply_node_map({v: k for k, v in last.items()}, record=False)
        self.log('Undid last renumbering.')

    # ------------------------------------------------------------------ #
    # plots
    # ------------------------------------------------------------------ #
    @guarded
    def plot_fe_mesh(self):
        self._need_fe_mesh()
        self.host.adopt_plotter(self.fe_geom.plot(), 'FE mesh')

    @guarded
    def plot_fe_modes(self):
        self._need_fe_mesh()
        self._need_fe_modes()
        self.host.adopt_plotter(self.fe_geom.plot_shape(self.fe_shapes), 'FE modes')

    # ------------------------------------------------------------------ #
    # SEREP
    # ------------------------------------------------------------------ #
    @guarded
    def run_serep(self):
        self._need_fe_mesh()
        self._need_fe_modes()
        self._need_geometry()
        self._need_shapes()
        nmap = self._serep_id_map()
        known = self.fe_geom.node.id if nmap is None else nmap.to_ids
        missing = np.setdiff1d(self.geom.node.id, known)
        if missing.size:
            raise RuntimeError('Test geometry nodes {} are not in the FE mesh / node map. Build the '
                               'test geometry from FE nodes or renumber them first (FE / SEREP '
                               'menu).'.format(missing.tolist()))
        n_dofs = int(np.asarray(self.shapes['coordinate'][0]).size)
        dlg = SerepDialog(self.fe_shapes.frequency, getattr(self, '_basis_sel', None),
                          n_dofs, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        sel = dlg.values()
        self._basis_sel = sel                      # remembered for next time
        basis = self.fe_shapes[np.array(sel)]
        self.expanded = self.shapes.expand(self.geom, self.fe_geom, basis, node_id_map=nmap)
        self.log('SEREP: expanded {} test shapes using FE modes {}.'.format(
            self.expanded.size, ', '.join(str(i + 1) for i in sel)))
        self.refresh_project()
        self.plot_expanded()

    @guarded
    def plot_expanded(self):
        self._need_fe_mesh()
        if self.expanded is None:
            raise RuntimeError('Run the SEREP expansion first.')
        self.host.adopt_plotter(self.fe_geom.plot_shape(self.expanded), 'Expanded shapes')

    @guarded
    def plot_overlay(self):
        self._need_fe_mesh()
        if self.expanded is None:
            raise RuntimeError('Run the SEREP expansion first.')
        geom, shp = sdn.shape.overlay_shapes((self.geom, self.fe_geom),
                                             (self.shapes, self.expanded), [1, 7])
        opts = {'node_size': 6, 'line_width': 4, 'show_edges': False, 'view_up': [0, 1, 0]}
        plotter = geom.plot_shape(shp, opts, deformed_opacity=0.5, undeformed_opacity=0)
        self.host.adopt_plotter(plotter, 'Test vs expanded')

    @staticmethod
    def _mac_matrix(a, b):
        """MAC (len(a) x len(b)) over the (node, direction) DOFs the two shape sets share."""
        ca, cb = a['coordinate'][0], b['coordinate'][0]
        ka = list(zip(np.asarray(ca['node']).astype(int).tolist(),
                      np.asarray(ca['direction']).astype(int).tolist()))
        kb = list(zip(np.asarray(cb['node']).astype(int).tolist(),
                      np.asarray(cb['direction']).astype(int).tolist()))
        pos_b = {k: i for i, k in enumerate(kb)}
        ia, ib = [], []
        for i, k in enumerate(ka):
            if k in pos_b:
                ia.append(i)
                ib.append(pos_b[k])
        if not ia:
            raise RuntimeError('The expanded and FE shapes share no DOFs.')
        A = np.asarray(a['shape_matrix'])[:, ia]
        B = np.asarray(b['shape_matrix'])[:, ib]
        num = np.abs(A.conj() @ B.T) ** 2
        den = np.outer(np.sum(np.abs(A) ** 2, axis=1), np.sum(np.abs(B) ** 2, axis=1))
        return num / den, len(ia)

    @guarded
    def plot_expanded_fe_mac(self):
        self._need_fe_modes()
        if self.expanded is None:
            raise RuntimeError('Run the SEREP expansion first.')
        mac, ndof = self._mac_matrix(self.expanded, self.fe_shapes)
        fa, fb = np.asarray(self.expanded.frequency), np.asarray(self.fe_shapes.frequency)
        fig = plt.figure('MAC: expanded vs FE')
        fig.clf()
        ax = fig.add_subplot(111)
        im = ax.imshow(mac, vmin=0, vmax=1, cmap='viridis', origin='lower', aspect='auto')
        fig.colorbar(im, ax=ax, label='MAC')

        def ticks(n):
            step = int(np.ceil(n / 30))
            return np.arange(0, n, step)
        tx, ty = ticks(len(fb)), ticks(len(fa))
        ax.set_xticks(tx)
        ax.set_xticklabels(['{}\n{:.0f} Hz'.format(i + 1, fb[i]) for i in tx], fontsize=7)
        ax.set_yticks(ty)
        ax.set_yticklabels(['{} ({:.0f} Hz)'.format(i + 1, fa[i]) for i in ty], fontsize=7)
        ax.set_xlabel('FE mode')
        ax.set_ylabel('Expanded test mode')
        ax.set_title('MAC over {} shared DOFs'.format(ndof))
        if mac.size <= 500:
            for (i, j), v in np.ndenumerate(mac):
                ax.text(j, i, '{:.2f}'.format(v), ha='center', va='center', fontsize=7,
                        color='white' if v < 0.6 else 'black')
        fig.tight_layout()
        self.host.adopt_current_figure('MAC: expanded vs FE')

    @guarded
    def save_expanded(self):
        if self.expanded is None:
            raise RuntimeError('Run the SEREP expansion first.')
        path, _ = QFileDialog.getSaveFileName(self, 'Save expanded shapes', '',
                                              'SDynPy shapes (*.npy)')
        if path:
            self.expanded.save(path)
            self.log('Saved expanded shapes to ' + path)
