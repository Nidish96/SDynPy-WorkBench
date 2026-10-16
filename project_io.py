"""
project_io.py - "workbench dump": save / load the whole Workbench state in ONE file.

The file (*.sdwb) is a zip archive that holds SDynPy's own save formats
(geometry, FRFs, shapes) plus the FE mesh, mesh groups, node maps, selections
and the log, so a session can be restored exactly as it was.

Used as a mixin:   class Workbench(ProjectMixin, SerepMixin, QMainWindow): ...
and after _build_menus():   self._build_project_menu()
"""
import json
import os
import tempfile
import time
import zipfile

import meshio
import numpy as np
import sdynpy as sdn
from qtpy.QtWidgets import QAction, QFileDialog, QMessageBox

from serep_tools import guarded

FORMAT_VERSION = 1
EXT = '.sdwb'


class ProjectMixin:
    # ------------------------------------------------------------------ #
    # menu
    # ------------------------------------------------------------------ #
    def _build_project_menu(self):
        """Add Save / Load workbench at the top of the existing File menu."""
        menu = next((a.menu() for a in self.menuBar().actions()
                     if a.menu() is not None and a.text().replace('&', '') == 'File'), None)
        if menu is None:
            menu = self.menuBar().addMenu('&File')
        first = menu.actions()[0] if menu.actions() else None
        for text, slot, key, tip in (
                ('Save workbench...', self.save_workbench, 'Ctrl+S',
                 'Dump FRFs, geometry, shapes, FE data and selections into one file'),
                ('Load workbench...', self.load_workbench, 'Ctrl+Shift+O',
                 'Restore a workbench dump')):
            act = QAction(text, self)
            act.setShortcut(key)
            act.setStatusTip(tip)
            act.triggered.connect(slot)
            menu.insertAction(first, act)
        menu.insertSeparator(first)

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    @guarded
    def save_workbench(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save workbench', '',
                                              'SDynPy workbench (*{})'.format(EXT))
        if not path:
            return
        if not path.lower().endswith(EXT):
            path += EXT
        self.write_workbench(path)
        self.log('Saved workbench to ' + path)

    def write_workbench(self, path):
        meta = dict(format=FORMAT_VERSION, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                    sdynpy=str(getattr(sdn, '__version__', '')), parts=[])
        with tempfile.TemporaryDirectory() as tmp:
            def put(name):
                meta['parts'].append(name)
                return os.path.join(tmp, name)

            if self.geom.node.size or self.geom.traceline.size or self.geom.element.size:
                self.geom.save(put('geometry.npz'))
            if self.frfs is not None:
                self.frfs.save(put('frfs.npz'))
            if self.shapes is not None and self.shapes.size:
                self.shapes.save(put('shapes.npy'))
            if self.fe_mesh is not None:
                d = {'points': np.asarray(self.fe_mesh.points)}
                for i, b in enumerate(self.fe_mesh.cells):
                    d['ctype_{}'.format(i)] = np.array(b.type)
                    d['cdata_{}'.format(i)] = np.asarray(b.data)
                if self.fe_faces is not None:
                    d['faces'] = np.asarray(self.fe_faces)
                names = []
                for i, (name, idx) in enumerate(self.fe_groups.items()):
                    d['group_{}'.format(i)] = np.asarray(idx)
                    names.append(name)
                meta['fe_group_names'] = names
                np.savez_compressed(put('fe_mesh.npz'), **d)
                if self.fe_geom is not None:
                    self.fe_geom.save(put('fe_geometry.npz'))
            if self.fe_shapes is not None:
                self.fe_shapes.save(put('fe_shapes.npy'))
            if self.expanded is not None:
                self.expanded.save(put('expanded.npy'))

            builder_state = self._saved_builder_state
            if self.builder is not None:
                try:
                    builder_state = self.builder.state()
                except Exception:                      # noqa: BLE001
                    pass
            meta.update(
                frf_source=self.frf_source,
                test_fe_map={str(k): int(v) for k, v in self.test_fe_map.items()},
                node_map_history=[{str(k): int(v) for k, v in m.items()}
                                  for m in self._node_map_history],
                basis_sel=[int(i) for i in (getattr(self, '_basis_sel', None) or [])],
                builder=builder_state,
                log=self.logview.toPlainText()[-200000:])
            with open(os.path.join(tmp, 'meta.json'), 'w') as fh:
                json.dump(meta, fh)
            tmp_zip = path + '.tmp'
            with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED) as z:
                for name in meta['parts'] + ['meta.json']:
                    z.write(os.path.join(tmp, name), name)
            os.replace(tmp_zip, path)

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def _has_state(self):
        return bool(self.geom.node.size or self.frfs is not None or self.fe_mesh is not None
                    or (self.shapes is not None and self.shapes.size))

    @guarded
    def load_workbench(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load workbench', '',
                                              'SDynPy workbench (*{});;All files (*)'.format(EXT))
        if not path:
            return
        if self._has_state() and QMessageBox.question(
                self, 'Load workbench', 'This replaces everything currently in the workbench. '
                'Continue?') != QMessageBox.Yes:
            return
        self.read_workbench(path)
        self.log('Loaded workbench from ' + path)

    def read_workbench(self, path):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                if 'meta.json' not in names:
                    raise ValueError('Not a workbench file (meta.json missing).')
                for n in names:
                    if n in ('meta.json', 'geometry.npz', 'frfs.npz', 'shapes.npy',
                             'fe_mesh.npz', 'fe_geometry.npz', 'fe_shapes.npy', 'expanded.npy'):
                        z.extract(n, tmp)
            with open(os.path.join(tmp, 'meta.json')) as fh:
                meta = json.load(fh)
            if meta.get('format', 0) > FORMAT_VERSION:
                self.log('WARNING: workbench file was written by a newer version (format {}).'
                         .format(meta.get('format')))

            def have(name):
                return os.path.exists(os.path.join(tmp, name))

            def where(name):
                return os.path.join(tmp, name)

            # -- close everything that depends on the old state ------------ #
            for attr in ('polypy', 'smac'):
                self.host.close_window(getattr(self, attr))
                setattr(self, attr, None)
            self._dispose_fe_tools()

            # -- test side --------------------------------------------------- #
            self.replace_geometry(sdn.Geometry.load(where('geometry.npz')) if have('geometry.npz')
                                  else sdn.Geometry())
            self.frfs = sdn.TransferFunctionArray.load(where('frfs.npz')) if have('frfs.npz') \
                else None
            self.frf_source = meta.get('frf_source', '')
            self.shapes = sdn.shape.load(where('shapes.npy')) if have('shapes.npy') else None

            # -- FE side ------------------------------------------------------ #
            self.fe_mesh = self.fe_geom = self.fe_faces = None
            self.fe_groups = {}
            if have('fe_mesh.npz'):
                with np.load(where('fe_mesh.npz'), allow_pickle=False) as d:
                    cells, i = [], 0
                    while 'ctype_{}'.format(i) in d:
                        cells.append((str(d['ctype_{}'.format(i)]), d['cdata_{}'.format(i)]))
                        i += 1
                    self.fe_mesh = meshio.Mesh(d['points'], cells)
                    self.fe_faces = d['faces'] if 'faces' in d else None
                    for i, name in enumerate(meta.get('fe_group_names', [])):
                        self.fe_groups[name] = d['group_{}'.format(i)]
                if have('fe_geometry.npz'):
                    self.fe_geom = sdn.Geometry.load(where('fe_geometry.npz'))
            self.fe_shapes = sdn.shape.load(where('fe_shapes.npy')) if have('fe_shapes.npy') \
                else None
            self.expanded = sdn.shape.load(where('expanded.npy')) if have('expanded.npy') \
                else None

            # -- selections / maps ----------------------------------------------- #
            self.test_fe_map = {int(k): int(v) for k, v in meta.get('test_fe_map', {}).items()}
            self._node_map_history = [{int(k): int(v) for k, v in m.items()}
                                      for m in meta.get('node_map_history', [])]
            self._basis_sel = meta.get('basis_sel') or None
            self._saved_builder_state = meta.get('builder')
            if meta.get('log'):
                self.logview.setPlainText(meta['log'])
            self.refresh_project()
