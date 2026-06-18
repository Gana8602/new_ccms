# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all database migrations, settings submodules and apps
hidden_imports = [
    'ccms_project.settings',
    'ccms_project.urls',
    'ccms_project.wsgi',
    'dashboard.apps',
    'dashboard.views',
    'dashboard.urls',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
# Auto-collect all submodules for PyTorch, Transformers, and CV2 if needed
hidden_imports += collect_submodules('django')
hidden_imports += collect_submodules('cv2')
hidden_imports += collect_submodules('PIL')

# Define paths to templates, static files, and models
project_dir = os.path.abspath('.')
datas = [
    ('dashboard/templates', 'dashboard/templates'),
    ('staticfiles', 'staticfiles'),
    ('db.sqlite3', '.'),  # Bundle initial database file
]

a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CCMS_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False if you want to hide the command prompt/terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CCMS_App',
)
