a = Analysis(
    ["native_app.py"],
    pathex=["."],
    binaries=[],
    datas=[("docs/user_guide.html","docs")],
    hiddenimports=["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"] + [
        "sem_metrology.calibration", "sem_metrology.io", "sem_metrology.lines",
        "sem_metrology.measurements", "sem_metrology.models", "sem_metrology.overlay",
        "sem_metrology.reporting", "sem_metrology.shapes",
        "sem_metrology.profile",
    ],
    excludes=["streamlit", "pytest", "matplotlib", "pyarrow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="SEM Metrology Workbench", console=False, debug=False,
    strip=False, upx=True,
)
