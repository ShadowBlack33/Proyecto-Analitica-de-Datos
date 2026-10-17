"""Diseño del dashboard (solo apariencia): hoja de estilos y marca de la barra lateral.

No cambia ningún control ni cálculo del dashboard. El logo es el del reto PENGWIN
(Pelvic Bone Fragments with Injuries Segmentation Challenge, MICCAI 2024).
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

_DIR = Path(__file__).resolve().parent
ICONO = str(_DIR / "pengwin_logo.png")


def aplicar_diseno() -> None:
    """Inyecta la hoja de estilos (colores, tipografía, bordes y sombras)."""
    css = (_DIR / "estilo.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def marca_lateral() -> None:
    """Logo y nombre del sistema en la parte superior de la barra lateral."""
    logo = base64.b64encode((_DIR / "pengwin_logo.png").read_bytes()).decode()
    st.markdown(
        f'<div class="pw-marca"><div class="pw-logo"><img src="data:image/png;base64,{logo}" alt="Logo de PENGWIN"></div>'
        '<div class="pw-nombre"><strong>PENGWIN</strong><span>Fracturas pélvicas en TC</span></div></div>',
        unsafe_allow_html=True,
    )
