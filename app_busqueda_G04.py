# -*- coding: utf-8 -*-
"""
Comparación de algoritmos de búsqueda (DFS, BFS, A*) - Sistemas Inteligentes 2026-2
Salón A1T1 · Grupo G04 · Factor: numero_aglomerados (2 / 5 / 9, radio fijo = 2)

Aplicación de escritorio hecha con Tkinter + Matplotlib.
La generación de mapas y los algoritmos son los de la plantilla
(busqueda_plantilla_estudiantes_v3), de modo que el mapa "bajo, repetición 0"
de esta aplicación es el mismo mapa "G04_bajo_00" del experimento y del CSV.

Ejecutar:   python app_busqueda_G04.py
Empaquetar: pyinstaller --onefile --windowed --name BusquedaG04 app_busqueda_G04.py
"""

import time
import tracemalloc
from collections import deque
from heapq import heappush, heappop

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------------------------------------------------------
# Identificación y configuración del grupo (igual que en la plantilla)
# ----------------------------------------------------------------------------
SALON = "A1T1"
GRUPO = "G04"
INTEGRANTES = [
    "Alanya Rojas, Jhon Cristopher",
    "Cisneros Sopan, Khenngie Jonel",
    "Pampa Llamucuri, Joel Estefano",
    "Romero Vasquez, Juan David",
]
SEMILLA = 202602
REPETICIONES = 30
FACTOR = "numero_aglomerados"
NIVELES = {"bajo": 2, "medio": 5, "alto": 9}
RADIO_AGLOMERADO = 2
TAMANO = 25

LIBRE, MURO = 0, 1
MOV = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ----------------------------------------------------------------------------
# Utilidades de la plantilla
# ----------------------------------------------------------------------------
def vecinos(m, u):
    for di, dj in MOV:
        v = (u[0] + di, u[1] + dj)
        if 0 <= v[0] < m.shape[0] and 0 <= v[1] < m.shape[1] and m[v] != MURO:
            yield v


def reconstruir(padres, meta):
    if meta not in padres:
        return []
    r = []
    u = meta
    while u is not None:
        r.append(u)
        u = padres[u]
    return r[::-1]


def ruta_valida(m, ruta, s, t):
    return (
        bool(ruta)
        and ruta[0] == s
        and ruta[-1] == t
        and all(m[x] != MURO for x in ruta)
        and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(ruta, ruta[1:]))
    )


# ----------------------------------------------------------------------------
# Generación del mapa (misma lógica de la plantilla para numero_aglomerados)
# ----------------------------------------------------------------------------
def generar_mapa(nivel, repeticion):
    valor = NIVELES[nivel]
    rng = np.random.default_rng(SEMILLA + repeticion)
    n = TAMANO
    m = (rng.random((n, n)) < 0.15).astype(np.uint8)  # ruido base de la plantilla
    s = (0, 0)
    t = (n - 1, n - 1)
    centros = [tuple(x) for x in rng.integers(4, n - 4, size=(9, 2))]
    k = int(valor)
    rad = RADIO_AGLOMERADO
    for i, j in centros[:k]:
        m[max(0, i - rad):min(n, i + rad + 1), max(0, j - rad):min(n, j + rad + 1)] = 1
    # ruta común garantizada: fila 0 y última columna siempre libres (48 pasos)
    m[s[0], :t[1] + 1] = 0
    m[:t[0] + 1, t[1]] = 0
    m[s] = 0
    m[t] = 0
    return m, s, t, valor


# ----------------------------------------------------------------------------
# Agentes de búsqueda (los de la plantilla; "traza" es opcional y solo sirve
# para dibujar en la interfaz los nodos en el orden en que se expandieron)
# ----------------------------------------------------------------------------
def dfs(m, inicio, meta, traza=None):
    frontera = [inicio]
    padres = {inicio: None}
    expandidos = 0
    while frontera:
        u = frontera.pop()
        if u == meta:
            return reconstruir(padres, meta), expandidos
        expandidos += 1
        if traza is not None:
            traza.append(u)
        for v in vecinos(m, u):
            if v not in padres:
                padres[v] = u
                frontera.append(v)
    return [], expandidos


def bfs(m, inicio, meta, traza=None):
    frontera = deque([inicio])
    padres = {inicio: None}
    expandidos = 0
    while frontera:
        u = frontera.popleft()
        if u == meta:
            return reconstruir(padres, meta), expandidos
        expandidos += 1
        if traza is not None:
            traza.append(u)
        for v in vecinos(m, u):
            if v not in padres:
                padres[v] = u
                frontera.append(v)
    return [], expandidos


def heuristica_manhattan(u, meta):
    return abs(u[0] - meta[0]) + abs(u[1] - meta[1])


def astar(m, inicio, meta, traza=None):
    frontera = []
    heappush(frontera, (heuristica_manhattan(inicio, meta), 0, inicio))
    padres = {inicio: None}
    costo_g = {inicio: 0}
    expandidos = 0
    while frontera:
        _, g_u, u = heappop(frontera)
        if g_u > costo_g.get(u, float("inf")):
            continue
        if u == meta:
            return reconstruir(padres, meta), expandidos
        expandidos += 1
        if traza is not None:
            traza.append(u)
        for v in vecinos(m, u):
            nuevo_costo_g = g_u + 1
            if nuevo_costo_g < costo_g.get(v, float("inf")):
                padres[v] = u
                costo_g[v] = nuevo_costo_g
                f_score = nuevo_costo_g + heuristica_manhattan(v, meta)
                heappush(frontera, (f_score, nuevo_costo_g, v))
    return [], expandidos


ALGS = {"DFS": dfs, "BFS": bfs, "A*": astar}


def medir(fn, m, s, t):
    """Misma medición que el experimento de la plantilla (tiempo + pico de memoria)."""
    tracemalloc.start()
    tic = time.perf_counter()
    ruta, exp = fn(m, s, t)
    ms = (time.perf_counter() - tic) * 1000
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ruta, exp, ms, pico / 1024


# ----------------------------------------------------------------------------
# Colores del dibujo
# ----------------------------------------------------------------------------
COL_LIBRE = (1.0, 1.0, 1.0)
COL_MURO = (0.0, 0.0, 0.0)
COL_EXPLORADO = (1.0, 0.88, 0.45)   # amarillo suave
COL_RUTA = "#d62728"                # rojo
BG_CABECERA = "#f0f0f0"


def mapa_a_rgb(m):
    img = np.empty(m.shape + (3,), dtype=float)
    img[m == LIBRE] = COL_LIBRE
    img[m == MURO] = COL_MURO
    return img


# ----------------------------------------------------------------------------
# Interfaz gráfica
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title(f"Comparación de algoritmos de búsqueda - {GRUPO}")
        root.minsize(1000, 700)

        self.nivel = tk.StringVar(value="bajo")
        self.rep = 0
        self.anim = None
        self.m = self.s = self.t = None
        self.img = None

        self._construir_interfaz()
        self._cargar_mapa()
        root.protocol("WM_DELETE_WINDOW", self._cerrar)

    # -- construcción de la ventana ------------------------------------------
    def _construir_interfaz(self):
        cab = tk.Frame(self.root, bg=BG_CABECERA, pady=8)
        cab.pack(fill="x")
        tk.Label(
            cab,
            text=f"Comparación de algoritmos de búsqueda - {GRUPO}",
            font=("Arial", 16, "bold"),
            bg=BG_CABECERA,
        ).pack(pady=(4, 8))

        fila1 = tk.Frame(cab, bg=BG_CABECERA)
        fila1.pack()
        tk.Label(fila1, text="Nivel de dificultad:", bg=BG_CABECERA).pack(side="left", padx=(0, 6))
        self.combo = ttk.Combobox(
            fila1, textvariable=self.nivel, values=list(NIVELES), state="readonly", width=10
        )
        self.combo.pack(side="left", padx=(0, 14))
        self.combo.bind("<<ComboboxSelected>>", lambda e: self._cargar_mapa())
        ttk.Button(fila1, text="Nuevo mapa", command=self._nuevo_mapa).pack(side="left")

        fila2 = tk.Frame(cab, bg=BG_CABECERA)
        fila2.pack(pady=(8, 0))
        for nombre in ALGS:
            ttk.Button(
                fila2, text=f"Ejecutar {nombre}", command=lambda n=nombre: self.ejecutar(n)
            ).pack(side="left", padx=6)

        cuerpo = tk.Frame(self.root)
        cuerpo.pack(fill="both", expand=True, padx=10, pady=8)

        # Mapa (izquierda)
        izq = tk.Frame(cuerpo)
        izq.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(5.8, 5.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=izq)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel de información y resultados (derecha)
        der = tk.Frame(cuerpo)
        der.pack(side="right", fill="y", padx=(14, 0))

        caja_info = ttk.LabelFrame(der, text="Mapa actual", padding=8)
        caja_info.pack(fill="x")
        self.lbl_info = tk.Label(caja_info, text="", justify="left", anchor="w")
        self.lbl_info.pack(fill="x")

        caja_res = ttk.LabelFrame(der, text="Resultados en este mapa", padding=8)
        caja_res.pack(fill="x", pady=(10, 0))
        columnas = [
            ("alg", "Algoritmo", 75),
            ("exito", "Éxito", 50),
            ("pasos", "Pasos", 55),
            ("nodos", "Nodos exp.", 75),
            ("tiempo", "Tiempo (ms)", 85),
            ("mem", "Memoria (KB)", 95),
        ]
        self.tabla = ttk.Treeview(
            caja_res, columns=[c[0] for c in columnas], show="headings", height=3
        )
        for cid, texto, ancho in columnas:
            self.tabla.heading(cid, text=texto)
            self.tabla.column(cid, width=ancho, anchor="center")
        self.tabla.pack(fill="x")

        caja_ult = ttk.LabelFrame(der, text="Última ejecución", padding=8)
        caja_ult.pack(fill="x", pady=(10, 0))
        self.lbl_ultima = tk.Label(
            caja_ult, text="Elige un algoritmo para ejecutarlo.", justify="left",
            anchor="w", wraplength=420,
        )
        self.lbl_ultima.pack(fill="x")

        tk.Label(
            der,
            text=(
                "Amarillo = nodos expandidos (en orden de expansión)\n"
                "Línea roja = ruta encontrada\n"
                "Los valores de tiempo y memoria se miden igual que en el experimento."
            ),
            justify="left", anchor="w", fg="#555555",
        ).pack(fill="x", pady=(10, 0))

        pie = tk.Label(
            self.root,
            text=f"Sistemas Inteligentes 2026-2 · Salón {SALON} · Grupo {GRUPO} · " + " | ".join(INTEGRANTES),
            fg="#666666", wraplength=980, justify="center",
        )
        pie.pack(fill="x", pady=(0, 6))

    # -- control del mapa -------------------------------------------------------
    def _nuevo_mapa(self):
        self.rep = (self.rep + 1) % REPETICIONES
        self._cargar_mapa()

    def _cargar_mapa(self):
        self._cancelar_anim()
        nivel = self.nivel.get()
        self.m, self.s, self.t, valor = generar_mapa(nivel, self.rep)
        n = self.m.shape[0]
        self.img = mapa_a_rgb(self.m)

        ax = self.ax
        ax.clear()
        self.im = ax.imshow(self.img, interpolation="nearest", origin="upper")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#d0d0d0", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        ax.set_title(
            f"{GRUPO} - Aglomerados - Nivel {nivel} ({valor} aglomerados de radio {RADIO_AGLOMERADO})",
            fontsize=11,
        )

        h_ini, = ax.plot(self.s[1], self.s[0], "o", color="green", markersize=10, zorder=5, label="Inicio")
        h_meta, = ax.plot(self.t[1], self.t[0], "*", color="blue", markersize=15, zorder=5, label="Meta")
        self.linea, = ax.plot([], [], color=COL_RUTA, linewidth=3, zorder=4, label="Ruta")
        h_exp = Patch(facecolor=COL_EXPLORADO, edgecolor="#999999", label="Nodo expandido")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.legend(
            handles=[h_ini, h_meta, h_exp, self.linea],
            loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9, frameon=False,
        )
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.13)
        self.canvas.draw_idle()

        self.lbl_info.config(
            text=(
                f"Mapa: {GRUPO}_{nivel}_{self.rep:02d}  (repetición {self.rep} de {REPETICIONES - 1})\n"
                f"Semilla: {SEMILLA + self.rep}\n"
                f"Aglomerados: {valor}  (radio {RADIO_AGLOMERADO})\n"
                f"Densidad de muros: {float(self.m.mean()):.3f}"
            )
        )
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)
        for nombre in ALGS:
            self.tabla.insert("", "end", iid=nombre, values=(nombre, "-", "-", "-", "-", "-"))
        self.lbl_ultima.config(text="Elige un algoritmo para ejecutarlo.")

    def _restaurar_vista(self):
        self.img = mapa_a_rgb(self.m)
        self.im.set_data(self.img)
        self.linea.set_data([], [])

    # -- ejecución de un algoritmo -------------------------------------------------
    def ejecutar(self, nombre):
        self._cancelar_anim()
        self._restaurar_vista()
        fn = ALGS[nombre]

        # 1) medición (igual que el experimento): tiempo y pico de memoria
        ruta, exp, ms, kb = medir(fn, self.m, self.s, self.t)
        ok = ruta_valida(self.m, ruta, self.s, self.t)
        pasos = max(0, len(ruta) - 1)

        # 2) segunda corrida, fuera de la medición, solo para animar el recorrido
        traza = []
        fn(self.m, self.s, self.t, traza)

        self.tabla.item(
            nombre,
            values=(nombre, "Sí" if ok else "No", pasos, exp, f"{ms:.3f}", f"{kb:.1f}"),
        )
        estado = "ruta encontrada" if ok else "sin ruta"
        self.lbl_ultima.config(
            text=f"{nombre}: {estado} · {pasos} pasos · {exp} nodos expandidos · "
                 f"{ms:.3f} ms · {kb:.1f} KB"
        )
        self._animar(traza, ruta)

    # -- animación ------------------------------------------------------------------
    def _animar(self, traza, ruta):
        paso = max(1, len(traza) // 70)  # ~70 cuadros: la animación dura ~1 s
        estado = {"i": 0}

        def cuadro():
            ini = estado["i"]
            fin = min(len(traza), ini + paso)
            for u in traza[ini:fin]:
                self.img[u] = COL_EXPLORADO
            estado["i"] = fin
            self.im.set_data(self.img)
            if fin < len(traza):
                self.canvas.draw_idle()
                self.anim = self.root.after(15, cuadro)
            else:
                self.anim = None
                if ruta:
                    self.linea.set_data([p[1] for p in ruta], [p[0] for p in ruta])
                self.canvas.draw_idle()

        cuadro()

    def _cancelar_anim(self):
        if self.anim is not None:
            self.root.after_cancel(self.anim)
            self.anim = None

    def _cerrar(self):
        self._cancelar_anim()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
