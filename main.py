import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math


# =============================================================================
# ФУНКЦИИ ИЗ ЛАБОРАТОРНОЙ РАБОТЫ №6
# =============================================================================

def to_h(point3):
    """Возвращает однородный 4x1 вектор из 3D точки (x, y, z)."""
    x, y, z = point3
    return np.array([x, y, z, 1.0], dtype=float)


def from_h(vec4):
    """Возвращает 3D точку из однородного вектора после перспективного деления."""
    w = vec4[3]
    if w == 0:
        raise ZeroDivisionError("Однородная координата w == 0 при дегомогенизации")
    return (vec4[:3] / w)


def normalize(v):
    """Нормализует вектор."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def T(dx, dy, dz):
    """Матрица переноса (смещения)."""
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M


def S(sx, sy, sz):
    """Матрица масштабирования."""
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = sx, sy, sz
    return M


def Rx(angle_deg):
    """Матрица поворота вокруг оси X."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[1, 1], M[1, 2] = ca, -sa
    M[2, 1], M[2, 2] = sa, ca
    return M


def Ry(angle_deg):
    """Матрица поворота вокруг оси Y."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0, 0], M[0, 2] = ca, sa
    M[2, 0], M[2, 2] = -sa, ca
    return M


def Rz(angle_deg):
    """Матрица поворота вокруг оси Z."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0, 0], M[0, 1] = ca, -sa
    M[1, 0], M[1, 1] = sa, ca
    return M


def reflect(plane: str):
    """Отражение относительно координатной плоскости: 'xy', 'yz', или 'xz'."""
    plane = plane.lower()
    if plane == "xy":
        return S(1, 1, -1)
    if plane == "yz":
        return S(-1, 1, 1)
    if plane == "xz":
        return S(1, -1, 1)
    raise ValueError("Плоскость должна быть: 'xy', 'yz', 'xz'")


def rodrigues_axis_angle(axis, angle_deg):
    """Поворот 3x3 (формула Родрига) вокруг единичной оси на угол в градусах."""
    axis = normalize(np.asarray(axis, dtype=float))
    a = radians(angle_deg)
    c, s = cos(a), sin(a)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K
    return R


def R_around_line(p1, p2, angle_deg):
    """Матрица поворота 4x4 вокруг произвольной 3D линии p1->p2 на угол."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    R3 = rodrigues_axis_angle(axis, angle_deg)  # 3x3
    M = np.eye(4)
    M[:3, :3] = R3
    # Сэндвич-преобразование для поворота вокруг линии, проходящей через p1
    return T(*p1) @ M @ T(*(-p1))


def perspective(f=1.5):
    """
    Простая матрица перспективной проекции.
    Камера в начале координат смотрит вдоль +Z; точки сцены должны иметь z > 0.
    """
    M = np.eye(4)
    M[3, 2] = 1.0 / f  # w' = z/f  -> x' = x / (z/f) = f*x/z
    return M


def ortho_xy():
    """Ортографическая проекция на плоскость XY (отбрасывание Z)."""
    M = np.eye(4)
    M[2, 2] = 0.0
    return M


def isometric_projection_matrix():
    """Аксонометрическая (изометрическая) проекция = поворот + ортографическая проекция."""
    # Классическая изометрия: поворот вокруг Z на 45°, затем вокруг X на ~35.264°
    alpha = 35.264389682754654
    beta = 45.0
    R = Rx(alpha) @ Rz(beta)
    return ortho_xy() @ R


class Point:
    """Класс для представления точки в 3D пространстве."""

    def __init__(self, x, y, z):
        self.v = to_h((x, y, z))

    @property
    def xyz(self):
        return from_h(self.v)

    def as_array(self):
        return self.v.copy()


class PolygonFace:
    """Класс для представления грани многогранника."""

    def __init__(self, vertex_indices):
        self.indices = list(vertex_indices)


class Polyhedron:
    """Класс для представления многогранника."""

    def __init__(self, vertices, faces):
        """
        vertices: список вершин (кортежи (x,y,z))
        faces: список граней (списки индексов вершин)
        """
        self.V = np.array([to_h(p) for p in vertices], dtype=float).T  # 4xN (столбец = вершина)
        self.faces = [PolygonFace(f) for f in faces]

    def copy(self):
        """Создает копию многогранника."""
        P = Polyhedron([(0, 0, 0)], [[]])
        P.V = self.V.copy()
        P.faces = [PolygonFace(f.indices.copy()) for f in self.faces]
        return P

    def center(self):
        """Вычисляет центр многогранника."""
        pts = self.V[:3, :] / self.V[3, :]
        return np.mean(pts, axis=1)

    def apply(self, M):
        """Применяет матричное преобразование 4x4."""
        self.V = M @ self.V
        return self

    def translate(self, dx, dy, dz):
        """Перенос (смещение)."""
        return self.apply(T(dx, dy, dz))

    def scale(self, sx, sy, sz):
        """Масштабирование."""
        return self.apply(S(sx, sy, sz))

    def scale_about_center(self, s):
        """Масштабирование относительно центра."""
        c = self.center()
        return self.apply(T(*(-c)) @ S(s, s, s) @ T(*c))

    def rotate_x(self, angle_deg):
        """Поворот вокруг оси X."""
        return self.apply(Rx(angle_deg))

    def rotate_y(self, angle_deg):
        """Поворот вокруг оси Y."""
        return self.apply(Ry(angle_deg))

    def rotate_z(self, angle_deg):
        """Поворот вокруг оси Z."""
        return self.apply(Rz(angle_deg))

    def reflect(self, plane: str):
        """Отражение относительно координатной плоскости."""
        return self.apply(reflect(plane))

    def rotate_around_axis_through_center(self, axis: str, angle_deg):
        """Поворот вокруг оси, проходящей через центр."""
        axis = axis.lower()
        c = self.center()
        R = {'x': Rx, 'y': Ry, 'z': Rz}[axis](angle_deg)
        return self.apply(T(*(-c)) @ R @ T(*c))

    def rotate_around_line(self, p1, p2, angle_deg):
        """Поворот вокруг произвольной линии."""
        return self.apply(R_around_line(p1, p2, angle_deg))

    def edges(self):
        """Вычисляет список ребер многогранника."""
        es = set()
        if self.faces and len(self.faces[0].indices) > 0:
            # Строим ребра из граней
            for f in self.faces:
                idx = f.indices
                for i in range(len(idx)):
                    a = idx[i]
                    b = idx[(i + 1) % len(idx)]
                    es.add(tuple(sorted((a, b))))
        else:
            # Резервный метод: соединение ближайших соседей
            pts = (self.V[:3, :] / self.V[3, :]).T
            n = len(pts)
            D = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
            for i in range(n):
                neigh = list(np.argsort(D[i])[1:4])  # 3 ближайших соседа
                for j in neigh:
                    es.add(tuple(sorted((i, j))))
        return sorted(list(es))

    def projected(self, matrix4x4):
        """Возвращает 2D точки (x,y) после применения матрицы проекции."""
        Pv = matrix4x4 @ self.V
        # Перспективное деление
        Pv = Pv / Pv[3, :]
        # возвращаем только (x,y)
        return Pv[0, :], Pv[1, :]


def wireframe_2d(ax, P: Polyhedron, proj='perspective', f=1.5):
    """Отрисовывает многогранник в 2D после проекции."""
    if proj == 'perspective':
        M = perspective(f)
    elif proj == 'axonometric' or proj == 'isometric':
        M = isometric_projection_matrix()
    else:
        raise ValueError("proj должна быть 'perspective' или 'axonometric'")
    x, y = P.projected(M)
    # рисуем ребра
    for a, b in P.edges():
        ax.plot([x[a], x[b]], [y[a], y[b]])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)


# =============================================================================
# ЛАБОРАТОРНАЯ РАБОТА №7
# =============================================================================


# =========================================================================
# 1. OBJ файл
# =========================================================================

class OBJModel:
    """Класс для работы с OBJ моделями на основе Polyhedron"""

    def __init__(self, polyhedron=None):
        self.polyhedron = polyhedron

    def load_from_file(self, filename):
        """Загрузка модели из OBJ файла"""
        try:
            vertices = []
            faces = []

            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0] == 'v':  # вершина
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertices.append(vertex)

                    elif parts[0] == 'f':  # грань
                        face = []
                        for part in parts[1:]:
                            # Обработка формата vertex/texture/normal
                            vertex_index = part.split('/')[0]
                            if vertex_index:
                                face.append(int(vertex_index) - 1)  # OBJ использует 1-based индексацию
                        if len(face) >= 3:
                            faces.append(face)

            self.polyhedron = Polyhedron(vertices, faces)
            return True

        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False

    def save_to_file(self, filename):
        """Сохранение модели в OBJ файл"""
        try:
            if self.polyhedron is None:
                return False

            with open(filename, 'w') as file:
                file.write("# OBJ файл\n")

                # Запись вершин
                vertices = self.polyhedron.V[:3, :] / self.polyhedron.V[3, :]
                for i in range(vertices.shape[1]):
                    vertex = vertices[:, i]
                    file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

                # Запись граней
                for face in self.polyhedron.faces:
                    face_line = "f"
                    for vertex_index in face.indices:
                        face_line += f" {vertex_index + 1}"  # OBJ использует 1-based индексацию
                    file.write(face_line + "\n")

            return True

        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
            return False

# =========================================================================
# 2. модель вращения
# =========================================================================

class SurfaceOfRevolution:
    """Класс для создания фигур вращения"""

    @staticmethod
    def create_from_profile(profile_points, axis='y', segments=36):
        """
        Создание фигуры вращения

        Args:
            profile_points: список точек образующей [[x, y], ...]
            axis: ось вращения ('x', 'y', 'z')
            segments: количество сегментов
        """
        vertices = []
        faces = []

        # Преобразуем точки профиля в 3D ПРАВИЛЬНО
        profile_3d = []
        for point in profile_points:
            x, y = point
            if axis == 'x':  # вращение вокруг X
                # Точка (x,y) -> (0, x, y), вращаем вокруг X
                profile_3d.append([0, x, y])
            elif axis == 'y':  # вращение вокруг Y
                # Точка (x,y) -> (x, 0, y), вращаем вокруг Y
                profile_3d.append([x, 0, y])
            elif axis == 'z':  # вращение вокруг Z
                # Точка (x,y) -> (x, y, 0), вращаем вокруг Z
                profile_3d.append([x, y, 0])

        angle_step = 360.0 / segments

        # Создаем вершины ПРАВИЛЬНЫМ способом
        for i in range(segments + 1):
            angle = i * angle_step

            for point in profile_3d:
                # Преобразуем точку в однородные координаты
                point_h = to_h(point)

                # Применяем матрицу вращения
                if axis == 'x':
                    rotated_point = from_h(Rx(angle) @ point_h)
                elif axis == 'y':
                    rotated_point = from_h(Ry(angle) @ point_h)
                elif axis == 'z':
                    rotated_point = from_h(Rz(angle) @ point_h)

                vertices.append(rotated_point)

        print(f"Создано вершин: {len(vertices)}")
        print("Первые 10 вершин:")
        for i in range(min(10, len(vertices))):
            print(f"  {i}: {vertices[i]}")

        # Создаем грани
        profile_len = len(profile_3d)
        for i in range(segments):
            for j in range(profile_len - 1):
                # Индексы вершин для текущего сегмента
                v1 = i * profile_len + j
                v2 = i * profile_len + j + 1
                v3 = ((i + 1) % segments) * profile_len + j + 1  # исправлено: берем по модулю segments
                v4 = ((i + 1) % segments) * profile_len + j  # исправлено: берем по модулю segments

                # Два треугольника образуют четырехугольник
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        return Polyhedron(vertices, faces)

# =========================================================================
# 3. график двух переменных
# =========================================================================

class FunctionSurface:
    """Класс для создания поверхности функции двух переменных"""

    @staticmethod
    def create_surface(func, x_range, y_range, x_steps, y_steps):
        """
        Создание поверхности функции z = f(x, y)

        Args:
            func: функция f(x, y)
            x_range: кортеж (x_min, x_max)
            y_range: кортеж (y_min, y_max)
            x_steps: количество шагов по X
            y_steps: количество шагов по Y
        """
        vertices = []
        faces = []

        x_min, x_max = x_range
        y_min, y_max = y_range

        # Создаем сетку точек
        x_values = np.linspace(x_min, x_max, x_steps)
        y_values = np.linspace(y_min, y_max, y_steps)

        # Создаем вершины
        vertex_grid = []
        for i, x in enumerate(x_values):
            row = []
            for j, y in enumerate(y_values):
                try:
                    z = func(x, y)
                    vertex = [x, y, z]
                    vertices.append(vertex)
                    row.append(len(vertices) - 1)
                except:
                    vertex = [x, y, 0]
                    vertices.append(vertex)
                    row.append(len(vertices) - 1)
            vertex_grid.append(row)

        # Создаем грани
        for i in range(x_steps - 1):
            for j in range(y_steps - 1):
                # Индексы вершин для текущего квадрата
                v1 = vertex_grid[i][j]
                v2 = vertex_grid[i][j + 1]
                v3 = vertex_grid[i + 1][j + 1]
                v4 = vertex_grid[i + 1][j]

                # Два треугольника образуют четырехугольник
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        return Polyhedron(vertices, faces)


class ModelViewer:
    """Графический интерфейс для просмотра моделей"""

    def __init__(self, root):
        self.root = root
        self.root.title("3D Model Viewer - Лабораторная работа №7")
        self.root.geometry("1200x800")

        self.current_model = None
        self.fig = None
        self.ax = None

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 1. Загрузка OBJ модели
        obj_frame = ttk.LabelFrame(control_frame, text="1. Загрузка OBJ модели", padding=10)
        obj_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(obj_frame, text="Загрузить OBJ",
                   command=self.load_obj_model).pack(fill=tk.X)
        ttk.Button(obj_frame, text="Сохранить OBJ",
                   command=self.save_obj_model).pack(fill=tk.X, pady=(5, 0))

        # 2. Фигура вращения
        revolution_frame = ttk.LabelFrame(control_frame, text="2. Фигура вращения", padding=10)
        revolution_frame.pack(fill=tk.X, pady=(0, 10))

        # Параметры профиля
        profile_frame = ttk.Frame(revolution_frame)
        profile_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(profile_frame, text="Профиль (x,y):").pack(anchor=tk.W)
        self.profile_text = tk.Text(profile_frame, height=4, width=30)
        self.profile_text.pack(fill=tk.X, pady=(5, 0))
        self.profile_text.insert(tk.END, "0.0,0.0\n1.0,0.0\n1.0,2.0\n0.0,2.0")

        # Параметры вращения
        params_frame = ttk.Frame(revolution_frame)
        params_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(params_frame, text="Ось:").grid(row=0, column=0, sticky=tk.W)
        self.axis_var = tk.StringVar(value="y")
        ttk.Combobox(params_frame, textvariable=self.axis_var,
                     values=["x", "y", "z"], width=5).grid(row=0, column=1, padx=(5, 0))

        ttk.Label(params_frame, text="Сегменты:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.segments_var = tk.StringVar(value="36")
        ttk.Entry(params_frame, textvariable=self.segments_var, width=10).grid(row=1, column=1, padx=(5, 0),
                                                                               pady=(5, 0))

        ttk.Button(revolution_frame, text="Построить фигуру",
                   command=self.create_revolution).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(revolution_frame, text="Сохранить фигуру",
                   command=self.save_revolution).pack(fill=tk.X, pady=(5, 0))

        # 3. Поверхность функции
        surface_frame = ttk.LabelFrame(control_frame, text="3. Поверхность функции", padding=10)
        surface_frame.pack(fill=tk.X, pady=(0, 10))

        # Функция
        ttk.Label(surface_frame, text="Функция z = f(x,y):").pack(anchor=tk.W)
        self.func_var = tk.StringVar(value="math.sin(x) * math.cos(y)")
        ttk.Entry(surface_frame, textvariable=self.func_var).pack(fill=tk.X, pady=(5, 0))

        # Диапазоны
        range_frame = ttk.Frame(surface_frame)
        range_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(range_frame, text="X диапазон:").grid(row=0, column=0, sticky=tk.W)
        self.x_min_var = tk.StringVar(value="-3")
        self.x_max_var = tk.StringVar(value="3")
        ttk.Entry(range_frame, textvariable=self.x_min_var, width=5).grid(row=0, column=1, padx=(5, 0))
        ttk.Entry(range_frame, textvariable=self.x_max_var, width=5).grid(row=0, column=2, padx=(5, 0))

        ttk.Label(range_frame, text="Y диапазон:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.y_min_var = tk.StringVar(value="-3")
        self.y_max_var = tk.StringVar(value="3")
        ttk.Entry(range_frame, textvariable=self.y_min_var, width=5).grid(row=1, column=1, padx=(5, 0), pady=(5, 0))
        ttk.Entry(range_frame, textvariable=self.y_max_var, width=5).grid(row=1, column=2, padx=(5, 0), pady=(5, 0))

        # Шаги
        steps_frame = ttk.Frame(surface_frame)
        steps_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(steps_frame, text="Шаги по X:").grid(row=0, column=0, sticky=tk.W)
        self.x_steps_var = tk.StringVar(value="20")
        ttk.Entry(steps_frame, textvariable=self.x_steps_var, width=10).grid(row=0, column=1, padx=(5, 0))

        ttk.Label(steps_frame, text="Шаги по Y:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.y_steps_var = tk.StringVar(value="20")
        ttk.Entry(steps_frame, textvariable=self.y_steps_var, width=10).grid(row=0, column=3, padx=(5, 0))

        ttk.Button(surface_frame, text="Построить поверхность",
                   command=self.create_surface).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(surface_frame, text="Сохранить поверхность",
                   command=self.save_surface).pack(fill=tk.X, pady=(5, 0))

        # Аффинные преобразования
        transform_frame = ttk.LabelFrame(control_frame, text="Аффинные преобразования", padding=10)
        transform_frame.pack(fill=tk.X)

        # Вращение
        rotation_frame = ttk.Frame(transform_frame)
        rotation_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(rotation_frame, text="Вращение (°):").pack(anchor=tk.W)
        rot_buttons_frame = ttk.Frame(rotation_frame)
        rot_buttons_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(rot_buttons_frame, text="X+",
                   command=lambda: self.rotate_model('x', 10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(rot_buttons_frame, text="X-",
                   command=lambda: self.rotate_model('x', -10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(rot_buttons_frame, text="Y+",
                   command=lambda: self.rotate_model('y', 10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(rot_buttons_frame, text="Y-",
                   command=lambda: self.rotate_model('y', -10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(rot_buttons_frame, text="Z+",
                   command=lambda: self.rotate_model('z', 10)).pack(side=tk.LEFT, expand=True)
        ttk.Button(rot_buttons_frame, text="Z-",
                   command=lambda: self.rotate_model('z', -10)).pack(side=tk.LEFT, expand=True)

        # Масштабирование и перенос
        other_frame = ttk.Frame(transform_frame)
        other_frame.pack(fill=tk.X)

        ttk.Button(other_frame, text="Увеличить",
                   command=lambda: self.scale_model(1.2)).pack(side=tk.LEFT, expand=True)
        ttk.Button(other_frame, text="Уменьшить",
                   command=lambda: self.scale_model(0.8)).pack(side=tk.LEFT, expand=True)
        ttk.Button(other_frame, text="Сброс",
                   command=self.reset_view).pack(side=tk.LEFT, expand=True)

        # Правая панель - отображение
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Создаем фигуру для 3D отображения
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Начальное сообщение
        self.ax.text(0.5, 0.5, "Загрузите модель\nили создайте новую",
                     ha='center', va='center', fontsize=12, transform=self.ax.transAxes)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal', 'box')
        self.ax.grid(True)
        self.canvas.draw()

    # =========================================================================
    # 1. Преобразования и отображение, сохранение и загрузка модели
    # =========================================================================

    def load_obj_model(self):
        """Загрузка OBJ модели из файла"""
        filename = filedialog.askopenfilename(
            title="Выберите OBJ файл",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            obj_model = OBJModel()
            if obj_model.load_from_file(filename):
                self.current_model = obj_model.polyhedron
                self.plot_model()
                messagebox.showinfo("Успех",
                                    f"Модель загружена: {self.current_model.V.shape[1]} вершин, {len(self.current_model.faces)} граней")
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить модель")

    def save_obj_model(self):
        """Сохранение текущей модели в OBJ файл"""
        if self.current_model is None:
            messagebox.showwarning("Предупреждение", "Нет модели для сохранения")
            return

        filename = filedialog.asksaveasfilename(
            title="Сохранить OBJ файл",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            obj_model = OBJModel(self.current_model)
            if obj_model.save_to_file(filename):
                messagebox.showinfo("Успех", "Модель сохранена")
            else:
                messagebox.showerror("Ошибка", "Не удалось сохранить модель")


    def plot_model(self):
            """Отрисовка текущей модели с использованием wireframe_2d из лабы 6"""
            if self.current_model is None:
                return

            self.ax.clear()

            # Используем функцию из лабы 6 для отображения
            wireframe_2d(self.ax, self.current_model, proj='isometric')

            self.canvas.draw()

    def rotate_model(self, axis, angle):
            """Вращение модели с использованием методов из лабы 6"""
            if self.current_model is None:
                return

            if axis == 'x':
                self.current_model.rotate_x(angle)
            elif axis == 'y':
                self.current_model.rotate_y(angle)
            elif axis == 'z':
                self.current_model.rotate_z(angle)

            self.plot_model()

    def scale_model(self, factor):
            """Масштабирование модели с использованием методов из лабы 6"""
            if self.current_model is None:
                return

            self.current_model.scale(factor, factor, factor)
            self.plot_model()

    def reset_view(self):
            """Сброс вида"""
            if self.current_model is None:
                return

            self.plot_model()

    # =========================================================================
    # 2. Создание и сохранение модели вращения
    # =========================================================================

    def create_revolution(self):
        """Создание фигуры вращения"""
        try:
            # Чтение точек профиля
            profile_text = self.profile_text.get("1.0", tk.END).strip()
            profile_points = []

            for line in profile_text.split('\n'):
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        profile_points.append([x, y])

            if len(profile_points) < 2:
                messagebox.showerror("Ошибка", "Недостаточно точек профиля")
                return

            axis = self.axis_var.get()
            segments = int(self.segments_var.get())

            # Создание фигуры вращения
            self.current_model = SurfaceOfRevolution.create_from_profile(
                profile_points, axis, segments
            )

            self.plot_model()
            messagebox.showinfo("Успех",
                                f"Фигура вращения создана: {self.current_model.V.shape[1]} вершин, {len(self.current_model.faces)} граней")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания фигуры: {str(e)}")

    def save_revolution(self):
        """Сохранение фигуры вращения"""
        self.save_obj_model()

    # =========================================================================
    # 3. Создание и сохранение функции
    # =========================================================================

    def create_surface(self):
        """Создание поверхности функции"""
        try:
            # Получение параметров
            func_str = self.func_var.get()
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_min = float(self.y_min_var.get())
            y_max = float(self.y_max_var.get())
            x_steps = int(self.x_steps_var.get())
            y_steps = int(self.y_steps_var.get())

            # Создание функции из строки
            def surface_func(x, y):
                return eval(func_str)

            # Создание поверхности
            self.current_model = FunctionSurface.create_surface(
                surface_func, (x_min, x_max), (y_min, y_max), x_steps, y_steps
            )

            self.plot_model()
            messagebox.showinfo("Успех",
                                f"Поверхность создана: {self.current_model.V.shape[1]} вершин, {len(self.current_model.faces)} граней")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания поверхности: {str(e)}")

    def save_surface(self):
        """Сохранение поверхности"""
        self.save_obj_model()



def main():
    """Основная функция"""
    root = tk.Tk()
    app = ModelViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()