"""初中小板块题模板生成器（补齐「几何图像」与「二元一次方程」）。

**为什么用模板生成**：TAL-SCQ5K 的初中子集里几何题只有 126 条、二元一次方程 175 条
（该库以小学奥数竞赛题为主），达不到方案「每板块 200 题」的目标；
而 cn-k12 等备选源是英文且偏高中，不可用。

初中小板块题型高度规范化，用「参数化模板 + 程序计算答案」可以做到：
  * 答案 **100% 正确**（由代码算出，不存在标注错误）
  * 干扰项来自**典型错法**（如勾股定理忘记开方、面积公式漏除 2、消元后忘记除以 2），有教学价值
  * 解析是**真实的分步推导**，不是空话

产物写入 ``data/processed/synth_questions.jsonl``（parsed 中间格式），
由 ``build_question_bank.py`` 与 TAL 数据合并入库。
source 标记为 ``GEO_TEMPLATE``（几何）/ ``ALG_TEMPLATE``（代数）可追溯。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "processed" / "synth_questions.jsonl"

LETTERS = "ABCD"

# 常用勾股数（保证答案是整数，避免出现开不尽的根号）
_PYTH = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17),
         (12, 16, 20), (7, 24, 25), (10, 24, 26), (20, 21, 29), (12, 35, 37),
         (15, 20, 25), (18, 24, 30), (16, 30, 34), (21, 28, 35), (24, 32, 40)]


def _coef(k: int, sym: str = "x") -> str:
    """把系数 k 格式化成 kx / ky（k=±1 时省略 1，避免出现 y=-1x 这种写法）。"""
    if k == 1:
        return sym
    if k == -1:
        return "-" + sym
    return f"{k}{sym}"


def _kx(k: int) -> str:
    return _coef(k, "x")


def _lin(k: int, b: int) -> str:
    """一次函数 y=kx+b 的可读形式。"""
    return f"y={_kx(k)}{b:+d}"


def _fmt(x: float) -> str:
    """整数不带小数点；小数保留 1 位并去掉多余的 0。"""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


# ==========================================================================
# 模板定义
# 每个模板返回：(题干, 正确答案, [3个干扰项], 解析, 知识点, 难度)
# ==========================================================================
def t_pythag_hyp(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, b, c = rng.choice(_PYTH)
    a, b = rng.sample([a, b], 2)
    stem = (f"在 Rt△ABC 中，∠C=90°，两直角边分别为 {a} 和 {b}，"
            f"则斜边 AB 的长为（　）")
    wrongs = [_fmt(a + b), _fmt(a * a + b * b), _fmt(abs(a - b))]
    ana = (f"由勾股定理：AB²=AC²+BC²={a}²+{b}²={a*a}+{b*b}={c*c}，"
           f"所以 AB=√{c*c}={c}。")
    return stem, str(c), wrongs, ana, ["勾股定理", "直角三角形"], 2


def t_pythag_leg(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, b, c = rng.choice(_PYTH)
    leg, other = rng.sample([a, b], 2)
    stem = (f"在 Rt△ABC 中，∠C=90°，斜边 AB={c}，一条直角边 AC={leg}，"
            f"则另一条直角边 BC 的长为（　）")
    wrongs = [_fmt(c - leg), _fmt(c + leg), _fmt(c * c + leg * leg)]
    ana = (f"由勾股定理：BC²=AB²-AC²={c}²-{leg}²={c*c}-{leg*leg}={other*other}，"
           f"所以 BC=√{other*other}={other}。")
    return stem, str(other), wrongs, ana, ["勾股定理", "直角三角形"], 3


def t_triangle_angle(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a = rng.randint(30, 90)
    b = rng.randint(20, 100 - a if a < 80 else 60)
    b = max(10, min(b, 170 - a))
    c = 180 - a - b
    if c <= 0:
        b, c = 30, 180 - a - 30
    stem = f"在△ABC 中，∠A={a}°，∠B={b}°，则∠C 的度数为（　）"
    wrongs = [str(a + b), str(180 - a), str(abs(a - b))]
    ana = f"三角形内角和为 180°，所以 ∠C=180°-{a}°-{b}°={c}°。"
    return stem, str(c), wrongs, ana, ["三角形内角和定理"], 1


def t_isosceles_apex(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    apex = rng.choice([40, 50, 60, 70, 80, 90, 100, 120])
    base = (180 - apex) // 2
    stem = f"等腰三角形的顶角为 {apex}°，则它的一个底角为（　）"
    # 注意：不能用 90-apex//2 —— 它恒等于底角 (180-apex)/2，会与正确答案重复
    wrongs = [str(180 - apex), str(apex), str(apex // 2)]
    ana = (f"等腰三角形两底角相等，且内角和为 180°，"
           f"所以底角=(180°-{apex}°)÷2={base}°。")
    return stem, str(base), wrongs, ana, ["等腰三角形", "三角形内角和定理"], 2


def t_isosceles_base(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    base = rng.choice([30, 40, 45, 50, 55, 60, 65, 70])
    apex = 180 - 2 * base
    stem = f"等腰三角形的一个底角为 {base}°，则它的顶角为（　）"
    wrongs = [str(180 - base), str(2 * base), str(90 - base)]
    ana = (f"两底角相等，顶角=180°-{base}°×2={apex}°。")
    return stem, str(apex), wrongs, ana, ["等腰三角形", "三角形内角和定理"], 2


def t_rect_perimeter(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, b = rng.randint(3, 20), rng.randint(2, 15)
    stem = f"一个长方形的长为 {a} cm，宽为 {b} cm，则它的周长为（　）"
    wrongs = [_fmt(a * b), _fmt(a + b), _fmt(2 * a + b)]
    ana = f"长方形周长=2×(长+宽)=2×({a}+{b})={2*(a+b)} cm。"
    return stem, str(2 * (a + b)), wrongs, ana, ["长方形", "周长"], 1


def t_rect_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, b = rng.randint(3, 20), rng.randint(2, 15)
    stem = f"一个长方形的长为 {a} cm，宽为 {b} cm，则它的面积为（　）"
    wrongs = [_fmt(2 * (a + b)), _fmt(a + b), _fmt(2 * a * b)]
    ana = f"长方形面积=长×宽={a}×{b}={a*b} cm²。"
    return stem, str(a * b), wrongs, ana, ["长方形", "面积"], 1


def t_triangle_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    h = rng.randint(3, 16)
    a = rng.choice([x for x in range(4, 25) if x % 2 == 0])
    stem = f"一个三角形的底边长为 {a} cm，对应高为 {h} cm，则它的面积为（　）"
    wrongs = [_fmt(a * h), _fmt(a + h), _fmt(a * h // 4 if a * h % 4 == 0 else a * h / 4)]
    ana = f"三角形面积=底×高÷2={a}×{h}÷2={a*h//2} cm²。"
    return stem, str(a * h // 2), wrongs, ana, ["三角形", "面积"], 2


def t_para_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, h = rng.randint(4, 20), rng.randint(3, 15)
    stem = (f"一个平行四边形的底边长为 {a} cm，这条底边上的高为 {h} cm，"
            f"则它的面积为（　）")
    wrongs = [_fmt(a * h // 2), _fmt(2 * (a + h)), _fmt(a + h)]
    ana = f"平行四边形面积=底×高={a}×{h}={a*h} cm²。"
    return stem, str(a * h), wrongs, ana, ["平行四边形", "面积"], 2


def t_trapezoid_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a = rng.randint(3, 12)
    b = a + rng.randint(2, 10)
    h = rng.choice([x for x in range(4, 15) if (a + b) * x % 2 == 0])
    stem = (f"一个梯形的上底为 {a} cm，下底为 {b} cm，高为 {h} cm，"
            f"则它的面积为（　）")
    wrongs = [_fmt((a + b) * h), _fmt((a + b) * h // 4), _fmt(a * h + b)]
    ana = (f"梯形面积=(上底+下底)×高÷2=({a}+{b})×{h}÷2={(a+b)*h//2} cm²。")
    return stem, str((a + b) * h // 2), wrongs, ana, ["梯形", "面积"], 2


def t_circle_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    r = rng.randint(2, 12)
    stem = f"一个圆的半径为 {r} cm，则它的面积为（　）"
    # 干扰项：① 与周长混淆 2rπ ② 忘记乘 π ③ 错用 2r²π
    wrongs = [f"{2*r}π", f"{r*r}", f"{2*r*r}π"]
    ana = f"圆面积=πr²=π×{r}²={r*r}π cm²。"
    return stem, f"{r*r}π", wrongs, ana, ["圆", "面积"], 2


def t_circle_circum(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    r = rng.randint(2, 15)
    stem = f"一个圆的半径为 {r} cm，则它的周长为（　）"
    wrongs = [f"{r*r}π", f"{r}π", f"{4*r}π"]
    ana = f"圆周长=2πr=2×π×{r}={2*r}π cm。"
    return stem, f"{2*r}π", wrongs, ana, ["圆", "周长"], 1


def t_polygon_sum(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    n = rng.randint(5, 10)
    stem = f"{n} 边形的内角和是（　）"
    wrongs = [str(n * 180), str((n - 1) * 180), str(360)]
    ana = f"n 边形内角和=(n-2)×180°=({n}-2)×180°={(n-2)*180}°。"
    return stem, str((n - 2) * 180), wrongs, ana, ["多边形", "内角和"], 2


def t_regular_angle(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    n = rng.choice([5, 6, 8, 9, 10, 12])
    v = (n - 2) * 180 / n
    stem = f"正 {n} 边形的每个内角为（　）"
    wrongs = [str(360 // n), _fmt(v / 2), str((n - 2) * 180)]
    ana = (f"正 n 边形每个内角=(n-2)×180°÷n=({n}-2)×180°÷{n}={_fmt(v)}°。")
    return stem, _fmt(v), wrongs, ana, ["正多边形", "内角和"], 3


def t_similar(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    k = rng.choice([2, 3, 4, 5])
    # 只取能被 k 整除的边长，避免出现 5÷3=1.7 这类除不尽的小数
    a = rng.choice([x for x in range(k, 25) if x % k == 0])
    stem = (f"已知△ABC∽△DEF，相似比为 {k}:1，若 AB={a} 且 AB 与 DE 是对应边，"
            f"则 DE 的长为（　）")
    wrongs = [_fmt(a * k), _fmt(a + k), _fmt(a * k * 2)]
    val = a / k
    ana = (f"相似比 {k}:1 表示 AB:DE={k}:1，所以 DE=AB÷{k}={a}÷{k}={_fmt(val)}。")
    return stem, _fmt(val), wrongs, ana, ["相似三角形", "比例"], 3


def t_right_median(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    c = rng.choice([x for x in range(6, 31) if x % 2 == 0])
    stem = (f"在 Rt△ABC 中，∠C=90°，斜边 AB={c}，CD 是斜边上的中线，"
            f"则 CD 的长为（　）")
    wrongs = [str(c), str(2 * c), _fmt(c / 4)]
    ana = (f"直角三角形斜边上的中线等于斜边的一半，"
           f"所以 CD=AB÷2={c}÷2={c//2}。")
    return stem, str(c // 2), wrongs, ana, ["直角三角形", "斜边中线"], 3


def t_midline(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    b = rng.choice([x for x in range(6, 31) if x % 2 == 0])
    stem = (f"在△ABC 中，D、E 分别是 AB、AC 的中点，若 BC={b}，"
            f"则中位线 DE 的长为（　）")
    wrongs = [str(b), str(2 * b), _fmt(b / 4)]
    ana = f"三角形中位线平行于第三边且等于第三边的一半，所以 DE=BC÷2={b}÷2={b//2}。"
    return stem, str(b // 2), wrongs, ana, ["三角形中位线定理"], 3


def t_parallel_angle(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    x = rng.randint(25, 145)
    stem = (f"如图，直线 a∥b，直线 c 与 a 相交所成的一个角为 {x}°，"
            f"则 c 与 b 相交所成的同位角为（　）")
    wrongs = [str(180 - x), str(90 - x if x < 90 else 2 * x), str(360 - x)]
    ana = f"两直线平行，同位角相等，所以所求角={x}°。"
    return stem, str(x), wrongs, ana, ["平行线", "同位角"], 2


def t_cuboid_volume(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a, b, c = rng.randint(2, 10), rng.randint(2, 10), rng.randint(2, 10)
    stem = (f"一个长方体的长、宽、高分别为 {a} cm、{b} cm、{c} cm，"
            f"则它的体积为（　）")
    wrongs = [_fmt(2 * (a * b + b * c + a * c)), _fmt(a + b + c), _fmt(a * b + c)]
    ana = f"长方体体积=长×宽×高={a}×{b}×{c}={a*b*c} cm³。"
    return stem, str(a * b * c), wrongs, ana, ["长方体", "体积"], 2


def t_sector_area(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    # 只取 n·r²/360 为整数或 .5 的组合，避免 0.75 被四舍五入成 0.8 的精度错误
    pairs = [(n, r) for n in (30, 45, 60, 90, 120) for r in range(2, 13)
             if (n * r * r * 2) % 360 == 0]
    n, r = rng.choice(pairs)
    v = n * r * r / 360.0
    stem = f"一个扇形的半径为 {r} cm，圆心角为 {n}°，则它的面积为（　）"
    wrongs = [f"{r*r}π", f"{n*r}π", f"{n*r*r//180}π"]
    ana = (f"扇形面积=nπr²/360={n}×π×{r}²÷360={_fmt(v)}π cm²。")
    return stem, f"{_fmt(v)}π", wrongs, ana, ["扇形", "面积"], 3


def t_square_diagonal(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    a = rng.randint(2, 12)
    stem = f"一个正方形的边长为 {a} cm，则它的对角线长为（　）"
    wrongs = [_fmt(2 * a), _fmt(a * a), _fmt(a * 3)]
    ana = (f"正方形对角线=边长×√2，由勾股定理 d²={a}²+{a}²={2*a*a}，"
           f"所以 d={a}√2 cm。")
    return stem, f"{a}√2", wrongs, ana, ["正方形", "勾股定理"], 3


# ==========================================================================
# 二元一次方程 / 一次函数 模板
# 同样保证：答案由程序算出、干扰项来自典型错法、解析为真实推导
# ==========================================================================
def a_system_add(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """x+y=S, x−y=D 型方程组（加减消元）。"""
    while True:
        x = rng.randint(10, 25)
        y = rng.randint(1, 9)
        S, D = x + y, x - y
        wrongs = [str(y), str(S), str(S + D)]       # S+D=2x：只加不除以 2
        if len({str(x), *wrongs}) == 4:
            break
    stem = f"已知 x、y 满足方程组 x+y={S}，x−y={D}，则 x 的值为（　）"
    ana = (f"两式相加得 (x+y)+(x−y)={S}+{D}，即 2x={S+D}，所以 x={x}；"
           f"代入 x+y={S} 得 y={S}−{x}={y}。")
    return stem, str(x), wrongs, ana, ["二元一次方程组", "加减消元法"], 2


def a_system_std(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """一般二元一次方程组：先随机整数解，再反推常数项，保证可解且为整数。"""
    while True:
        x0, y0 = rng.randint(1, 9), rng.randint(1, 9)
        a1, b1 = rng.randint(2, 6), rng.randint(1, 6)
        a2, b2 = rng.randint(1, 6), rng.randint(2, 6)
        det = a1 * b2 - a2 * b1
        if det <= 0:                                # 保持 det>0，解析里不出现负系数
            continue
        c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
        wrongs = [str(y0), str(x0 + y0), str(x0 * y0)]
        if len({str(x0), *wrongs}) == 4:
            break
    e1 = f"{_coef(a1, 'x')}+{_coef(b1, 'y')}={c1}"
    e2 = f"{_coef(a2, 'x')}+{_coef(b2, 'y')}={c2}"
    stem = f"解方程组 {e1}，{e2}，则 x 的值为（　）"
    ana = (f"①×{b2} 得 {_coef(a1*b2, 'x')}+{_coef(b1*b2, 'y')}={c1*b2}，"
           f"②×{b1} 得 {_coef(a2*b1, 'x')}+{_coef(b1*b2, 'y')}={c2*b1}；"
           f"两式相减得 {_coef(det, 'x')}={c1*b2-c2*b1}，"
           f"所以 x={x0}，再代入①得 y={y0}。")
    return stem, str(x0), wrongs, ana, ["二元一次方程组", "加减消元法"], 3


def a_substitute(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """代入消元法：x=ky 且 x+y=S。"""
    while True:
        k = rng.randint(2, 5)
        y0 = rng.randint(2, 12)
        x0, S = k * y0, k * y0 + y0
        wrongs = [str(x0), str(S), str(S - k)]
        if len({str(y0), *wrongs}) == 4:
            break
    stem = f"已知 x={k}y，且 x+y={S}，则 y 的值为（　）"
    ana = (f"把 x={k}y 代入 x+y={S} 得 {k}y+y={S}，即 {k+1}y={S}，"
           f"所以 y={y0}，进而 x={k}×{y0}={x0}。")
    return stem, str(y0), wrongs, ana, ["二元一次方程组", "代入消元法"], 2


def a_slope(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """已知两点求一次函数斜率。"""
    while True:
        x1, x2 = rng.randint(-5, 5), rng.randint(-5, 5)
        if abs(x2 - x1) < 2:                        # 间距为 1 时 y₂−y₁ 会等于 k
            continue
        k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        y1 = rng.randint(-6, 8)
        y2 = y1 + k * (x2 - x1)
        wrongs = [str(-k), str(y2 - y1), str(x2 - x1)]
        if len({str(k), *wrongs}) == 4:
            break
    stem = (f"已知一次函数 y=kx+b 的图象经过点 ({x1}, {y1}) 和 ({x2}, {y2})，"
            f"则 k 的值为（　）")
    ana = (f"k=(y₂−y₁)/(x₂−x₁)=({y2}−({y1}))/({x2}−({x1}))="
           f"{y2-y1}/{x2-x1}={k}。")
    return stem, str(k), wrongs, ana, ["一次函数", "斜率"], 3


def a_intercept_x(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """一次函数与 x 轴交点横坐标。"""
    while True:
        k = rng.randint(2, 6)
        t = rng.randint(-6, 6)
        if t == 0:
            continue
        b = -k * t
        wrongs = [str(-b), str(b), str(t + 1)]
        if len({str(t), *wrongs}) == 4:
            break
    stem = f"一次函数 {_lin(k, b)} 的图象与 x 轴交点的横坐标是（　）"
    ana = (f"与 x 轴交点处 y=0，令 {_kx(k)}{b:+d}=0 得 {_kx(k)}={-b}，"
           f"所以 x={_fmt(-b / k)}。")
    return stem, str(t), wrongs, ana, ["一次函数", "图象与坐标轴交点"], 2


def a_point_on_line(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """判断哪个点在给定直线上。"""
    k, b = rng.randint(2, 5), rng.randint(-5, 5)
    x0 = rng.randint(1, 6)
    y0 = k * x0 + b
    correct = f"({x0}, {y0})"
    wrongs, seen = [], {correct}
    while len(wrongs) < 3:
        wx, wy = rng.randint(-4, 8), rng.randint(-8, 20)
        cand = f"({wx}, {wy})"
        if cand in seen or wy == k * wx + b:
            continue
        seen.add(cand)
        wrongs.append(cand)
    stem = f"下列四个点中，在一次函数 {_lin(k, b)} 图象上的是（　）"
    ana = (f"把 x={x0} 代入 {_lin(k, b)} 得 y={k}×{x0}{b:+d}={y0}，"
           f"所以点 ({x0}, {y0}) 在图象上；其余点代入后左右两边不相等。")
    return stem, correct, wrongs, ana, ["一次函数", "图象上的点"], 2


def a_chicken_rabbit(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """应用题：鸡兔同笼（本质是二元一次方程组）。"""
    while True:
        r, c = rng.randint(3, 20), rng.randint(3, 20)
        if r == c or 2 * r == c:
            continue
        H, F = r + c, 4 * r + 2 * c
        wrongs = [str(c), str(H), str(F - 2 * H)]   # F−2H=2r：忘了再除以 2
        if len({str(r), *wrongs}) == 4:
            break
    stem = f"笼中有鸡和兔共 {H} 个头、{F} 只脚，则兔有（　）只"
    ana = (f"设兔 x 只，则鸡 {H}−x 只。由脚数列方程 4x+2({H}−x)={F}，"
           f"化简得 2x={F-2*H}，所以 x={r}，即兔 {r} 只、鸡 {c} 只。")
    return stem, str(r), wrongs, ana, ["二元一次方程组", "鸡兔同笼"], 3


def a_ticket(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """应用题：两种票价的总价问题。"""
    while True:
        a, b = rng.randint(8, 20), rng.randint(3, 7)
        x, y = rng.randint(3, 20), rng.randint(3, 20)
        n, m = x + y, a * x + b * y
        wrongs = [str(y), str(n), str(m // a)]
        if len({str(x), *wrongs}) == 4:
            break
    stem = (f"购买单价 {a} 元的 A 票和单价 {b} 元的 B 票共 {n} 张，"
            f"一共花了 {m} 元，则 A 票买了（　）张")
    ana = (f"设 A 票 x 张，则 B 票 {n}−x 张。列方程 {a}x+{b}({n}−x)={m}，"
           f"化简得 {a-b}x={m-b*n}，所以 x={x}。")
    return stem, str(x), wrongs, ana, ["二元一次方程组", "列方程解应用题"], 3


def a_age(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """应用题：年龄倍数问题。"""
    while True:
        s = rng.randint(5, 15)
        n = rng.randint(2, 12)
        m = rng.randint(2, 4)
        f = m * (s + n) - n
        if not (s + 18 <= f <= 70):                 # 父亲年龄要合理
            continue
        wrongs = [str(m - 1), str(m + 1), str(f - s)]
        if len({str(m), *wrongs}) == 4:
            break
    stem = (f"今年儿子 {s} 岁，父亲 {f} 岁，{n} 年后父亲的年龄是儿子的（　）倍")
    ana = (f"{n} 年后儿子 {s+n} 岁，父亲 {f+n} 岁；"
           f"{f+n}÷{s+n}={m}，所以是 {m} 倍。")
    return stem, str(m), wrongs, ana, ["一元一次方程", "年龄问题"], 2


def a_two_digit(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """数字问题：十位个位交换。"""
    while True:
        s = rng.choice([5, 7, 9, 11, 13, 15])
        a, b = (s - 3) // 2, (s + 3) // 2
        if a < 1 or b > 9:
            continue
        num = 10 * a + b
        wrongs = [str(10 * b + a), str(s), str(num - 27)]
        if len({str(num), *wrongs}) == 4:
            break
    stem = (f"一个两位数，十位数字与个位数字之和为 {s}；把十位数字与个位数字"
            f"交换后，得到的新数比原数大 27，则原来的两位数是（　）")
    ana = (f"设十位数字为 a、个位数字为 b。交换后大 27，即 "
           f"(10b+a)−(10a+b)=9(b−a)=27，得 b−a=3；又 a+b={s}，"
           f"解得 a={a}、b={b}，所以原数是 {num}。")
    return stem, str(num), wrongs, ana, ["二元一次方程组", "数字问题"], 3


# ==========================================================================
# 函数 模板（以一次函数为主，控制在初中课标内）
# ==========================================================================
_QUADRANT = {(1, 1): "一、二、三", (1, -1): "一、三、四",
             (-1, 1): "一、二、四", (-1, -1): "二、三、四"}


def f_eval(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """已知一次函数与 x，求 y。"""
    while True:
        k, b = rng.randint(2, 8) * rng.choice([1, -1]), rng.randint(-9, 9)
        x0 = rng.randint(-6, 9)
        y0 = k * x0 + b
        wrongs = [str(k * x0), str(x0 + b), str(k + b + x0)]   # 漏 b / 漏 k / 乱加
        if len({str(y0), *wrongs}) == 4:
            break
    stem = f"已知一次函数 {_lin(k, b)}，当 x={x0} 时，y 的值为（　）"
    ana = f"把 x={x0} 代入 {_lin(k, b)}，得 y={k}×({x0}){b:+d}={y0}。"
    return stem, str(y0), wrongs, ana, ["一次函数", "函数值"], 1


def f_find_x(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """已知一次函数与 y，求 x（保证整除）。"""
    while True:
        k = rng.choice([2, 3, 4, 5, 6]) * rng.choice([1, -1])
        x0 = rng.randint(-8, 8)
        b = rng.randint(-9, 9)
        y0 = k * x0 + b
        wrongs = [str(y0 // k), str(-x0), str(x0 + k)]
        if len({str(x0), *wrongs}) == 4:
            break
    stem = f"已知一次函数 {_lin(k, b)}，当 y={y0} 时，x 的值为（　）"
    ana = (f"把 y={y0} 代入得 {_kx(k)}{b:+d}={y0}，移项得 {_kx(k)}={y0-b}，"
           f"所以 x={x0}。")
    return stem, str(x0), wrongs, ana, ["一次函数", "解一元一次方程"], 2


def f_two_points(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """已知两点求一次函数解析式。"""
    while True:
        x1, x2 = rng.randint(-5, 5), rng.randint(-5, 5)
        if abs(x2 - x1) < 2:
            continue
        k = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        y1 = rng.randint(-6, 8)
        y2 = y1 + k * (x2 - x1)
        b = y1 - k * x1
        correct = f"{_lin(k, b)}"
        # 典型错法：斜率取反、截距符号错、代入 y₂ 时符号错
        bk = y1 + k * x1
        wrongs = [f"{_lin(-k, y1 + k * x1)}",
                  f"{_lin(k, bk)}" if bk != b else f"{_lin(k, b + 1)}",
                  f"{_lin(k, y2 + k * x2)}"]
        if len({correct, *wrongs}) == 4:
            break
    stem = (f"已知一次函数的图象经过点 ({x1}, {y1}) 和 ({x2}, {y2})，"
            f"则这个一次函数的解析式为（　）")
    ana = (f"k=(y₂−y₁)/(x₂−x₁)=({y2}−({y1}))/({x2}−({x1}))={k}；"
           f"把 ({x1}, {y1}) 代入 y={_kx(k)}+b 得 {y1}={k}×({x1})+b，解得 b={b}。"
           f"所以解析式为 {_lin(k, b)}。")
    return stem, correct, wrongs, ana, ["一次函数", "待定系数法"], 3


def f_quadrant(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """判断一次函数图象经过的象限。"""
    k = rng.choice([2, 3, 4, 5]) * rng.choice([1, -1])
    b = rng.randint(1, 9) * rng.choice([1, -1])
    correct = _QUADRANT[(1 if k > 0 else -1, 1 if b > 0 else -1)]
    wrongs = [v for key, v in _QUADRANT.items() if v != correct]
    rng.shuffle(wrongs)
    stem = f"一次函数 {_lin(k, b)} 的图象经过（　）"
    ana = (f"k={k}>0" if k > 0 else f"k={k}<0")
    ana = (f"因为 k={k}＞0，y 随 x 增大而增大；又因为 b={b}，图象与 y 轴交于"
           if k > 0 else
           f"因为 k={k}＜0，y 随 x 增大而减小；又因为 b={b}，图象与 y 轴交于")
    side = "正半轴" if b > 0 else "负半轴"
    ana += f"{side}（0, {b}）。所以图象经过 {correct} 象限。"
    return stem, correct, wrongs[:3], ana, ["一次函数", "图象与象限"], 2


def f_area_axis(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """一次函数与两条坐标轴围成的三角形面积。"""
    while True:
        k = rng.choice([1, 2, 3, 4])
        x0 = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        if (k * x0 * x0) % 2 != 0:          # 面积 = k·x₀²/2，保证整数
            continue
        b = -k * x0
        area = k * x0 * x0 // 2
        wrongs = [str(k * abs(x0)), str(abs(b) * 2), str(area + abs(b))]
        if len({str(area), *wrongs}) == 4:
            break
    stem = (f"一次函数 {_lin(k, b)} 的图象与 x 轴、y 轴围成的三角形面积是（　）")
    ana = (f"与 x 轴交点：令 y=0 得 x={x0}，即 ({x0}, 0)；与 y 轴交点：(0, {b})。"
           f"两交点到原点的距离分别为 {abs(x0)} 和 {abs(b)}，"
           f"所以面积={abs(x0)}×{abs(b)}÷2={area}。")
    return stem, str(area), wrongs, ana, ["一次函数", "图象与坐标轴交点"], 3


def f_proportional(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """正比例函数求系数。"""
    while True:
        k = rng.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        x0 = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        y0 = k * x0
        wrongs = [str(-k), str(y0 + x0), str(x0 + k)]
        if len({str(k), *wrongs}) == 4:
            break
    stem = f"已知正比例函数 y=kx 的图象经过点 ({x0}, {y0})，则 k 的值为（　）"
    ana = (f"把 ({x0}, {y0}) 代入 y=kx 得 {y0}=k×({x0})，所以 k={y0}÷({x0})={k}。")
    return stem, str(k), wrongs, ana, ["正比例函数", "待定系数法"], 1


def f_inverse_prop(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """反比例函数求系数。"""
    while True:
        x0 = rng.choice([-6, -4, -3, -2, 2, 3, 4, 6])
        y0 = rng.choice([-6, -5, -4, -3, 3, 4, 5, 6])
        k = x0 * y0
        wrongs = [str(-k), str(x0 + y0), str(abs(k) // 2)]
        if len({str(k), *wrongs}) == 4:
            break
    stem = f"已知反比例函数 y=k/x 的图象经过点 ({x0}, {y0})，则 k 的值为（　）"
    ana = f"把 ({x0}, {y0}) 代入 y=k/x 得 {y0}=k/({x0})，所以 k=({x0})×({y0})={k}。"
    return stem, str(k), wrongs, ana, ["反比例函数", "待定系数法"], 2


def f_intersection(rng) -> Tuple[str, str, List[str], str, List[str], int]:
    """两条一次函数图象的交点坐标。"""
    while True:
        x0, y0 = rng.randint(-5, 5), rng.randint(-6, 6)
        k1, k2 = rng.sample([-4, -3, -2, -1, 1, 2, 3, 4], 2)
        b1, b2 = y0 - k1 * x0, y0 - k2 * x0
        correct = f"({x0}, {y0})"
        wrongs = [f"({x0}, {y0 + k1})", f"({y0}, {x0})", f"({-x0}, {y0})"]
        if len({correct, *wrongs}) == 4:
            break
    stem = (f"直线 {_lin(k1, b1)} 与直线 {_lin(k2, b2)} 的交点坐标是（　）")
    ana = (f"联立两式：{_kx(k1)}{b1:+d}={_kx(k2)}{b2:+d}，移项得 "
           f"{_kx(k1-k2)}={b2-b1}，所以 x={x0}，代入得 y={y0}。")
    return stem, correct, wrongs, ana, ["一次函数", "两直线交点"], 3


TEMPLATES: List[Callable] = [
    t_pythag_hyp, t_pythag_leg, t_triangle_angle, t_isosceles_apex,
    t_isosceles_base, t_rect_perimeter, t_rect_area, t_triangle_area,
    t_para_area, t_trapezoid_area, t_circle_area, t_circle_circum,
    t_polygon_sum, t_regular_angle, t_similar, t_right_median,
    t_midline, t_parallel_angle, t_cuboid_volume, t_sector_area,
    t_square_diagonal,
    # ---- 二元一次方程 / 一次函数 ----
    a_system_add, a_system_std, a_substitute, a_slope, a_intercept_x,
    a_point_on_line, a_chicken_rabbit, a_ticket, a_age, a_two_digit,
    # ---- 函数 ----
    f_eval, f_find_x, f_two_points, f_quadrant, f_area_axis,
    f_proportional, f_inverse_prop, f_intersection,
]

# 模板 → 板块 / 是否应用题（函数属性，供 build_question 读取）
for _t in (a_system_add, a_system_std, a_substitute, a_slope, a_intercept_x,
           a_point_on_line, a_chicken_rabbit, a_ticket, a_age, a_two_digit):
    _t.topic = "二元一次方程"
for _t in (f_eval, f_find_x, f_two_points, f_quadrant, f_area_axis,
           f_proportional, f_inverse_prop, f_intersection):
    _t.topic = "函数"
for _t in (a_chicken_rabbit, a_ticket, a_age):
    _t.is_app = True


# ==========================================================================
def build_question(tpl: Callable, rng: random.Random) -> Dict[str, Any] | None:
    """生成一个 parsed 结构的题目；选项重复或非法时返回 None。"""
    try:
        stem, correct, wrongs, ana, kps, diff = tpl(rng)
    except Exception:
        return None

    # 干扰项去重且不得与正确答案相同
    pool, seen = [], {str(correct)}
    for w in wrongs:
        w = str(w)
        if w and w not in seen:
            seen.add(w)
            pool.append(w)

    # 模板自带的干扰项撞车时，用数值扰动兜底（保证任何模板都不会静默丢失）
    if len(pool) < 3:
        try:
            base_v = float(str(correct).replace("π", ""))
            suffix = "π" if "π" in str(correct) else ""
            for delta in (1, -1, 2, -2, 3, 5, -3, 10):
                v = base_v + delta
                if v <= 0:
                    continue
                cand = f"{_fmt(v)}{suffix}"
                if cand not in seen:
                    seen.add(cand)
                    pool.append(cand)
                if len(pool) >= 3:
                    break
        except ValueError:      # 含 √ 的答案无法数值扰动，改用前缀变化
            for cand in (f"{correct}/2", f"2{correct}", f"{correct}+1"):
                if cand not in seen:
                    seen.add(cand)
                    pool.append(cand)
                if len(pool) >= 3:
                    break
    if len(pool) < 3:
        return None

    contents = [str(correct)] + pool[:3]
    rng.shuffle(contents)
    idx = contents.index(str(correct))
    answer_letter = LETTERS[idx]
    options = [f"{LETTERS[i]}. {c}" for i, c in enumerate(contents)]
    if len({c for c in contents}) != 4:
        return None

    topic = getattr(tpl, "topic", "几何图像")
    return {
        "stem": stem,
        "options": options,
        "answer": answer_letter,
        "answer_content": str(correct),
        "analysis": ana,
        "knowledge_points": kps,
        "difficulty": diff,
        "topic": topic,
        "is_app": bool(getattr(tpl, "is_app", False)),
        "uid": "",
        "source": "GEO_TEMPLATE" if topic == "几何图像" else "ALG_TEMPLATE",
    }


def generate(per_template: int = 9, seed: int = 20260908) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out, seen = [], set()
    for tpl in TEMPLATES:
        got = 0
        for _ in range(per_template * 12):        # 重复采样直到凑够不重复题干
            if got >= per_template:
                break
            q = build_question(tpl, rng)
            if not q:
                continue
            key = q["stem"]
            if key in seen:
                continue
            seen.add(key)
            q["uid"] = f"{tpl.__name__}_{got}"
            out.append(q)
            got += 1
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=9)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    qs = generate(args.per_template, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"✔ 生成几何题 {len(qs)} 条 → {OUT}")
    from collections import Counter
    print("  模板分布：", len({q['uid'].rsplit('_', 1)[0] for q in qs}), "类")
    print("  难度分布：", sorted(Counter(q["difficulty"] for q in qs).items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
