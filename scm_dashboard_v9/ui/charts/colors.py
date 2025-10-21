"""차트 색상 관리 모듈.

SKU 및 센터별 색상 매핑, 색상 변환 유틸리티를 제공합니다.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

# 계단식 차트와 비슷한 톤의 기본 팔레트 (20+ 색상)
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

# Step 차트 전용 확장 팔레트
STEP_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",
    "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD",
    "#CCEBC5", "#FFED6F",
]

# 센터별 음영 계수 (밝기 조정)
CENTER_SHADE: Dict[str, float] = {
    "태광KR": 0.85,
    "AMZUS": 1.00,
    "CJ서부US": 1.15,
    "품고KR": 0.90,
    "AcrossBUS": 1.10,
    "SBSPH": 1.05,
    "SBSSG": 1.05,
    "SBSMY": 1.05,
}
DEFAULT_SHADE_STEP = 0.10


def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    """16진수 색상 코드를 RGB 튜플로 변환합니다.

    Args:
        hx: "#RRGGBB" 또는 "#RGB" 형식의 16진수 색상 코드

    Returns:
        (R, G, B) 튜플 (각 값은 0-255 범위)
    """
    hx = hx.lstrip("#")
    if len(hx) == 3:
        hx = "".join(ch * 2 for ch in hx)
    return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """RGB 튜플을 16진수 색상 코드로 변환합니다.

    Args:
        rgb: (R, G, B) 튜플 (각 값은 0-255 범위)

    Returns:
        "#RRGGBB" 형식의 16진수 색상 코드
    """
    r, g, b = [max(0, min(255, int(round(v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def tint(hex_color: str, factor: float) -> str:
    """기본 색상의 밝기를 조정합니다.

    factor가 1.0보다 크면 밝게, 작으면 어둡게 조정합니다.

    Args:
        hex_color: "#RRGGBB" 형식의 색상 코드
        factor: 밝기 조정 계수 (1.0 = 원본)

    Returns:
        조정된 색상의 16진수 코드
    """
    r, g, b = hex_to_rgb(hex_color)
    if factor >= 1.0:
        # 밝게: 흰색(255)에 가까워짐
        r = r + (255 - r) * (factor - 1.0)
        g = g + (255 - g) * (factor - 1.0)
        b = b + (255 - b) * (factor - 1.0)
    else:
        # 어둡게: 검정색(0)에 가까워짐
        r = r * factor
        g = g * factor
        b = b * factor
    return rgb_to_hex((r, g, b))


def shade_for(center: str, index: int) -> float:
    """센터 이름에 따라 적절한 음영 계수를 반환합니다.

    특정 센터는 고정된 밝기를 사용하고, 나머지는 인덱스에 따라 자동 계산됩니다.

    Args:
        center: 센터 이름
        index: 센터 인덱스 (같은 SKU의 여러 센터를 구분)

    Returns:
        밝기 조정 계수 (0.4~1.3 범위)
    """
    if center in CENTER_SHADE:
        return CENTER_SHADE[center]
    if index <= 0:
        return 1.0
    step = ((index + 1) // 2) * DEFAULT_SHADE_STEP
    if index % 2 == 1:
        return 1.0 + step
    return max(0.4, 1.0 - step)


def sku_colors(skus: Sequence[str], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """SKU별 고정 색상 매핑 딕셔너리를 생성합니다.

    Args:
        skus: SKU 코드 리스트
        base: 기존 매핑 딕셔너리 (있으면 유지)

    Returns:
        SKU → 색상 코드 매핑
    """
    cmap = {} if base is None else dict(base)
    i = 0
    for s in skus:
        if s not in cmap:
            cmap[s] = PALETTE[i % len(PALETTE)]
            i += 1
    return cmap


def sku_color_map(skus: Sequence[str]) -> Dict[str, str]:
    """SKU별 색상 매핑을 생성합니다 (간단 버전).

    Args:
        skus: SKU 코드 리스트

    Returns:
        SKU → 색상 코드 매핑
    """
    m: Dict[str, str] = {}
    for i, sku in enumerate([str(s) for s in skus]):
        if sku not in m:
            m[sku] = STEP_PALETTE[i % len(STEP_PALETTE)]
    return m


def step_sku_color_map(labels: list[str]) -> Dict[str, str]:
    """'SKU @ Center' 라벨에서 SKU만 추출하여 색상을 매핑합니다.

    계단식 차트에서 사용되며, 같은 SKU는 센터에 관계없이 같은 기본 색을 사용합니다.

    Args:
        labels: "SKU @ Center" 형식의 라벨 리스트

    Returns:
        SKU → 색상 코드 매핑
    """
    m: Dict[str, str] = {}
    i = 0
    for lb in labels:
        sku = lb.split(" @ ", 1)[0] if " @ " in lb else lb
        if sku not in m:
            m[sku] = STEP_PALETTE[i % len(STEP_PALETTE)]
            i += 1
    return m
