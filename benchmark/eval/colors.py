"""Fixed color palettes for HSI classification maps.

Standard palettes used in the remote sensing community.
Each dataset maps class index (1-based) to an RGB tuple (0-255).
"""
from __future__ import annotations

import numpy as np


# ── Per-dataset palettes (class_id → RGB) ────────────────────────
# Index 0 is always background (black).

DATASET_COLORS: dict[str, list[tuple[int, int, int]]] = {
    "Trento": [
        (0, 0, 0),          # 0: background
        (0, 128, 0),        # 1: Apple trees
        (255, 0, 0),        # 2: Buildings
        (192, 192, 192),    # 3: Ground
        (34, 139, 34),      # 4: Woods
        (128, 0, 128),      # 5: Vineyard
        (64, 64, 64),       # 6: Roads
    ],
    "Houston2013": [
        (0, 0, 0),          # 0: background
        (0, 205, 0),        # 1: Healthy grass
        (127, 255, 0),      # 2: Stressed grass
        (46, 139, 87),      # 3: Synthetic grass
        (0, 100, 0),        # 4: Trees
        (160, 82, 45),      # 5: Soil
        (0, 0, 255),        # 6: Water
        (255, 0, 0),        # 7: Residential
        (255, 127, 80),     # 8: Commercial
        (128, 128, 128),    # 9: Road
        (169, 169, 169),    # 10: Highway
        (255, 215, 0),      # 11: Railway
        (255, 105, 180),    # 12: Parking lot 1
        (219, 112, 147),    # 13: Parking lot 2
        (0, 255, 255),      # 14: Tennis court
        (255, 69, 0),       # 15: Running track
    ],
    "MUUFL": [
        (0, 0, 0),          # 0: background
        (0, 100, 0),        # 1: Trees
        (0, 205, 0),        # 2: Mostly grass
        (160, 82, 45),      # 3: Mixed ground surface
        (210, 180, 140),    # 4: Dirt and sand
        (128, 128, 128),    # 5: Road
        (0, 0, 255),        # 6: Water
        (75, 0, 130),       # 7: Buildings shadow
        (255, 0, 0),        # 8: Buildings
        (192, 192, 192),    # 9: Sidewalk
        (255, 255, 0),      # 10: Yellow curb
        (255, 165, 0),      # 11: Cloth panels
    ],
    "Augsburg": [
        (0, 0, 0),          # 0: background
        (0, 100, 0),        # 1: Forest
        (255, 0, 0),        # 2: Residential area
        (255, 127, 80),     # 3: Industrial area
        (0, 205, 0),        # 4: Low plants
        (160, 82, 45),      # 5: Soil
        (127, 255, 0),      # 6: Allotment
        (255, 165, 0),      # 7: Commercial area
        (0, 0, 255),        # 8: Water
    ],
    "Houston2018": [
        (0, 0, 0),          # 0: background
        (0, 205, 0),        # 1: Healthy grass
        (127, 255, 0),      # 2: Stressed grass
        (46, 139, 87),      # 3: Artificial turf
        (0, 100, 0),        # 4: Evergreen trees
        (34, 139, 34),      # 5: Deciduous trees
        (160, 82, 45),      # 6: Bare earth
        (0, 0, 255),        # 7: Water
        (255, 0, 0),        # 8: Residential buildings
        (255, 69, 0),       # 9: Non-residential buildings
        (128, 128, 128),    # 10: Roads
        (192, 192, 192),    # 11: Sidewalks
        (255, 255, 0),      # 12: Crosswalks
        (169, 169, 169),    # 13: Major thoroughfares
        (105, 105, 105),    # 14: Highways
        (255, 215, 0),      # 15: Railways
        (255, 105, 180),    # 16: Paved parking lots
        (210, 180, 140),    # 17: Unpaved parking lots
        (0, 255, 255),      # 18: Cars
        (255, 165, 0),      # 19: Trains
        (75, 0, 130),       # 20: Stadium seats
    ],
    "IndianPines": [
        (0, 0, 0),          # 0: background
        (140, 67, 46),      # 1: Alfalfa
        (0, 0, 255),        # 2: Corn-notill
        (255, 100, 0),      # 3: Corn-mintill
        (0, 255, 123),      # 4: Corn
        (164, 75, 155),     # 5: Grass-pasture
        (101, 174, 255),    # 6: Grass-trees
        (118, 254, 172),    # 7: Grass-pasture-mowed
        (60, 91, 112),      # 8: Hay-windrowed
        (255, 255, 0),      # 9: Oats
        (255, 255, 125),    # 10: Soybean-notill
        (255, 0, 255),      # 11: Soybean-mintill
        (100, 0, 255),      # 12: Soybean-clean
        (0, 172, 254),      # 13: Wheat
        (0, 255, 0),        # 14: Woods
        (171, 175, 80),     # 15: Buildings-Grass-Trees-Drives
        (101, 193, 60),     # 16: Stone-Steel-Towers
    ],
    "PaviaU": [
        (0, 0, 0),          # 0: background
        (128, 128, 128),    # 1: Asphalt
        (0, 255, 0),        # 2: Meadows
        (255, 255, 0),      # 3: Gravel
        (0, 128, 0),        # 4: Trees
        (255, 0, 0),        # 5: Painted metal sheets
        (160, 82, 45),      # 6: Bare soil
        (0, 0, 0),          # 7: Bitumen  (dark)
        (255, 127, 80),     # 8: Self-blocking bricks
        (75, 0, 130),       # 9: Shadows
    ],
    "Salinas": [
        (0, 0, 0),          # 0: background
        (0, 100, 0),        # 1: Brocoli_green_weeds_1
        (0, 200, 0),        # 2: Brocoli_green_weeds_2
        (160, 82, 45),      # 3: Fallow
        (210, 180, 140),    # 4: Fallow_rough_plow
        (244, 164, 96),     # 5: Fallow_smooth
        (255, 255, 0),      # 6: Stubble
        (0, 255, 0),        # 7: Celery
        (128, 0, 128),      # 8: Grapes_untrained
        (255, 0, 255),      # 9: Soil_vinyard_develop
        (0, 255, 255),      # 10: Corn_senesced_green_weeds
        (255, 0, 0),        # 11: Lettuce_romaine_4wk
        (255, 69, 0),       # 12: Lettuce_romaine_5wk
        (255, 140, 0),      # 13: Lettuce_romaine_6wk
        (255, 215, 0),      # 14: Lettuce_romaine_7wk
        (0, 0, 255),        # 15: Vinyard_untrained
        (65, 105, 225),     # 16: Vinyard_vertical_trellis
    ],
    "Berlin": [
        (0, 0, 0),          # 0: background
        (0, 100, 0),        # 1: Forest
        (255, 0, 0),        # 2: Residential area
        (255, 127, 80),     # 3: Industrial area
        (0, 205, 0),        # 4: Low plants
        (160, 82, 45),      # 5: Soil
        (127, 255, 0),      # 6: Allotment
        (255, 165, 0),      # 7: Commercial area
        (0, 0, 255),        # 8: Water
    ],
    "WHU-Hi-LongKou": [
        (0, 0, 0),          # 0: background
        (255, 255, 0),      # 1: Corn
        (255, 255, 255),    # 2: Cotton
        (176, 48, 96),      # 3: Sesame
        (0, 255, 0),        # 4: Broad-leaf soybean
        (0, 128, 0),        # 5: Narrow-leaf soybean
        (0, 255, 255),      # 6: Rice
        (0, 0, 255),        # 7: Water
        (255, 0, 0),        # 8: Roads and houses
        (128, 128, 128),    # 9: Mixed weed
    ],
}


def get_colormap(dataset_name: str, num_classes: int) -> np.ndarray:
    """Return (num_classes+1, 3) uint8 array: index 0 = background.

    Falls back to a generated qualitative palette if dataset not in registry.
    """
    if dataset_name in DATASET_COLORS:
        colors = DATASET_COLORS[dataset_name]
        arr = np.array(colors[:num_classes + 1], dtype=np.uint8)
        return arr

    # Fallback: generate using HSV
    arr = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for i in range(1, num_classes + 1):
        hue = (i - 1) / num_classes
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        arr[i] = (int(r * 255), int(g * 255), int(b * 255))
    return arr


def label_map_to_rgb(label_map: np.ndarray, dataset_name: str,
                     num_classes: int) -> np.ndarray:
    """Convert (H, W) label map to (H, W, 3) RGB image.

    Args:
        label_map:    (H, W) int array, 0 = background, 1..C = classes.
        dataset_name: Name of dataset for color lookup.
        num_classes:  Total number of classes.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    cmap = get_colormap(dataset_name, num_classes)
    clipped = np.clip(label_map, 0, num_classes)
    return cmap[clipped]
