"""Geometry contracts for the Prod2 native reviewer."""

import json
from pathlib import Path

import numpy as np


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return area-weighted normals, oriented toward the +X cavity side."""
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Surface contains vertices with undefined normals")
    normals /= lengths[:, None]
    if np.median(normals[:, 0]) < 0:
        normals *= -1
    if np.any(normals[:, 0] <= 0):
        raise ValueError("Surface normals are not consistently oriented toward +X")
    return normals


def image_geometry(
    path: Path, level: str
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Validate the completed spatial ZYX pyramid contract and return its geometry."""
    root = json.loads((path / "zarr.json").read_text())["attributes"]
    if root.get("squisher_complete") is not True:
        raise ValueError(f"Image store is not marked complete: {path}")
    multiscales = root["ome"]["multiscales"]
    if len(multiscales) != 1:
        raise ValueError(f"Expected one multiscale image, found {len(multiscales)}")
    axes = multiscales[0]["axes"]
    axis_contract = [(axis["name"], axis["type"], axis["unit"]) for axis in axes]
    if axis_contract != [
        ("z", "space", "micrometer"),
        ("y", "space", "micrometer"),
        ("x", "space", "micrometer"),
    ]:
        raise ValueError(
            f"Expected spatial ZYX axes in micrometers, found {axis_contract}"
        )
    datasets = {item["path"]: item for item in multiscales[0]["datasets"]}
    if level not in datasets:
        raise ValueError(f"Pyramid level {level} is absent from {path}")
    transforms = datasets[level]["coordinateTransformations"]
    scale = [item["scale"] for item in transforms if item["type"] == "scale"]
    origin = [
        item["translation"] for item in transforms if item["type"] == "translation"
    ]
    if len(scale) != 1 or len(origin) != 1:
        raise ValueError(f"Level {level} must have exactly one scale and translation")
    array_meta = json.loads((path / level / "zarr.json").read_text())
    shape = tuple(array_meta["shape"])
    if len(shape) != 3 or array_meta["data_type"] != "uint16":
        raise ValueError(
            f"Expected a 3-D uint16 array, found {shape} {array_meta['data_type']}"
        )
    return (
        shape,
        np.asarray(scale[0], dtype=np.float64),
        np.asarray(origin[0], dtype=np.float64),
    )


def mesh_vertex_normals(vertices_xyz_um: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return unit area-weighted normals preserving the mesh face orientation."""
    vertices = np.asarray(vertices_xyz_um, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("surface contains vertices with undefined normals")
    return normals / lengths[:, None]

