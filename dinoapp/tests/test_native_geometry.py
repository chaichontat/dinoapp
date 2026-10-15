import numpy as np

from dinoapp.native_geometry import mesh_vertex_normals, vertex_normals


def test_mesh_normals_preserve_face_orientation_while_prod2_faces_cavity():
    vertices = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1]])
    np.testing.assert_array_equal(mesh_vertex_normals(vertices, faces), [[-1., 0., 0.]] * 3)
    np.testing.assert_array_equal(vertex_normals(vertices, faces), [[1., 0., 0.]] * 3)
