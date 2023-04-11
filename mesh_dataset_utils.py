import numpy as np
import trimesh
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import robust_laplacian
import os

# ======================================
# ========= SPECTRAL EMBEDDING =========
# ======================================

# gr : nx Graph
# k : k-largest eigenvalues

# returns:
# ndarray with shape (k,)
# k eigenvalues

# ndarray with shape (N, k)
# for N = number of nodes in gr 
#     k = number of eigenvalues 
def laplacian_eigenmap(gr, k, lap=None):
    lap = nx.laplacian_matrix(gr)
    lap = lap.asfptype()
    eigs = scipy.sparse.linalg.eigsh(lap, k=k)
    e_vals = eigs[0]
    e_vecs = eigs[1]
    
    return e_vals, e_vecs


def mesh_laplacian_eigenmap(verts, faces, k, lap=None):
    L, M = robust_laplacian.mesh_laplacian(verts, faces)
    # L = L.asfptype()
    eigs = scipy.sparse.linalg.eigsh(L, k, M)
    e_vals = eigs[0]
    e_vecs = eigs[1]
    
    return e_vals, e_vecs

# ======================================
# ========= POINTWISE OPERATIONS =======
# ======================================

def get_dists(obj1, obj2):
    if obj1.vertices.shape[0] != obj2.vertices.shape[0]:
        raise Exception('objects are not one-to-one')
    
    offsets = np.zeros(obj1.vertices.shape[0])
    for i, face in enumerate(obj1.vertices):
        offsets[i] = np.linalg.norm(obj1.vertices[i]-obj2.vertices[i])
    return offsets

def get_offsets(obj1, obj2):
    if obj1.vertices.shape[0] != obj2.vertices.shape[0]:
        raise Exception('objects are not one-to-one')
    
    offsets = np.zeros((obj1.vertices.shape[0], 3))
    for i, face in enumerate(obj1.vertices):
        offsets[i] = obj1.vertices[i]-obj2.vertices[i]
    return offsets

# ======================================
# ============= SMOOTHING ==============
# ======================================

# smoothes a mesh using trimesh laplacian smoothing
def smooth_mesh_tm(mesh, iterations):
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_laplacian(smooth, iterations=iterations)
    print("laplacian")
    return smooth, orig
# smoothes a mesh using mutable diffusion laplacian smoothing
def smooth_mesh_mut_dif(mesh, iterations):
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_mut_dif_laplacian(smooth, iterations=iterations)
    print("mut dif")
    return smooth, orig
#smoothes a mesh using taubin smoothing
def smooth_mesh_taubin(mesh, iterations):
    orig = mesh
    smooth = mesh.copy()
    smooth = trimesh.smoothing.filter_taubin(smooth, iterations=iterations)
    print("taubin")

    return smooth, orig
# ======================================
# ========= DATASET BUILDERS ===========
# ======================================


def build_offset_dataset(mesh, smooth_iter=200, lap_type='mesh'):
    smooth, orig = smooth_mesh_mut_dif(mesh, iterations=smooth_iter)
    offsets = get_offsets(smooth, orig)
    
    if lap_type == 'mesh':
        e_vals, e_vecs = mesh_laplacian_eigenmap(np.array(smooth.vertices), np.array(smooth.faces), 100)
    else:
        # note that laplacian_eigenmap implementation is very inneficient at the moment. 
        # Pretty sure the slowness comes from using nx laplacian calculation which is really slow on larger graphs.
        gr = smooth.vertex_adjacency_graph
        e_vals, e_vecs = laplacian_eigenmap(gr, 100)
    
    dataset = {
        'fourier' : e_vecs,
        'points' : smooth.vertices,
        'target' : offsets      
          }
    return dataset, smooth

def build_blank_dataset(blank, DIR_NAME, lap_type='mesh'):
    if lap_type == 'mesh':
        e_vals, e_vecs = mesh_laplacian_eigenmap(np.array(blank.vertices), np.array(blank.faces), 100)
    else:
        gr = blank.vertex_adjacency_graph
        e_vals, e_vecs = laplacian_eigenmap(gr, 100)
    
    dataset = {
    'fourier' : e_vecs,
    'target' : np.zeros(blank.vertices.shape[0]),
    'points' : blank.vertices   
          }
    
    if not os.path.exists(f'{DIR_NAME}/npz_files'):
        os.makedirs(f'{DIR_NAME}/npz_files')
    np.savez(f'{DIR_NAME}/npz_files/{DIR_NAME}', fourier=dataset['fourier'], points=dataset['points'], target=dataset['target'])
    
    return f'/Users/maxperozek/GINR-texture/{DIR_NAME}'

# ======================================
# ========== INFERENCE MESH ============
# ======================================

def get_sphere(size=0):
    sphere = trimesh.primitives.Sphere()
    if size > 0:
        new_verts, new_faces = trimesh.remesh.subdivide_loop(sphere.vertices, sphere.faces, iterations=size)
        sphere = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
    return sphere

# ======================================
# ========= APPLY INFERENCE ============
# ======================================


# blank: trimesh mesh object
# offset_preds: np array of model predictions. shape (N,3) where N=number of vertices in blank mesh
def generate_pred_mesh(offset_preds, blank):
    pred_verts =  blank.vertices - offset_preds
    inference = trimesh.Trimesh(vertices=pred_verts, faces=blank.faces)
    return inference